from enum import Enum
import functools
import logging
import os
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable

from .utils import (
    MultiProcessAdapter,
    communicate,
    create_process_group,
)
from .utils.cuda_metering import EventRecorder, create_event_recorder

BROADCAST_BUCKET_SIZE = 10 * 1024 * 1024


class CO2BaseAlgorithm(str, Enum):
    LOCALSGD = "localsgd"

class CO2DistributedDataParallel(Module):
    """
    Example:
        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = fairscale.data_parallel.CO2DistributedDataParallel(model, nprocs_per_node=8)
        >>> loss = criterion(net(inputs), targets)
        >>> loss.backward()
        >>> optimizer.step()
        >>> net.perform_co2(optimizer)
    """

    def __init__(
        self,
        module: torch.nn.Module,
        nprocs_per_node: int,
        broadcast_buffers: bool = True,
        # CO2 Args
        co2_base_algorithm: CO2BaseAlgorithm = CO2BaseAlgorithm.LOCALSGD,
        outer_momentum: float = 0.5,
        outer_momentum_memory_efficient: bool = True,
        outer_frequency: int = 48,
        outer_lr: float = 1.0,
        outer_momentum_num_shards: int = 32,
        localsgd_frequency: int = 3,
        co2_clip: bool = False,
        co2_gap_penalty: Optional[float] = 1.0,
        co2_clip_threshold: Optional[float] = 1.0,
        co2_use_streams: bool = False,
        # Debugging Args
        verbose: bool = False,
        profile_mode: bool = False,
        # Args for advanced users (these are automatically handled otherwise)
        process_rank: Optional[int] = None,
        process_world_size: Optional[int] = None,
        global_group: Optional[torch.distributed.ProcessGroup] = None,
        master_group: Optional[torch.distributed.ProcessGroup] = None,
        local_node_group: Optional[torch.distributed.ProcessGroup] = None,
        comm_device: Optional[torch.device] = None,
    ) -> None:
        super(CO2DistributedDataParallel, self).__init__()

        # NCCL_BLOCKING_WAIT causes issues with using multiple process groups
        assert os.environ.get("NCCL_BLOCKING_WAIT", "0") == "0"

        assert nprocs_per_node >= 1
        self.nprocs_per_node = nprocs_per_node

        if process_world_size is None or process_rank is None:
            assert dist.is_initialized()
            process_rank = dist.get_rank()
            process_world_size = dist.get_world_size()
        assert process_world_size is not None and process_rank is not None
        self.process_rank = process_rank
        self.process_world_size = process_world_size

        self._initialize_logger(verbose, self.process_rank)

        # The logical prefix in the following variables denotes the variable value if nprocs_per_node processes
        # were treated as one process and then the following variables were calculated for the resulting process
        # group. This is how they are being treated for optimization purposes because intra-node communication is
        # very efficient with NVLink.
        logical_rank, logical_world_size = self._maybe_create_process_groups(
            self.process_rank, self.process_world_size, nprocs_per_node, global_group, master_group, local_node_group
        )
        self.logical_rank = logical_rank
        self.logical_world_size = logical_world_size

        self.module = module
        self.broadcast_buffers = broadcast_buffers
        first_param_dtype = next(self.module.parameters()).dtype

        # prepare local intra-node all-reduce objects
        self.broadcast_bucket_size = BROADCAST_BUCKET_SIZE  # bytes
        self.module_buffers = list(self.module.buffers())

        # choose communication device based on backend
        if comm_device is None:
            cpu_comm = dist.get_backend() == "gloo"
            comm_device = torch.device("cpu") if cpu_comm else torch.device("cuda")
        self._cpu_comm = comm_device.type == "cpu"

        # distributed backend config
        self.dist_config = {
            "verbose": verbose,
            "comm_device": comm_device,
            "logical_rank": logical_rank,
            "process_rank": self.process_rank,
            "logical_world_size": logical_world_size,
            "cpu_comm": self._cpu_comm,
        }
        self.profile_mode = profile_mode
        self.num_updates = 0
        self.portion_start: Optional[int] = None

        # CO2 being set to False is equivalent to outer_lr being set to 1 and outer_momentum being set to 0
        # This condition is ensuring the values are safe to use even when CO2 is disabled
        self.co2 = outer_lr != 1 or outer_momentum != 0

        self.outer_lr = outer_lr if self.co2 else 1
        self.outer_momentum = outer_momentum if self.co2 else 0

        self.outer_frequency = outer_frequency

        self.localsgd = co2_base_algorithm == CO2BaseAlgorithm.LOCALSGD

        self.localsgd_frequency = localsgd_frequency
        self.ef1: Optional[List[torch.Tensor]] = None
        self.global_momentum_buffers_initialized = False

        # Comment this block to use CO2 inside a node
        # if self.master_group is None:
        #     assert self.localsgd
        #     self.localsgd = False
        #     self.logger.warning("Disabling LocalSGD since a local allreduce will suffice")

        if self.co2 and not self.localsgd:
            self.logger.warning("CO2 is being used without LocalSGD")

        self.outer_momentum_memory_efficient = outer_momentum_memory_efficient
        self.outer_momentum_num_shards = min(self.process_world_size, outer_momentum_num_shards) if self.outer_momentum_memory_efficient else 1
        self.is_current_node_a_co2_shard = (
            self.process_rank < self.outer_momentum_num_shards if self.outer_momentum_memory_efficient else True
        )

        self.nprocs_per_node_device = torch.tensor([self.nprocs_per_node], device=comm_device, dtype=first_param_dtype)

        # register ps/grad-reduction hooks
        self._register_hooks()

        self.co2_clip = co2_clip
        self.co2_gap_penalty = co2_gap_penalty
        self.co2_clip_threshold = co2_clip_threshold
        self.co2_use_streams = co2_use_streams
        print('self.co2_use_streams', self.co2_use_streams)
        # import pdb; pdb.set_trace()

        if cast(torch.device, self.dist_config["comm_device"]).type != "cpu" and self.co2_use_streams:
            self.co2_stream = torch.cuda.Stream()
        else:
            self.co2_stream = torch.cuda.current_stream()
        
        self.co2_end = threading.Event()
        self.parameter_ready = threading.Event()
        self.co2_end.set()
        self.parameter_ready.clear()

        self.co2_thread = threading.Thread(
                target=self._CO2_AAR,
                args=(self.parameter_ready, self.co2_end, self.co2_stream))
        self.co2_thread.daemon = True
        self.co2_thread.name = 'CO2-Communication-Thread'
        self.co2_thread.start()

        self.logger.debug("Initialization of CO2DistributedDataParallel complete")

    def _initialize_logger(self, verbose: bool, process_rank: int) -> None:
        """Initializes the logger"""
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)

        # Only create an adapter if debug logging is enabled to avoid additional overhead
        if self.logger.isEnabledFor(logging.DEBUG):
            # Set custom adapter on top of logger
            self.logger = cast(logging.Logger, MultiProcessAdapter(self.logger, {"process_num": process_rank}))

    def _maybe_create_process_groups(
        self,
        process_rank: int,
        process_world_size: int,
        nprocs_per_node: int,
        global_group: Optional[torch.distributed.ProcessGroup],
        master_group: Optional[torch.distributed.ProcessGroup],
        local_node_group: Optional[torch.distributed.ProcessGroup],
    ) -> Tuple[int, int]:
        """Creates the process groups required for the CO2 implementation"""

        self.local_rank = process_rank % self.nprocs_per_node
        assert (
            process_world_size % self.nprocs_per_node == 0
        )  # total world size must be a multiple of `nprocs_per_node`
        logical_world_size = process_world_size // self.nprocs_per_node
        logical_rank = process_rank // self.nprocs_per_node

        self._maybe_initialize_global_group(global_group, process_world_size)
        self._maybe_initialize_local_node_group(local_node_group, process_rank, logical_world_size)
        self._maybe_initialize_master_group(master_group, process_rank, process_world_size, nprocs_per_node)

        self.logger.debug("Initialization of all process groups complete")
        return logical_rank, logical_world_size

    def _maybe_initialize_global_group(
        self, global_group: Optional[torch.distributed.ProcessGroup], process_world_size: int
    ) -> None:
        if global_group is None:
            all_processes = list(range(process_world_size))
            self.global_group = create_process_group(all_processes)
            self.logger.debug("Initialization of global group complete")
        else:
            self.global_group = global_group
        self.logger.debug("Global group set")
        self.process_group = self.global_group

    def _maybe_initialize_master_group(
        self,
        master_group: Optional[torch.distributed.ProcessGroup],
        process_rank: int,
        process_world_size: int,
        nprocs_per_node: int,
    ) -> None:
        if master_group is not None:
            self.master_group: Optional[torch.distributed.ProcessGroup] = master_group
            return

        if self.nprocs_per_node > 1:
            self.logger.debug("Initializing master process group")
            master_nodes = [i for i in range(process_world_size) if i % nprocs_per_node == 0]
            self.master_group = create_process_group(master_nodes) if len(master_nodes) > 1 else None
            if self.master_group is not None and process_rank in master_nodes:
                self.logger.debug("Initialization of master group complete")
        else:
            self.master_group = self.global_group

    def _maybe_initialize_local_node_group(
        self, local_node_group: Optional[torch.distributed.ProcessGroup], process_rank: int, logical_world_size: int
    ) -> None:
        if self.nprocs_per_node == 1:
            self.local_node_group = None
            return

        if local_node_group is not None:
            self.local_node_group = local_node_group
            return

        self.logger.debug("Initializing local process groups")
        for node in range(logical_world_size):
            node_processes_ranks = list(
                range(
                    node * self.nprocs_per_node,
                    (node + 1) * self.nprocs_per_node,
                )
            )
            # Process group to communicate between processes on this machine
            new_local_group = create_process_group(node_processes_ranks)
            if process_rank in node_processes_ranks:
                self.local_node_group = new_local_group
        assert self.local_node_group is not None
        self.logger.debug("Initialization of local groups complete")

    def forward(self, *inputs: Any, **kwargs: Any) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass performed in parallel across all devices on node"""
        return self.module(*inputs, **kwargs)

    def _sync_params(self) -> None:
        """Synchronize parameters across devices (intra-node)"""
        if self.local_node_group is None:
            return

        # intra-node parameter sync
        params = cast(List[torch.Tensor], list(self.module.parameters()))
        communication_op = functools.partial(
            dist.broadcast,
            src=self.logical_rank * self.nprocs_per_node,
            group=self.local_node_group,
        )
        communicate(params, communication_op)
        self.logger.debug("Intra-node param sync complete")

    def _sync_buffers(self) -> None:
        """Synchronize buffers across nodes"""
        # module buffer sync
        if self.broadcast_buffers and len(self.module_buffers) > 0:
            # Synchronize buffers across processes.
            # The process with rank 0 is considered the authoritative copy.
            self._distributed_broadcast_coalesced(self.process_group, self.module_buffers, self.broadcast_bucket_size)
        self.logger.debug("Intra-node buffer sync complete")

    def _distributed_broadcast_coalesced(
        self, process_group: torch.distributed.ProcessGroup, tensors: List[torch.Tensor], buffer_size: int
    ) -> None:
        dist._broadcast_coalesced(process_group, tensors, buffer_size)

    def _create_event_recorder(self, event_name: str) -> EventRecorder:
        """Creates an cuda event recorder which helps in profiling"""
        return create_event_recorder(event_name, dummy=not self.profile_mode)

    def _fp16_fp32_iterator(
        self, optimizer: torch.optim.Optimizer, fp32_params: Optional[torch.Tensor]
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterator for those fp16 parameters which have a fp32 copy"""
        # Handle apex fp16 optimizer
        if hasattr(optimizer, "_amp_stash") and hasattr(optimizer._amp_stash, "fp16_groups"):
            for p_fp16_group, p_fp32_group in zip(
                optimizer._amp_stash.fp16_groups,
                optimizer._amp_stash.fp32_from_fp16_groups,
            ):
                for p_fp16, p_fp32 in zip(p_fp16_group, p_fp32_group):
                    yield p_fp16, p_fp32

        # Handle fairseq fp16 optimizer
        elif fp32_params is not None:
            if isinstance(fp32_params, dict):
                fp32_params_list = list(fp32_params.values())
                assert len(fp32_params_list) == 1
                fp32_params = fp32_params_list[0]

            if isinstance(fp32_params, list):
                for p, fp32_param in zip(self.parameters(), fp32_params):
                    yield p.view(-1), fp32_param
            else:
                offset = 0
                for p in self.parameters():
                    yield p.view(-1), fp32_params[offset : offset + p.numel()]
                    offset += p.numel()

    def _should_perform_co2(self) -> bool:
        return self.co2 and (self.num_updates + 1) % self.outer_frequency == 0

    def _should_perform_localsgd(self) -> bool:
        return self.localsgd and (self.num_updates + 1) % self.localsgd_frequency == 0

    def _skip_averaging_memory_efficient_co2(self) -> bool:
        return self.outer_momentum_memory_efficient and self._should_perform_co2()

    def _should_use_error_feedback(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        return bool(fp16_fp32_list) and self._should_allreduce_params()

    def _should_allreduce_params(self) -> bool:
        # We do not all-reduce parameters with local SGD if a outer momentum step is
        # performed, since this step contains a reduce operation already. Note that this
        # also means there is no error feedback correction in that case: it is not needed
        # since communication within the outer momentum step happens in fp32.
        return (self._should_perform_localsgd() and not self._skip_averaging_memory_efficient_co2())

    def _maybe_pre_communicate_error_feedback(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        ef_rec = self._create_event_recorder("Error feedback")
        if self._should_use_error_feedback(fp16_fp32_list):
            with torch.no_grad():
                for p_fp16, p_fp32 in fp16_fp32_list:
                    if self._should_allreduce_params():
                        # This division and multiplication with the same number is done
                        # to ensure that we do not lose bits of information when we divide
                        # before the all_reduce. In order to preserve these bits in an
                        # error feedback (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1050.5040&rep=rep1&type=pdf)
                        # like manner, we are forcing the bits to be lost
                        # initially, and storing the lost information in error feedback
                        p_fp16.div_(self.logical_world_size)
                        p_fp16.mul_(self.logical_world_size)
                    p_fp32 -= p_fp16.float()

                if self.ef1 is not None:
                    for idx, (_, p_fp32) in enumerate(fp16_fp32_list):
                        p_fp32 += self.ef1[idx]
                        p_fp32.div_(2)
        ef_rec.stop()
        self.logger.debug("Error feedback completed")

    def _maybe_post_communicate_error_feedback(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        ef_unroll_rec = self._create_event_recorder("Sync and error feedback unroll rec")
        if self._should_use_error_feedback(fp16_fp32_list):
            # Error Feedback Reversal
            with torch.no_grad():
                for p, p_fp32 in fp16_fp32_list:
                    p_fp32 += p.float()
        ef_unroll_rec.stop()
        self.logger.debug("Error feedback unroll completed")

    def _maybe_allreduce(self) -> None:
        localsgd_rec = self._create_event_recorder("Localsgd communication time")
        if self._should_allreduce_params():
            communication_op = functools.partial(dist.all_reduce, group=self.master_group)
            params = cast(List[torch.Tensor], list(self.parameters()))
            with torch.no_grad():
                for p in params:
                    p.div_(self.logical_world_size)
            self.logger.debug("Params normalized before localsgd step")

            communicate(params, communication_op, self.logger)
            self.logger.debug("Allreduce completed")
        localsgd_rec.stop()

    def _maybe_sync_locally(self) -> None:
        if self._should_allreduce_params():
            self._sync_params()

    def _maybe_perform_co2(self, optimizer: torch.optim.Optimizer) -> None:
        co2_rec = self._create_event_recorder("CO2")
        if self._should_perform_co2():
            self.co2_end.wait()
            self._global_momentum_step(optimizer)
            self.co2_end.clear()
            self.parameter_ready.set()
        co2_rec.stop()
        self.logger.debug("Global momentum step completed")

    def _maybe_copy_back_fp32_parameters(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        ef_copy_rec = self._create_event_recorder("Error feedback copy back")
        if (
            self._should_allreduce_params() or self._should_perform_co2()
        ) and fp16_fp32_list:
            with torch.no_grad():
                for idx, (p_fp16, p_fp32) in enumerate(fp16_fp32_list):
                    p_fp16.copy_(p_fp32)
        ef_copy_rec.stop()
        self.logger.debug("Error feedback copy-back completed")

    def perform_co2(self, optimizer: torch.optim.Optimizer, fp32_params: Optional[torch.Tensor] = None) -> None:
        """This is to be called after optimizer.step(). It performs the approximate averaging using
        the base algorithm LocalSGD and the outer momentum step. Since LocalSGD and the outer
        momentum step are not performed every iteration, it only performs those when needed.

        It is recommended to call ``model.zero_grad(set_to_none=True)`` just before calling this function. This
        is because ``model.zero_grad(set_to_none=True)`` frees up the memory occupied by the gradients, some of which
        may be reused by this function.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer being used for training the model
            fp32_params (Optional[torch.Tensor]): To be used when performing fp16 training. Needs to be
                        set to the fp16 copy of the parameters (default: None)
        """
        # Done here in case the global momentum buffers have not been initialized by the caller.
        # In an ideal implementation, this would be called by the caller. We do it here instead of
        # waiting for it to happen in the global_momentum step function so that we store a copy of
        # the version of the parameters at iteration 0 and can use them for an outer momentum step later.
        if not self.global_momentum_buffers_initialized:
            self._init_global_momentum_buffers(optimizer)

        fp16_fp32_list = list(self._fp16_fp32_iterator(optimizer, fp32_params))
        self.logger.debug("Created a list of fp16 and fp32 corresponding parameters")

        self.logger.debug(
            "Booleans set. Values - self._should_perform_co2()=%r, self._should_perform_localsgd()=%r, self._should_allreduce_params()=%r",
            self._should_perform_co2(),
            self._should_perform_localsgd(),
            self._should_allreduce_params(),
        )
        self.logger.debug("Step number(0-indexed)=%d", self.num_updates)

        if (
            self.num_updates == 0
            and fp32_params is None
            and not hasattr(optimizer, "_amp_stash")
            and any(p.dtype == torch.float16 for p in self.parameters())
        ):
            self.logger.warning("WARNING: please set fp32_params in perform_co2() in order to avoid accuracy loss")

        self._maybe_pre_communicate_error_feedback(fp16_fp32_list)
        # Here the asynchronize communication is moved into communication thread
        self._maybe_post_communicate_error_feedback(fp16_fp32_list)
        self._maybe_perform_co2(optimizer)
        self._maybe_copy_back_fp32_parameters(fp16_fp32_list)

        self.num_updates += 1

    def _init_global_momentum_buffers(self, optimizer: torch.optim.Optimizer) -> None:
        """Initializes the outer momentum buffers"""
        self.global_momentum_buffers_initialized = True

        if not self.co2:
            return

        total_elements = 0
        params_dtype = None
        for group in optimizer.param_groups:
            for p in group["params"]:
                total_elements += p.numel()

                # Assert that all parameters have the same device and dtype
                if params_dtype is None:
                    params_dtype, params_device = p.dtype, p.device
                # Check that dtype is fp32 since outer mometum is to be performed in fp32
                assert p.dtype == params_dtype == torch.float32
                assert p.device == params_device

        self.world_portion_length = (total_elements + self.outer_momentum_num_shards - 1) // self.outer_momentum_num_shards

        if not self.is_current_node_a_co2_shard:
            return

        self.portion_start = self.process_rank * self.world_portion_length if self.outer_momentum_memory_efficient else 0
        self.portion_end = (
            min((self.process_rank + 1) * self.world_portion_length, total_elements)
            if self.outer_momentum_memory_efficient
            else total_elements
        )

        self.old_params = torch.empty(self.world_portion_length, dtype=params_dtype).to(params_device).detach()

        # copy params to old_params to initialize old_params
        offset = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                numel = p.numel()

                if offset + numel > self.portion_start and offset < self.portion_end:

                    # start and end for each
                    overall_start = max(self.portion_start, offset)
                    overall_end = min(self.portion_end, offset + numel)

                    p_start = overall_start - offset
                    p_end = overall_end - offset

                    buffer_start = overall_start - self.portion_start
                    buffer_end = overall_end - self.portion_start

                    # let's see size of p and split based on that
                    current_p = p.view(-1)[p_start:p_end]
                    current_p_old = self.old_params[buffer_start:buffer_end]

                    current_p_old.copy_(current_p)

                offset += numel

        self.global_momentum_buffer = torch.zeros_like(self.old_params).detach()

    def _distributed_comm(self, optimizer: torch.optim.Optimizer, mode: str) -> None:
        """Performs the communication needed for the efficient CO2 implementation"""
        offset = 0
        co2_comm_lists: List[List[torch.Tensor]] = [[] for _ in range(self.outer_momentum_num_shards)]
        with torch.no_grad():
            for group in optimizer.param_groups:
                # aggregate different parts of p in required node
                for p in group["params"]:
                    numel = p.numel()

                    # gather has a reduce operation so division by world size is needed
                    if mode == "gather":
                        p /= self.process_world_size

                    current_start = offset
                    while current_start < offset + numel:
                        main_node = current_start // self.world_portion_length

                        main_node_end = (main_node + 1) * self.world_portion_length
                        current_end = min(offset + numel, main_node_end)

                        p_start = current_start - offset
                        p_end = current_end - offset

                        co2_comm_lists[main_node].append(p.view(-1)[p_start:p_end])

                        current_start = current_end
                    offset += numel

            for co2_rank, co2_comm_list in enumerate(co2_comm_lists):
                if mode == "gather":
                    communication_op = functools.partial(dist.reduce, dst=co2_rank)
                elif mode == "scatter":
                    communication_op = functools.partial(dist.broadcast, src=co2_rank)
                communicate(co2_comm_list, communication_op)

    def _global_momentum_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Performs the outer momentum step"""
        if not self.co2:
            return

        if not self.global_momentum_buffers_initialized:
            self._init_global_momentum_buffers(optimizer)

        if self.outer_momentum_memory_efficient:
            self._distributed_comm(optimizer, mode="gather")

        if self.is_current_node_a_co2_shard:
            self._perform_local_optimization(optimizer)

        if self.outer_momentum_memory_efficient:
            self._distributed_comm(optimizer, mode="scatter")
            
    def _co2_clip(self, input: torch.Tensor, clip_threshold: float) -> None:
        torch.clamp(input, min=-clip_threshold, max=clip_threshold)

    def _perform_local_optimization(self, optimizer: torch.optim.Optimizer) -> None:
        """Performs the outer momentum on the local shard"""
        assert self.portion_start is not None

        with torch.no_grad():
            offset = 0
            for group in optimizer.param_groups:
                for p in group["params"]:
                    numel = p.numel()

                    if offset + numel > self.portion_start and offset < self.portion_end:

                        # start and end for each
                        overall_start = max(self.portion_start, offset)
                        overall_end = min(self.portion_end, offset + numel)

                        p_start = overall_start - offset
                        p_end = overall_end - offset

                        buffer_start = overall_start - self.portion_start
                        buffer_end = overall_end - self.portion_start

                        # let's see size of p and split based on that
                        current_p = p.view(-1)[p_start:p_end]
                        current_p_gmb = self.global_momentum_buffer[buffer_start:buffer_end]
                        current_p_old = self.old_params[buffer_start:buffer_end]

                        current_p_gmb.mul_(self.outer_momentum).sub_(current_p, alpha=1 / self.co2_gap_penalty).add_(
                            current_p_old, alpha=1 / self.co2_gap_penalty
                        )
                        if self.co2_clip:
                            self._co2_clip(current_p_gmb, self.co2_clip_threshold)
                        current_p_old.add_(current_p_gmb, alpha=-self.outer_lr)
                        current_p.copy_(current_p_old)

                    offset += numel

    def _register_hooks(self) -> None:
        """
        Registers push-sum de-bias/bias hooks in pre-forward/post-backward
        passes in all leaf modules
        """
        self.register_forward_pre_hook(self.__make_forward_pre_hook())
        self.register_backward_hook(self.__make_backward_hook())

    def __make_backward_hook(self) -> Callable[..., None]:
        self.logger.debug("making backward hook")

        def hook(*unused: Any) -> None:
            # reduce gradients across devices on a single machine
            if self.local_node_group is not None:
                grads = []
                for p in self.module.parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    p.grad.div_(self.nprocs_per_node)
                    grads.append(p.grad)
                self.logger.debug("Gradients ready for syncing")

                communication_op = functools.partial(dist.all_reduce, group=self.local_node_group)
                communicate(grads, communication_op, self.logger)
                self.logger.debug("Gradient sync during backward pass in local_group complete")

        def queue_hook(*unused: Any) -> None:
            Variable._execution_engine.queue_callback(hook)

        return queue_hook

    def __make_forward_pre_hook(self) -> Callable[..., None]:
        self.logger.debug("making forward pre-hook")

        def hook(*unused: Any) -> None:
            """Query gossip queue and de-bias during forward pass"""
            # sync buffers before the forward pass
            self._sync_buffers()

        return hook

    def _CO2_AAR(self, parameter_ready, co2_end, co2_stream):
        while True:
            with torch.cuda.stream(co2_stream):
                parameter_ready.wait()
                self._maybe_allreduce()
                self._maybe_sync_locally()
                parameter_ready.clear()
                co2_end.set()
