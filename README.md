# Fairscale-CO2

This repository shows an simple implementation version of CO2 within [Fairscale](https://github.com/facebookresearch/fairscale).

--------------------------------------------------------------------------------

# Requirements and Installation

* PyTorch>= 1.8.1


Installing `fairscale-CO2` from source:

``` bash
git clone https://github.com/weigao266/fairscale-CO2.git
cd fairscale-CO2
pip install -e .
```

To build with GPU-support enabled, be sure to set ``BUILD_CUDA_EXTENSIONS=1``
as well as an appropriate ``TORCH_CUDA_ARCH_LIST``.

Note: If the above installation fails, add ``--no-build-isolation`` to the ``pip install`` command.


# Usage
If you have code that is setup to use Distributed Data Parallel (DDP), using CO2 Distributed Data Parallel
is simply replacing the DDP call with a call to
``fairscale.experimental.nn.data_parallel.CO2DistributedDataParallel``, and adding a
``model.perform_co2(optimizer)`` call after ``optimizer.step()`` -- preceded by
``model.zero_grad(set_to_none=True)`` in order to reduce peak memory usage. Like:


``` python
import torch
from fairscale.experimental.nn.data_parallel import CO2DistributedDataParallel


def train(
    rank: int,
    world_size: int,
    epochs: int,
    use_slowmo: bool):

    # process group init
    dist_init(rank, world_size)

    # Problem statement
    model = MyAwesomeModel().to(rank)
    if use_co2:
        # Wrap the model into CO2DistributedDataParallel
        model = CO2DistributedDataParallel(model, outer_momentum=0.2, nprocs_per_node=8)
    else:
        model = DDP(model, device_ids=[rank])

    dataloader = MySuperFastDataloader()
    loss_ln = MyVeryRelevantLoss()
    optimizer = MyAmazingOptimizer()

    # Any relevant training loop, with a line at the very end specific to CO2DistributedDataParallel, e.g.:
    model.train()
    for e in range(epochs):
        for (data, target) in dataloader:
            data, target = data.to(rank), target.to(rank)
            # Train
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=use_co2)  # free memory for the perform_co2() call below
            if use_co2:
                model.perform_co2(optimizer)
```

In the example above, when using CO2DistributedDataParallel, the communication of parameter all-reducing is overlapped with local computation steps (default 3 as setting by ``localsgd_frequency``). Users are able to tune it for a tradeoff between the convergence/generalization performance and overlapping ratio. With a big enough setting of ``localsgd_frequency``, the communication is able to be fully overlapped by local computation steps.

CO2DistributedDataParallel takes in ``outer_momentum`` as a parameter. This parameter may need to be tuned
depending on your use case. It also takes in ``nproces_per_node`` which should be typically set
to the number of GPUs on a node.

# Citation
If you find our work useful, please cite the following paper:

``` bibtex
@article{sun2024co2,
  title={CO2: Efficient Distributed Training with Full Communication-Computation Overlap},
  author={Sun, Weigao and Qin, Zhen and Sun, Weixuan and Li, Shidi and Li, Dong and Shen, Xuyang and Qiao, Yu and Zhong, Yiran},
  journal={arXiv preprint arXiv:2401.16265},
  year={2024}
}
```
