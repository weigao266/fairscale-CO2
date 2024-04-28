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
