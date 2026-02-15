# CS336 Spring 2025 Assignment 2: Systems

### 完成代码
``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO: code that you'll write for assignment 2 
│   ├── __init__.py
│   ├── benchmarking_script.py  # End-to-end enchmarking of the forward and backward passes.   
│   ├── benchmarking_nvtx.py  # Memory profiles.
│   ├── submit_benchmark.py  # Call benchmarking_script.py, sweep.
│   ├── benchmarking_flash.py  # Call FlashAttention_v2.py, benchmarking script for flashattention.
│   ├── FlashAttention_v2.py  # (1). Pure PyTorch implementation of FlashAttention-2; (2). and using Triton kernels.
│   ├── benchmark_allreduce.py  # A simple bechmarking for all-reduce operation; Gloo + CPU, NCCL + GPU.
│   ├── naive_dpp.py  # A simple test for naively Distributed Data Parallel (DDP) training by all-reducing individual parameter gradients after the backward pass.
│   ├── benchmarking_naive_dpp.py  # .
│   ├── ddp_utils.py  # Implementation of (1). DDP with communication call for each parameter; (2). DDP with single communication call for all parameter.
│   ├── dpp_overlap.py  # Implementation of DDP, overlapping computation with communication; (1). overlap_individual_parameters; (2). overlap_bucketed_parameters.
│   ├── benchmarking_dpp_overlap.py  # .
│   ├── optimizer_state_sharding.py  # A simple implementation of optimizer state sharding.
│   ├── optimizer_state_sharding_accounting.py  # accounting the peak memory usage when training language models with and without optimizer state shardin.
│   └── .
├── README.md
├── pyproject.toml

```


For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
