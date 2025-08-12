```bazel run //:requirements_lock.update```
```bazel test //src/...```

# Repository setup

Run `python3 ./collect_pytorch_cuda_libs.py` to collect runtime dependencies of Pytorch. 
It is a hack to make it possible to use Pytorch with CUDA in Bazel environment.

# Mamba SSM

Mamba-SSM from pypi didn't work out of the box. Solution is to build Mamba from source github 
project into virtual python environemnt. 

It is important that:
  - Mamba is built using the same Python version as the one configured in the project
  - Mamba is built with the same Pytorch + CUDA combination as in the hermetic project environment   

In practice, easiest way to achieve this:
  - Match project's Python version to the system Python version
  - Run requirements lock update and inspect generated file to see version of Pytorch. You will see e.g. `torch==2.6.0+cu126`, which means Pytorch is 2.6.0 with CUDA 12.6
  - Install CUDA with the version above to your system
  - Create Python venv using your system Python
  - Install Pytorch with the version above to the venv
  - Build isolated Mamba wheel from inside venv, e.g. `cd mamba && pip wheel . --no-deps -w dist/`
  - Unpack the wheel to `third_party` folder
  - Create Bazel BUILD file with two `py_binary` targets:

```
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "selective_scan_cuda",
    srcs = [],
    data = ["selective_scan_cuda.cpython-312-x86_64-linux-gnu.so"],
    imports = ["."],
    visibility = ["//visibility:public"],
)

py_library(
    name = "mamba_ssm",
    srcs = glob([
        "mamba_ssm/**/*.py",
    ]),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [":selective_scan_cuda"],
)
```
# Tensorboard

```
source .venv/bin/activate
tensorboard --logdir=/PATH/TO/LOGDIR
```