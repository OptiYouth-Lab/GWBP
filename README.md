# GWBP: Accelerating Weighted Back-Projection via Efficient GPU Parallel Optimization

This repository contains the implementation of **GWBP**, a GPU-accelerated weighted back-projection (WBP) pipeline for tomographic reconstruction.  
GWBP is designed to preserve the semantics and output conventions of an existing WBP workflow while improving end-to-end performance through GPU kernelization, system-level bottleneck reduction, and overlapped execution.

## Software Environment

- GCC 10.5.0
- ICC 2021.1 Beta
- MKL 2021.0 Update 1

The code has been tested on V100 and A100 platforms.

| Attribute | Platform 1 | Platform 2 |
|---|---|---|
| Operating system | Ubuntu 22.04.5 | Ubuntu 24.04.3 |
| CPU model | Xeon Gold 6542Y | Xeon Gold 6530 |
| CPU physical cores | 48 | 64 |
| CPU threads | 96 | 128 |
| GPU model | V100-PCIE-32GB | A100-SXM4-40GB |
| GPU memory | 32 GB | 40 GB |

## Result

![image text](https://github.com/OptiYouth-Lab/GWBP/blob/main/img/runtime_speedup_2x2_singlecol.png)

