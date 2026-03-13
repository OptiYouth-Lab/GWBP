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

## Build and Run

### 1.Compile the CPU Version(baseline)
```bash
cd baseline
mkdir build
cd build
cmake ../
make -j8
```

### 2.Compile the GPU Version(eg.GWBP-O)
```bash
cd GWBP-O
make -f Makefile.gpu clean
make -f Makefile.gpu tomo_gpu
```

### 3.Run Reconstruction
Before running, update the executable paths in `run.sh` and `check.sh` to match the locations of the compiled `tomo_gpu` and `validate_cpu` binaries in your local environment.
#### 3.1 Run on proteasome-bin6
```bash
cd ../data/proteasome-bin6
bash run.sh
bash check.sh
```
#### 3.2 Run on proteasome-bin6
```bash
cd ../data/TS_038_WBP_bin6
bash run.sh
bash check.sh
```










