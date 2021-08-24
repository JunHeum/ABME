# How to set the nvcc version (Default:CUDA-11.0)

## Pre-requisite

Check installed `cuda-11.0` path:
```bash
  $ cd /usr/local
  $ find . -maxdepth 1 -name 'cuda-11.0'
```
- If there is no `cuda-11.0` folder in the directory, install `cuda-11.0` first.

## Change environments of the terminal (temporal)
Change your terminal `PATH` and `LD_LIBRARY_PATH`:
```bash
  $ export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
  $ export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64
  $ nvcc --version
```
## Change default environments of the terminal
For change default `nvcc` version of your terminal, you should add below two lines in your `~/.bashrc`.
```bash
  $ gedit ~/.bashrc
  export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64
```
Save and then open new terminal:
```bash
  $ nvcc --version
```
