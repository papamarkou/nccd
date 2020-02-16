# Installation of packages required by the project

## Setup conda environment

```
conda create -n torch_gpu python=3.6
```

## Install PyTorch via conda

```
conda activate torch_gpu
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

## Test PyTorch GPU access from Python REPL

```
import torch
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
```

## Install other dependencies

```
conda install -c fastai fastai
conda install -c anaconda scikit-learn
```

## Install nccd from github

You may `git clone` at a directory of your choice.

```
git clone git@github.com:papamarkou/nccd.git
cd nccd

python setup.py develop --user
```
