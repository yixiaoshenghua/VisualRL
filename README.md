<img src="imgs/logo2.png" align="right" width="40%"/>

# VisualRL: A Toolbox for Visual Reinforcement Learning
![CI](https://github.com/DLR-RM/stable-baselines3/workflows/CI/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines3.readthedocs.io/en/master/?badge=master) 
[![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/tianshou)](https://pypi.org/project/tianshou/) 
[![Conda](https://img.shields.io/conda/vn/conda-forge/tianshou)](https://github.com/conda-forge/tianshou-feedstock) 
[![Unittest](https://github.com/thu-ml/tianshou/workflows/Unittest/badge.svg?branch=master)](https://github.com/thu-ml/tianshou/actions) [![GitHub issues](https://img.shields.io/github/issues/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/issues) 
[![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers) 
[![GitHub forks](https://img.shields.io/github/forks/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/network) 
[![GitHub license](https://img.shields.io/github/license/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/blob/master/LICENSE)


👋 Welcome to VisualRL, perhaps the **first** toolbox for visual reinforcement learning with the **most** implemeted methods in PyTorch. 

If you use any content of this repo for your work, please cite:

```
[some paper]
```

## Introduction

Visual reinforcement learning focuses on addressing the challenge of training agents in reinforcement learning tasks where the input information is visual, typically in the form of image or video frames. In traditional reinforcement learning, agents usually receive structed and numerical input data, such as state vectors. However, in many real-world scenarios, the input is high-dimensional visual information, instead of states. Visual RL
aims to enable agents to learn and make decisions from visual input.

## Methods Implemented

Model-based:

- **BIRD:** Bridging Imagination and Reality for Model-Based Deep Reinforcement Learning. NeurIPS 2020. [[paper](https://arxiv.org/abs/2010.12142)]
- **Dreamer:** 
  - **Dreamerv1:** Dream to Control: Learning Behaviors by Latent Imagination. ICLR 2020 [[paper](https://arxiv.org/abs/1912.01603)]
  - **Dreamerv2:** 
- **PLANET:** Learning Latent Dynamics for Planning from Pixels
. ICML 2019. [[paper](https://arxiv.org/abs/1811.04551)]
- **TIA:** Learning Task Informed Abstractions. PMLR 2021. [[paper](https://arxiv.org/abs/2106.15612)]
- **World Model:**

Model-free:

- **Baseline model:**
- **CURL:** CURL: Contrastive Unsupervised Representations for Reinforcement Learning. PMLR 2020. [[paper](https://arxiv.org/abs/2004.04136)]
- **DBC:** Learning Invariant Representations for Reinforcement Learning without Reconstruction. ICLR 2021. [[paper](https://arxiv.org/abs/2006.10742)]
- **DeepMDP:** DeepMDP: Learning Continuous Latent Space Models for Representation Learning. PMLR 2019. [[paper](https://arxiv.org/abs/1906.02736)]
- **DRIBO:** DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck. PMLR 2022. [[paper](https://arxiv.org/abs/2102.13268)]
- **DRQ:** Regularizing Deep Reinforcement Learning from Pixels. ICLR 2021. [[paper](https://arxiv.org/pdf/2004.13649.pdf)]
- **FLARE:** Reinforcement Learning with Latent Flow. NeurIPS 2021. [[paper](https://arxiv.org/abs/2101.01857)]
- **RAD:** Reinforcement Learning with Augmented Data. NeurIPS 2020. [[paper](https://arxiv.org/abs/2004.14990)]
- **SAC+AE:** Improving Sample Efficiency in Model-Free Reinforcement Learning from Images. AAAI 2021. [[paper](https://arxiv.org/pdf/1910.01741.pdf)]

## Reproduced Result

## How to use


<!-- **Improving Sample Efficiency in Model-Free Reinforcement Learning from Images** by

[Denis Yarats](https://cs.nyu.edu/~dy1042/), [Amy Zhang](https://mila.quebec/en/person/amy-zhang/), [Ilya Kostrikov](https://github.com/ikostrikov), [Brandon Amos](http://bamos.github.io/), [Joelle Pineau](https://www.cs.mcgill.ca/~jpineau/), [Rob Fergus](https://cs.nyu.edu/~fergus/pmwiki/pmwiki.php).

[[Paper]](https://arxiv.org/abs/1910.01741) [[Webpage]](https://sites.google.com/view/sac-ae/home) -->

<!-- ## Citation
If you use this repo in your research, please consider citing the paper as follows
```

```

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with:
```
source activate pytorch_sac_ae
```

## Instructions
To train an SAC+AE agent on the `cheetah run` task from image-based observations  run:
```
python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir ./log \
    --seed 1
```
This will produce 'log' folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir log
```
and opening up tensorboad in your browser.

The console output is also available in a form:
```
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000
```
a training entry decodes as:
```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
RLOSS - average reconstruction loss (only if is trained from pixels and decoder)
```
while an evaluation entry:
```
| eval | S: 0 | ER: 21.1676
```
which just tells the expected reward `ER` evaluating current policy after `S` steps. Note that `ER` is average evaluation performance over `num_eval_episodes` episodes (usually 10). -->

<!-- ## Results
Our method demonstrates significantly improved performance over the baseline SAC:pixel. It matches the state-of-the-art performance of model-based algorithms, such as PlaNet (Hafner et al., 2018) and SLAC (Lee et al., 2019), as well
as a model-free algorithm D4PG (Barth-Maron et al., 2018), that also learns from raw images. Our
algorithm exhibits stable learning across ten random seeds and is extremely easy to implement.
![Results](results/graph.png) -->
