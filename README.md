# Differentiable Factor Graph Optimization for Learning Smoothers

[![mypy](https://github.com/brentyi/dfgo/actions/workflows/mypy.yml/badge.svg)](https://github.com/brentyi/dfgo/actions/workflows/mypy.yml)

![Figure describing the overall training pipeline proposed by our IROS paper. Contains five sections, arranged left to right: (1) system models, (2) factor graphs for state estimation, (3) MAP inference, (4) state estimates, and (5) errors with respect to ground-truth. Arrows show how gradients are backpropagated from right to left, starting directly from the final stage (error with respect to ground-truth) back to parameters of the system models.](./data/paper_figure1.png)

<!-- vim-markdown-toc GFM -->

* [Overview](#overview)
* [Status](#status)
* [Setup](#setup)
* [Datasets](#datasets)
* [Training](#training)
* [Evaluation](#evaluation)
* [Acknowledgements](#acknowledgements)

<!-- vim-markdown-toc -->

## Overview

Code release for our IROS 2021 conference paper:

<table><tr><td>
    Brent Yi<sup>1</sup>, Michelle A. Lee<sup>1</sup>, Alina Kloss<sup>2</sup>,
    Roberto Mart&iacute;n-Mart&iacute;n<sup>1</sup>, and Jeannette
    Bohg<sup>1</sup>.
    <strong>
        <a href="https://sites.google.com/view/diffsmoothing">
            Differentiable Factor Graph Optimization for Learning Smoothers.
        </a>
    </strong>
    Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), October 2021.
</td></tr></table>

<sup>1</sup><em>Stanford University,
`{brentyi,michellelee,robertom,bohg}@cs.stanford.edu`</em><br />
<sup>2</sup><em>Max Planck Institute for Intelligent Systems,
`akloss@tue.mpg.de`</em>

---

This repository contains models, training scripts, and experimental results, and
can be used to either reproduce our results or as a reference for implementation
details.

Significant chunks of the code written for this paper have been factored out of
this repository and released as standalone libraries, which may be useful for
building on our work. You can find each of them linked here:

- **[jaxfg](https://github.com/brentyi/jaxfg)** is our core factor graph
  optimization library.
- **[jaxlie](https://github.com/brentyi/jaxlie)** is our Lie theory library for
  working with rigid body transformations.
- **[jax_dataclasses](https://github.com/brentyi/jax_dataclasses)** is our
  library for building JAX pytrees as dataclasses. It's similar to
  `flax.struct`, but has workflow improvements for static analysis and nested
  structures.
- **[jax-ekf](https://github.com/brentyi/jax-ekf)** contains our EKF
  implementation.

## Status

Included in this repo for the disk task:

- [x] Smoother training & results
  - [x] Training: `python train_disk_fg.py --help`
  - [x] Evaluation:
        `python cross_validate.py --experiment-paths ./experiments/disk/fg/**/`
- [x] Filter baseline training & results
  - [x] Training: `python train_disk_ekf.py --help`
  - [x] Evaluation:
        `python cross_validate.py --experiment-paths ./experiments/disk/ekf/**/`
- [x] LSTM baseline training & results
  - [x] Training: `python train_disk_lstm.py --help`
  - [x] Evaluation:
        `python cross_validate.py --experiment-paths ./experiments/disk/lstm/**/`

And, for the visual odometry task:

- [x] Smoother training & results (including ablations)
  - [x] Training: `python train_kitti_fg.py --help`
  - [x] Evaluation:
        `python cross_validate.py --experiment-paths ./experiments/kitti/fg/**/`
- [x] EKF baseline training & results
  - [x] Training: `python train_kitti_ekf.py --help`
  - [x] Evaluation:
        `python cross_validate.py --experiment-paths ./experiments/kitti/ekf/**/`
- [x] LSTM baseline training & results
  - [x] Training: `python train_kitti_lstm.py --help`
  - [x] Evaluation:
        `python cross_validate.py --experiment-paths ./experiments/kitti/lstm/**/`

Note that `**/` indicates a recursive glob in zsh. This can be emulated in
bash>4 via the globstar option (`shopt -q globstar`).

We've done our best to make our research code easy to parse, but it's still
being iterated on! If you have questions, suggestions, or any general comments,
please reach out or file an issue.

## Setup

We use Python 3.8 and miniconda for development.

1. Any calls to CHOLMOD (via `scikit-sparse`, sometimes used for eval but never
   for training itself) will require SuiteSparse:

   ```bash
   # Mac
   brew install suite-sparse

   # Debian
   sudo apt-get install -y libsuitesparse-dev
   ```

2. Dependencies can be installed via pip:

   ```bash
   pip install -r requirements.txt
   ```

   In addition to JAX and the first-party dependencies listed above, note that
   this also includes various other helpers:

   - **[torch](https://github.com/pytorch/pytorch)**'s `Dataset` and
     `DataLoader` interfaces are used for training.
   - **[fannypack](https://github.com/brentyi/fannypack)** contains some
     utilities for working with hdf5 files.

The `requirements.txt` provided will install the CPU version of JAX by default.
For CUDA support, please see [instructions](http://github.com/google/jax) from
the JAX team.

## Datasets

Datasets synced from Google Drive and loaded via [h5py](https://www.h5py.org/)
automatically as needed. If you're interested in downloading them manually, see
[`lib/kitti/data_loading.py`](lib/kitti/data_loading.py) and
[`lib/disk/data_loading.py`](lib/disk/data_loading.py).

## Training

The naming convention for training scripts is as follows:
`train_{task}_{model type}.py`.

All of the training scripts provide a command-line interface for configuring
experiment details and hyperparameters. The `--help` flag will summarize these
settings and their default values. For example, to run the training script for
factor graphs on the disk task, try:

```bash
> python train_disk_fg.py --help

Factor graph training script for disk task.

optional arguments:
  -h, --help            show this help message and exit
  --experiment-identifier EXPERIMENT_IDENTIFIER
                        (default: disk/fg/default_experiment/fold_{dataset_fold})
  --random-seed RANDOM_SEED
                        (default: 94305)
  --dataset-fold {0,1,2,3,4,5,6,7,8,9}
                        (default: 0)
  --batch-size BATCH_SIZE
                        (default: 32)
  --train-sequence-length TRAIN_SEQUENCE_LENGTH
                        (default: 20)
  --num-epochs NUM_EPOCHS
                        (default: 30)
  --learning-rate LEARNING_RATE
                        (default: 0.0001)
  --warmup-steps WARMUP_STEPS
                        (default: 50)
  --max-gradient-norm MAX_GRADIENT_NORM
                        (default: 10.0)
  --noise-model {CONSTANT,HETEROSCEDASTIC}
                        (default: CONSTANT)
  --loss {JOINT_NLL,SURROGATE_LOSS}
                        (default: SURROGATE_LOSS)
  --pretrained-virtual-sensor-identifier PRETRAINED_VIRTUAL_SENSOR_IDENTIFIER
                        (default: disk/pretrain_virtual_sensor/fold_{dataset_fold})

```

When run, train scripts serialize experiment configurations to an
`experiment_config.yaml` file. You can find hyperparameters in the
`experiments/` directory for all results presented in our paper.

## Evaluation

All evaluation metrics are recorded at train time. The `cross_validate.py`
script can be used to compute metrics across folds:

```bash
# Summarize all experiments with means and standard errors of recorded metrics.
python cross_validate.py

# Include statistics for every fold -- this is much more data!
python cross_validate.py --disaggregate

# We can also glob for a partial set of experiments; for example, all of the
# disk experiments.
# Note that the ** wildcard may fail in bash; see above for a fix.
python cross_validate.py --experiment-paths ./experiments/disk/**/
```

## Acknowledgements

We'd like to thank [Rika Antonova](https://contactrika.github.io/),
[Kevin Zakka](https://github.com/kevinzakka),
[Nick Heppert](https://github.com/SuperN1ck),
[Angelina Wang](https://angelina-wang.github.io/), and
[Philipp Wu](https://github.com/wuphilipp) for discussions and feedback on both
our paper and codebase. Our software design also benefits from ideas from
several open-source projects, including
[Sophus](https://github.com/strasdat/Sophus), [GTSAM](https://gtsam.org/),
[Ceres Solver](http://ceres-solver.org/),
[minisam](https://github.com/dongjing3309/minisam), and
[SwiftFusion](https://github.com/borglab/SwiftFusion).

This work is partially supported by the Toyota Research Institute (TRI) and
Google. This article solely reflects the opinions and conclusions of its authors
and not TRI, Google, or any entity associated with TRI or Google.
