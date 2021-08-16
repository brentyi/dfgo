"""Helpers for loading and manipulating data."""

import dataclasses
import enum
import pathlib
from typing import Iterable, List

import fannypack
import jax
import numpy as onp
import torch

from .. import utils
from . import data, experiment_config

fannypack.data.set_cache_path(
    # ./lib/kitti/data.py => ./data/.cache
    str(pathlib.Path(__file__).parent.parent.parent.absolute() / "data/.cache/")
)

_DATASET_URLS = {
    "kitti_00.hdf5": "https://drive.google.com/file/d/1DSbwoYPpD9sAKnazBHa5EY2MxI4dCEc0/view?usp=sharing",
    "kitti_01.hdf5": "https://drive.google.com/file/d/1zDqi1oTcIrUhwcUe-xWphVNcBu-Y00Og/view?usp=sharing",
    "kitti_02.hdf5": "https://drive.google.com/file/d/1h1nqyXLP-TuUHJe7q97Plt90vvP3p8yS/view?usp=sharing",
    "kitti_03.hdf5": "https://drive.google.com/file/d/1ls0lIT1nN7eXmOI-1ZQDcD0tBZ9foCgr/view?usp=sharing",
    "kitti_04.hdf5": "https://drive.google.com/file/d/1YcRVSD9FCL6ZP_bt1Q0BVyqWTkaQxTK7/view?usp=sharing",
    "kitti_05.hdf5": "https://drive.google.com/file/d/1xFRJKS8k56UhMYrKVWw6eg7GKPtoJ9df/view?usp=sharing",
    "kitti_06.hdf5": "https://drive.google.com/file/d/1GBGh39gcsLofaT63UgXjCOgPL1n8Rpii/view?usp=sharing",
    "kitti_07.hdf5": "https://drive.google.com/file/d/1Dmr7gXFXX4Iiec3JWRrNGrV1FXoBWPEy/view?usp=sharing",
    "kitti_08.hdf5": "https://drive.google.com/file/d/1TTIlFjxXf-YpyRodB88rS49ncHGJ6C5o/view?usp=sharing",
    "kitti_09.hdf5": "https://drive.google.com/file/d/1GKJHCMj6q5hZol_gAZX9Iw5oLSXQpYIc/view?usp=sharing",
    "kitti_10.hdf5": "https://drive.google.com/file/d/1HCKczAcknVZFSfbT4W5138EzLz3EaNeD/view?usp=sharing",
}


def load_trajectories_from_paths(
    paths: Iterable[str], verbose: bool = True
) -> List[data.KittiStructRaw]:
    trajectories: List[data.KittiStructRaw] = []
    for path in paths:
        with fannypack.data.TrajectoriesFile(
            path=path,
            verbose=verbose,
        ) as traj_file:
            for trajectory in traj_file:
                assert len(trajectory.keys()) == len(
                    dataclasses.fields(data.KittiStructRaw)
                )
                trajectories.append(data.KittiStructRaw(**trajectory))
    return trajectories


def load_trajectories_from_ids(
    trajectory_ids: Iterable[int],
    verbose: bool = True,
) -> List[data.KittiStructRaw]:
    for number in trajectory_ids:
        assert 0 <= number <= 10

    file_names = map(lambda n: f"kitti_{n:02d}.hdf5", trajectory_ids)
    file_paths = [
        fannypack.data.cached_drive_file(f, _DATASET_URLS[f]) for f in file_names
    ]
    return load_trajectories_from_paths(file_paths, verbose=verbose)


def load_all_trajectories_and_print_stats() -> None:
    """Load all trajectories, then print mean and standard deviations. Used to populated
    `_DATASET_MEANS` and `_DATASET_STD_DEVS`."""
    trajectories: List[data.KittiStructRaw] = load_trajectories_from_ids(range(11))
    concat: data.KittiStructRaw = jax.tree_map(
        lambda *x: onp.concatenate(x, axis=0), *trajectories
    ).normalize()

    for k, v in {
        "image": concat.image.reshape((-1, 3)),
        "image_diff": concat.image_diff.reshape((-1, 3)),
        "x": concat.x.reshape((-1, 1)),
        "y": concat.y.reshape((-1, 1)),
        "linear_vel": concat.linear_vel.reshape((-1, 1)),
        "angular_vel": concat.angular_vel.reshape((-1, 1)),
    }.items():
        print(
            k,
            "mean, std:",
            onp.mean(v, axis=0).squeeze(),
            onp.std(v, axis=0).squeeze(),
        )


class DatasetSplit(utils.StringEnum):
    TRAIN_ALL = enum.auto()
    """Complete training set."""

    TRAIN_VIRTUAL_SENSOR = enum.auto()
    """Portion of training set used for virtual sensor pretraining."""

    TRAIN_VIRTUAL_SENSOR_HOLDOUT = enum.auto()
    """Portion of training set that's held out from pretraining."""

    VALIDATION = enum.auto()
    """Validation trajectory."""

    OVERFIT = enum.auto()
    """Tiny overfitting dataset for making sure that things are working!"""

    EVERYTHING = enum.auto()
    """Everything :)"""


def load_trajectories_from_split(
    split: DatasetSplit,
    fold: int,
) -> List[data.KittiStructRaw]:
    assert 0 <= fold <= 9

    # Figure which trajectory files we want to read
    # Note that we exclude trajectory #1
    validation_trajectory_id: int = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10)[fold]
    if split in (
        DatasetSplit.TRAIN_ALL,
        DatasetSplit.TRAIN_VIRTUAL_SENSOR,
        DatasetSplit.TRAIN_VIRTUAL_SENSOR_HOLDOUT,
    ):
        trajectories = load_trajectories_from_ids(
            set(range(11)) - {validation_trajectory_id} - {1}
        )
    elif split is DatasetSplit.VALIDATION:
        trajectories = load_trajectories_from_ids([validation_trajectory_id])
    elif split is DatasetSplit.OVERFIT:
        print("Using small data sample!")
        trajectories = load_trajectories_from_paths(["./data/_kitti_small_sample.hdf5"])
    else:
        assert False

    # For virtual sensor pretraining, we include both a primary train set and a holdout
    # set for validation.
    pretrain_holdout_padding: int = 5
    pretrain_holdout_size: int = 200
    if split is DatasetSplit.TRAIN_VIRTUAL_SENSOR:
        trajectories = jax.tree_map(
            lambda x: x[: -pretrain_holdout_size - pretrain_holdout_padding],
            trajectories,
        )
    elif split is DatasetSplit.TRAIN_VIRTUAL_SENSOR_HOLDOUT:
        trajectories = jax.tree_map(lambda x: x[-pretrain_holdout_size:], trajectories)

    return trajectories


def make_disjoint_subsequences(
    trajectories: List[data.KittiStructRaw],
    subsequence_length: int = 100,
) -> List[data.KittiStructRaw]:
    return make_overlapping_subsequences(
        trajectories,
        subsequence_length=subsequence_length,
        overlap_length=0,
    )


def make_overlapping_subsequences(
    trajectories: List[data.KittiStructRaw],
    subsequence_length: int = 100,
    overlap_length: int = 50,  # 0 => no overlap
) -> List[data.KittiStructRaw]:

    assert overlap_length < subsequence_length

    # Load trajectories!
    subsequences: List[data.KittiStructRaw] = []
    for traj in trajectories:
        t = 0
        while t + subsequence_length <= traj.x.shape[0]:
            subsequences.append(
                jax.tree_map(lambda x: x[t : t + subsequence_length], traj)
            )
            t += subsequence_length - overlap_length

    return subsequences


def make_subsequence_eval_dataset(
    config: experiment_config.DatasetConfig,
    subsequence_length: int = 100,
) -> torch.utils.data.Dataset[data.KittiStructNormalized]:
    """Returns a dataset for computing validation losses.

    We use a dataset here instead of a dataloader for cases where our model isn't yet
    capable of batching. (for factor graphs: when we use sparse Cholesky for linear
    solves)
    """

    trajectories = load_trajectories_from_split(
        split=DatasetSplit.VALIDATION
        if not config.use_overfitting_dataset
        else DatasetSplit.OVERFIT,
        fold=config.dataset_fold,
    )
    subsequences = make_disjoint_subsequences(
        trajectories,
        subsequence_length=subsequence_length,  # We evaluate over subsequences of length 100
    )
    return data.KittiSubsequenceDataset(subsequences=subsequences)


def make_subsequence_dataloader(
    config: experiment_config.SequenceDatasetConfig,
    split: DatasetSplit,
) -> torch.utils.data.DataLoader[data.KittiStructNormalized]:
    """Returns a dataloader for end-to-end training on sequences."""

    trajectories = load_trajectories_from_split(
        split=split if not config.use_overfitting_dataset else DatasetSplit.OVERFIT,
        fold=config.dataset_fold,
    )
    subsequences = make_overlapping_subsequences(
        trajectories,
        subsequence_length=config.train_sequence_length,
        overlap_length=config.train_sequence_length // 2,
    )
    dataset = data.KittiSubsequenceDataset(subsequences=subsequences)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=4,
        collate_fn=utils.collate_fn,
        shuffle=True,
        drop_last=True,
    )


def make_single_step_dataloader(
    config: experiment_config.DatasetConfig,
    split: DatasetSplit,
) -> torch.utils.data.DataLoader[data.KittiStructNormalized]:
    """Returns a dataloader for KITTI CNN pre-training."""

    assert not isinstance(config, experiment_config.SequenceDatasetConfig), (
        "We probably should not be loading single-step datasets if the goal is to "
        "train on sequences!"
    )

    return torch.utils.data.DataLoader(
        data.KittiSingleStepDataset(
            trajectories=load_trajectories_from_split(
                split=split,
                fold=config.dataset_fold,
            )
        ),
        batch_size=config.batch_size,
        collate_fn=utils.collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
