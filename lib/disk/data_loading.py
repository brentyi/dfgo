import dataclasses
import pathlib
from typing import List, Optional

import fannypack
import jax

# For future projects, we probably want to use fifteen.data.DataLoader instead of the
# torch DataLoader, but keeping the torch one because that's what was used for the paper
# results.
import torch.utils.data

from .. import utils
from . import data, experiment_config

# Download Google Drive files to same directory as this file
fannypack.data.set_cache_path(
    # ./lib/disk/data.py => ./data/.cache
    str(pathlib.Path(__file__).parent.parent.parent.absolute() / "data/.cache/")
)

_DATASET_URLS = {
    "disk_tracking_0.hdf5": "https://drive.google.com/file/d/1BGa7VRNto6LCnD6dcCc_tbAR_Gpj3uQl/view?usp=sharing",
    "disk_tracking_1.hdf5": "https://drive.google.com/file/d/1GP-5rKQiYROicy8ac6tosq4KOnS9DUJn/view?usp=sharing",
    "disk_tracking_2.hdf5": "https://drive.google.com/file/d/1mZEZXiO52HlHVWD3zgH8Cy3PK-z-Db0N/view?usp=sharing",
    "disk_tracking_3.hdf5": "https://drive.google.com/file/d/1PJw9BEEKOudgds_KzUsJ0S-s_tYWZ02G/view?usp=sharing",
    "disk_tracking_4.hdf5": "https://drive.google.com/file/d/1NPHrgSENiV6DQiJw_E4g3jvbnpZC_rbo/view?usp=sharing",
    "disk_tracking_5.hdf5": "https://drive.google.com/file/d/1CHI3fn3ZaCzCykB-4GERUv1P9EeBd6k9/view?usp=sharing",
    "disk_tracking_6.hdf5": "https://drive.google.com/file/d/1vw56aXEw9G69h-PKFy-w5FWJEbhviCtY/view?usp=sharing",
    "disk_tracking_7.hdf5": "https://drive.google.com/file/d/1BsKHlxrJKjoyqU5xaDp58o2FpAR23XQO/view?usp=sharing",
    "disk_tracking_8.hdf5": "https://drive.google.com/file/d/1vU1nWijM4Eud5w6nJo-EKBXXgUb8Wby_/view?usp=sharing",
    "disk_tracking_9.hdf5": "https://drive.google.com/file/d/1GYev2mPHO5kK2J0RxNnwFHOl4DkXQZv0/view?usp=sharing",
}


def make_single_step_dataloader(
    config: experiment_config.BasicDatasetConfig, train: bool
) -> torch.utils.data.DataLoader[data.DiskStructNormalized]:
    """Returns a dataloader for virtual sensor pretraining."""
    return torch.utils.data.DataLoader(
        DiskSingleStepDataset(train=train, fold=config.dataset_fold),
        batch_size=config.batch_size,
        collate_fn=utils.collate_fn,
        shuffle=True,
        drop_last=True,
    )


def make_subsequence_dataloader(
    config: experiment_config.SequenceDatasetConfig, train: bool
) -> torch.utils.data.DataLoader[data.DiskStructNormalized]:
    """Returns a dataloader for training on sequences."""
    return torch.utils.data.DataLoader(
        DiskSubsequenceDataset(
            train=train,
            fold=config.dataset_fold,
            subsequence_length=config.train_sequence_length,
        ),
        batch_size=config.batch_size,
        collate_fn=utils.collate_fn,
        shuffle=True,
        drop_last=True,
    )


def load_trajectories(
    train: bool, fold: int, subsequence_length: Optional[int] = None
) -> List[data.DiskStructNormalized]:
    """Grabs a list of trajectories from a set of input files.

    Set `train` to False to load validation set.
    """
    assert 0 <= fold < len(_DATASET_URLS)

    # We intentionally exclude 01 from all datasets, because it's very different
    # (highway driving)
    files: List[str] = list(_DATASET_URLS.keys())
    if train:
        files = files[0:fold] + files[fold + 1 : len(_DATASET_URLS)]
    else:
        files = files[fold : fold + 1]

    assert len(set(files) - set(_DATASET_URLS.keys())) == 0

    trajectories: List[data.DiskStructNormalized] = []
    for filename in files:
        with fannypack.data.TrajectoriesFile(
            fannypack.data.cached_drive_file(filename, _DATASET_URLS[filename]),
            verbose=False,
        ) as traj_file:
            for trajectory in traj_file:
                # assert len(trajectory.keys()) == len(dataclasses.fields(data.DiskStructRaw))

                traj = data.DiskStructRaw(
                    **{
                        field.name: trajectory[field.name]
                        for field in dataclasses.fields(data.DiskStructRaw)
                    }
                ).normalize()
                assert traj.image is not None

                if subsequence_length is None:
                    # Return full trajectories
                    trajectories.append(traj)
                else:
                    # Split trajectory into overlapping subsequences
                    timesteps = traj.image.shape[0]
                    index = 0
                    while index + subsequence_length <= timesteps:
                        end_index = index + subsequence_length
                        trajectories.append(
                            jax.tree_map(
                                lambda x: x[index:end_index],
                                traj,
                            )
                        )
                        index += subsequence_length // 2

        print(f"Loaded {filename}, total trajectories: {len(trajectories)}")

    # # Print some data statistics
    # for field in ("image", "position", "velocity", "visible_pixels_count"):
    #     values = jax.tree_multimap(
    #         lambda *x: onp.concatenate(x, axis=0), *trajectories
    #     ).__getattribute__(field)
    #
    #     if field != "visible_pixels_count":
    #         values = values.reshape((-1, values.shape[-1]))
    #
    #     print(
    #         f"({field}) Mean, std dev:",
    #         onp.mean(values, axis=0),
    #         onp.std(values, axis=0),
    #     )

    return trajectories


class DiskSubsequenceDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, fold: int, subsequence_length: int = 5):
        self.samples: List[data.DiskStructNormalized] = []

        for trajectory in load_trajectories(train=train, fold=fold):
            assert trajectory.image is not None
            timesteps = len(trajectory.image)
            index = 0
            while index + subsequence_length <= timesteps:
                self.samples.append(
                    jax.tree_map(
                        lambda x: x[index : index + subsequence_length], trajectory
                    )
                )
                index += subsequence_length // 2

    def __getitem__(self, index: int) -> data.DiskStructNormalized:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class DiskSingleStepDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, fold: int):
        self.samples: List[data.DiskStructNormalized] = []

        for trajectory in load_trajectories(train=train, fold=fold):
            assert trajectory.image is not None
            timesteps = len(trajectory.image)
            for t in range(timesteps):
                self.samples.append(jax.tree_map(lambda x: x[t], trajectory))

    def __getitem__(self, index: int) -> data.DiskStructNormalized:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)
