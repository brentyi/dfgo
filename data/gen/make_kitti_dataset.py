"""Data preparation for the KITTI dataset. Processes raw dataset and produce and hdf5
file for each trajectory.

Expects a raw data `data/` directory in the current working directory, which can be
downloaded via:
```bash
wget -N 'https://tubcloud.tu-berlin.de/s/0fU32cq0ppqdGXe/download'
echo 'Unpacking data .. '
echo
unzip download
rm download
```
(source: https://github.com/tu-rbo/differentiable-particle-filters)
"""

import pathlib
import sys

import fannypack
import fifteen
import jax_dataclasses
import numpy as onp
from PIL import Image
from tqdm.auto import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent.parent))
print(sys.path)

from lib import kitti


def load_data(
    pose_txt: pathlib.Path, image_dir: pathlib.Path
) -> kitti.data.KittiStructRaw:
    """Load and pre-process KITTI data from a paths. Returns a trajectory."""
    poses = onp.loadtxt(pose_txt)
    N = poses.shape[0]
    assert poses.shape == (N, 12)

    # Reshape poses to standard 3x4 pose matrix
    poses = poses.reshape((N, 3, 4))

    # Extract 2D poses
    # Note that we treat the XZ plane as the XY plane
    xs = poses[:, 2, 3]
    ys = -poses[:, 0, 3]

    # Extra y-axis rotation
    thetas = -onp.arctan2(-poses[:, 2, 0], poses[:, 2, 2])

    # Validate shapes
    assert xs.shape == ys.shape == thetas.shape == (N,)

    # Load images
    image_paths = sorted(image_dir.iterdir())
    for i in range(len(image_paths) - 1):

        def timestep_from_path(path: pathlib.Path) -> int:
            """Input: something/image1_0000003.png
            Output: 3
            """
            assert str(path).endswith(".png")
            return int(str(path).rpartition("_")[2][:-4])

        assert (
            timestep_from_path(image_paths[i])
            == timestep_from_path(image_paths[i + 1]) - 1
        )

    images = onp.array(
        [onp.array(Image.open(image_path)) for image_path in tqdm(image_paths)]
    )
    assert images.shape == (N, 50, 150, 3)
    assert images.dtype == onp.uint8

    # Cast to prevent overflow when computing difference images
    images_int16 = images.astype(onp.int16)

    # Consolidate all data, matching conventions from Kloss et al:
    # > How to Train Your Differentiable Filter
    # > https://arxiv.org/pdf/2012.14313.pdf

    # time between frames is really 0.103, but life feels easier if we just don't divide
    time_delta = 1.0

    data = kitti.data.KittiStructRaw(
        image=images[1:-1],
        image_diff=(
            # image_diff[i] = image[i] - image[i - 1]
            # => after subtracting, we're missing the first timestep
            # => to align with velocities, we need to chop off the last timestep
            images_int16[1:]
            - images_int16[:-1]
        )[:-1],
        x=xs[1:-1],
        y=ys[1:-1],
        theta=thetas[1:-1],
        linear_vel=(
            # Note that we want: positions[i + 1] = positions[i] + velocity[i]
            #
            # velocity[i] = positions[i + 1] - positions[i]
            # => after subtracting, we're missing the last timestep
            # => to align with image differences, we need to chop off the first timestep
            onp.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2)
            / time_delta
        )[1:],
        angular_vel=(
            # Same alignment logic as linear velocity
            kitti.math_utils.wrap_angle(thetas[1:] - thetas[:-1])
            / time_delta
        )[1:],
    )

    # Validate alignment
    assert onp.all(
        data.image_diff[1]
        == data.image[1].astype(onp.int16) - data.image[0].astype(onp.int16)
    )
    assert data.angular_vel[0] == kitti.math_utils.wrap_angle(
        data.theta[1] - data.theta[0]
    )

    return data


if __name__ == "__main__":

    fifteen.utils.pdb_safety_net()

    path: pathlib.Path
    directories = sorted(
        filter(
            lambda path: path.is_dir(),
            (pathlib.Path.cwd() / "data" / "kitti").iterdir(),
        )
    )
    assert len(directories) == 11

    # Make sure output directory exists
    output_dir = pathlib.Path.cwd() / "data_out"
    if not output_dir.exists():
        output_dir.mkdir()
    assert output_dir.is_dir()

    for directory in directories:

        dataset_id: str = directory.stem
        assert (
            dataset_id.isdigit() and len(dataset_id) == 2
        ), "Dataset subdirectories should be two digit numbers!"

        print("Handling", directory.stem)

        with fannypack.data.TrajectoriesFile(
            str(output_dir / f"kitti_{dataset_id}.hdf5"),
            read_only=False,
        ) as traj_file:
            traj_file.resize(2)

            # Load data from first camera
            traj_file[0] = jax_dataclasses.asdict(
                load_data(
                    pose_txt=directory.parent / f"{dataset_id}_image1.txt",
                    image_dir=directory / "image_2",
                )
            )

            # Load data from second camera
            traj_file[1] = jax_dataclasses.asdict(
                load_data(
                    pose_txt=directory.parent / f"{dataset_id}_image2.txt",
                    image_dir=directory / "image_3",
                )
            )
