"""Script for computing the expected RMSE of the analytical dynamics model. Uses entire
dataset."""

import jax
import jaxlie
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm

from lib import kitti

Scalar = jnp.ndarray


@jax.jit
def compute_subsequence_sum_squared_error(
    subsequence: kitti.data.KittiStructRaw,
) -> Scalar:
    # Quick shape checks
    (timesteps,) = subsequence.x.shape
    assert subsequence.y.shape == subsequence.theta.shape == (timesteps,)

    # Make stacked state object + do predictions
    states: kitti.fg_system.State
    states = jax.vmap(kitti.fg_system.State.make)(
        pose=jax.vmap(jaxlie.SE2.from_xy_theta)(
            subsequence.x, subsequence.y, subsequence.theta
        ),
        linear_vel=subsequence.linear_vel,
        angular_vel=subsequence.angular_vel,
    )
    predicted_states: kitti.fg_system.State
    predicted_states = jax.vmap(kitti.fg_system.State.predict_next)(states)

    # Align true states and predicted states
    states = jax.tree_map(lambda x: x[1:], states)
    predicted_states = jax.tree_map(lambda x: x[:-1], predicted_states)

    # Compute squared errors
    squared_errors = (
        jax.vmap(kitti.fg_system.State.manifold_minus)(states, predicted_states) ** 2
    )
    assert squared_errors.shape == (timesteps - 1, 5)
    return jnp.sum(squared_errors, axis=0)


def main() -> None:
    print("Loading data...")
    trajectories = kitti.data_loading.load_trajectories_from_ids(
        range(11), verbose=False
    )
    subsequences = kitti.data_loading.make_disjoint_subsequences(
        trajectories, subsequence_length=100
    )

    print("Computing statistics...")
    sse = onp.zeros(5)
    for subsequence in tqdm(subsequences):
        sse = sse + compute_subsequence_sum_squared_error(subsequence)
    mse = sse / len(subsequences)
    rmse = onp.sqrt(mse)

    print("RMSE:", rmse)


if __name__ == "__main__":
    main()
