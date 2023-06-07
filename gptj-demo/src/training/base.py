from __future__ import annotations

import os
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from src.types import Metrics
from src.utils.parameters import load_params, save_params


class TrainingState(NamedTuple):
    """
    Contains a full training state.

    step: Epoch.
    params: Network parameters as PyTree.
    optimizer_state: Optimizer state as Pytree.
    random_key: Jax random PRNG Key.
    """

    step: jnp.ndarray
    params: hk.Params
    optimizer_state: optax.MultiStepsState
    random_key: jnp.ndarray

    def save(self, save_dir: str) -> None:
        """
        Saves state using joblib. Filename should end with .joblib.
        It returns the tree structure of the state, which is used to
        load the state.

        Args:
            save_dir: Directory where the training state is to be saved.

        Returns:
            Pytree definition of the parameters.
            Pytree definition of the optimizer state.
        """
        os.makedirs(save_dir, exist_ok=True)
        save_params(self.params, os.path.join(save_dir, "params.joblib"))
        save_params(self.optimizer_state, os.path.join(save_dir, "opt_state.joblib"))
        jnp.save(os.path.join(save_dir, "step.npy"), self.step)
        jnp.save(os.path.join(save_dir, "key.npy"), self.random_key)

    @property
    def tree_def(self) -> jax.tree_util.PyTreeDef:
        return jax.tree_util.tree_structure(self)

    @classmethod
    def load(
        cls,
        save_dir: str,
    ) -> TrainingState:
        """
        Load the training state from a given directory.

        Args:
            save_dir: Directory where the training state is saved.

        Returns:
            Loaded training state.
        """

        params = load_params(os.path.join(save_dir, "params.joblib"))
        optimizer_state = load_params(os.path.join(save_dir, "opt_state.joblib"))
        step = jnp.load(os.path.join(save_dir, "step.npy"))
        key = jnp.load(os.path.join(save_dir, "key.npy"))

        return TrainingState(
            step=step, params=params, optimizer_state=optimizer_state, random_key=key
        )


@jax.jit
def aggregate_metrics(metrics_list: list[Metrics]) -> Metrics:
    """
    Aggregates a list of metrics into a single pytree that contains mean metrics.

    Args:
         metrics_list: List of metrics.

    Returns:
        Aggregated metrics.
    """
    metrics: Metrics = jax.tree_map(
        lambda *leaf: jnp.mean(jnp.stack(leaf, axis=0), axis=0),
        *metrics_list,
    )
    return metrics
