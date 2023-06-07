import haiku as hk
import jax
import jax.numpy as jnp
import joblib


def get_num_parameters(params: hk.Params) -> int:
    """
    Returns the number of parameters of a model.

    Args:
        params: Model's haiku parameters.

    Returns:
        The number of parameters.
    """
    return sum([x.size for x in jax.tree_util.tree_leaves(params)])


def save_params(params: hk.Params, filename: str) -> jax.tree_util.PyTreeDef:
    """
    Save a neural network's parameters to a joblib archive.

    Args:
        params: Pytree of the parameters to be saved.
        filename: File name where it is to be saved.

    """
    joblib.dump(
        jax.tree_map(lambda x: x.__array__(), params),
        filename,
        compress=False,
    )
    return


def load_params(filename: str) -> hk.Params:
    """
    Load params from a joblib archive.

    Args:
        filename: File name where it is to be saved

    Returns:
        Parameters pytree.
    """
    return joblib.load(filename)


def load_params_npz(filename: str, tree_def: jax.tree_util.PyTreeDef) -> hk.Params:
    """
    Load params from a .npz file (legacy version).

    Args:
        filename: File name where it is to be saved
        tree_def: Parameters pytree definition.

    Returns:
        Parameters pytree.
    """
    with open(filename, "rb") as f:
        uploaded = jnp.load(f)
        arrays = [jnp.asarray(uploaded[file]) for file in uploaded.files]
        reconstructed_params = jax.tree_util.tree_unflatten(tree_def, arrays)
        return reconstructed_params
