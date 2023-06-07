from __future__ import annotations

import functools
from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from src.training.base import TrainingState
from src.training.losses import causal_lm_loss
from src.training.utils import synchronize_accumulated_gradients
from src.types import Metrics, RNGKey, SequenceMask, Tokens


class DecoderCLMTrainer:
    """
    A stateless abstraction around an init_fn/update_fn/compute_metrics+fn set of funcs.
    This extracts some common boilerplate from the training loop. This trainer trains a
    Decoder language model through causal language modelling (CLM).
    """

    def __init__(
        self,
        apply_fn: hk.Transformed.apply,
        init_fn: hk.Transformed.apply,
        pad_token_id: int,
        eos_token_id: int,
        bos_token_id: int,
        optimizer: optax.MultiSteps,
        parameters_partition_fn: Callable[[str, str, Any], bool] | None = None,
    ):
        """
        Args:
            apply_fn: The transformed forward function of the model.
            init_fn: The transformed init function of the model.
            pad_token_id: Pad token id.
            eos_token_id: Eos token id.
            bos_token_id: Bos token id.
            optimizer: Optimizer with gradient accumulation.
                Should be a optax.Multisteps object. Set every_k_schedule to 1
                if you do not want to accumulate.
            parameters_partition_fn: if specified, this function is used to split
                parameters into trainable and non-trainable parameters. Can be used
                to specify to the trainer not to train some parameters from the model.
                The function takes as inputs the name of the module,
                the name of a given entry in the module data bundle
                (e.g. parameter name) and the corresponding data and returns a boolean.
                See haiku.data_structures.partition documentation for more details.
        """
        self._init_fn = init_fn

        self._loss_fn = functools.partial(
            causal_lm_loss,
            apply_fn=apply_fn,
        )

        self._optimizer = optimizer
        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id
        self._bos_token_id = bos_token_id

        self._parameters_partition_fn = parameters_partition_fn

    def build_init_fn(self) -> Callable:
        xmapped_init_fn: Callable = jax.experimental.maps.xmap(
            fun=self.init,
            in_axes=(
                ["shard", "batch", ...],
                [
                    ...,
                ],
            ),
            out_axes=["shard", "batch", ...],
            axis_resources={"shard": "shard"},
            donate_argnums=(0,),
        )
        return xmapped_init_fn

    def build_init_fn_from_params(self) -> Callable:
        xmapped_init_fn: Callable = jax.experimental.maps.xmap(
            fun=self.init,
            in_axes=(
                ["shard", "batch", ...],
                [
                    ...,
                ],
                ["shard", "batch", ...],
            ),
            out_axes=["shard", "batch", ...],
            axis_resources={"shard": "shard"},
            donate_argnums=(0,),
        )
        return xmapped_init_fn

    def build_xmapped_update_fn(self) -> Callable:
        """
        Xmaps the self._update function nover the defined devices_mesh.
            The training_state has in_axis= ["shard", "batch", ...] since the first axis
        corresponds to the number of shards and the second corresponds to the number of
        data_parallel_ways axis.
            The tokens correspond to 'axis_name="batch"' since the first axis
        corresponds to the number of data_parallel_ways.

        The function returns a new training_state and training metrics.

        For updates where the optimizer updates are different along the batch_axis,
        if the out_axis indicate ["shard", ...], the following error is raised by xmap:
        "One of xmap results has an out_axes specification of ['shard', ...], but is
        actually mapped along more axes defined by this xmap call: batch". The out_axis
        needs to indicate ["shard", "batch", ...].

        For updates where the accumulated gradients are synchronized and the
        optimizer_state have been pmean along the "batch" axis, the out_axis
        should be ["shard", ...], but we observe that it also works with
        ["shard", "batch", ...].

        Since the optimizer state is different along "batch" axis in all training steps
        except those that call the pmean operation on this "batch" axis, i.e the steps
        where the gradient accumulation is applied, the training state needs to have its
        first two axis being [num_shards, num_data_parallel]. Therefore, the xmapped
        update function has in_axes ["shard", "batch", ...] for the training_state and
        equal out_axes.


        Returns:
            The jitted function with xmap
        """

        xmapped_update_fn: Callable = jax.experimental.maps.xmap(
            fun=self.update,
            # in_axes=(["shard", ...], ["batch", ...]),
            in_axes=(["shard", "batch", ...], ["batch", ...]),
            out_axes=(
                # ["shard", ...],
                ["shard", "batch", ...],
                [
                    "shard",
                    ...,
                ],
            ),
            axis_resources={"shard": "shard", "batch": "batch"},
            donate_argnums=(0,),
        )
        return xmapped_update_fn

    def build_xmapped_metrics_fn(self) -> Callable:
        """
        Xmaps the self._compute_metrics function nover the defined devices_mesh.
            The training_state has in_axis= ["shard", "batch", ...] since the first axis
        corresponds to the number of shards and the second corresponds to the number of
        data_parallel_ways axis.
            The tokens correspond to 'axis_name="batch"' since the first axis
        corresponds to the number of data_parallel_ways.

        The function returns a new training_state and training metrics.

        Returns:
            The jitted function with xmap
        """

        xmapped_metrics_fn: Callable = jax.experimental.maps.xmap(
            fun=self.compute_metrics,
            in_axes=(["shard", "batch", ...], ["batch", ...]),
            out_axes=[
                "shard",
                ...,
            ],
            axis_resources={"shard": "shard", "batch": "batch"},
        )
        return xmapped_metrics_fn

    @functools.partial(jax.jit, static_argnums=0)
    def init(
        self,
        random_key: RNGKey,
        tokens: Tokens,
        pretrained_params: hk.Params | None = None,
    ) -> TrainingState:
        """
        Initializes the Training State.

        Args:
            random_key: Random JAX key.
            tokens: Tokens (batch_size,seq_length).

        Returns:
            Initialized Training state.
        """
        random_key, subkey = jax.random.split(random_key)
        if pretrained_params is None:
            params = self._init_fn(subkey, tokens)
        else:
            params = pretrained_params

        if self._parameters_partition_fn is None:
            trainable_params = params
        else:
            trainable_params, _ = hk.data_structures.partition(
                self._parameters_partition_fn, params
            )

        optimizer_state = self._optimizer.init(trainable_params)

        return TrainingState(
            step=jnp.array(0),
            random_key=random_key,
            optimizer_state=optimizer_state,
            params=params,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def update(
        self,
        state: TrainingState,
        tokens: Tokens,
        sequence_mask: SequenceMask | None = None,
    ) -> tuple[TrainingState, Metrics]:
        """
        Updates the training state. This function is to be called inside
        a pmap operator with `axis_name="batch"`.

        Args:
            state: Current training state.
            tokens: Tokens (batch_size, seq_length).
            sequence_mask: Optional sequence mask to mask the loss function
                ( batch_size, seq_length). Expect 0 and 1 values.
                Default to full of ones (=no mask) if not defined.

        Returns:
            Updated training state.
            Metrics.
        """

        random_key = state.random_key
        params = state.params

        random_key, subkey = jax.random.split(random_key)

        if sequence_mask is None:
            sequence_mask = jnp.ones_like(tokens, dtype=tokens.dtype)

        # partition params into trainable and non-trainable if needed
        if self._parameters_partition_fn is None:
            trainable_params = params
            non_trainable_params = {}
        else:
            trainable_params, non_trainable_params = hk.data_structures.partition(
                self._parameters_partition_fn, params
            )

        def _split_loss_fn(  # type: ignore
            trainable_params, non_trainable_params, *args, **kwargs
        ):
            params = hk.data_structures.merge(trainable_params, non_trainable_params)
            return self._loss_fn(params, *args, **kwargs)

        (loss, metrics), gradient = jax.value_and_grad(_split_loss_fn, has_aux=True)(
            trainable_params,
            non_trainable_params,
            random_key,
            tokens=tokens,
            sequence_mask=sequence_mask,  # type: ignore
        )
        metrics["loss"] = loss
        metrics["perplexity"] = jnp.exp(loss)

        # update optimizer and weights
        optimizer_state = state.optimizer_state

        # synchronize workers only when done accumulating gradients
        optimizer_state, gradient = jax.lax.cond(
            optimizer_state.mini_step == self._optimizer._every_k_schedule(0) - 1,
            lambda x, y: synchronize_accumulated_gradients(x, y),
            lambda x, y: (x, y),
            optimizer_state,
            gradient,
        )

        updates, optimizer_state = self._optimizer.update(
            gradient, optimizer_state, params
        )

        # update parameters
        trainable_params = optax.apply_updates(trainable_params, updates)
        params = hk.data_structures.merge(trainable_params, non_trainable_params)

        new_state = TrainingState(
            step=state.step + 1,
            random_key=subkey,
            optimizer_state=optimizer_state,
            params=params,
        )

        # gather metrics across devices
        metrics = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="batch"), metrics)

        return new_state, metrics

    @functools.partial(jax.jit, static_argnums=0)
    def compute_metrics(
        self,
        state: TrainingState,
        tokens: Tokens,
        sequence_mask: SequenceMask | None = None,
    ) -> Metrics:
        """
        Carries out inference over a batch of data and returns metrics.
        This function is to be called inside a pmap operator
        with `axis_name="batch"`.

        Args:
            state: Training state.
            tokens: Tokens (batch_size, seq_length).
            sequence_mask: Optional sequence mask to mask the loss function
                (batch_size, seq_length). Expect 0 and 1 values.
                Default to full of ones (=no mask) if not defined.

        Returns:
            Metrics.
        """

        random_key = state.random_key
        params = state.params

        if sequence_mask is None:
            sequence_mask = jnp.ones_like(tokens, dtype=tokens.dtype)

        # Calculate loss
        loss, metrics = self._loss_fn(params, random_key, tokens, sequence_mask)

        metrics["loss"] = loss
        metrics["perplexity"] = jnp.exp(loss)

        # gather losses and gradients across devices
        metrics = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="batch"), metrics)

        return metrics  # type:ignore
