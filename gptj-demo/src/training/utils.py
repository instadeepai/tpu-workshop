from typing import List, Tuple

import haiku as hk
import jax
import numpy as np
import optax
from jax import numpy as jnp
from optax._src import base

from src.types import RNGKey, Tokens


def split_sequences(
    sequences: List[str],
    train_proportion: float = 0.8,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Splits sequences into train and test sequences.


    Args:
        sequences: List of sequences.
        train_proportion: Training set proportion. Defaults to 0.8.
        seed: Seed. Defaults to 0.

    Returns:
        Training sequences.
        Validation sequences.
    """
    num_train_sequences = int(np.round(train_proportion * len(sequences)))

    random_key = np.random.default_rng(seed)
    indices = np.arange(len(sequences))
    random_key.shuffle(indices)

    train_indices = indices[:num_train_sequences]
    val_indices = indices[num_train_sequences:]

    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]

    return train_sequences, val_sequences


def square_decay(
    init_value: float,
    warmup_end_lr: float,
    warmup_updates: int,
) -> base.Schedule:
    """
    Implements square root decay warmup.

    Args:
        init_value: Initial learning rate.
        warmup_end_lr: Peak learning rate.
        warmup_updates: Number of updates to linearly increase from the initial
            to the peak learning rate.

    Returns:
        A function that takes the step as input and returns the learning rate.
    """

    if warmup_updates <= 0:
        raise ValueError("Please provide a non-negative number of warmup steps")

    def schedule(count: int) -> float:
        decreased_count = count - warmup_updates
        ratio = jnp.where(warmup_updates == 0, 0.0, count / (warmup_updates))
        linear_lr = init_value + (warmup_end_lr - init_value) * ratio
        decayed_value: float = jnp.where(
            decreased_count <= 0,
            linear_lr,
            warmup_end_lr * (warmup_updates**0.5) * ((count + 1e-30) ** -0.5),
        )
        return decayed_value

    return schedule


def mask_sequence_bert_style(
    tokens: Tokens,
    random_key: RNGKey,
    random_token_indices: jnp.ndarray,
    mask_id: int,
    pad_id: int,
    noising_ratio: float,
    masking_prob: float = 0.8,
    random_token_prob: float = 0.1,
) -> Tuple[Tokens, Tokens]:
    """
    Add noise(mask) to a sequence of tokens for BERT-style training.

    Args:
        tokens: Input tokens to be masked.
        random_key: Random JAX key.
        random_token_indices: Collection of tokens indices in which we will sample from
            for setting random tokens in masking.
        mask_id: Mask token id.
        pad_id: Padding token id.
        noising_ratio: Ratio of tokens to be noised as either mask or random tokens.
        masking_prob: Among the tokens selected for noising (with noising_ratio), the
            proportion of tokens that actually will be masked with <mask> token.
        random_token_prob: Among the tokens selected for noising (with noising_ratio),
            the proportion of tokens that actually will be set to a random token. The
            rest of the tokens will be kept equal to the input.


    Returns:
        Noised tokens with some elements replaced by the mask_id, some elements set to a
            random token and some elements kept unchanged.
        Targets indicating which tokens are taken into account
            in the loss function. This is the same vector as tokens but the non-noised
            tokens are signaled by the padding token id.
    """

    def _mask_one_sequence(
        tokens: Tokens,
        random_key: RNGKey,
    ) -> Tuple[Tokens, Tokens]:
        """
        Mask one sequence, function to be vmapped over all tokens
        """

        # Build attention mask of sequence
        padding_mask = 1 * (tokens != pad_id)
        seq_length = jnp.sum(padding_mask)  # sequence length excluding padding tokens
        fixed_length = tokens.shape[0]  # sequence length including padding tokens

        random_key, subkey1, subkey2 = jax.random.split(random_key, num=3)

        # Determine number of masked tokens (mask+random tokens)
        # and number of true mask tokens
        num_mask = jnp.maximum(jnp.floor(seq_length * noising_ratio), 1)
        num_true_mask = jnp.maximum(
            jnp.floor(seq_length * noising_ratio * masking_prob), 1
        )
        num_random = jnp.floor(seq_length * noising_ratio * random_token_prob)

        # Shuffle the indices of the sequence.
        # We build a re-arrangement of the indices such that the
        # indices of non-padding tokens and the indices of padding tokens are separated
        aranges = jnp.arange(start=0, stop=fixed_length)
        aranges = jax.random.permutation(subkey1, aranges, independent=True)
        masked_aranges = aranges * (1 * (tokens != pad_id) + (-1) * (tokens == pad_id))
        # here it guarantees that the indices of the pad_tokens are at the end of
        # aranges
        aranges = jax.numpy.argsort(-masked_aranges, axis=0)

        # Build targets vectors by padding all non-mask tokens
        def pad_fn(i: jnp.int8, v: jnp.ndarray) -> jnp.ndarray:
            return jax.lax.cond(
                (i > num_mask),
                lambda v, i: v.at[aranges[i]].set(pad_id),
                lambda v, i: v,
                v,
                i,
            )  # noqa: E731,

        targets = jax.lax.fori_loop(
            lower=0,
            upper=fixed_length - 1,
            body_fun=pad_fn,
            init_val=tokens,
        )

        # Build new_tokens vector by masking true mask tokens
        def mask_fn(i: jnp.int8, v: jnp.ndarray) -> jnp.ndarray:
            return jax.lax.cond(
                (i < num_true_mask),
                lambda v, i: v.at[aranges[i]].set(mask_id),
                lambda v, i: v,
                v,
                i,
            )  # noqa: E731,

        new_tokens = jax.lax.fori_loop(
            lower=0,
            upper=fixed_length - 1,
            body_fun=mask_fn,
            init_val=tokens,
        )

        # Continue building new_tokens vector by setting  tokens to random tokens
        random_tokens = jax.random.choice(
            subkey2, random_token_indices, shape=(fixed_length,), replace=True
        )

        def random_fn(i: jnp.int8, v: jnp.ndarray) -> jnp.ndarray:
            return jax.lax.cond(
                jnp.logical_and(i < num_true_mask + num_random, i >= num_true_mask),
                lambda v, i: v.at[aranges[i]].set(random_tokens[i]),
                lambda v, i: v,
                v,
                i,
            )  # noqa: E731,

        new_tokens = jax.lax.fori_loop(
            lower=0,
            upper=fixed_length - 1,
            body_fun=random_fn,
            init_val=new_tokens,
        )
        new_tokens = jnp.where(
            tokens == pad_id, pad_id, new_tokens
        )  # prevent from setting pad tokens as targets
        # to ensure that there is at least 1 sequence position that is masked, as to not
        # cause nans in accuracy prediction

        return new_tokens, targets

    random_keys_stacked = jnp.stack(
        jax.random.split(random_key, num=tokens.shape[0]), axis=0
    )

    new_tokens, targets = jax.vmap(_mask_one_sequence)(tokens, random_keys_stacked)

    return new_tokens, targets


def get_causal_labels(
    tokens: Tokens, eos_token_id: int, pad_token_id: int, prefix: int = 0
) -> Tokens:
    """
    Obtains the labels of the input tokens for causal language modeling.

    Args:
        tokens: Tokens (batch_size, sequence_length, *).
        eos_token_id: ID of <eos> token.
        pad_token_id: ID of <pad> token.
        prefix: Number of prefix tokens to use as context.

    Returns:
        Causal labels with shape same as tokens.
    """

    # labels are input tokens shifted by one to the right,
    # with <eos> token id as final label
    initial_shape = tokens.shape
    tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)
    if prefix == 0:
        labels = jnp.concatenate(
            [
                tokens[:, 1:, :],
                jnp.full_like(tokens, eos_token_id)[:, :1, :],
            ],
            axis=1,
        )

    # labels are <pad> token ids of length prefix followed by input tokens
    # shifted by 1 + prefix to the right, with <eos> token id appended
    elif prefix > 0:
        labels = jnp.concatenate(
            [
                jnp.full_like(tokens, pad_token_id)[:, :prefix, :],
                tokens[:, prefix + 1 :, :],
                jnp.full_like(tokens, eos_token_id)[:, :1, :],
            ],
            axis=1,
        )

    else:
        raise ValueError("prefix argument must be positive")

    return labels.reshape(initial_shape)


def synchronize_accumulated_gradients(
    optimizer_state: optax.MultiStepsState,
    gradients: hk.Params,
    axis_name: str = "batch",
) -> optax.MultiStepsState:
    """
    Synchronize accumulated gradients from a multi steps optimizer state and last
    gradients computed. To be used before the last accumulation steps, where updates
    are computed.

    Args:
        optimizer_state: Optimizer state of a MultiSteps optimizer, with gradients
            accumulated in acc_grads .
        gradients: Gradients computed at the last iteration (not yet accumulated).
        axis_name: Axis name to pmean values.

    Returns:
        Synchronized optimizer state.
        Synchronized last gradients.
    """

    synchronized_accumulated_gradients, gradients = jax.tree_util.tree_map(
        lambda y: jax.lax.pmean(y, axis_name=axis_name),
        (optimizer_state.acc_grads, gradients),
    )

    new_state = optax.MultiStepsState(
        mini_step=optimizer_state.mini_step,
        gradient_step=optimizer_state.gradient_step,
        inner_opt_state=optimizer_state.inner_opt_state,
        acc_grads=synchronized_accumulated_gradients,
        skip_state=optimizer_state.skip_state,
    )

    return new_state, gradients
