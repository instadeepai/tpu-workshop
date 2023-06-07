from typing import Union

import jax
import jax.numpy as jnp

from src.types import AttentionMask, Tokens


def build_causal_attention_mask(batch_size: int, seq_len: int) -> AttentionMask:
    """
    Builds a batch of causal masks of shape (batch_size, 1, seq_len, seq_len) to feed
    to an attention layer.

    Args:
        batch_size: Batch size.
        seq_len: Length of the sequences.

    Returns:
        Batch of causal masks.
    """
    mask = jnp.ones((batch_size, 1, seq_len, seq_len))
    causal_mask = jnp.tril(mask)
    return causal_mask


def _causal_with_prefix_mask(seq_len: int, prefix: int) -> AttentionMask:
    """
    Builds a causal mask with prefix of shape (1, seq_len, seq_len). Function to be
    vmapped.

    Args:
        seq_len: Length of the sequences.
        prefix: Prefix up to which the mask is fully connected.

    Returns:
        A causal mask with prefix.
    """
    mask = jnp.ones((1, seq_len, seq_len))
    causal_mask = jnp.tril(mask)
    keep = jnp.tile(jnp.arange(seq_len), (1, seq_len, 1))
    prefix_mask = (keep < prefix) * mask + (keep >= prefix) * causal_mask
    return prefix_mask


def build_prefix_causal_attention_mask(
    batch_size: int, seq_len: int, prefix: Union[int, jnp.ndarray]
) -> AttentionMask:
    """
    Builds a batch of causal mask with prefix of shape (batch_size, 1, seq_len, seq_len)
    to feed to an attention layer.

    Args:
        batch_size: Batch size.
        seq_len: Length of the sequences.
        prefix: Prefix up to which the mask is fully connected.

    Returns:
        Batch of causal masks with prefix.
    """
    if type(prefix) is int:
        prefix = jnp.array([prefix] * batch_size)
    prefix_mask = jax.vmap(_causal_with_prefix_mask, in_axes=(None, 0))(seq_len, prefix)
    return prefix_mask


def build_padding_attention_mask(tokens: Tokens, pad_token_id: int) -> AttentionMask:
    """
    Builds a padding mask from a sequence of tokens by masking <pad> in the attention.

    Args:
        tokens: Batch of sequences of shape (batch_size, seq_len).
        pad_token_id: Int corresponding to the <pad> token to mask.

    Returns:
        Batch of attention masks, masking out <pad> tokens.
    """
    padding_mask = tokens != pad_token_id
    padding_mask = padding_mask[:, None, :]
    padding_mask = jnp.einsum("bhT, bht->bhtT", padding_mask, padding_mask)
    return padding_mask
