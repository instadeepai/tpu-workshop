"""
Utils for autoregressive decoding with a decoder network.
"""
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from src.types import RNGKey, Tokens


def update_tokens_ids_greedy(
    tokens_ids: Tokens,
    time_step: jnp.ndarray,
    random_key: RNGKey,
    params: hk.Params,
    apply_fn: hk.Transformed.apply,
) -> Tuple[Tokens, RNGKey]:
    """
    Update the input sequence of tokens using greedy decoding with a decoder model.
    Typical inputs could be a prompt with end of tokens appended. This function can
    then be called recursively after the prompt to generate the rest of the sentence.

    Args:
        tokens_ids: Input tokens ids, shape = (batch_size, sequence_length).
        time_step: Time step at which to decode, shape = (,).
        random_key: Random key.
        params: Decoder parameters.
        apply_fn: Decoder apply fn.

    Returns:
        Tokens ids with decoded token at position time_step + 1 and updated random key.
    """
    logits = apply_fn(params, random_key, tokens_ids)["logits"]
    logits = logits[:, time_step, :]
    new_token_id = jnp.argmax(logits, axis=-1)
    tokens_ids = tokens_ids.at[:, time_step + 1].set(new_token_id)
    return tokens_ids, random_key


def update_tokens_ids_temperature_sampling(
    tokens_ids: Tokens,
    time_step: jnp.ndarray,
    random_key: RNGKey,
    params: hk.Params,
    apply_fn: hk.Transformed.apply,
    temperature: float = 1.0,
) -> Tuple[Tokens, RNGKey]:
    """
    Update the input sequence of tokens using temperature sampling decoding
    with a decoder model. Typical inputs could be a prompt with end of tokens appended.
    This function can then be called recursively after the prompt to generate the
    rest of the sentence.

    Args:
        tokens_ids: Input tokens ids, shape = (batch_size, sequence_length).
        time_step: Time step at which to decode, shape = (,).
        random_key: Random key.
        params: Decoder parameters.
        apply_fn: Decoder apply fn.
        temperature: temperature coefficient for sampling.

    Returns:
        Tokens ids with decoded token at position time_step + 1 and updated random key.
    """
    logits = apply_fn(params, random_key, tokens_ids)["logits"]
    logits = logits[:, time_step, :]
    logits = logits / temperature
    random_key, sub_key = jax.random.split(random_key)
    new_token_id = jax.random.categorical(sub_key, logits, axis=-1)
    tokens_ids = tokens_ids.at[:, time_step + 1].set(new_token_id)
    return tokens_ids, random_key


def decode_greedy(
    init_tokens_ids: Tokens,
    random_key: RNGKey,
    params: hk.Params,
    apply_fn: hk.Transformed.apply,
    num_tokens_to_decode: int,
    eos_token_id: int,
) -> Tuple[Tokens, RNGKey]:
    """
    Takes a decoder network (assumes causal attention, e.g. a GPT network) and use
    it to decode a sequence of tokens starting from a tokenized prompt. Decoding
    is done greedily.

    Args:
        init_tokens_ids: Tokens ids of the prompt. Shape = (batch_size, seq_length).
        random_key: Random key.
        params: Decoder parameters.
        apply_fn: Decoder apply function.
        num_tokens_to_decode: Number of tokens to decode.
        eos_token_id: Id of the end of sequence token.

    Returns:
        The decoded tokens of shape (batch_size, seq_length + num_tokens_to_decode)
        and a new random key.

    """
    batch_size, init_seq_length = init_tokens_ids.shape[0], init_tokens_ids.shape[1]
    complete_tokens_ids = jnp.full(
        shape=(batch_size, num_tokens_to_decode),
        fill_value=eos_token_id,
    )
    tokens_ids = jnp.concatenate(
        [
            init_tokens_ids,
            complete_tokens_ids,
        ],
        axis=-1,
    )

    def scan_loop(carry_tokens_ids, time_step):  # type: ignore
        carry_tokens_ids, _ = update_tokens_ids_greedy(
            tokens_ids=carry_tokens_ids,
            time_step=time_step + init_seq_length - 1,
            random_key=random_key,
            params=params,
            apply_fn=apply_fn,
        )
        return carry_tokens_ids, None

    tokens_ids, _ = jax.lax.scan(
        f=scan_loop, init=tokens_ids, xs=jnp.arange(num_tokens_to_decode)
    )

    return tokens_ids, random_key


def decode_temperature_sampling(
    init_tokens_ids: Tokens,
    random_key: RNGKey,
    params: hk.Params,
    apply_fn: hk.Transformed.apply,
    num_tokens_to_decode: int,
    eos_token_id: int,
    temperature: float = 1.0,
) -> Tuple[Tokens, RNGKey]:
    """
    Takes a decoder network (assumes causal attention, e.g. a GPT network) and use
    it to decode a sequence of tokens starting from a tokenized prompt. Decoding
    is done using temperature sampling.

    Args:
        init_tokens_ids: Tokens ids of the prompt. Shape = (batch_size, seq_length).
        random_key: Random key.
        params: Decoder parameters.
        apply_fn: Decoder apply function.
        num_tokens_to_decode: Number of tokens to decode.
        eos_token_id: Id of the end of sequence token.
        temperature: Temperature coefficient for temperature sampling.

    Returns:
        The decoded tokens of shape (batch_size, seq_length + num_tokens_to_decode)
        and a new random key.

    """
    batch_size, init_seq_length = init_tokens_ids.shape[0], init_tokens_ids.shape[1]
    complete_tokens_ids = jnp.full(
        shape=(batch_size, num_tokens_to_decode),
        fill_value=eos_token_id,
    )
    tokens_ids = jnp.concatenate(
        [
            init_tokens_ids,
            complete_tokens_ids,
        ],
        axis=-1,
    )

    def scan_loop(carry, time_step):  # type: ignore
        carry_tokens_ids, key = carry
        carry_tokens_ids, key = update_tokens_ids_temperature_sampling(
            tokens_ids=carry_tokens_ids,
            time_step=time_step + init_seq_length - 1,
            random_key=key,
            params=params,
            apply_fn=apply_fn,
            temperature=temperature,
        )
        return (carry_tokens_ids, key), None

    (tokens_ids, random_key), _ = jax.lax.scan(
        f=scan_loop, init=(tokens_ids, random_key), xs=jnp.arange(num_tokens_to_decode)
    )

    return tokens_ids, random_key
