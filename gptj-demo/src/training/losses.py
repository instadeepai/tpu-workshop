from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from src.types import (
    AttentionMask,
    Dict,
    Labels,
    Metrics,
    RNGKey,
    SequenceMask,
    Tokens,
)


def causal_lm_loss(
    params: hk.Params,
    random_key: RNGKey,
    tokens: Tokens,
    sequence_mask: SequenceMask,
    apply_fn: hk.Transformed.apply,
) -> Tuple[jnp.ndarray, Metrics]:
    """
    Computes the causal language modelling loss using standard teacher forcing.

    Args:
        params: Network parameters.
        random_key: Random JAX key.
        tokens: Tokens (batch,seq_length).
        sequence_mask: To mask the loss (batch,seq_length).
        apply_fn: Apply forward function.

    Returns:
        Pytree of mean cross entropy over all output positions masked by
        sequence mask and mean accuracy of the predictions over all
        output positions masked by sequence mask.
    """
    # here we make the assumption that the causal attention mask is computed within
    # the call function of the model
    logits = apply_fn(params, random_key, tokens)["logits"]  # noqa
    # logits shape = (batch_size, seq_length, vocabulary_size)

    # we do not compute loss for final element in the sequence
    logits = logits[:, :-1, :]
    targets = tokens[:, 1:]
    mask = sequence_mask[:, :-1]
    logits = jnp.reshape(logits, (-1, logits.shape[-1]))
    targets = jnp.reshape(targets, -1)  # B X L
    loss = -jnp.sum(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits), jnp.expand_dims(targets, axis=1), axis=1
        ),
        axis=-1,
    )  # (B*L)
    loss = jnp.reshape(loss, newshape=tokens[:, 1:].shape)  # (batch_size, seq_length)
    # mask elements to not compute the loss on
    loss = loss * mask
    loss = jnp.sum(loss, axis=-1) / jnp.sum(mask, axis=-1)
    loss = jnp.mean(loss)
    return loss, {}


def cross_entropy_loss(
    params: hk.Params,
    random_key: RNGKey,
    tokens: Tokens,
    targets: Tokens,
    apply_fn: hk.Transformed.apply,
    sequence_mask: Optional[SequenceMask] = None,
    attention_mask: Optional[AttentionMask] = None,
) -> Tuple[jnp.ndarray, Metrics]:
    """
    Computes the cross entropy loss only for output positions depending on the
    sequence mask.

    WARNING: If no sequence mask is given then all output positions are
    considered in the loss.

    Args:
        params: Network parameters.
        random_key: Random JAX key.
        tokens: Tokens (batch,seq_length).
        targets: Targets (batch,seq_length).
        apply_fn: Apply forward function, must return a dictionary with a key named
            'logits'.
        sequence_mask: Sequence mask that dictates which output position are taken into
            account in the loss of shape (batch_size, seq_len).
        attention_mask: Attention Mask of shape (batch_size, 1, seq_len, seq_len).

    Returns:
        Pytree of mean cross entropy over all output positions masked by
        sequence mask and mean accuracy of the predictions over all
        output positions masked by sequence mask.
    """
    batch_size, sequence_length = tokens.shape[:2]

    logits = apply_fn(params, random_key, tokens, attention_mask)["logits"]  # (B,L,V)

    # if no sequence mask provided, consider all output positions
    if sequence_mask is None:
        sequence_mask = jnp.ones_like(tokens)

    accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets)  # (B,L)
    accuracy = jnp.sum(accuracy * sequence_mask, axis=-1) / jnp.sum(
        sequence_mask, axis=-1
    )  # (B,1)/(B,1)

    # Remove targets with where all tokens are <pad>
    nan_accuracy = jnp.isnan(accuracy)
    accuracy = jnp.where(~nan_accuracy, accuracy, 0)

    accuracy = jnp.sum(accuracy / jnp.sum(~nan_accuracy))

    logits = jnp.reshape(logits, (-1, logits.shape[-1]))
    targets = jnp.reshape(targets, -1)  # B X L
    loss = -jnp.sum(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits), jnp.expand_dims(targets, axis=1), axis=1
        ),
        axis=-1,
    )  # (B*L)
    loss = loss.reshape(batch_size, sequence_length)

    loss = jnp.sum(loss * sequence_mask, axis=-1) / jnp.sum(sequence_mask, axis=-1)
    loss = jnp.mean(loss)

    metrics = {}
    metrics["accuracy"] = accuracy
    return loss, metrics


def cross_entropy_loss_classification(
    params: hk.Params,
    random_key: RNGKey,
    tokens: Tokens,
    targets: Labels,
    forward_fn: Callable[
        [hk.Params, RNGKey, Tokens, Optional[AttentionMask], Optional[SequenceMask]],
        Dict[str, jnp.ndarray],
    ],
    attention_mask: Optional[AttentionMask] = None,
    sequence_mask: Optional[SequenceMask] = None,
) -> Tuple[jnp.ndarray, Metrics]:
    """
    Computes the cross entropy loss only for output positions depending on the
    sequence mask. The difference with the previous version is that it also returns
    the predictions.

    Args:
        params: Network parameters.
        random_key: Random JAX key.
        tokens: Tokens (batch, seq_length).
        targets: Targets (batch,).
        forward_fn: Apply forward function.
        attention_mask: Attention Mask of shape (batch_size, 1, seq_len, seq_len).
        sequence_mask: Sequence mask of shape (batch_size, seq_len).

    Returns:
        Mean cross entropy and a pytree containing the predictions for each sample.
    """

    logits = forward_fn(params, random_key, tokens, attention_mask, sequence_mask,)[
        "logits"
    ]  # (B, C)
    loss = -jnp.sum(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(targets, axis=1),
            axis=1,
        ),
        axis=-1,
    )  # (B,)
    loss = jnp.mean(loss)  # (B*L)
    predictions = jax.nn.softmax(logits, axis=-1)
    metrics = {}
    metrics["predictions"] = predictions
    return loss, metrics


def cross_entropy_loss_multilabel_classification(
    params: hk.Params,
    random_key: jnp.ndarray,
    tokens: jnp.ndarray,
    targets: jnp.ndarray,
    forward_fn: Callable[
        [
            hk.Params,
            RNGKey,
            Tokens,
            Optional[AttentionMask],
            Optional[AttentionMask],
        ],
        Dict[str, jnp.ndarray],
    ],
    num_classes: int,
    weight_pos: int = 1,
    weight_neg: int = 1,
    attention_mask: Optional[AttentionMask] = None,
    sequence_mask: Optional[SequenceMask] = None,
) -> jnp.ndarray:
    """Computes the cross entropy loss by taking the inputs as logits with the
    logsumexp subtracted to them. This loss assumes that all the labels predict
    the same number of classes.

    Args:
        params: Network parameters.
        random_key: Random JAX key.
        tokens: Tokens (batch, seq_length).
        targets: Targets (batch, num_labels).
        forward_fn: Apply forward function.
        num_classes: Number of classes per label,
        weight_pos: Weight assigned to the positive samples loss,
        weight_neg: Weight assigned to the negative samples loss,
        attention_mask: Attention Mask of shape (batch_size, 1, seq_len, seq_len).
        sequence_mask: Sequence mask of shape (batch_size, seq_len).

    Returns:
        Mean cross entropy and a pytree containing the predictions for each sample.
    """

    logits_per_label = forward_fn(
        params,
        random_key,
        tokens,
        attention_mask,
        sequence_mask,
    )[
        "logits"
    ]  # (batch_size, num_labels, num_classes)

    # Computing the cross entropy loss per label per sample
    loss_per_label = jnp.sum(
        jax.nn.log_softmax(logits_per_label, axis=-1)
        * jax.nn.one_hot(targets, num_classes=num_classes, axis=-1),
        axis=-1,
    )  # [batch_size, num_labels]

    # Weights_tensor
    weights = jnp.where(targets == 1, weight_pos, weight_neg)

    # Ponderate the losses
    loss_per_label = loss_per_label * weights

    # Sum or average the loss over the labels and batch
    total_loss = jnp.mean(loss_per_label)

    # Computing the predictions
    preds_per_label = jax.nn.softmax(
        logits_per_label, axis=-1
    )  # batch_size, num_labels, num_class

    metrics = {}
    metrics["predictions"] = preds_per_label
    return -total_loss, metrics


def cross_entropy_loss_multiregression(
    params: hk.Params,
    random_key: jnp.ndarray,
    tokens: jnp.ndarray,
    targets: jnp.ndarray,
    forward_fn: Callable,
    attention_mask: Optional[AttentionMask] = None,
    sequence_mask: Optional[SequenceMask] = None,
) -> jnp.ndarray:
    """
    Computes the cross entropy loss between the logits outputted by the network
    and the continuous labels represented by their bins probabilities.



    Args:
        params: Network parameters.
        random_key: Random JAX key.
        tokens: Tokens (batch, seq_length).
        targets: Targets (batch, num_labels, num_bins + 1).
        forward_fn: Apply forward function.
        attention_mask: Attention Mask of shape (batch_size, 1, seq_len, seq_len).
        sequence_mask: Sequence mask of shape (batch_size, seq_len).

    Returns:
        Mean cross entropy and a pytree containing the predictions for each sample.
    """

    logits_per_label = forward_fn(
        params,
        random_key,
        tokens=tokens,
        attention_mask=attention_mask,
        sequence_mask=sequence_mask,
    )[
        "logits"
    ]  # (batch_size, num_labels, num_classes)
    # Computing the cross entropy loss
    loss_per_label = jnp.mean(
        jax.nn.log_softmax(logits_per_label, axis=-1) * targets,
        axis=0,
    )  # (num_labels, num_classes)

    # Can be changed to jnp.mean
    total_loss = jnp.mean(loss_per_label)

    # Compute the predictions as floats
    predictions = jax.nn.softmax(
        logits_per_label, axis=-1
    )  # (batch_size, num_labels, num_bins)

    metrics = {}
    metrics["predictions"] = predictions
    return -total_loss, metrics


def cross_entropy_loss_encoder_decoder(
    params: hk.Params,
    random_key: jnp.ndarray,
    encoder_tokens: jnp.ndarray,
    decoder_tokens: jnp.ndarray,
    targets: jnp.ndarray,
    forward_fn: Callable[
        [
            hk.Params,
            RNGKey,
            Tokens,
            Tokens,
            Optional[AttentionMask],
            Optional[AttentionMask],
            Optional[AttentionMask],
        ],
        Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    ],
    denoising_sequence_mask: SequenceMask,
    decoder_sequence_mask: Optional[SequenceMask] = None,
    encoder_attention_mask: Optional[AttentionMask] = None,
    decoder_attention_mask: Optional[AttentionMask] = None,
    cross_attention_mask: Optional[AttentionMask] = None,
) -> Tuple[jnp.ndarray, Metrics]:

    """
    Computes the cross entropy loss only for output positions depending on the decoder
    tokens sequence mask. Used for encoder decoder models where the forward function
    takes a set of tokens for the encoder and a set of tokens for the decoder.

    WARNING: If no sequence mask is given then all output positions are
    considered in loss.

    Args:
        params: Network parameters.
        random_key: Random JAX key.
        encoder_tokens: Tokens (batch,encoder_seq_length).
        decoder_tokens: Tokens (batch,decoder_seq_length).
        targets: Targets (batch,seq_length).
        forward_fn: Apply forward function.
        decoder_sequence_mask: Sequence Mask that dictates which output position are
        taken into account in the loss of shape (batch_size, decoder_seq_len).
        encoder_attention_mask: Attention Mask of shape
        (batch_size, 1, encoder_seq_len, encoder_seq_len).
        decoder_attention_mask: Attention Mask of shape
        (batch_size, 1, decoder_seq_len, decoder_seq_len).
        cross_attention_mask: Attention Mask of shape
        (batch_size, 1, decoder_seq_len, encoder_seq_len).

    Returns:
        Metrics as a pytree, containing mean cross entropy , mean accuracy, mean
        denoising accuracy, and mean causal accuracy. All of these metrics are over all
        output positions masked by decoder sequence mask.
    """
    batch_size, sequence_length = decoder_tokens.shape[:2]

    _, decoder_outs = forward_fn(  # type: ignore
        params,
        random_key,
        encoder_tokens,
        decoder_tokens,
        encoder_attention_mask=encoder_attention_mask,
        decoder_attention_mask=decoder_attention_mask,
        decoder_cross_attention_mask=cross_attention_mask,
    )

    logits = decoder_outs["logits"]  # (B,L,V)

    # if no sequence mask provided, consider all output positions
    if decoder_sequence_mask is None:
        decoder_sequence_mask = jnp.ones_like(decoder_tokens)

    accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets)  # (B,L)

    denoising_accuracy = jnp.sum(
        accuracy * decoder_sequence_mask * denoising_sequence_mask, axis=-1
    ) / jnp.sum(
        decoder_sequence_mask * denoising_sequence_mask, axis=-1
    )  # (B,1)/(B,1)

    causal_accuracy = jnp.sum(
        accuracy * decoder_sequence_mask * ~denoising_sequence_mask, axis=-1
    ) / jnp.sum(
        decoder_sequence_mask * ~denoising_sequence_mask, axis=-1
    )  # (B,1)/(B,1)

    accuracy = jnp.sum(accuracy * decoder_sequence_mask, axis=-1) / jnp.sum(
        decoder_sequence_mask, axis=-1
    )  # (B,1)/(B,1)

    # Remove targets with only padding
    nan_accuracy = jnp.isnan(accuracy)
    accuracy = jnp.where(~nan_accuracy, accuracy, 0)
    accuracy = jnp.sum(accuracy / jnp.sum(~nan_accuracy))

    nan_accuracy = jnp.isnan(denoising_accuracy)
    bert_accuracy = jnp.where(~nan_accuracy, denoising_accuracy, 0)
    bert_accuracy = jnp.sum(bert_accuracy / jnp.sum(~nan_accuracy))

    nan_accuracy = jnp.isnan(causal_accuracy)
    causal_accuracy = jnp.where(~nan_accuracy, causal_accuracy, 0)
    causal_accuracy = jnp.sum(causal_accuracy / jnp.sum(~nan_accuracy))

    logits = jnp.reshape(logits, (-1, logits.shape[-1]))
    targets = jnp.reshape(targets, -1)  # B X L

    loss = -jnp.sum(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits), jnp.expand_dims(targets, axis=1), axis=1
        ),
        axis=-1,
    )  # (B*L)

    loss = loss.reshape(batch_size, sequence_length)

    loss = jnp.mean(loss * decoder_sequence_mask, axis=-1)
    loss = jnp.mean(loss)

    metrics: Metrics = {}
    metrics["accuracy"] = accuracy
    metrics["bert_accuracy"] = bert_accuracy
    metrics["causal_accuracy"] = causal_accuracy

    return loss, metrics


def mse_loss(
    params: hk.Params,
    random_key: RNGKey,
    tokens: Tokens,
    targets: Tokens,
    forward_fn: Callable[
        [hk.Params, RNGKey, Tokens, Optional[AttentionMask]],
        Dict[str, jnp.ndarray],
    ],
    sequence_mask: AttentionMask,
    attention_mask: AttentionMask,
) -> Tuple[jnp.ndarray, Metrics]:
    """
    Mean-squared error loss.

    Args:
        params: Model parameters.
        random_key: Random key.
        tokens: A batch of tokens.
        targets: Ground truths.
        forward_fn: Model forward function.
        sequence_mask: Sequence mask to discard loss terms computed on pad values.
        attention_mask: Attention mask.

    Returns:
        The average loss over the inputted batch.
        A dictionary with metrics.
    """
    # Predictions
    predictions = forward_fn(params, random_key, tokens, attention_mask)["predictions"]

    error = targets - predictions
    loss = 0.5 * jnp.mean(jnp.square(error), where=sequence_mask)
    metrics: Dict[str, jnp.ndarray] = {}
    return loss, metrics
