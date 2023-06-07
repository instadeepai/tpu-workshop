from typing import Optional, Callable

import haiku as hk
import jax
import jax.numpy as jnp

from src.model.positional import (
    apply_rotary_pos_emb,
    create_sinusoidal_positions,
)
from src.types import AttentionMask, Embedding

SUPPORTED_FFN_ACTIVATIONS = ["gelu", "gelu-no-approx", "relu", "swish", "silu"]

def get_activation_fn(activation_name: str) -> Callable:
    """
    Return activation fn given its name.
    Args:
        activation_name: Activation name.

    Returns:
        activation function.
    """
    if activation_name not in SUPPORTED_FFN_ACTIVATIONS:
        raise NotImplementedError(
            f"Activation {activation_name} not supported yet. "
            f"Supported activations for feed forward "
            f"block are {SUPPORTED_FFN_ACTIVATIONS}"
        )
    if activation_name == "gelu-no-approx":
        activation_fn = lambda x: jax.nn.gelu(x, approximate=False)  # noqa: E731
    else:
        activation_fn = getattr(jax.nn, activation_name)
    return activation_fn


class GPTMultiHeadAttention(hk.Module):
    """
    Multi-head attention with masking applied. Modified from Haiku implementation to
    be able to support relative positional embeddings.  Computes the keys, queries, and
    values from the input embeddings.  Future versions could compute and store these
    to prevent redundant calculations during autoregressive inference. The keys, queries
    , and value sizes are fixed to be embed_dim/num_heads in accordance with the
    standard GPT model.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rotary_dim: Optional[int],
        max_position_embeddings: int,
        key_size: Optional[int] = None,
        name: Optional[str] = "attention",
    ):
        """
        Initializes the attention layer.

        Args:
            embed_dim: Length of the token embedding at each position in the sequence.
            num_heads: Number of independent attention heads.
            rotary_dim: The dimension of the rotary positional embedding in each key
                space
            max_position_embeddings: the maximum positions used for the computation of
                the RoPE
            key_size: dimension of the key vectors
            name: Optional name for this module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = embed_dim
        self.key_size = key_size or self.embed_dim // self.num_heads
        self.rotary_dim = rotary_dim or self.key_size

        self.key_linear = hk.Linear(
            output_size=self.key_size * num_heads, with_bias=False, name="key_linear"
        )
        self.query_linear = hk.Linear(
            output_size=self.key_size * num_heads, with_bias=False, name="query_linear"
        )
        self.value_linear = hk.Linear(
            output_size=self.key_size * num_heads, with_bias=False, name="value_linear"
        )

        self.out_linear = hk.Linear(
            output_size=embed_dim, with_bias=False, name="out_linear"
        )

        self.sincos_positions = None

    def get_sincos_positions(self) -> jnp.ndarray:
        """
        Generate the sincos_positions tensor
        Returns:
            array of shape (max_position_embeddings, rotary_dim) containing the sinus
            and cosinus for the RoPE embedding
        """
        if self.sincos_positions is None:
            self.sincos_positions = create_sinusoidal_positions(
                self.max_position_embeddings, self.rotary_dim
            )
        return self.sincos_positions

    def __call__(
        self,
        query_inputs: jnp.ndarray,
        key_inputs: jnp.ndarray,
        value_inputs: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Computes the result of multiheaded dot-product attention, using
        pre-computed projections for the queries, keys, and values.

        Args:
            query_inputs: Embeddings that will be projected to become the queries.
            key_inputs: Embeddings that will be projected to become the keys.
            value_inputs: Embeddings that will be projected to become the values.
            attention_mask: Mask to be applied in the attention layers.
                Triangular for autoregressive models.
                shape : (1, 1, seq_len, seq_len)

        Returns:
            The standard output of multi-headed attention
        """
        position_ids = jnp.arange(0, key_inputs.shape[1], 1, dtype=jnp.int32)
        position_ids = jnp.expand_dims(position_ids, 0).repeat(key_inputs.shape[0], 0)

        keys = self.key_linear(key_inputs)
        queries = self.query_linear(query_inputs)
        values = self.value_linear(value_inputs)

        keys = keys.reshape(keys.shape[0], keys.shape[1], self.num_heads, -1)
        queries = queries.reshape(
            queries.shape[0], queries.shape[1], self.num_heads, -1
        )
        values = values.reshape(values.shape[0], values.shape[1], self.num_heads, -1)

        sincos = jnp.take(self.get_sincos_positions(), position_ids, axis=0)
        sincos = jnp.split(sincos, 2, axis=-1)

        if self.rotary_dim is not None:
            k_rot = keys[:, :, :, : self.rotary_dim]
            k_pass = keys[:, :, :, self.rotary_dim :]

            q_rot = queries[:, :, :, : self.rotary_dim]
            q_pass = queries[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)

            keys = jnp.concatenate([k_rot, k_pass], axis=-1)
            queries = jnp.concatenate([q_rot, q_pass], axis=-1)
        else:

            keys = apply_rotary_pos_emb(keys, sincos)
            queries = apply_rotary_pos_emb(queries, sincos)

        attention_logits = jnp.einsum("...thd,...Thd->...htT", queries, keys)
        sqrt_key_size = jnp.sqrt(keys.shape[-1]).astype(queries.dtype)
        attention_logits = attention_logits / sqrt_key_size

        attention_logits = jnp.where(attention_mask, attention_logits, -1e30)

        attention_weights = jax.nn.softmax(attention_logits, axis=-1)

        values = jnp.einsum("...htT,...Thd->...thd", attention_weights, values)
        values = jnp.reshape(values, (values.shape[0], values.shape[1], -1))

        return self.out_linear(values)


class GPTDecoderLayer(hk.Module):
    """
    Single layer in the encoder, including self-attention and feed-forward operations.
    The feed-forward network uses a ReLU activation and has no biases.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        num_heads: int,
        rotary_dim: Optional[int],
        max_position_embeddings: int,
        norm_type: str,
        parallel_attention_ff: bool,
        add_bias_ffn: bool,
        ffn_activation_name: str,
        use_glu_in_ffn: bool,
        name: Optional[str] = None,
    ):
        """
        Initializes the encoder layer, including the projections needed for
        self-attention and the linear layers applied in the fully connected portion

        Args:
            embed_dim: Dimension of the embeddings
            ffn_embed_dim: Dimension of the hidden layer in the MLP
            num_heads: Number of independent attention heads.
            rotary_dim: The dimension in key space to apply the rotary positional
                embeddings
            max_position_embeddings: The maximum length to apply rotary positional
                embeddings
            norm_type: The type of norm used ( pre normalization scheme ) used. can be
                one of ["layer_norm", "RMS_norm"]
            parallel_attention_ff: Whether to do the attention and the MLP in parallel,
                and then sum up the results as it is done in GPT-NeoX :
                Black, Sid, et al. "Gpt-neox-20b: An open-source autoregressive
                language model." arXiv preprint arXiv:2204.06745 (2022).
                It is said to improve the training time of 15% when compiling with JAX
            add_bias_ffn: Add bias in feed forward network block.
            ffn_activation_name: Activation function to be used in FFN block. Supported
                names are "gelu", "gelu-no-approx", "relu", "swish", and "silu"
            use_glu_in_ffn: Whether to use Gated Linear Unit (GLU) in Feed
                Forward Network (FFN) block. To do a swiGLU (gated-swish) put this arg
                to True and use swish as ffn_activation_name.
                Same principle for a gated-relu.
            name: Optional name for this module.
        """
        super().__init__(name=name)

        self.num_heads = num_heads
        self.parallel_attention_ff = parallel_attention_ff
        self.sa_layer = GPTMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            name="self_attn",
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
        )

        if norm_type == "layer_norm":
            self.attn_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name="attn_layer_norm"
            )
            if not (self.parallel_attention_ff):
                self.ffn_norm = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name="ffn_layer_norm",
                )
        elif norm_type == "RMS_norm":
            self.attn_norm = hk.RMSNorm(
                axis=-1, create_scale=True, name="attn_RMS_norm", eps=1e-6
            )
            if not (self.parallel_attention_ff):
                self.ffn_norm = hk.RMSNorm(
                    axis=-1, create_scale=True, name="ffn_RMS_norm", eps=1e-6
                )
        else:
            raise ValueError(f"unrecognized norm_type : {norm_type}")

        # Get ffn activation function
        self._ffn_activation_fn = get_activation_fn(activation_name=ffn_activation_name)
        self._use_glu_in_fnn = use_glu_in_ffn

        # Define layers
        if use_glu_in_ffn:
            # user should multiply ffn_embed_dim by 2/3 when using GLU
            # to keep total number of parameters equal
            # see https://arxiv.org/pdf/2002.05202.pdf. for more details
            # we multiply by 2 here as the output will be split in 2 for GLU
            ffn_embed_dim = int(2 * ffn_embed_dim)

        self.fc1_linear = hk.Linear(
            output_size=ffn_embed_dim,
            with_bias=add_bias_ffn,
            name="fc1_linear_glu" if use_glu_in_ffn else "fc1_linear",
        )
        self.fc2_linear = hk.Linear(
            output_size=embed_dim, with_bias=add_bias_ffn, name="fc2_linear"
        )

    @hk.transparent
    def mlp(self, x: Embedding) -> Embedding:
        """
        Applies one linear layer, a ReLU activation, dropout, then a final linear layer.

        Args:
            x: Embeddings of shape (batch_size, seq_len, embed_dim).

        Returns:
            The transformed sequence embedding.
        """
        if self._use_glu_in_fnn:
            x1, x2 = jnp.split(self.fc1_linear(x), indices_or_sections=2, axis=-1)
            x = self._ffn_activation_fn(x1) * x2
        else:
            x = self._ffn_activation_fn(self.fc1_linear(x))

        x = self.fc2_linear(x)
        return x

    def __call__(
        self,
        embeddings: Embedding,
        attention_mask: AttentionMask,
    ) -> Embedding:
        """
        Computes the output embeddings of the encoder layer.
        if self.parallel_attention_ff, the model uses parallel MLP and attention

        Args:
            embeddings: Decoder layer input embeddings of shape
                (batch_size,seq_len,embed_dim).
            attention_mask: Mask to be applied in the attention layers.
                Triangular for autoregressive models.
                shape = (1, 1, seq_len, seq_len)

        Returns:
            The output embeddings that result from the application of the layer
        """
        if self.parallel_attention_ff:
            residuals = embeddings

            embeddings = self.attn_norm(embeddings)
            attn_outputs = self.sa_layer(
                query_inputs=embeddings,
                key_inputs=embeddings,
                value_inputs=embeddings,
                attention_mask=attention_mask,
            )
            mlp_ouputs = self.mlp(embeddings)
            return residuals + attn_outputs + mlp_ouputs
        else:
            normed_embeddings = self.attn_norm(embeddings)
            # print("embeddings", embeddings.shape)
            # print("normed_embeddings", normed_embeddings.shape)
            # tmp = self.sa_layer(
            #     query_inputs=normed_embeddings,
            #     key_inputs=normed_embeddings,
            #     value_inputs=normed_embeddings,
            #     attention_mask=attention_mask,
            # )
            # print("tmp", tmp.shape)
            attn_outputs = embeddings + self.sa_layer(
                query_inputs=normed_embeddings,
                key_inputs=normed_embeddings,
                value_inputs=normed_embeddings,
                attention_mask=attention_mask,
            )
            mlp_ouputs = attn_outputs + self.mlp(self.ffn_norm(attn_outputs))

            return mlp_ouputs
