from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jmp

from src.model.positional import apply_rotary_pos_emb
from src.model.layers import GPTDecoderLayer, GPTMultiHeadAttention
from src.model.model import GptConfig, GPTDecoder
from src.types import Embedding, TransformerOutput


class GPTMultiHeadAttentionIA3Rescaling(GPTMultiHeadAttention):
    """
    GPT multi-head attention with IA3 rescaling.
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
        GPTMultiHeadAttention.__init__(
            self,
            embed_dim=embed_dim,
            num_heads=num_heads,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            key_size=key_size,
            name=name,
        )

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

        # IA3 rescaling
        key_ia3_rescaling = hk.get_parameter(
            "key_ia3_rescaling",
            shape=[keys.shape[-2], keys.shape[-1]],
            dtype=keys.dtype,
            init=hk.initializers.Constant(1.0),
        )
        keys = keys * key_ia3_rescaling

        value_ia3_rescaling = hk.get_parameter(
            "value_ia3_rescaling",
            shape=[values.shape[-2], values.shape[-1]],
            dtype=values.dtype,
            init=hk.initializers.Constant(1.0),
        )
        values = values * value_ia3_rescaling

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


class GPTDecoderLayerIA3Rescaling(GPTDecoderLayer):
    """
    GPT decoder layer with IA3 rescaling.
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
        add_bias_ffn: bool = True,
        ffn_activation_name: str = "gelu",
        use_glu_in_ffn: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            num_heads=num_heads,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            norm_type=norm_type,
            parallel_attention_ff=parallel_attention_ff,
            add_bias_ffn=add_bias_ffn,
            ffn_activation_name=ffn_activation_name,
            use_glu_in_ffn=use_glu_in_ffn,
            name=name,
        )
        self.sa_layer = GPTMultiHeadAttentionIA3Rescaling(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            name=hk.experimental.force_name(self.sa_layer.module_name),  # type: ignore
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

        # IA3 rescaling
        ffn_ia3_rescaling = hk.get_parameter(
            "ffn_ia3_rescaling",
            shape=[x.shape[-1]],
            dtype=x.dtype,
            init=hk.initializers.Constant(1.0),
        )
        x = x * ffn_ia3_rescaling

        x = self.fc2_linear(x)
        return x


class GPTDecoderIA3Rescaling(GPTDecoder):
    """
    GPT-J model with IA3 rescaling.
    """

    @hk.experimental.name_like("__call__")
    def decoder_layer(self, layer_idx: int) -> GPTDecoderLayer:
        """
        gives the GPTJ encoder layer
        Args:
            layer_idx: the layer index

        Returns:
            The named GPTJDecoderLayer
        """
        return GPTDecoderLayerIA3Rescaling(
            embed_dim=self._config.embed_dim,
            ffn_embed_dim=self._config.ffn_embed_dim,
            num_heads=self._config.num_heads,
            rotary_dim=self._config.rope_dimensions,
            max_position_embeddings=self._config.max_position_embeddings,
            norm_type=self._config.norm_type,
            parallel_attention_ff=self._config.parallel_attention_ff,
            add_bias_ffn=self._config.add_bias_ffn,
            ffn_activation_name=self._config.ffn_activation_name,
            use_glu_in_ffn=self._config.use_glu_in_ffn,
            name=f"gpt_decoder_layer_{layer_idx}",
        )


def build_gpt_ia3_rescaling_fn(
    config: GptConfig,
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    output_dtype: jnp.dtype = jnp.float32,
    name: Optional[str] = None,
) -> Callable:
    """
    Create the model's forward pass. No classification head in this function
    as for decoder only models, one simply keep tuning the model language head.

    Args:
        config: Configuration data class containing the hyperparameters for the GPT
            forward function.
        compute_dtype: the type of the activations. fp16 runs faster and is lighter in
            memory. bf16 handles better large int, and is hence more stable ( it avoids
            float overflows ).
        param_dtype: if compute_dtype is fp16, the model weights will be cast to fp16
            during the forward pass anyway. So in inference mode ( not training mode ),
            it is better to use params in fp16 if compute_dtype is fp16 too
        output_dtype: the output type of the model. it determines the float precioson
            of the gradient when training the model.
            NOTE: when training, the gradient is often accumulated in fp32, therefore
            output_dtype need to be in fp32.
        name: the name of the model. example: gpt_j_decoder.


        # NOTE: in inference, the model could be in fp16 without too much degradation
        # NOTE: on NVIDIA accelerator, XLA inter-device operation ( psum, all_gather,
        etc ... ) are not always implemented for bf16. but on TPU hardware yes


    Returns:
        Gpt with IA3 finetuning model forward function.
    """

    assert {compute_dtype, param_dtype, output_dtype}.issubset(
        {
            jnp.bfloat16,
            jnp.float32,
            jnp.float16,
        }
    ), f"provide a dtype in {jnp.bfloat16, jnp.float32, jnp.float16}"

    policy = jmp.Policy(
        compute_dtype=compute_dtype, param_dtype=param_dtype, output_dtype=output_dtype
    )
    hk.mixed_precision.set_policy(GPTDecoderIA3Rescaling, policy)

    # Remove it in batch norm to avoid instabilities
    norm_policy = jmp.Policy(
        compute_dtype=compute_dtype, param_dtype=param_dtype, output_dtype=output_dtype
    )
    hk.mixed_precision.set_policy(hk.LayerNorm, norm_policy)
    hk.mixed_precision.set_policy(hk.RMSNorm, norm_policy)

    def gptj_fn(
        token_ids: jnp.ndarray,
    ) -> TransformerOutput:
        # rename model to based model name to ensure compatibility between
        # parameters dictionaries
        model = GPTDecoderIA3Rescaling(config, name=name)

        outs = model(token_ids=token_ids)
        return outs

    return gptj_fn
