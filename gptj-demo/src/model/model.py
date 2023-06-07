""" Implement the abstract Class of a GPT-like models.
It is an autoregressive decoder.
It will be used for architecture such as LLAMA, GPTJ, BLOOM, Gopher """


from dataclasses import dataclass
from typing import Callable, Optional

import haiku as hk
import jax.numpy as jnp
import jmp

from src.model.lm_head import SimpleLMHead
from src.model.layers import GPTDecoderLayer
from src.types import Embedding, Tokens, TransformerOutput
from src.utils.masking import build_causal_attention_mask


@dataclass
class GptConfig:
    """
    Parameters to initialize a Gpt model.

    NOTE: the pad token is not defined

    Args:
        vocab_size: Token vocabulary.
        eos_token_id: used to stop sentence generation
        embed_dim: Embedding dimension.
        ffn_embed_dim: Feed forward embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of Decoder layer_stack
        rope_dimensions: the number of dimension per attention head to which the
                RoPE ( rotary positional embeddings ) is applied
                If None, the RoPE is applied on every dimensions in the key space
        max_position_embeddings: the maximum input length
        add_bias_ffn: Add bias in feed forward network block.
        ffn_activation_name: Activation function to be used in FFN block. Supported
            names are "gelu", "gelu-no-approx", "relu", "swish".
        use_glu_in_ffn: whether to use Gated Linear Unit (GLU) in Feed
            Forward Network (FFN) block.
            example: To do a swiGLU (gated-swish) put this arg
            to True and use swish as ffn_activation_name.
            Same principle for a gated-relu.
        add_bias_lm_head: whether to use bias in the final LM layer
    """

    # vocabulary
    vocab_size: int
    eos_token_id: int

    # architecture
    embed_dim: int
    ffn_embed_dim: int
    num_heads: int
    num_layers: int
    rope_dimensions: Optional[int]
    max_position_embeddings: int
    add_bias_ffn: bool
    ffn_activation_name: str
    use_glu_in_ffn: bool
    add_bias_lm_head: bool
    norm_type: str
    parallel_attention_ff: bool

    use_gradient_checkpointing: bool

    def __post_init__(self) -> None:
        """
        Checks that the given values are compatible.
        """
        if not self.embed_dim % self.num_heads == 0:
            raise ValueError(
                f"The embedding dimension should be "
                f"divisible by the number of heads, however provided embedding "
                f"dimension is {self.embed_dim} and the number of heads is "
                f"{self.num_heads}."
            )

        if not self.embed_dim // self.num_heads > 1:
            raise ValueError(
                "embed_dim / num_heads must be higher than 2 to apply rotary embeddings"
            )
        if not (self.rope_dimensions is None):
            if not self.embed_dim // self.num_heads >= self.rope_dimensions:
                raise ValueError(
                    "embed_dim // num_heads must be higher than rope_dimensions "
                    "to apply rotary embeddings"
                )


class GPTDecoder(hk.Module):
    """
    Creates the Gpt model ( decoder only ).
    """

    def __init__(
        self,
        config: GptConfig,
        name: Optional[str] = None,
    ):
        """
        Initializes the Decoder stack.

        Args:
            config: Configuration data class
            name: haiku module name
        """

        self._config = config

        super().__init__(name=name)

        self.token_embed = hk.Embed(
            vocab_size=config.vocab_size, embed_dim=config.embed_dim, name="token_embed"
        )
        self.lm_head = SimpleLMHead(
            embed_dim=config.embed_dim,
            alphabet_size=config.vocab_size,
            add_bias_lm_head=self._config.add_bias_lm_head,
        )
        if self._config.norm_type == "layer_norm":
            self.final_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name="final_layer_norm"
            )
        elif self._config.norm_type == "RMS_norm":
            self.final_norm = hk.RMSNorm(
                axis=-1, create_scale=True, name="final_RMS_norm"
            )
        else:
            raise ValueError(
                f"unrecognized norm_type in config {self._config.norm_type}"
            )

    @hk.experimental.name_like("__call__")
    def decoder_layer(self, layer_idx: int) -> GPTDecoderLayer:
        """
        Returns the GPT encoder layer.

        Args:
            layer_idx: the layer index

        Returns:
            The named GPTDecoderLayer
        """
        return GPTDecoderLayer(
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

    @hk.transparent
    def apply_transformer_layers(self, tokens_embeddings: Embedding) -> Embedding:
        """
        Takes as inputs the tokens embeddings and apply successively an attention mask
        and the transformer layers to obtain final embeddings ready to be decoded.

        Args:
            tokens_embeddings: tokens embeddings

        Returns:
            Embeddings transformed through successive transformer layers.
        """
        # compute causal attention mask
        attention_mask = build_causal_attention_mask(1, tokens_embeddings.shape[1])

        # go through the transformer layer_stack
        layer_stack = [self.decoder_layer(i) for i in range(self._config.num_layers)]

        # use gradient checkpointing if required
        if self._config.use_gradient_checkpointing:
            # the remat-ed function cannot take control flow arguments
            layer_stack = [hk.remat(layer) for layer in layer_stack]

        embeddings = tokens_embeddings
        for i in range(len(layer_stack)):
            embeddings = layer_stack[i](
                embeddings=embeddings,
                attention_mask=attention_mask,
            )
        return embeddings

    def __call__(self, token_ids: Tokens) -> TransformerOutput:
        """
        Compute the logits and embeddings from a sequence of tokens

        Args:
            token_ids: Sequence of token ids delivered to the decoder of shape
                (batch_size, seq_len)

        Returns:
             The logits over the token vocabulary at each time step.
        """
        # (batch_size,seq_len) -> (batch_size,seq_len,embed_dim)
        tokens_embeddings = self.token_embed(token_ids)
        # (batch_size,seq_len,embed_dim) -> (batch_size,seq_len,embed_dim)
        embeddings = self.apply_transformer_layers(tokens_embeddings)
        embeddings = self.final_norm(embeddings)

        # get outputs
        outs = {}
        outs["embeddings"] = embeddings

        # compute logits
        logits = self.lm_head(outs["embeddings"])["logits"]
        outs["logits"] = logits

        return outs  # type: ignore


def build_gpt_fn(
    config: GptConfig,
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    output_dtype: jnp.dtype = jnp.float32,
    name: Optional[str] = None,
) -> Callable:
    """
    Create the model's forward pass.

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
        Gpt Decoder model forward function.
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
    hk.mixed_precision.set_policy(GPTDecoder, policy)

    # Remove it in batch norm to avoid instabilities
    norm_policy = jmp.Policy(
        compute_dtype=compute_dtype, param_dtype=param_dtype, output_dtype=output_dtype
    )
    hk.mixed_precision.set_policy(hk.LayerNorm, norm_policy)
    hk.mixed_precision.set_policy(hk.RMSNorm, norm_policy)

    def gpt_fn(token_ids: jnp.ndarray) -> TransformerOutput:
        model = GPTDecoder(config, name=name)
        return model(token_ids=token_ids)

    return gpt_fn
