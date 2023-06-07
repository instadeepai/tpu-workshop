"""Implementation of utilities to load a pretrained gptj model in Trix."""
import gc
from collections import defaultdict
from typing import Callable, Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from src.model.model import GptConfig, build_gpt_fn

# in order to disable hugging face transformer module's warning
logging.set_verbosity_error()

GPTJ_MODEL_NAME = "gpt_j_decoder"


def translate_torch_params(
    torch_params: torch.nn.ParameterDict, num_layers: int = 28, dtype: jnp.dtype = jnp.float32,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Converts the full Hugging Face PyTorch gptj model to Haiku parameters.
    Note that the parameters names match the defaults defined in the Trix gptj
    architecture.  Other choices may cause errors.  See:
    https://github.com/huggingface/transformers/tree/main/src/transformers/models/gptj
    The torch_params can be retrieved using
    T5Model.from_pretrained(<model_name>).state_dict()

    Args:
        torch_params: Pretrained PyTorch gptj model state_dict().
        num_layers: Number of decoder layers in the desired pre-trained model.

    Returns:
       Dictionary of Haiku parameters.
    """
    translate_dict = {}
    gptj_prfix = f"{GPTJ_MODEL_NAME}/"

    translate_dict["transformer.wte.weight"] = (
        gptj_prfix + "~/token_embed",
        "embeddings",
    )

    for k in range(num_layers):
        prefix_layer = gptj_prfix + f"gpt_decoder_layer_{k}/~/"
        torch_prefix = f"transformer.h.{k}."

        # Layer norm
        translate_dict[torch_prefix + "ln_1.weight"] = (
            prefix_layer + "attn_layer_norm",
            "scale",
        )
        translate_dict[torch_prefix + "ln_1.bias"] = (
            prefix_layer + "attn_layer_norm",
            "offset",
        )
        # attention K, Q, V projections
        translate_dict[torch_prefix + "attn.k_proj.weight"] = (
            prefix_layer + "self_attn/~/key_linear",
            "w",
        )
        translate_dict[torch_prefix + "attn.v_proj.weight"] = (
            prefix_layer + "self_attn/~/value_linear",
            "w",
        )
        translate_dict[torch_prefix + "attn.q_proj.weight"] = (
            prefix_layer + "self_attn/~/query_linear",
            "w",
        )
        translate_dict[torch_prefix + "attn.out_proj.weight"] = (
            prefix_layer + "self_attn/~/out_linear",
            "w",
        )
        # MLP dense layers
        translate_dict[torch_prefix + "mlp.fc_in.weight"] = (
            prefix_layer + "fc1_linear",
            "w",
        )
        translate_dict[torch_prefix + "mlp.fc_in.bias"] = (
            prefix_layer + "fc1_linear",
            "b",
        )
        translate_dict[torch_prefix + "mlp.fc_out.weight"] = (
            prefix_layer + "fc2_linear",
            "w",
        )
        translate_dict[torch_prefix + "mlp.fc_out.bias"] = (
            prefix_layer + "fc2_linear",
            "b",
        )

    translate_dict["lm_head.weight"] = (
        gptj_prfix + "~/simple_lm_head/~/lm_final_fc",
        "w",
    )
    translate_dict["lm_head.bias"] = (
        gptj_prfix + "~/simple_lm_head/~/lm_final_fc",
        "b",
    )
    translate_dict["transformer.ln_f.weight"] = (
        gptj_prfix + "~/final_layer_norm",
        "scale",
    )
    translate_dict["transformer.ln_f.bias"] = (
        gptj_prfix + "~/final_layer_norm",
        "offset",
    )

    params: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for torch_key, (trix_key, weight_key) in translate_dict.items():
        if "weight" in torch_key and not ("wte" in torch_key):
            # in pytorch, the weights of dense matrices indexation is transposes
            # compared to haiku, except for word-token-embedding
            params[trix_key][weight_key] = jnp.array(torch_params[torch_key], dtype=dtype).transpose()
        else:
            params[trix_key][weight_key] = jnp.array(torch_params[torch_key], dtype=dtype)

    return dict(params)


def get_pretrained_gptj_model(
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    output_dtype: jnp.dtype = jnp.float32,
) -> Tuple[hk.Params, Callable, AutoTokenizer, GptConfig]:
    """
    Create a Haiku gptj model by downloading the pytorch weights hosted by huggin face
        and translating them.

    Args:
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

    Returns:
        Model parameters.
        Haiku function to call the model.
        Tokenizer.
        Model config
    """

    if param_dtype == jnp.float16 or param_dtype == jnp.bfloat16:
        model_params = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
        ).state_dict()
    elif param_dtype == jnp.float32:
        model_params = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
        ).state_dict()
    else:
        raise ValueError(f"unaccepted param_dtype {param_dtype}")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token_id == 50256

    # NOTE: the vocab_size is 50400 compared to 50257 in the tokenizer, this is done
    # because 50400 is a multiple of Z^5, powers of 2 tensor dimensions are preferred
    # for TPU hardware
    config = GptConfig(
        vocab_size=50400,
        eos_token_id=tokenizer.eos_token_id,
        embed_dim=4096,
        ffn_embed_dim=16384,
        num_heads=16,
        num_layers=28,
        rope_dimensions=64,
        max_position_embeddings=2048,
        add_bias_ffn=True,
        ffn_activation_name="gelu",
        use_glu_in_ffn=False,
        add_bias_lm_head=True,
        norm_type="layer_norm",
        parallel_attention_ff=True,
        use_gradient_checkpointing=False,
    )

    parameters = translate_torch_params(model_params, config.num_layers)
    # remove the torch parameters from the RAM
    del model_params
    gc.collect()
    torch.cuda.empty_cache()

    gptj_fn = build_gpt_fn(
        config=config,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        output_dtype=output_dtype,
        name=GPTJ_MODEL_NAME,
    )

    return parameters, gptj_fn, tokenizer, config
