from typing import Dict, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from haiku import initializers


class SimpleLMHead(hk.Module):
    """
    Basic Language Model head. Transforms final attention block output
    into a distribution over tokens at each sequence position.
    """

    def __init__(
        self,
        embed_dim: int,
        alphabet_size: int,
        add_bias_lm_head: bool = True,
        name: Optional[str] = None,
    ):
        """
        Args:
            embed_dim: Embedding dimension.
            alphabet_size: Number of tokens in the alphabet.
            name: Name of the layer. Defaults to None.
        """
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.alphabet_size = alphabet_size

        # Define layers
        w_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        b_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        self._final_fc = hk.Linear(
            self.alphabet_size,
            w_init=w_init,
            b_init=b_init,
            with_bias=add_bias_lm_head,
            name="lm_final_fc",
        )

    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Compute logits
        logits = self._final_fc(x)
        return {"logits": logits}


class RobertaLMHead(hk.Module):
    """
    Roberta Language Model head. Transform final attention layer output into a
    distribution over tokens at each position.
    """

    def __init__(self, embed_dim: int, alphabet_size: int, name: Optional[str] = None):
        """
        Args:
            embed_dim: Embedding dimension.
            alphabet_size: Number of tokens in the alphabet.
            name: Name of the layer. Defaults to None.
        """
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.alphabet_size = alphabet_size

        # Define layers
        self._first_layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_after"
        )
        self._fc1 = hk.Linear(self.embed_dim, name="lm_head_fc_1")
        self._final_fc = hk.Linear(self.alphabet_size, name="lm_final_fc")
        self._second_layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="lm_head_layer_norm"
        )

    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        x = self._first_layer_norm(x)
        # Embeddings are computed after the first layer norm to be consistent with ESM
        embeddings = x
        x = self._fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self._second_layer_norm(x)

        # Compute logits
        logits = self._final_fc(x)
        return {"embeddings": embeddings, "logits": logits}
