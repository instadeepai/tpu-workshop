from typing import Dict, Iterator, Tuple

import jax.numpy as jnp
from datasets import load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class HFTextDataset:
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        tokenized_sequence_length: int,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        drop_last: bool = True,
        seed: int = 0,
        streaming: bool = True,
    ):
        """
        Abstraction over HF datasets to train language models. Note that this dataset
        assumes the presence of a column "text" in the dataset which is the case for
        the_pile or C4 for instance.

        Args:
            dataset_name: Name of the dataset in HuggingFace datasets.
            split: name of the split, e.g. "train".
            tokenizer: Tokenizer.
            batch_size: Batch size.
            tokenized_sequence_length: Tokenized sequence length, will be used to crop
                sequences.
            shuffle: Determines if the sequences are to be shuffled.
                Defaults to True.
            shuffle_buffer_size: Buffer size for on the fly shuffling with iterable
                dataset. See https://huggingface.co/docs/datasets/v2.10.0/en/package_reference/main_classes#datasets.IterableDataset.shuffle  # noqa
                for more details.
            drop_last: Determines if the last batch is dropped. This is
                important because a difference in the input's shape will force JAX to
                recompile. Defaults to True.
            seed: Seed to use to shuffle sequences.
            streaming: Whether to download full dataset or batch per batch.
        """

        # internalize parameters
        self._seed = seed
        self._shuffle = shuffle
        self._shuffle_buffer_size = shuffle_buffer_size
        self._tokenized_sequence_length = tokenized_sequence_length
        self._tokenizer = tokenizer
        self._drop_last = drop_last
        self._batch_size = batch_size
        self._streaming = streaming

        # get iterable dataset
        self._iterable_dataset = load_dataset(
            dataset_name, split=split, streaming=streaming
        )

    def update_seed(self, seed: int) -> None:
        """
        Updates the seed of the dataset (useful between epochs).
        Args:
            seed: New seed of the dataset.
        """
        self._seed = seed

    def get_iterator(self) -> Iterator[jnp.ndarray]:
        """
        Yields successive batches of tokens ids of shape (batch_size,
        tokenized_sequence_length).
        """
        if self._shuffle:
            if self._streaming:
                iterable_dataset = self._iterable_dataset.shuffle(
                    seed=self._seed, buffer_size=self._shuffle_buffer_size
                )
            else:
                # shuffle buffer size argument does not exist when not streaming dataset
                iterable_dataset = self._iterable_dataset.shuffle(
                    seed=self._seed,
                )
        else:
            iterable_dataset = self._iterable_dataset

        def tokenize_function(examples):  # type: ignore
            return self._tokenizer(
                [example + self._tokenizer.eos_token for example in examples["text"]]
            )

        iterable_dataset = iterable_dataset.select_columns("text")
        tokenized_iterable_dataset = iterable_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        block_size = self._tokenized_sequence_length

        # aggregate data in dataset into blocks of fixed size
        def group_texts(examples):  # type: ignore
            # Concatenate all texts.
            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()
            }  # type: ignore
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported
            # it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        # get resulting iterable dataset and generator, ready to be used for training
        lm_iterable_dataset = tokenized_iterable_dataset.map(
            group_texts,
            batched=True,
            batch_size=1000,
        )
        batch_generator = lm_iterable_dataset.iter(
            batch_size=self._batch_size, drop_last_batch=self._drop_last
        )

        def get_tokens_ids_array(example: Dict) -> jnp.ndarray:
            tokens_ids = jnp.asarray(example["input_ids"])
            return tokens_ids

        return (get_tokens_ids_array(example) for example in batch_generator)


class HFInstructionDataset(HFTextDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        tokenized_sequence_length: int,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        drop_last: bool = True,
        seed: int = 0,
        streaming: bool = False,
    ):
        """
        Abstraction over HF dataset to train language models over instructions.
        Make the assumption that the instruction are structured with 'instruction',
        'input', 'output' and 'text' keys.
        """
        if dataset_name != "tatsu-lab/alpaca":
            raise UserWarning(
                "This utility has been made following tatsu-lab/alpaca structure. You "
                "are using a different dataset and may encounter errors. If that is "
                "the case, you can can check column names in the associated datasets to"
                "modify this class. "
            )

        # pad token are set here if it is None as setting it before fails with
        # HF tokenizers
        if (tokenizer.pad_token is None) or (tokenizer.pad_token_id is None):
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token

        HFTextDataset.__init__(
            self,
            dataset_name=dataset_name,
            split=split,
            tokenizer=tokenizer,
            batch_size=batch_size,
            tokenized_sequence_length=tokenized_sequence_length,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            drop_last=drop_last,
            seed=seed,
            streaming=streaming,
        )

    def get_iterator(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Yields successive batches of tokens ids of shape (batch_size,
        tokenized_sequence_length) and sequence_mask of the same shape to mask queries
        in the loss function.
        """
        if self._shuffle:
            if self._streaming:
                iterable_dataset = self._iterable_dataset.shuffle(
                    seed=self._seed, buffer_size=self._shuffle_buffer_size
                )
            else:
                # shuffle buffer size argument does not exist when not streaming dataset
                iterable_dataset = self._iterable_dataset.shuffle(
                    seed=self._seed,
                )
        else:
            iterable_dataset = self._iterable_dataset

        def tokenize_function(examples):  # type: ignore
            query_tokens = self._tokenizer(
                examples["text"],
                max_length=self._tokenized_sequence_length,
                padding="max_length",
            )
            output_tokens = self._tokenizer(
                examples["output"],
                max_length=self._tokenized_sequence_length,
                padding="max_length",
            )
            num_query_tokens = jnp.sum(
                jnp.array(query_tokens["input_ids"]) != self._tokenizer.pad_token_id
            )
            num_output_tokens = jnp.sum(
                jnp.array(output_tokens["input_ids"]) != self._tokenizer.pad_token_id
            )
            # compute sequence mask to compute loss only over outputs
            sequence_mask = (
                [0 for _ in range(num_query_tokens - num_output_tokens)]
                + [1 for _ in range(num_output_tokens + 1)]  # +1 to add one endoftext
                + [
                    0
                    for _ in range(
                        self._tokenized_sequence_length - num_query_tokens - 1
                    )
                ]
            )
            examples.update(query_tokens)
            examples.update({"sequence_mask": sequence_mask})
            return examples

        iterable_dataset = iterable_dataset.select_columns(["text", "output"])
        # batched = False otherwise there is an issue with the way we compute the mask
        lm_iterable_dataset = iterable_dataset.map(
            tokenize_function, remove_columns=["text", "output"]
        )

        batch_generator = lm_iterable_dataset.iter(
            batch_size=self._batch_size, drop_last_batch=self._drop_last
        )

        def get_tokens_ids_array(example: Dict) -> jnp.ndarray:
            tokens_ids = jnp.asarray(example["input_ids"])
            return tokens_ids

        def get_sequence_mask_array(example: Dict) -> jnp.ndarray:
            sequence_mask = jnp.asarray(example["sequence_mask"])
            return sequence_mask

        return (
            (get_tokens_ids_array(example), get_sequence_mask_array(example))
            for example in batch_generator
        )
