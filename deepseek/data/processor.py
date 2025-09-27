import tiktoken
import os
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import torch
from typing import List


class TinyStoriesProcesssor:

    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 1024):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_length = max_length

        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Data directory: {self.data_dir}")

    def tokenize(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        return tokens

    def detokenize(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def process(self, example):
        text = example["text"]
        tokens = self.tokenize(text)
        return {"input_ids": tokens, "len": len(tokens)}

    def prepare_dataset(
        self,
        dataset_name: str = "roneneldan/TinyStories",
        split: str = "train",
        debug: bool = False,
    ):
        train_path = os.path.join(self.data_dir, "train.bin")
        validation_path = os.path.join(self.data_dir, "val.bin")
        test_path = os.path.join(self.data_dir, "test.bin")

        ds = load_dataset(dataset_name, split=split)

        if debug:
            print("Debug mode: using a small subset of the data")
            ds = ds.select(range(1024))

        if (
            os.path.exists(train_path)
            and os.path.exists(validation_path)
            and os.path.exists(test_path)
        ):

            print("Found existing processed files!")
            print(f"Train file: {os.path.getsize(train_path) / (1024*1024):.2f} MB")
            print(
                f"Validation file: {os.path.getsize(validation_path) / (1024*1024):.2f} MB"
            )
            print(f"Finetune file: {os.path.getsize(test_path) / (1024*1024):.2f} MB")

            return {
                "train": train_path,
                "validation": validation_path,
                "finetune": test_path,
            }

        train_val_test = ds.train_test_split(test_size=0.2, seed=42)
        val_finetune = train_val_test["test"].train_test_split(test_size=0.5, seed=42)

        # Create a new dataset dictionary with all splits
        ds = {
            "train": train_val_test["train"],
            "validation": val_finetune["train"],
            "test": val_finetune["test"],
        }

        for split_name, split_data in ds.items():
            print(f"\nProcessing {split_name} split...")

            # Process the data
            tokenized = split_data.map(
                self.process,
                desc=f"tokenizing {split_name} split",
                num_proc=8,
            )

            tokenized = tokenized.filter(lambda x: x["len"] > 0)
            print(f"After processing: {len(tokenized)} valid examples")

            filename = os.path.join(self.data_dir, f"{split_name}.bin")
            print(f"Saving {split_name} split to: {filename}")

            arr_len = np.sum(tokenized["len"], dtype=np.uint64)
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = tokenized.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["input_ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

            if os.path.exists(filename):
                print(f"Successfully created {filename}")
                print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
            else:
                raise RuntimeError(f"Failed to create {filename}")

        return {
            "train": train_path,
            "validation": validation_path,
            "test": test_path,
        }

    def load_binary_data(self, filepath: str) -> torch.Tensor:
        """Load binary data file as tensor"""
        try:
            data = np.memmap(filepath, dtype=np.uint16, mode="r")
            return torch.from_numpy(data.copy())
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            raise

    def get_batch(self, data: torch.Tensor, batch_size: int, block_size: int) -> tuple:
        """Get a batch of data for training"""

        ix = torch.randint(len(data) - block_size, (batch_size,))

        x = torch.stack([data[i : i + block_size].long() for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size].long() for i in ix])

        return x, y

    def prepare_dataset_memory(
        self,
        dataset_name: str = "roneneldan/TinyStories",
        debug: bool = False,
        splits: List[str] = ["train", "validation", "test"],
    ):
        """Load, tokenize, and keep dataset fully in memory."""
        print("Loading dataset into memory...")
        ds = load_dataset(dataset_name)

        if debug:
            print("Debug mode: using a small subset of the data")
            for split in ds:
                ds[split] = ds[split].select(range(min(10240, len(ds[split]))))

        for split in splits:
            print(f"\nProcessing {split} split (in memory)...")
            tokenized = ds[split].map(
                self.process,
                desc=f"tokenizing {split} split",
            )
            tokenized = tokenized.filter(lambda x: x["len"] > 0)
            print(f"After processing: {len(tokenized)} valid examples")

            # Flatten into one long array of token IDs
            arr = np.concatenate(tokenized["input_ids"])
            arr = torch.tensor(arr, dtype=torch.long)
            self.memory_datasets[split] = arr

        return self.memory_datasets

    def get_dataset(self, split: str = "train") -> torch.Tensor:
        """Return in-memory dataset tensor for a split."""
        if split not in self.memory_datasets:
            raise ValueError(f"Split {split} not found. Call prepare_dataset_memory first.")
        return self.memory_datasets[split]


if __name__ == "__main__":
    processor = TinyStoriesProcesssor(tokenizer_name="gpt2", max_length=512)
    processor.prepare_dataset(split="train", debug=True)
