import json
import os
from pathlib import Path
from typing import Any

import tiktoken
import numpy as np
from datasets import Dataset
from tqdm import tqdm

GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")

def tokenize(example: dict[str, Any]):
    ids = GPT2_TOKENIZER.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(GPT2_TOKENIZER.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {"ids": ids, "len": len(ids)}
    return out
   

def main() -> None:
    current_working_dir = Path(__file__).parents[0]
    pile_path = current_working_dir / "pile-sample.jsonl"
    num_cpus = os.cpu_count()

    dataset = Dataset.from_generator(
        lambda: map(json.loads, open(pile_path, "rt")),
        num_proc=num_cpus,
    )

    split_dataset = dataset.train_test_split(test_size=0.002, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test") # rename the test split to val

    tokenized = split_dataset.map(
        tokenize,
        remove_columns=["text", "meta"],
        desc="tokenizing the splits",
        num_proc=os.cpu_count(),
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = current_working_dir / f"{split}.bin"
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # tokenizer = tiktoken.get_encoding("gpt2")
    # with open(path.parents[0] / "pile-sample.jsonl", "rt") as fin:
    #     count = 0
    #     for record in map(json.loads, fin):
    #         if record["meta"]["pile_set_name"] == "Books3":
    #             print(len(record["text"]))


if __name__ == "__main__":
    main()
