from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import Dataset


class OpenWebTextDataset(Dataset):
    """Define custom class for the OpenWebText dataset.
    Due to the large dataset size,
    we need to think of how to store this dataset locally and load it.
    """

    def __init__(
        self,
        dataset_path: Path,
        train: bool,
        block_size: int = 1024,
        device: torch.device = "cpu",
        **kwargs,
    ):
        """Dataset class for the OpenWebText dataset.

        Args:
            dataset_path (Path): Directory containing of the binary files.
            train (bool): If we look at the test or train/valid set.
            block_size (int): Size used to format the binary file.
            device (device): Device used to train the data. Relevant for DDP.
        """
        print(kwargs)
        assert (
            dataset_path
        ).exists(), (
            "The dataset was not found. Please run env_utils/nanoGPT/prepare_dataset.py"
        )
        mode = "train" if train else "test"
        self.block_size = block_size
        self.device = device

        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        self.data = np.memmap(dataset_path / f"{mode}.bin", dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.stack(
            [
                torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64))
                for i in index
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in index
            ]
        )
        if "cuda" in self.device:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y