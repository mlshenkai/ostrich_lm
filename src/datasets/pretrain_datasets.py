import json
from typing import Any, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer,max_length=522, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lines = self._load_dataset(data_path=data_path)
        self.tokenzer = tokenizer
        self.padding = 0
        self.max_length = max_length
    

    def _load_dataset(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return lines
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 
        item = json.loads(self.lines[index % len(self.lines)])
        text = item["text"]

        # 每一个sentences 开始都要添加一个bos tonken
        text = f"{self.tokenzer.bos_token}{text}"

        # 对text进行编码
        text_tokens: list = self.tokenzer(text=text)["input_ids"] # type: ignore
        input_ids = text_tokens[:self.max_length]
        curent_length = len(input_ids)
        if curent_length < self.max_length:
            # 进行padding
            input_ids = input_ids + [self.padding] * (self.max_length - curent_length)
            loss_mask = [1] * curent_length + [0] * (self.max_length - curent_length)
        else:
            loss_mask = [1] * curent_length
        # 开始构建mask和target
        X = np.array(input_ids[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

