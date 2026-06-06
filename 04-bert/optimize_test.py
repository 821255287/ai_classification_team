# 导包
import torch
from utils import build_dataloader
from config import Config

train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
# dev_dataloader(input_ids, attention_mask, labels)
for i, batch in enumerate(test_dataloader):
    print(batch)
    break
