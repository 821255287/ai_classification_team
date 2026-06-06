import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from config import Config
import time
conf = Config()

# 读取原始数据train.txt等相关文件，进行切分之后，以元组的形式保存样本对到列表中。
def load_raw_data(file_path):
    data = []
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f, desc="Loading data"):
            line = line.strip()
            if not line:
                continue
            text, label = line.split("\t")
            data.append((text, int(label)))

    return data

# 构建成成符合pytorch模型需求的数据集。
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text=self.data[idx][0]
        label=self.data[idx][1]
        return text, label

# 构建dataloader
def build_dataloader():
    # 加载原始数据
    train_data = load_raw_data(conf.train_datapath)
    test_data = load_raw_data(conf.test_datapath)
    dev_data = load_raw_data(conf.dev_datapath)

    # 创建 Dataset
    train_dataset = TextDataset(train_data)
    dev_dataset = TextDataset(dev_data)
    test_dataset = TextDataset(test_data)

    # 创建 DataLoader，批次为batch_size=64
    train_dataloader = DataLoader(train_dataset,batch_size=conf.batch_size,shuffle=False,collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader,test_dataloader,dev_dataloader

# collate_fn是dataloader为了解决进入模型训练的数据不符合要求的进一步处理，例如batch级别数据处理长度、数据数值化等。


def collate_fn(batch):
    texts, labels = zip(*batch)


    text_tokens = conf.tokenizer.batch_encode_plus(
        texts,
        padding="max_length",  # 直接用字符串，最稳
        truncation=True,  # 建议加上，超长自动截断
        max_length=conf.pad_size, # 填充长度为32
        return_attention_mask=True
    )

    # 转 tensor
    input_ids = torch.tensor(text_tokens['input_ids'])
    attention_mask = torch.tensor(text_tokens['attention_mask'])
    labels = torch.tensor(labels)

    return input_ids, attention_mask, labels


if __name__ == "__main__":
    # 构建 DataLoader
    train_dataloader,test_dataloader,dev_dataloader = build_dataloader()
    # #遍历 DataLoader
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        print("input_ids=>",input_ids.shape)
        print("labels=>",labels.shape)
        print("attention_mask=>",attention_mask.shape)
        breakpoint()