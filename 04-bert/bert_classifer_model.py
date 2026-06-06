import torch
import torch.nn as nn
from transformers import BertModel
from config import Config

conf = Config()
class BertClassifier(nn.Module):
    """
    BERT + 全连接层的分类模型。
    """
    def __init__(self):
        """
        初始化模型，包括BERT和全连接层。
        """
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(conf.bert_path)
        self.fc1 = nn.Linear(conf.bert_model.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        out = self.fc1(pooled)
        out = self.fc2(out)
        return out
if __name__ == '__main__':
    text = ["朝鲜要求日本对过去罪行道歉和赔偿", "《口袋妖怪 黑白》日本首周贩售255万"]
    text_tokens = conf.tokenizer.batch_encode_plus(
        text,
        padding="max_length",  # 直接用字符串，最稳
        truncation=True,  # 建议加上，超长自动截断
        max_length=conf.pad_size,  # 填充长度为32
        return_attention_mask=True
    )
    input_ids = torch.tensor(text_tokens['input_ids']).to(device=conf.device)
    attention_mask = torch.tensor(text_tokens['attention_mask']).to(device=conf.device)
    # 创建模型对象
    model = BertClassifier()
    model.to(conf.device)

    # 模型预测
    logits = model.forward(input_ids, attention_mask)

    logits = torch.softmax(logits, dim=-1)
    print(f"预测结果：{logits}")
    pred = logits.argmax(dim=-1)
    print(f"预测结果：{conf.class_list[int(pred[0])]}")
    print(f"预测结果：{conf.class_list[int(pred[1])]}")

