import os
import torch
import datetime

from transformers import BertModel, BertTokenizer, BertConfig

class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert"  # 模型名称
        # 路径
        self.root_path = r'C:\github_project\ai_classification_team'
        # 原始数据路径（使用 os.path.join 保证跨平台和分隔符正确）
        self.train_datapath = os.path.join(self.root_path, '01-data', 'train.txt')
        self.test_datapath = os.path.join(self.root_path, '01-data', 'test.txt')
        self.dev_datapath = os.path.join(self.root_path, '01-data', 'dev.txt')
        # 类别文档
        self.class_path = os.path.join(self.root_path, '01-data', 'class.txt')

        # class.txt 格式: "数字\t标签名"，提取纯标签名列表
        self.class_list = [
            line.strip().split("\t")[1]
            for line in open(self.class_path, encoding="utf-8")
            if line.strip()
        ]

        # 模型训练保存路径
        self.model_save_path = os.path.join(self.root_path, '04-bert', 'save_models', 'bertclassifer_model.pt')

        # 模型训练+预测的时候
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu

        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 5  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 3e-5  # 学习率
        self.bert_path = os.path.join(self.root_path, '04-bert', 'bert-base-chinese')  # 预训练BERT模型的路径
        self.bert_model = BertModel.from_pretrained(self.bert_path) # 加载预训练BERT模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path)  # BERT模型的配置

        # 量化模型存放地址
        self.quantization_bert_model_path = os.path.join(
            self.root_path,
            '04-bert',
            'save_models',
            'bertclassifer_quantization_model.pt'
        )


if __name__ == '__main__':
    conf = Config()
    print(conf.quantization_bert_model_path)
    with open(conf.train_datapath, "r", encoding="utf-8") as f:
        # 打印前10行
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(line.strip())
