import torch
from transformers.utils import PaddingStrategy

from bert_classifer_model import BertClassifier
from config import Config

# 初始化配置
conf = Config()

# 设置设备参数
device = conf.device
# 定义模型
model = BertClassifier()
# 指定模型到设备
model.to(device)
# 加载模型参数
model.load_state_dict(torch.load(conf.model_save_path))
# 设置模型为评估模式
model.eval()


# 6、定义predict_fun函数
def predict_fun(data_dict):
    """
    预测函数
    :param data_dict: {"text":"状元心经：考前一周重点是回顾和整理"}
    :return: {"text":"状元心经：考前一周重点是回顾和整理", "pred_class":"education"}
    """
    # 获取文本
    text = data_dict['text']
    # 将文本转为id
    text_tokens = conf.tokenizer.batch_encode_plus(
        [text],
        padding="max_length",
        max_length=conf.pad_size )
    # 获取input_ids和attention_mask
    input_ids = text_tokens['input_ids']
    attention_mask = text_tokens['attention_mask']
    # 将input_ids和attention_mask转为tensor, 并指定到设备
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    # 设置不进行梯度计算
    with torch.no_grad():
        # 前向传播
        output = model(input_ids, attention_mask)
        output = torch.argmax(output, dim=1)
        # 获取预测结果
        pred_idx = output.item()
        # 获取类别
        pred_class = conf.class_list[pred_idx]
        # 将预测结果添加到data_dict中
        data_dict['pred_class'] = pred_class
    # 返回data_dict
    return data_dict


if __name__ == '__main__':
    text = {'text': '《口袋妖怪 黑白》日本首周贩售255万'}
    data = predict_fun(text)
    print(data)