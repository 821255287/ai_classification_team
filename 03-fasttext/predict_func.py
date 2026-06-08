# 导包
import jieba
import fasttext
from config import Config
config = Config()

# todo 1.加载模型       这里用的字符级别自动训练的模型
model = fasttext.load_model(config.ft_model_save_path + "fasttext_model_train_char_auto.bin")

# todo 2.定义预测函数
def predict(text):
    # 获取文本数据
    text_data = text["text"]
    words = " ".join(jieba.lcut(text_data)[:30])
    result = model.predict(words)
    result = result[0][0][9:]
    text["pred_class"] =  result
    return text


if __name__ == '__main__':
    text = {
        "text": "T 7 G 滨 崎 小 熊 碗 仔 糖 果 三 合 一"
    }
    predict(text)
    print(f"模型预测完成: {predict(text)['pred_class']}")