# 导包
import fasttext
from config import Config

# 创建文本配置对象
config = Config()

# 模型训练 - 词级别优化参
model = fasttext.train_supervised(
    input=config.process_train_datapath_word,
    dim=50,
    epoch=15,
    lr=0.3,
    wordNgrams=2,
    minCount=1,
    bucket=2000000,
    minn=2,
    maxn=4,
    thread=8,
    verbose=0
)
print("模型训练完成")

# 保存模型
model_name = config.ft_model_save_path+"fasttext_model_train_word_default.bin"
model.save_model(model_name)
print("模型保存完成")

# 模型预测
predict = model.predict("磁器 口 老 火锅 底料 牛油 手工 重庆 特产 四川 麻辣火锅 500g")
print("预测结果:",predict[0][0][9:])

# 模型评估
res = model.test(config.process_test_datapath_word)
print(f'测试样本总数: {res[0]}')
print(f'精确率: {res[1]:.4f}')
print(f'召回率: {res[2]:.4f}')
