# 导包
import fasttext
from config import Config

# 创建文本配置对象
config = Config()

# 模型训练 - 词级别优化后的自动调参配置
model = fasttext.train_supervised(
    input=config.process_train_datapath_word,                # 词级别训练数据路径
    autotuneValidationFile=config.process_dev_datapath_word, # 验证集用于自动调参
    autotuneDuration=600,                                    # 调参时长：10分钟充分搜索
    autotuneMetric='f1',                                     # 优化指标：F1分数
    autotunePredictions=1,                                   # Top-1预测准确率
    thread=8,                                                # 线程数：加速训练
    verbose=2,                                               # 日志级别：显示进度
    minCount=2,                                              # 最小词频：过滤低频噪声词
    bucket=2000000,                                          # hash桶大小：词级别需要更大空间
    wordNgrams=3                                             # N-gram特征：三元组捕捉词组搭
)
print("模型训练完成")

# 保存模型
# 修改模型文件名
model_name = config.ft_model_save_path + "fasttext_model_train_word_auto.bin"
model.save_model(model_name)
print("模型保存完成")

# 模型预测
predict = model.predict("潘婷 氨基酸 无 硅油 洗发水 微米 净透 排浊 赋能 530g 去 油 强韧 清爽")
print("预测结果:",predict[0][0][9:])

# 模型评估
res = model.test(config.process_test_datapath_word)
print(f'测试样本总数: {res[0]}')
print(f'精确率: {res[1]:.4f}')
print(f'召回率: {res[2]:.4f}')


