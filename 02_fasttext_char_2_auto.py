# 导包
import fasttext
from config import Config

# 创建文本配置对象
config = Config()

# 模型训练  自动调参
model = fasttext.train_supervised(
    input=config.process_train_datapath_char,                    # 训练数据路径
    autotuneValidationFile=config.process_dev_datapath_char,     # 验证集用于自动调参
    autotuneDuration=600,                                        # 调参时长：从120秒增加到600秒(10分钟)
    autotuneMetric='f1',                                         # 优化指标：使用F1分数
    autotunePredictions=1,                                       # Top-1预测准确率
    thread=8,                                                    # 线程数：提升训练速度
    verbose=2,                                                   # 日志级别：显示详细进度
    minCount=1,                                                  # 最小词频：保留所有词
    bucket=200000,                                               # hash桶大小：节省内存
    wordNgrams=2                                                 # N-gram特征：捕捉字符组合
)
print("模型训练完成")


# 保存模型
# 修改模型文件名
model_name = config.ft_model_save_path+"fasttext_model_train_char_auto.bin"
model.save_model(model_name)
print("模型保存完成")

# # 模型预测
print("开始预测.........")
predict = model.predict("新 西 兰 Q u e e n 苹 果 礼 盒 装 8 个 1 . 6 k g 起")
print("预测结果:",predict[0][0][9:])

# 模型评估
res = model.test(config.process_test_datapath_char)
print(f'测试样本总数: {res[0]}')
print(f'精确率: {res[1]:.4f}')
print(f'召回率: {res[2]:.4f}')
