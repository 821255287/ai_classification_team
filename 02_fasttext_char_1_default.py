# 导包
import fasttext
from config import Config

# 创建文本配置对象
config = Config()

# 模型训练
model = fasttext.train_supervised(
    input=config.process_train_datapath_char,     # 训练数据路径
    lr=0.7,                                        # 学习率：默认0.1，提高以加快收敛
    epoch=25,                                      # 训练轮数：默认5，增加到25提升效果
    dim=100,                                       # 词向量维度：从10提升到100，更好表达语义
    wordNgrams=3,                                  # N-gram特征：捕捉相邻字符组合
    minCount=1,                                    # 最小词频：保留所有词
    bucket=200000,                                 # hash桶大小：默认200万，降低节省内存
    minn=1,                                        # 最小子词长度：保持1，捕捉单字
    maxn=5,                                        # 最长子词长度：从6降5，避免噪声
    thread=8,                                      # 线程数：根据CPU核心数调整
    verbose=2,                                     # 日志级别：显示训练进度
    loss='softmax'                                 # 损失函数：多分类使用softmax
)
print("模型训练完成")

# 保存模型
# 修改模型文件名
model_name = config.ft_model_save_path+"fasttext_model_train_char_default.bin"
model.save_model(model_name)
print("模型保存完成")

# 模型预测
print("模型预测开始...")
predict = model.predict("伊 威 儿 童 零 食 入 口 即 化 冻 干 溶 豆 萌 化 了 益 生 菌 酸 奶 块 原 味")
print("预测结果:",predict[0][0][9:])

# 模型评估
res = model.test(config.process_test_datapath_char)
print(f'测试样本总数: {res[0]}')
print(f'精确率: {res[1]:.4f}')
print(f'召回率: {res[2]:.4f}')