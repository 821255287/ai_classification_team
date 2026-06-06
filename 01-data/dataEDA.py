"""
dataEDA: 探索性数据分析：文本长度，标签情况。。。。
"""
from collections import Counter
#导包
import pandas as pd
from config import Config

#创建配置对象
conf = Config()

#todo 1 读取数据
data = pd.read_csv(conf.train_datapath, sep = '\t',names = ['text', 'label'])


# # todo 2 打印前10行数据
print(data.head(10))

# todo 3  新增一列数据，统计数据文本的长度
data['text_len'] =data['text'].str.len()
print(data.head(10))

# todo 4 统计文本长度的分布情况
print(data['text_len'].describe())
# 打印文本长度的平均值
print(data['text_len'].mean())
#打印文本长度的标准差
print(data['text_len'].std())
#打印文本长的最大值
print(data['text_len'].max())
#打印文本长度的最小值
print(data['text_len'].min())

# 第二步：统计标签分布
label_counts = Counter(data['label'])  # 数一数每个标签出现了几次
print("\n标签分布：")
for label, count in label_counts.items():
    print(f"标签 {label}：{count} 次")  # 输出每个标签的次数

# 第三步：计算标签比例
total_rows = len(data)  # 总行数
print("\n标签比例：")
for label, count in label_counts.items():
    percent = (count / total_rows) * 100  # 计算百分比
    print(f"标签 {label}：{percent:.2f}%")  # 输出百分比，保留2位小数