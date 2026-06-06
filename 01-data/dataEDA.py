"""
dataEDA: 探索性数据分析：文本长度，标签分布，类别统计
数据源：每行一条json {"product name":"xxx","category":"xxx"}，NDJSON格式
方案二：直接读json，不生成任何txt中间文件
"""
from collections import Counter
import pandas as pd
import json
from config import Config

# 1.实例化配置
conf = Config()

# ----------------工具函数：加载单行json数据，文本+字符标签→数字标签----------------
def load_ndjson(file_path):
    """
    读取每行一条json的文件，返回df(text,label)、类别映射{类别名:数字id}
    :param file_path: json文件路径
    :return: df, cat2id
    """
    data_list = []
    cat_set = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item["product_name"]
            cat_name = item["category"]
            data_list.append([text, cat_name])
            cat_set.add(cat_name)

    # 生成类别→数字映射（按类别名称升序编号）
    sorted_cat = sorted(list(cat_set))
    cat2id = {cat: idx for idx, cat in enumerate(sorted_cat)}
    # 替换标签为数字
    final_data = [[txt, cat2id[cat]] for txt, cat in data_list]
    df = pd.DataFrame(final_data, columns=["text", "label"])
    return df, cat2id

# ----------------加载训练集做EDA分析----------------
train_df, category_mapping = load_ndjson(conf.train_json_path)
print("======【类别映射：文本标签→数字标签】======")
for k, v in category_mapping.items():
    print(f"{k} → {v}")

# todo 2 打印前10行数据
print("\n======训练集前10条样本======")
print(train_df.head(10))

# todo3 新增文本长度列
train_df["text_len"] = train_df["text"].str.len()
print("\n======新增文本长度字段后数据======")
print(train_df.head(10))

# todo4 文本长度整体分布统计
print("\n======文本长度统计describe======")
print(train_df["text_len"].describe())
print(f"文本平均长度：{train_df['text_len'].mean():.2f}")
print(f"文本长度标准差：{train_df['text_len'].std():.2f}")
print(f"文本最长：{train_df['text_len'].max()}")
print(f"文本最短：{train_df['text_len'].min()}")

# 标签频次统计
label_count = Counter(train_df["label"])
total = len(train_df)
print("\n======标签数量分布======")
for lab, cnt in label_count.items():
    print(f"数字标签{lab}：{cnt}条")

# 标签占比
print("\n======标签占比分布======")
for lab, cnt in label_count.items():
    rate = cnt / total * 100
    print(f"数字标签{lab}：{rate:.2f}%")

# =========可选：顺带EDA测试集数据（按需开启）=========
# test_df,_ = load_ndjson(conf.test_json_path)
# print("\n测试集样本总数：", len(test_df))