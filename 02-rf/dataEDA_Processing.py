"""
数据预处理模块 — 分词 + 保存处理后数据
输入: 原始 tsv 数据 (text, label)
输出: 分词后的数据 (text, label, words)
"""
import os
import pandas as pd
import jieba
from config import Config


def load_data(data_path):
    """加载原始数据"""
    df = pd.read_csv(data_path, sep='\t', names=['text', 'label'])
    return df


def tokenize_text(text, max_words=50):
    """对文本进行 jieba 分词, 保留前 max_words 个词"""
    if not isinstance(text, str):
        text = str(text)
    words = jieba.lcut(text)[:max_words]
    return ' '.join(words)


def process_dataset(df, conf):
    """对 DataFrame 进行分词处理, 添加 words 列"""
    df = df.copy()
    df['words'] = df['text'].apply(lambda x: tokenize_text(x, conf.max_words))
    return df


def save_processed_data(df, save_path):
    """保存预处理后的数据"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, sep='\t', index=False, encoding='utf-8')
    print(f'已保存: {save_path} ({len(df)} 条)')


def main():
    conf = Config()

    print('开始数据预处理...')
    print(f'最大词数: {conf.max_words}')

    # 处理训练集
    print('\n[1/3] 处理训练集...')
    train_df = load_data(conf.train_datapath)
    train_df = process_dataset(train_df, conf)
    save_processed_data(train_df, conf.process_train_datapath)
    print(f'  标签分布: {train_df["label"].value_counts().to_dict()}')

    # 处理测试集
    print('\n[2/3] 处理测试集...')
    test_df = load_data(conf.test_datapath)
    test_df = process_dataset(test_df, conf)
    save_processed_data(test_df, conf.process_test_datapath)

    # 处理验证集
    print('\n[3/3] 处理验证集...')
    dev_df = load_data(conf.dev_datapath)
    dev_df = process_dataset(dev_df, conf)
    save_processed_data(dev_df, conf.process_dev_datapath)

    print('\n数据预处理完成！')


if __name__ == '__main__':
    main()
