"""
数据探索性分析报告
覆盖：数据集概览、标签分布、文本特征、TF-IDF 分析
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

from config import Config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 40)


def analyze_splits(train_df, test_df, dev_df, n2c):
    """数据集划分分析"""
    total = len(train_df) + len(test_df) + len(dev_df)
    print('\n' + '=' * 70)
    print('一、数据集概览')
    print('=' * 70)
    print(f'  训练集: {len(train_df):>8,} 条  ({len(train_df)/total*100:.1f}%)')
    print(f'  测试集: {len(test_df):>8,} 条  ({len(test_df)/total*100:.1f}%)')
    print(f'  验证集: {len(dev_df):>8,} 条  ({len(dev_df)/total*100:.1f}%)')
    print(f'  总计:   {total:>8,} 条')
    print(f'  类别数: {len(n2c)}')

    # 分集分布一致性
    print(f'\n  分集标签分布:')
    print(f'  {"类别":<10s} {"train":>8s} {"train%":>8s} {"test%":>8s} {"dev%":>8s} {"偏差":>8s}')
    print(f'  {"-"*55}')
    all_labels = sorted(train_df['label'].unique())
    tc = train_df['label'].value_counts()
    tsc = test_df['label'].value_counts()
    dc = dev_df['label'].value_counts()
    dev_big = []
    for lid in all_labels:
        tp = tc[lid] / len(train_df) * 100
        tsp = tsc[lid] / len(test_df) * 100
        dp = dc[lid] / len(dev_df) * 100
        max_dev = max(abs(tp - tsp), abs(tp - dp))
        if max_dev > 1:
            dev_big.append((lid, max_dev))
        if lid < 10 or lid >= 25:
            name = n2c.get(lid, '?')
            flag = ' !' if max_dev > 1 else ''
            print(f'  {name:<8s} {tc[lid]:>8d} {tp:>7.1f}% {tsp:>7.1f}% {dp:>7.1f}% {max_dev:>7.1f}%{flag}')
    if dev_big:
        print(f'\n  ! 偏差 >1% 的类别: {len(dev_big)} 个')

    return tc


def analyze_imbalance(label_counts, n2c):
    """类别不平衡分析"""
    print('\n' + '=' * 70)
    print('二、类别不平衡分析')
    print('=' * 70)
    labels = label_counts.sort_values(ascending=False)

    top3 = labels.head(3).sum()
    top5 = labels.head(5).sum()
    top10 = labels.head(10).sum()
    total = labels.sum()
    top3_names = [n2c.get(i, '?') for i in labels.head(3).index]
    print(f'  Top3  类别占比: {top3/total*100:.1f}%  ({", ".join(top3_names)})')
    print(f'  Top5  类别占比: {top5/total*100:.1f}%')
    print(f'  Top10 类别占比: {top10/total*100:.1f}%')

    ratio = labels.max() / labels.min()
    print(f'\n  不平衡比 (max/min): {ratio:.0f}:1')
    print(f'  均值: {labels.mean():.0f}  中位数: {labels.median():.0f}  标准差: {labels.std():.0f}')

    # 长尾类别
    print(f'\n  长尾类别 (样本 < 500):')
    print(f'  {"类别":<10s} {"样本数":>8s} {"占比":>8s}')
    print(f'  {"-"*30}')
    for lid, cnt in labels[labels < 500].items():
        name = n2c.get(lid, '?')
        print(f'  {name:<8s} {cnt:>8d} {cnt/total*100:>7.2f}%')

    head_total = labels[labels >= labels.mean()].sum()
    tail_total = labels[labels < 500].sum()
    print(f'\n  头部(>=均值) vs 尾部(<500):')
    print(f'    头部: {head_total:,} ({head_total/total*100:.1f}%)')
    print(f'    尾部: {tail_total:,} ({tail_total/total*100:.1f}%)')


def analyze_text(train_df):
    """文本特征分析"""
    print('\n' + '=' * 70)
    print('三、文本特征分析')
    print('=' * 70)

    train_df = train_df.copy()
    train_df['text_len'] = train_df['text'].str.len()
    text_lens = train_df['text_len']

    print(f'  文本长度分布:')
    print(f'    均值: {text_lens.mean():.1f}  中位数: {text_lens.median():.0f}')
    print(f'    最小值: {text_lens.min()}  最大值: {text_lens.max()}  标准差: {text_lens.std():.1f}')
    print(f'    25%分位: {text_lens.quantile(0.25):.0f}  75%分位: {text_lens.quantile(0.75):.0f}')

    bins = [0, 10, 20, 30, 50, 100, 200, 500, 10000]
    bin_labels = ['1-10', '11-20', '21-30', '31-50', '51-100', '101-200', '201-500', '500+']
    train_df['len_bin'] = pd.cut(text_lens, bins=bins, labels=bin_labels)
    print(f'\n  长度区间分布:')
    for lb in bin_labels:
        cnt = (train_df['len_bin'] == lb).sum()
        bar = '|' * int(cnt / len(train_df) * 50)
        print(f'    {lb:>8s}: {cnt:>6d} ({cnt/len(train_df)*100:5.1f}%) {bar}')

    return train_df


def analyze_tfidf(train_df, stop_words):
    """TF-IDF 特征分析 — 比较不同参数配置"""
    print('\n' + '=' * 70)
    print('四、TF-IDF 特征分析')
    print('=' * 70)

    words_list = train_df['text'].apply(lambda x: ' '.join(jieba.lcut(str(x))[:50]))

    configs = {
        '当前 (1,2)-gram 50k':     {'ngram_range': (1, 2), 'max_features': 50000, 'min_df': 3, 'max_df': 0.8},
        '优化A (1,1)-gram 20k':    {'ngram_range': (1, 1), 'max_features': 20000, 'min_df': 5, 'max_df': 0.7},
        '优化B char (3,5)-gram':   {'ngram_range': (3, 5), 'max_features': 10000, 'min_df': 5, 'max_df': 0.7, 'analyzer': 'char'},
    }

    results = []
    for name, params in configs.items():
        tv = TfidfVectorizer(
            stop_words=stop_words if stop_words else None,
            sublinear_tf=True,
            **params
        )
        X = tv.fit_transform(words_list)
        sparsity = X.nnz / (X.shape[0] * X.shape[1]) * 100
        freqs = np.array(X.sum(axis=0)).flatten()
        low_freq = (freqs < 5).sum()
        results.append({
            'name': name,
            'n_features': X.shape[1],
            'sparsity': f'{sparsity:.3f}%',
            'low_freq': f'{low_freq} ({low_freq/X.shape[1]*100:.1f}%)'
        })

    header = f"  {'Config':<25} {'Features':>8} {'Sparsity':>10} {'LowFreq(<5)':>15}"
    print(f'\n{header}')
    print(f'  {"-"*65}')
    for r in results:
        print(f"  {r['name']:<25} {r['n_features']:>8} {r['sparsity']:>10} {r['low_freq']:>15}")

    return results


def analyze_word_quality(train_df, stop_words, n2c):
    """高频词与类别关联分析"""
    print('\n' + '=' * 70)
    print('五、高频词与类别关联')
    print('=' * 70)

    words_list = train_df['text'].apply(lambda x: ' '.join(jieba.lcut(str(x))[:50]))
    top_labels = train_df['label'].value_counts().head(5).index

    for lid in top_labels:
        name = n2c.get(lid, '?')
        mask = train_df['label'] == lid
        class_words = words_list[mask]
        all_words_list = []
        for w in class_words:
            all_words_list.extend(w.split())
        wc = Counter(all_words_list)
        top_words = [w for w, _ in wc.most_common(10) if w not in (stop_words or [])]
        print(f'  [{lid:2d}] {name:<8s}: {" | ".join(top_words[:8])}')


def print_summary(train, test, dev, label_counts, n2c):
    """总结"""
    total = len(train) + len(test) + len(dev)
    ratio = label_counts.max() / label_counts.min()
    print('\n' + '=' * 70)
    print('六、总结')
    print('=' * 70)
    print(f'''
  数据集: {total:,} 条 (train/test/dev)
  类别:   {len(n2c)} 类
  不平衡比: {ratio:.0f}:1
  策略:   class_weight='balanced' + TF-IDF ngram(1,2) 50k
''')


if __name__ == '__main__':
    conf = Config()

    print('=' * 70)
    print('  商品分类数据集 — 探索性分析报告')
    print(f'  生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    # 加载数据
    train = pd.read_csv(conf.train_datapath, sep='\t', names=['text', 'label'])
    test = pd.read_csv(conf.test_datapath, sep='\t', names=['text', 'label'])
    dev = pd.read_csv(conf.dev_datapath, sep='\t', names=['text', 'label'])

    # 加载类别映射
    _, n2c = conf.load_class_map()

    # 加载停用词
    stop_words = []
    if os.path.exists(conf.stopwords_path):
        with open(conf.stopwords_path, encoding='utf-8') as f:
            stop_words = [line.strip() for line in f if line.strip()]

    # 分析
    label_counts = analyze_splits(train, test, dev, n2c)
    analyze_imbalance(label_counts, n2c)
    train_df = analyze_text(train)
    analyze_tfidf(train_df, stop_words)
    analyze_word_quality(train_df, stop_words, n2c)
    print_summary(train, test, dev, label_counts, n2c)
