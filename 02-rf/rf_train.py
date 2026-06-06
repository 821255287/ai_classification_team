"""
随机森林基线模型 — 训练脚本
pipeline: 加载数据 → TF-IDF特征 → 训练RF → 评估(Dev+Test) → 保存模型
"""
import os
import sys
import time
import pickle
import logging
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from config import Config

# ============================================================
# Logger 配置: 同时输出到文件和控制台
# ============================================================
def setup_logger(conf):
    logger = logging.getLogger('rf_train')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    log_format = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件 handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(conf.result_dir, f'train_{timestamp}.log')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(log_format)
    logger.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    return logger, log_path


# ============================================================
# 模型评估函数
# ============================================================
def evaluate_model(model, vectorizer, X, y_true, dataset_name, conf, logger):
    """评估模型并输出详细指标"""
    start = time.time()
    X_vec = vectorizer.transform(X)
    vec_time = time.time() - start

    logger.debug(f'  向量化耗时: {vec_time:.2f}s')

    start = time.time()
    y_pred = model.predict(X_vec)
    pred_time = time.time() - start

    logger.info(f'  预测 {len(y_true)} 条, 耗时: {pred_time:.2f}s')

    # 基本指标
    acc = accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    p_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    r_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 输出
    logger.info(f'  ========================================')
    logger.info(f'  [{dataset_name}] 评估结果 (样本数: {len(y_true)})')
    logger.info(f'  ========================================')
    logger.info(f'  准确率 (Accuracy):           {acc:.4f}')
    logger.info(f'  精确率 (Precision-macro):    {p_macro:.4f}')
    logger.info(f'  召回率 (Recall-macro):       {r_macro:.4f}')
    logger.info(f'  F1-score (F1-macro):         {f1_macro:.4f}')
    logger.info(f'  精确率 (Precision-weighted):  {p_weighted:.4f}')
    logger.info(f'  召回率 (Recall-weighted):     {r_weighted:.4f}')
    logger.info(f'  F1-score (F1-weighted):       {f1_weighted:.4f}')

    # sklearn 分类报告
    n2c, _ = conf.load_class_map()
    labels = sorted(np.unique(y_true))
    target_names = [n2c.get(l, str(l)) for l in labels]
    cr = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    logger.info(f'\n  [{dataset_name}] 分类报告:\n{cr}')

    # 各类别正确率
    logger.info(f'  [{dataset_name}] 各类别正确率 (>=80% 高 / <40% 低):')
    class_acc = {}
    for label in labels:
        mask = y_true == label
        if mask.sum() > 0:
            class_acc[label] = (y_pred[mask] == label).sum() / mask.sum()

    high = {k: v for k, v in class_acc.items() if v >= 0.8}
    low = {k: v for k, v in class_acc.items() if v < 0.4}

    if high:
        logger.info(f'    高正确率:')
        for k, v in sorted(high.items(), key=lambda x: -x[1]):
            name = n2c.get(k, '?')
            cnt = (y_true == k).sum()
            correct = int(v * cnt)
            logger.info(f'      [{k:2d}] {name:<8s}      : {correct}/{cnt} ({v*100:.1f}%)')

    if low:
        logger.info(f'    低正确率:')
        for k, v in sorted(low.items(), key=lambda x: x[1]):
            name = n2c.get(k, '?')
            cnt = (y_true == k).sum()
            correct = int(v * cnt)
            logger.info(f'      [{k:2d}] {name:<8s}      : {correct}/{cnt} ({v*100:.1f}%)')

    # 保存指标和预测结果
    metrics = {
        'dataset': dataset_name,
        'n_samples': len(y_true),
        'accuracy': acc,
        'precision_macro': p_macro,
        'recall_macro': r_macro,
        'f1_macro': f1_macro,
        'precision_weighted': p_weighted,
        'recall_weighted': r_weighted,
        'f1_weighted': f1_weighted,
    }

    # 保存预测结果
    result_df = pd.DataFrame({
        'text': X,
        'true_label': y_true,
        'pred_label': y_pred,
    })
    if dataset_name == 'Dev':
        result_path = os.path.join(conf.result_dir, 'dev_predict_result.csv')
        metrics_path = conf.dev_metrics_path
    else:
        result_path = os.path.join(conf.result_dir, 'test_predict_result.csv')
        metrics_path = conf.test_metrics_path

    result_df.to_csv(result_path, index=False, encoding='utf-8')
    logger.info(f'  预测结果已保存至: {result_path}')

    # 保存指标
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f'[{dataset_name}] 评估结果\n')
        f.write('========================================\n')
        f.write(f'样本数: {len(y_true)}\n')
        f.write(f'准确率: {acc:.4f}\n')
        f.write(f'精确率 (macro): {p_macro:.4f}\n')
        f.write(f'召回率 (macro): {r_macro:.4f}\n')
        f.write(f'F1-score (macro): {f1_macro:.4f}\n')
        f.write(f'精确率 (weighted): {p_weighted:.4f}\n')
        f.write(f'召回率 (weighted): {r_weighted:.4f}\n')
        f.write(f'F1-score (weighted): {f1_weighted:.4f}\n')
        f.write(f'\n分类报告:\n')
        f.write(cr)
        f.write(f'\n混淆矩阵:\n')
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        f.write('     ' + ' '.join([f'{l:>4}' for l in labels]) + '\n')
        for i, row in enumerate(cm):
            f.write(f'{labels[i]:>5}' + ' '.join([f'{v:>4}' for v in row]) + '\n')

    logger.info(f'  指标已保存至: {metrics_path}')

    return {
        'acc': acc, 'p_macro': p_macro, 'r_macro': r_macro, 'f1_macro': f1_macro,
        'p_weighted': p_weighted, 'r_weighted': r_weighted, 'f1_weighted': f1_weighted,
    }


# ============================================================
# 主训练流程
# ============================================================
def main():
    conf = Config()
    logger, log_path = setup_logger(conf)

    logger.info('=' * 60)
    logger.info('随机森林基线模型 — 训练开始')
    logger.info(f'启动时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 60)

    # ============================
    # Step 1: 加载数据
    # ============================
    logger.info('=' * 60)
    logger.info('Step 1/5: 加载数据')

    train_df = pd.read_csv(conf.train_datapath, sep='\t', names=['text', 'label'])
    test_df = pd.read_csv(conf.test_datapath, sep='\t', names=['text', 'label'])
    dev_df = pd.read_csv(conf.dev_datapath, sep='\t', names=['text', 'label'])

    # 类别映射
    n2c, c2n = conf.load_class_map()
    n_classes = train_df['label'].nunique()

    logger.info(f'  训练集: {len(train_df)} 条')
    logger.info(f'  测试集: {len(test_df)} 条')
    logger.info(f'  验证集: {len(dev_df)} 条')
    logger.info(f'  类别数量: {n_classes}')

    # 标签分布
    label_counts = Counter(train_df['label'])
    sorted_labels = label_counts.most_common()
    logger.info(f'  标签分布 top5 / bottom5:')
    for label_id, cnt in sorted_labels[:5]:
        name = n2c.get(label_id, '?')
        logger.info(f'    ↑ [{label_id}] {cnt} 条')
    for label_id, cnt in sorted_labels[-5:]:
        name = n2c.get(label_id, '?')
        logger.info(f'    ↓ [{label_id}] {cnt} 条')
    logger.info(f'  类别映射: {n_classes} 类')

    # ============================
    # Step 2: 构建 TF-IDF 特征
    # ============================
    logger.info('=' * 60)
    logger.info('Step 2/5: 构建 TF-IDF 特征')

    # 加载停用词
    stopwords = conf.load_stopwords()
    logger.info(f'  停用词数量: {len(stopwords)}')
    logger.info(f'  TF-IDF 参数: {conf.tfidf_params}')

    # 文本分词预处理（jieba分词后用空格连接）
    import jieba
    train_texts = train_df['text'].apply(lambda x: ' '.join(jieba.lcut(str(x))[:conf.max_words]))

    tfidf_start = time.time()
    vectorizer = TfidfVectorizer(stop_words=stopwords if stopwords else None, **conf.tfidf_params)
    X_train = vectorizer.fit_transform(train_texts)
    tfidf_time = time.time() - tfidf_start

    logger.info(f'  TF-IDF 完成, 耗时: {tfidf_time:.2f}s')
    logger.info(f'  特征维度: {X_train.shape}')
    logger.debug(f'  稀疏度: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]) * 100:.4f}%')

    # ============================
    # Step 3: 训练随机森林
    # ============================
    logger.info('=' * 60)
    logger.info('Step 3/5: 训练随机森林模型')
    logger.info(f'  模型参数: {conf.rf_params}')

    train_start = time.time()
    model = RandomForestClassifier(**conf.rf_params)
    model.fit(X_train, train_df['label'])
    train_time = time.time() - train_start

    logger.info(f'  训练完成, 耗时: {train_time:.2f}s')
    logger.info(f'  树的数量: {model.n_estimators}')

    # 特征重要性
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]

    logger.info(f'  Top 15 特征重要性:')
    for rank, idx in enumerate(top_indices, 1):
        logger.info(f'    #{rank:>2}: {feature_names[idx]:<10s}     = {importances[idx]:.6f}')

    n_important = (importances > 0.001).sum()
    logger.debug(f'  特征重要性 > 0.001 的特征数: {n_important}')
    logger.debug(f'  特征重要性总和: {importances.sum():.6f}')

    # ============================
    # Step 4: 模型评估
    # ============================
    dev_texts = dev_df['text'].apply(lambda x: ' '.join(jieba.lcut(str(x))[:conf.max_words]))
    test_texts = test_df['text'].apply(lambda x: ' '.join(jieba.lcut(str(x))[:conf.max_words]))

    logger.info('=' * 60)
    logger.info('Step 4/5 [Dev]: 模型评估')
    dev_results = evaluate_model(model, vectorizer, dev_texts, dev_df['label'], 'Dev', conf, logger)

    logger.info('=' * 60)
    logger.info('Step 4/5 [Test]: 模型评估')
    test_results = evaluate_model(model, vectorizer, test_texts, test_df['label'], 'Test', conf, logger)

    # ============================
    # Step 5: 保存模型
    # ============================
    logger.info('=' * 60)
    logger.info('Step 5/5: 保存模型')

    with open(conf.rf_model_save_path, 'wb') as f:
        pickle.dump(model, f)
    model_size = os.path.getsize(conf.rf_model_save_path) / (1024 * 1024)
    logger.info(f'  模型已保存至: {conf.rf_model_save_path}')
    logger.info(f'  模型大小: {model_size:.1f} MB')

    with open(conf.tfidf_model_save_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    # ============================
    # 训练汇总
    # ============================
    logger.info('=' * 60)
    logger.info('训练汇总')
    logger.info('=' * 60)
    logger.info(f'  {"指标":<35s} {"Dev":>8s} {"Test":>10s}')
    logger.info(f'  {"-"*50}')
    metrics_keys = [
        ('accuracy', 'acc'),
        ('precision_macro', 'p_macro'),
        ('recall_macro', 'r_macro'),
        ('f1_macro', 'f1_macro'),
        ('precision_weighted', 'p_weighted'),
        ('recall_weighted', 'r_weighted'),
        ('f1_weighted', 'f1_weighted'),
    ]
    for display_name, key in metrics_keys:
        logger.info(f'  {display_name:<35s} {dev_results[key]:>8.4f} {test_results[key]:>10.4f}')

    logger.info(f'')
    logger.info('训练完成！')
    logger.info(f'完整日志已保存至: {log_path}')


if __name__ == '__main__':
    main()
