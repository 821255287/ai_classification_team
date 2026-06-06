"""
随机森林预测函数模块
提供: 加载模型、文本预处理、预测 等可复用函数
"""
import os
import pickle
import pandas as pd
import jieba
from config import Config


# ========== 全局缓存 ==========
_model = None
_vectorizer = None
_conf = None


def get_config():
    """获取配置(单例)"""
    global _conf
    if _conf is None:
        _conf = Config()
    return _conf


def load_model():
    """加载训练好的随机森林模型和TF-IDF向量化器"""
    global _model, _vectorizer
    conf = get_config()

    if os.path.exists(conf.rf_model_save_path):
        with open(conf.rf_model_save_path, 'rb') as f:
            _model = pickle.load(f)
    else:
        raise FileNotFoundError(f'模型文件不存在: {conf.rf_model_save_path}')

    if os.path.exists(conf.tfidf_model_save_path):
        with open(conf.tfidf_model_save_path, 'rb') as f:
            _vectorizer = pickle.load(f)
    else:
        raise FileNotFoundError(f'TF-IDF模型文件不存在: {conf.tfidf_model_save_path}')

    return _model, _vectorizer


def load_class_names():
    """加载类别名称映射"""
    conf = get_config()
    n2c, _ = conf.load_class_map()
    return n2c


def preprocess_text(text, max_words=None):
    """文本预处理: 分词并用空格连接"""
    if max_words is None:
        max_words = get_config().max_words
    if not isinstance(text, str):
        text = str(text)
    words = jieba.lcut(text)[:max_words]
    return ' '.join(words)


def predict_single(text):
    """单条文本预测, 返回 (label_id, label_name)"""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        load_model()

    processed = preprocess_text(text)
    X = _vectorizer.transform([processed])
    pred_id = _model.predict(X)[0]
    n2c = load_class_names()
    pred_name = n2c.get(pred_id, str(pred_id))
    return pred_id, pred_name


def predict_batch(texts):
    """批量文本预测, 返回预测标签列表"""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        load_model()

    processed = [preprocess_text(t) for t in texts]
    X = _vectorizer.transform(processed)
    pred_ids = _model.predict(X)
    n2c = load_class_names()
    pred_names = [n2c.get(pid, str(pid)) for pid in pred_ids]
    return pred_ids, pred_names


def predict_with_proba(texts):
    """批量预测并返回概率分布, 返回 (pred_ids, pred_names, proba)"""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        load_model()

    processed = [preprocess_text(t) for t in texts]
    X = _vectorizer.transform(processed)
    pred_ids = _model.predict(X)
    proba = _model.predict_proba(X)
    n2c = load_class_names()
    pred_names = [n2c.get(pid, str(pid)) for pid in pred_ids]
    return pred_ids, pred_names, proba


def evaluate_on_file(data_path, result_dir=None):
    """在数据集上批量预测并评估 (需要有标注的数据)"""
    conf = get_config()
    if result_dir is None:
        result_dir = conf.result_dir

    # 加载数据
    df = pd.read_csv(data_path, sep='\t', names=['text', 'label'])

    # 预测
    texts = df['text'].tolist()
    pred_ids, pred_names = predict_batch(texts)

    # 构建结果
    true_labels = df['label'].values
    df_result = pd.DataFrame({
        'text': texts,
        'label': true_labels,
        'pred_label': pred_ids,
        'correct': true_labels == pred_ids,
    })

    # 保存
    basename = os.path.splitext(os.path.basename(data_path))[0]
    result_path = os.path.join(result_dir, f'{basename}_predict.csv')
    df_result.to_csv(result_path, index=False, encoding='utf-8')

    # 计算准确率
    acc = (true_labels == pred_ids).mean()
    print(f'[{basename}] 准确率: {acc:.4f} ({acc*100:.2f}%), 样本数: {len(df)}')
    print(f'结果已保存至: {result_path}')

    return df_result
