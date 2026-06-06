"""
随机森林模型 — Flask API 服务
提供 RESTful 接口供外部调用
"""
import os
import sys
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from rf_predict_fun import load_model, load_class_names, predict_single, predict_batch, predict_with_proba

app = Flask(__name__)
CORS(app)

# 启动时加载模型
print('正在加载模型...')
model, vectorizer = load_model()
n2c = load_class_names()
print(f'模型加载成功, 类别数: {model.n_classes_}')


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_classes': model.n_classes_,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """单条预测接口

    Request JSON:
        {"text": "商品名称"}

    Response JSON:
        {"pred_label_id": 0, "pred_label_name": "家居", "elapsed": 0.05}
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '缺少 text 字段'}), 400

        text = data['text']
        start = time.time()
        pred_id, pred_name = predict_single(text)
        elapsed = time.time() - start

        return jsonify({
            'pred_label_id': int(pred_id),
            'pred_label_name': pred_name,
            'elapsed': round(elapsed, 4),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch_api():
    """批量预测接口

    Request JSON:
        {"texts": ["商品1", "商品2", ...]}

    Response JSON:
        {"results": [{"pred_label_id": 0, "pred_label_name": "家居"}, ...]}
    """
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': '缺少 texts 字段'}), 400

        texts = data['texts']
        start = time.time()
        pred_ids, pred_names = predict_batch(texts)
        elapsed = time.time() - start

        results = [
            {'pred_label_id': int(pid), 'pred_label_name': pname}
            for pid, pname in zip(pred_ids, pred_names)
        ]

        return jsonify({
            'results': results,
            'elapsed': round(elapsed, 4),
            'count': len(results),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_proba', methods=['POST'])
def predict_proba_api():
    """带概率的预测接口

    Request JSON:
        {"text": "商品名称", "top_k": 3}

    Response JSON:
        {"pred_label_id": 0, "pred_label_name": "家居",
         "top_k": [{"label_id": 0, "label_name": "家居", "prob": 0.85}, ...]}
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '缺少 text 字段'}), 400

        text = data['text']
        top_k = data.get('top_k', 3)

        pred_id, pred_name, proba = predict_with_proba([text])

        # 获取 Top-K 概率
        top_indices = proba[0].argsort()[::-1][:top_k]
        top_list = []
        for idx in top_indices:
            name = n2c.get(idx, str(idx))
            top_list.append({
                'label_id': int(idx),
                'label_name': name,
                'prob': round(float(proba[0][idx]), 4),
            })

        return jsonify({
            'pred_label_id': int(pred_id[0]),
            'pred_label_name': pred_name[0],
            'top_k': top_list,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """获取所有类别列表"""
    classes = [{'label_id': k, 'label_name': v} for k, v in sorted(n2c.items())]
    return jsonify({'classes': classes, 'count': len(classes)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', '1') == '1'
    print(f'API 服务启动: http://0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=debug)
