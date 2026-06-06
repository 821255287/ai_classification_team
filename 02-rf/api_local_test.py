"""
API 本地测试脚本
测试 Flask API 各接口是否正常工作
"""
import requests
import json
import time

BASE_URL = 'http://127.0.0.1:5000'

TEST_TEXTS = [
    '伊利纯牛奶250ml',
    '小米14手机壳透明防摔',
    '三只松鼠坚果大礼包',
    '维达抽纸3层120抽',
    '李宁运动鞋男款篮球鞋',
]


def test_health():
    """测试健康检查"""
    print('=' * 50)
    print('1. 测试健康检查 /health')
    print('=' * 50)
    resp = requests.get(f'{BASE_URL}/health')
    data = resp.json()
    print(f'  状态码: {resp.status_code}')
    print(f'  响应: {json.dumps(data, ensure_ascii=False)}')
    assert resp.status_code == 200
    assert data['status'] == 'ok'
    print('  ✅ 通过\n')


def test_single_predict():
    """测试单条预测"""
    print('=' * 50)
    print('2. 测试单条预测 /predict')
    print('=' * 50)

    for text in TEST_TEXTS:
        resp = requests.post(
            f'{BASE_URL}/predict',
            json={'text': text},
        )
        data = resp.json()
        print(f'  输入: {text}')
        print(f'  预测: [{data["pred_label_id"]}] {data["pred_label_name"]} ({data["elapsed"]}s)')
        print()
        assert resp.status_code == 200

    print('  ✅ 全部通过\n')


def test_batch_predict():
    """测试批量预测"""
    print('=' * 50)
    print('3. 测试批量预测 /predict_batch')
    print('=' * 50)

    resp = requests.post(
        f'{BASE_URL}/predict_batch',
        json={'texts': TEST_TEXTS},
    )
    data = resp.json()
    print(f'  输入: {len(TEST_TEXTS)} 条')
    print(f'  预测结果:')
    for i, r in enumerate(data['results']):
        print(f'    [{i}] {TEST_TEXTS[i][:30]:<30s} → [{r["pred_label_id"]}] {r["pred_label_name"]}')
    print(f'  总耗时: {data["elapsed"]}s')
    assert resp.status_code == 200
    assert data['count'] == len(TEST_TEXTS)
    print('  ✅ 通过\n')


def test_proba_predict():
    """测试带概率的预测"""
    print('=' * 50)
    print('4. 测试概率预测 /predict_proba')
    print('=' * 50)

    text = TEST_TEXTS[0]
    resp = requests.post(
        f'{BASE_URL}/predict_proba',
        json={'text': text, 'top_k': 3},
    )
    data = resp.json()
    print(f'  输入: {text}')
    print(f'  预测: [{data["pred_label_id"]}] {data["pred_label_name"]}')
    print(f'  Top 3:')
    for item in data['top_k']:
        print(f'    [{item["label_id"]}] {item["label_name"]:<10s} 概率: {item["prob"]:.4f}')
    assert resp.status_code == 200
    print('  ✅ 通过\n')


def test_classes():
    """测试类别接口"""
    print('=' * 50)
    print('5. 测试类别列表 /classes')
    print('=' * 50)

    resp = requests.get(f'{BASE_URL}/classes')
    data = resp.json()
    print(f'  类别总数: {data["count"]}')
    print(f'  前5个类别:')
    for c in data['classes'][:5]:
        print(f'    [{c["label_id"]}] {c["label_name"]}')
    assert resp.status_code == 200
    assert data['count'] == 30
    print('  ✅ 通过\n')


def test_performance():
    """性能测试"""
    print('=' * 50)
    print('6. 性能测试 (100次单条预测)')
    print('=' * 50)

    text = TEST_TEXTS[0]
    times = []
    for _ in range(100):
        start = time.time()
        resp = requests.post(f'{BASE_URL}/predict', json={'text': text})
        times.append(time.time() - start)

    avg_time = sum(times) / len(times) * 1000
    print(f'  平均耗时: {avg_time:.2f} ms')
    print(f'  最慢: {max(times)*1000:.2f} ms')
    print(f'  最快: {min(times)*1000:.2f} ms')
    print(f'  QPS: {1000/avg_time:.0f}')
    print('  ✅ 完成\n')


if __name__ == '__main__':
    print('\n' + '=' * 50)
    print(' 随机森林 API 本地测试')
    print('=' * 50)
    print(f' 服务地址: {BASE_URL}')
    print()

    try:
        test_health()
        test_single_predict()
        test_batch_predict()
        test_proba_predict()
        test_classes()
        test_performance()

        print('=' * 50)
        print(' 🎉 所有测试通过!')
        print('=' * 50)

    except requests.ConnectionError:
        print('❌ 无法连接到 API 服务!')
        print('请先启动服务: python api_server.py')
        print('或: python run_app.py --mode api')
    except AssertionError as e:
        print(f'❌ 测试断言失败: {e}')
    except Exception as e:
        print(f'❌ 测试出错: {e}')
