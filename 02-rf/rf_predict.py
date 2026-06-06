"""
随机森林模型 — 预测脚本
支持: 单条预测 / 批量预测 / 文件评估
"""
import os
import sys
import time
import argparse

from rf_predict_fun import (
    load_model, load_class_names,
    predict_single, predict_batch, predict_with_proba, evaluate_on_file,
    preprocess_text
)
from config import Config


def main():
    parser = argparse.ArgumentParser(description='随机森林商品分类预测')
    parser.add_argument('--text', '-t', type=str, default=None,
                        help='单条文本预测')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='批量预测文件路径 (tsv格式: text\\tlabel)')
    parser.add_argument('--proba', '-p', action='store_true',
                        help='输出预测概率')
    args = parser.parse_args()

    conf = Config()

    # 加载模型
    print('加载模型中...')
    t0 = time.time()
    model, vectorizer = load_model()
    n2c = load_class_names()
    print(f'模型加载完成, 耗时: {time.time()-t0:.2f}s')
    print(f'模型类别数: {model.n_classes_}')
    print()

    # 单条预测
    if args.text:
        pred_id, pred_name = predict_single(args.text)
        print(f'输入文本: {args.text}')
        print(f'预处理后: {preprocess_text(args.text)}')
        print(f'预测类别: [{pred_id}] {pred_name}')

        if args.proba:
            _, _, proba = predict_with_proba([args.text])
            print(f'\n各类别概率 Top5:')
            top_indices = proba[0].argsort()[::-1][:5]
            for idx in top_indices:
                name = n2c.get(idx, str(idx))
                print(f'  [{idx:2d}] {name:<8s}: {proba[0][idx]:.4f}')

    # 文件批量预测
    elif args.file:
        if not os.path.exists(args.file):
            print(f'文件不存在: {args.file}')
            sys.exit(1)

        evaluate_on_file(args.file)

    # 无参数: 交互模式
    else:
        print('=' * 50)
        print('  随机森林商品分类预测 - 交互模式')
        print('  输入商品名称, 按回车预测, 输入 quit 退出')
        print('=' * 50)
        print()

        while True:
            text = input('请输入商品名称: ').strip()
            if text.lower() in ('quit', 'exit', 'q'):
                print('再见！')
                break
            if not text:
                continue

            start = time.time()
            pred_id, pred_name = predict_single(text)
            elapsed = time.time() - start

            print(f'  预处理: {preprocess_text(text)}')
            print(f'  预测结果: [{pred_id}] {pred_name}')
            print(f'  耗时: {elapsed:.3f}s')

            # 输出 Top3 概率
            _, _, proba = predict_with_proba([text])
            top3 = proba[0].argsort()[::-1][:3]
            print(f'  Top3: ', end='')
            for i, idx in enumerate(top3):
                name = n2c.get(idx, str(idx))
                print(f'{name}({proba[0][idx]:.2f})', end='  ')
            print()
            print()


if __name__ == '__main__':
    main()
