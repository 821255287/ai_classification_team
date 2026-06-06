"""
加载已训练的 BERT 分类模型，在测试集上评估并输出 ACC / F1 / Precision / Recall。
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 中文字体设置（Windows 用 SimHei，否则回退）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import os
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from config import Config
from utils import build_dataloader
from bert_classifer_model import BertClassifier

conf = Config()


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制归一化混淆矩阵并保存"""
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # 在格子中标注数值
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i][j]
            if val > 0:
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=6)

    ax.set_title('Confusion Matrix')
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Chart] 混淆矩阵已保存 -> {save_path}")


def evaluate():
    # ---------- 1. 加载数据 ----------
    _, test_dataloader, _ = build_dataloader()
    print(f"测试集 batch 数: {len(test_dataloader)}")

    # ---------- 2. 加载模型 ----------
    device = conf.device
    print(f"设备: {device}")

    model = BertClassifier()
    model.load_state_dict(torch.load(conf.model_save_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型已加载: {conf.model_save_path}")

    # ---------- 3. 推理 ----------
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="评估中"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # ---------- 4. 计算指标 ----------
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')

    print("\n" + "=" * 60)
    print("                    测试集评估结果")
    print("=" * 60)
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  F1 (macro)       : {f1_macro:.4f}")
    print(f"  F1 (micro)       : {f1_micro:.4f}")
    print(f"  F1 (weighted)    : {f1_weighted:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro)   : {recall_macro:.4f}")
    print("=" * 60)

    # ---------- 5. 分类报告 ----------
    target_names = conf.class_list
    print("\n" + "=" * 60)
    print("                    分类报告 (per-class)")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names=target_names,
        digits=4,
    ))

    # ---------- 6. 混淆矩阵 ----------
    cm = confusion_matrix(all_labels, all_preds)
    chart_dir = os.path.join(conf.root_path, "04-bert", "charts")
    plot_confusion_matrix(cm, target_names, os.path.join(chart_dir, "confusion_matrix.png"))

    # ---------- 7. 每类 Top-5 漏报 & 误报 ----------
    print("\n" + "=" * 60)
    print("              每类 FN（漏报）& FP（误报）Top-5")
    print("=" * 60)

    # 建立每个样本的 text 索引
    test_texts = []
    with open(conf.test_datapath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = line.rsplit("\t", 1)[0]
            test_texts.append(text)

    for cls_id in range(conf.num_classes):
        cls_name = target_names[cls_id]

        # 漏报：真实是该类，预测不是
        fn_mask = [(t == cls_id and p != cls_id) for t, p in zip(all_labels, all_preds)]
        fn_texts = [test_texts[i] for i, m in enumerate(fn_mask) if m]

        # 误报：预测是该类，真实不是
        fp_mask = [(t != cls_id and p == cls_id) for t, p in zip(all_labels, all_preds)]
        fp_texts = [test_texts[i] for i, m in enumerate(fp_mask) if m]

        if fn_texts or fp_texts:
            print(f"\n  [{cls_name}]  FN={len(fn_texts)}  FP={len(fp_texts)}")
            if fn_texts:
                for t in fn_texts[:3]:
                    print(f"    FN: {t[:80]}")
            if fp_texts:
                for t in fp_texts[:3]:
                    print(f"    FP: {t[:80]}")

    print("\n🎉 评估完成！")


if __name__ == "__main__":
    evaluate()
