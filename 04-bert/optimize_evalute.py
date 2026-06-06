# 导包
import time
import warnings
import numpy as np
import pandas as pd
import torch
from bert_classifer_model import BertClassifier
from config import Config
from utils import build_dataloader
from train import model2dev

# 忽略弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

# TODO 1 加载配置和数据集
conf = Config()
train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

# TODO 2 定义计时函数
def count_time(model, dataloader, device='cpu', num_batches=50):
    """
    测量模型在指定设备上的平均推理时间（毫秒/批次）

    参数：
        model: 已设为 eval 模式的模型
        dataloader: 数据加载器
        device: 'cpu' 或 'cuda'
        num_batches: 用于测时的批次数

    返回：
        avg_time: 平均每批次耗时（毫秒）
        std_time: 耗时标准差
    """
    model.to(device)
    model.eval()
    times = []

    with torch.no_grad():
        for i, (input_ids, attention_mask, labels) in enumerate(dataloader):
            if i >= num_batches:
                break

            # 将所需数据传输到指定设备
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 第1批预热（不计入统计）
            if i == 0:
                _ = model(input_ids, attention_mask)
                continue

            # 计时
            start_time = time.time()
            _ = model(input_ids, attention_mask)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒

    avg_time = np.mean(times) if times else 0
    std_time = np.std(times) if times else 0
    return avg_time, std_time

# TODO 3 计算变化百分比
def compute_change(before, after):
    """计算前后变化百分比"""
    if before == 0:
        return "inf"
    change = (after - before) / before * 100
    return f"{change:+.1f}%"

# TODO 4 加载模型函数
def load_model(model_path, device='cpu', is_quantized=False):
    """
    加载模型并移到指定设备

    参数：
        model_path: 模型文件路径
        device: 目标设备
        is_quantized: 是否为量化模型（保存的是完整模型对象）

    返回：
        加载好的模型
    """
    if is_quantized:
        # 量化模型：直接加载完整模型对象
        model = torch.load(model_path, map_location=device, weights_only=False)
    else:
        # 原始模型：先创建实例，再加载 state_dict
        model = BertClassifier()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    
    model.to(device)
    model.eval()
    return model

# TODO 5 主评估函数
def evaluate_quantization():
    """评估量化前后的性能对比"""
    print("BERT 模型 FP32 → INT8 动态量化评估")
    
    # --- 量化前评估（FP32）---
    print("\n加载原始模型（FP32）...")
    original_model = load_model(conf.model_save_path, device='cpu', is_quantized=False)

    print("测量原始模型推理时间...")
    time_fp32, std_fp32 = count_time(original_model, dev_dataloader, device='cpu')
    print(f"  → 平均推理时间: {time_fp32:.2f} ± {std_fp32:.2f} ms/batch", flush=True)

    print("评估原始模型性能指标...")
    report_fp32, f1_fp32, acc_fp32, prec_fp32, rec_fp32 = model2dev(
        original_model, dev_dataloader, device='cpu'
    )
    print(f"  → F1: {f1_fp32:.4f}, Accuracy: {acc_fp32:.4f}")

    # --- 量化后评估（INT8）---
    print("\n加载量化模型（INT8）...")
    quantized_model = load_model(conf.test_save_path + 'quantized_model.pth', device='cpu', is_quantized=True)

    print("测量量化模型推理时间...")
    time_int8, std_int8 = count_time(quantized_model, dev_dataloader, device='cpu')
    print(f"  werty→ 平均推理时间: {time_int8:.2f} ± {std_int8:.2f} ms/batch", flush=True)

    print("评估量化模型性能指标...")
    report_int8, f1_int8, acc_int8, prec_int8, rec_int8 = model2dev(
        quantized_model, dev_dataloader, device='cpu'
    )
    print(f"  → F1: {f1_int8:.4f}, Accuracy: {acc_int8:.4f}")

    # --- 构建对比表格 ---
    print("\n量化效果对比（FP32 → INT8）")
    metrics = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'Inference Time (ms/batch)']
    before_vals = [f1_fp32, acc_fp32, prec_fp32, rec_fp32, time_fp32]
    after_vals = [f1_int8, acc_int8, prec_int8, rec_int8, time_int8]

    df = pd.DataFrame({
        'Metric': metrics,
        'FP32 (Before)': before_vals,
        'INT8 (After)': after_vals,
        'Change (%)': [compute_change(b, a) for b, a in zip(before_vals, after_vals)]
    })

    print(df.to_string(index=False))
    print("=" * 70)

    # --- 补充 ---
    print(f"\n推理时间加速比: {time_fp32 / time_int8:.2f}x" if time_int8 > 0 else "\n推理时间加速比: N/A")
    print(f"精度损失 (F1): {f1_fp32 - f1_int8:.4f}")


if __name__ == "__main__":
    evaluate_quantization()
