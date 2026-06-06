import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹窗
import os
from config import Config
from utils import build_dataloader
from bert_classifer_model import BertClassifier

# 加载配置对象，包含模型参数、路径等
conf = Config()
# 忽略的警告信息
import warnings

warnings.filterwarnings("ignore")
def model2train():
    """
    训练 BERT 分类模型并在验证集上评估，保存最佳模型。
    参数：无显式参数，所有配置通过全局 conf 对象获取。
    返回：无返回值，训练过程中保存最佳模型到指定路径。
    """
    # todo 1、准备数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 定义训练参数，包括：设备，轮数，学习率，模型保存路径
    device = conf.device
    epochs = conf.num_epochs
    lr = conf.learning_rate
    model_save_path = conf.model_save_path

    # todo 2、准备模型
    model = BertClassifier()

    # 将模型移动到指定的设备gpu
    model.to(device)

    # todo 3、定义优化器
    optimizer = AdamW(model.parameters(), lr=lr)
    # todo 4、定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 初始化F1分数，用于保存最好的模型
    best_f1 = 0.0

    # ========== 记录训练过程中的指标，用于最后画图 ==========
    train_records = []   # 每一项: {step, epoch, loss, acc, f1}
    val_records = []     # 每一项: {step, epoch, f1, accuracy, precision, recall}
    global_step = 0      # 全局 batch 计数器（横轴）

    # todo 2个遍历：外层遍历轮次
    for epoch in range(epochs):
        # 设置模型为训练模式
        model.train()
        # 初始化累计损失，初始化训练集预测和真实标签
        total_loss = 0.0
        train_preds, train_labels = [], []  # 存储训练集预测和真实标签

        # todo 2个遍历：内层遍历批次
        for i, batch in enumerate(tqdm(train_dataloader, desc="Training on train set")):
            global_step += 1

            #  提取批次数据
            input_ids, attention_mask, labels = batch
            # 移动到设备gpu
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)


            # todo 5.1前向传播：模型预测
            outputs = model(input_ids, attention_mask)

            # todo 5.2计算损失
            loss = loss_fn(outputs, labels)

            # todo 5.3梯度清零
            optimizer.zero_grad()

            # todo 5.4反向传播：计算梯度
            loss.backward()

            # todo 5.5参数更新
            optimizer.step()

            # 累计损失
            total_loss += loss.item()

            #
            # 获取预测结果（最大logits对应的类别）
            y_pred_list = torch.argmax(outputs, dim=1)

            # 存储预测和真实标签，用于计算训练集指标
            # y_pred_list 和 labels 都在 GPU（cuda） 上，GPU 张量 不能直接转列表
            # 必须先挪回 CPU，才能用 .tolist()
            train_preds.extend(y_pred_list.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

            # 每50个批次，计算训练集指标
            if (i + 1) % 10 == 0 or i == len(train_dataloader) - 1:
                # 计算准确率和f1值
                acc = accuracy_score(train_labels, train_preds)
                # macro（宏平均）的计算步骤：对每一个类别，单独计算它自己的 F1 分数把所有类别的 F1 分数直接相加，
                # 再除以类别总数不关心每个类别有多少样本（大类别、小类别权重一样）
                f1 = f1_score(train_labels, train_preds, average='macro')
                # 获取batch_count，并计算平均损失
                batch_count = i % 10 + 1
                avg_loss = total_loss / batch_count
                # 打印训练信息
                print(
                    f"\nEpoch: {epoch + 1}, Batch: {i + 1}, Loss: {avg_loss:.4f}, acc:{acc:.4f}, f1:{f1:.4f}")
                # 记录训练指标
                train_records.append({
                    'step': global_step, 'epoch': epoch + 1,
                    'loss': avg_loss, 'acc': acc, 'f1': f1
                })
                # 清空累计损失和预测和真实标签
                total_loss = 0.0
                train_preds, train_labels = [], []

            # 每200个批次，计算验证集指标，打印，保存模型
            if (i + 1) % 100 == 0 or i == len(train_dataloader) - 1:
                # 计算在测试集的评估报告，准确率，精确率，召回率，f1值
                report, f1score, accuracy, precision, recall = model2dev(model, dev_dataloader, device)
                print("验证集评估报告：\n", report)
                print(
                    f"验证集f1，accuracy, precision, recall: {f1score:.4f}, {accuracy:.4f}, {precision:.4f}, {recall:.4f}")
                # 记录验证指标
                val_records.append({
                    'step': global_step, 'epoch': epoch + 1,
                    'f1': f1score, 'accuracy': accuracy,
                    'precision': precision, 'recall': recall
                })
                # 将模型设置为训练模式
                model.train()
                # 如果验证F1分数优于历史最佳，保存模型
                if f1score > best_f1:
                    # 更新历史最佳F1分数
                    best_f1 = f1score
                    # todo 保存模型
                    torch.save(model.state_dict(), model_save_path)
                    print("保存模型, f1score:", best_f1)

    # ========== 训练结束，画图 ==========
    plot_training_curves(train_records, val_records, conf.root_path)
def plot_training_curves(train_records, val_records, save_dir):
    """
    训练结束后，根据记录的数据绘制 Loss、Accuracy、F1 等曲线并保存。

    参数：
        train_records (list): 训练集指标记录，每项含 step, epoch, loss, acc, f1。
        val_records (list): 验证集指标记录，每项含 step, epoch, f1, accuracy, precision, recall。
        save_dir (str): 图表保存目录。
    """
    chart_dir = os.path.join(save_dir, "04-bert", "charts")
    os.makedirs(chart_dir, exist_ok=True)

    # 设置中文字体（Windows 用 SimHei，否则回退）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    train_steps = [r['step'] for r in train_records]
    train_loss = [r['loss'] for r in train_records]
    train_acc = [r['acc'] for r in train_records]
    train_f1 = [r['f1'] for r in train_records]

    val_steps = [r['step'] for r in val_records]
    val_acc = [r['accuracy'] for r in val_records]
    val_f1 = [r['f1'] for r in val_records]
    val_precision = [r['precision'] for r in val_records]
    val_recall = [r['recall'] for r in val_records]

    # ====== 图1：训练 Loss 曲线 ======
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_loss, marker='o', color='#e74c3c', linewidth=1.5, markersize=3)
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print(f"[Chart] Loss 曲线已保存 -> {chart_dir}/loss_curve.png")

    # ====== 图2：训练集 Accuracy & F1 ======
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_acc, marker='o', color='#2ecc71', linewidth=1.5, markersize=3, label='Train Accuracy')
    plt.plot(train_steps, train_f1, marker='s', color='#3498db', linewidth=1.5, markersize=3, label='Train F1 (macro)')
    plt.xlabel('Global Step')
    plt.ylabel('Score')
    plt.title('Training Accuracy & F1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'train_acc_f1.png'), dpi=150)
    plt.close()
    print(f"[Chart] 训练集 Acc/F1 曲线已保存 -> {chart_dir}/train_acc_f1.png")

    # ====== 图3：验证集 Accuracy & F1 ======
    plt.figure(figsize=(10, 5))
    plt.plot(val_steps, val_acc, marker='o', color='#2ecc71', linewidth=1.5, markersize=5, label='Val Accuracy')
    plt.plot(val_steps, val_f1, marker='s', color='#e67e22', linewidth=1.5, markersize=5, label='Val F1 (macro)')
    plt.xlabel('Global Step')
    plt.ylabel('Score')
    plt.title('Validation Accuracy & F1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'val_acc_f1.png'), dpi=150)
    plt.close()
    print(f"[Chart] 验证集 Acc/F1 曲线已保存 -> {chart_dir}/val_acc_f1.png")

    # ====== 图4：验证集 Precision & Recall ======
    plt.figure(figsize=(10, 5))
    plt.plot(val_steps, val_precision, marker='^', color='#9b59b6', linewidth=1.5, markersize=5, label='Val Precision')
    plt.plot(val_steps, val_recall, marker='v', color='#1abc9c', linewidth=1.5, markersize=5, label='Val Recall')
    plt.xlabel('Global Step')
    plt.ylabel('Score')
    plt.title('Validation Precision & Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'val_precision_recall.png'), dpi=150)
    plt.close()
    print(f"[Chart] 验证集 Precision/Recall 曲线已保存 -> {chart_dir}/val_precision_recall.png")

    # ====== 图5：综合对比大图（2×2 子图） ======
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1：Loss
    axes[0, 0].plot(train_steps, train_loss, marker='o', color='#e74c3c', linewidth=1.2, markersize=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Global Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # 子图2：Train Acc + F1
    axes[0, 1].plot(train_steps, train_acc, marker='o', color='#2ecc71', linewidth=1.2, markersize=2, label='Acc')
    axes[0, 1].plot(train_steps, train_f1, marker='s', color='#3498db', linewidth=1.2, markersize=2, label='F1')
    axes[0, 1].set_title('Training Accuracy & F1')
    axes[0, 1].set_xlabel('Global Step')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 子图3：Val Acc + F1
    axes[1, 0].plot(val_steps, val_acc, marker='o', color='#2ecc71', linewidth=1.5, markersize=4, label='Acc')
    axes[1, 0].plot(val_steps, val_f1, marker='s', color='#e67e22', linewidth=1.5, markersize=4, label='F1')
    axes[1, 0].set_title('Validation Accuracy & F1')
    axes[1, 0].set_xlabel('Global Step')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 子图4：Val Precision + Recall
    axes[1, 1].plot(val_steps, val_precision, marker='^', color='#9b59b6', linewidth=1.5, markersize=4, label='Precision')
    axes[1, 1].plot(val_steps, val_recall, marker='v', color='#1abc9c', linewidth=1.5, markersize=4, label='Recall')
    axes[1, 1].set_title('Validation Precision & Recall')
    axes[1, 1].set_xlabel('Global Step')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('BERT Training Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'training_overview.png'), dpi=150)
    plt.close()
    print(f"[Chart] 综合对比大图已保存 -> {chart_dir}/training_overview.png")


def model2dev(model, data_loader, device):
    """
    在验证或测试集上评估 BERT 分类模型的性能。

    参数：
        model (nn.Module): BERT 分类模型。
        data_loader (DataLoader): 数据加载器（验证或测试集）。
        device (str): 设备（"cuda" 或 "cpu"）。

    返回：
        tuple: (分类报告, F1 分数, 准确度, 精确度，召回率)
            - report: 分类报告（包含每个类别的精确度、召回率、F1 分数等）。
            - f1score: 微平均 F1 分数。
            - accuracy: 准确度。
            - precision: 微平均精确度
            - recall: 微平均召回率
    """
    # todo 准备数据： 初始化列表，存储预测结果和真实标签
    all_preds, all_labels = [], []
    # todo 准备模型 设置模型为评估模式（禁用 dropout 和 batch norm）
    model.eval()

    #  禁用梯度计算以提高效率并减少内存占用
    with torch.no_grad():
        # todo 一个遍历：批次
        for i, batch in enumerate(tqdm(data_loader, desc="Evaluating model on dev set")):
            #  提取批次数据
            input_ids, attention_mask, labels = batch
            # 移动到设备gpu
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # todo 2.1前向传播：模型预测
            outputs = model(input_ids, attention_mask)

            #  获取预测结果（最大 logits 对应的类别）
            y_pred_list = torch.argmax(outputs, dim=1)

            #  存储预测和真实标签
            all_preds.extend(y_pred_list.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # todo 2.2计算分类报告、F1 分数、准确率，精确率，召回率
    report = classification_report(all_labels, all_preds)
    f1score = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # 返回评估结果
    return report, f1score, accuracy, precision, recall
if __name__ == '__main__':
    # 主程序入口
    model2train()