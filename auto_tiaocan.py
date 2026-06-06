import fasttext
import time
import logging
from itertools import product
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建配置对象
config = Config()

# 定义参数搜索空间（针对词级别数据优化）
param_grid = {
    'dim': [50, 100, 150],           # 词向量维度
    'epoch': [15, 25, 35],           # 训练轮数
    'lr': [0.3, 0.5, 0.7],           # 学习率
    'wordNgrams': [2, 3, 4],         # N-gram特征（词级别推荐2-4）
    'minCount': [1, 2, 3],           # 最小词频（过滤低频噪声词）
    'bucket': [1000000, 2000000],    # hash桶大小（词级别需要更大）
    'minn': [2, 3],                  # 最小子词长度
    'maxn': [4, 5, 6]                # 最长子词长度
}

# 记录最佳结果
best_score = 0
best_params = None
total_combinations = (
    len(param_grid['dim']) *
    len(param_grid['epoch']) *
    len(param_grid['lr']) *
    len(param_grid['wordNgrams']) *
    len(param_grid['minCount']) *
    len(param_grid['bucket']) *
    len(param_grid['minn']) *
    len(param_grid['maxn'])
)

logger.info("="*60)
logger.info("🚀 开始 FastText 词级别模型自动调参")
logger.info("="*60)
logger.info(f"📊 训练集: {config.process_train_datapath_word}")
logger.info(f"📊 验证集: {config.process_dev_datapath_word}")
logger.info(f"🔢 总参数组合数: {total_combinations}")
logger.info("="*60)

start_time = time.time()
combination_count = 0

# 遍历所有参数组合
for dim, epoch, lr, ngrams, mincount, bucket, minn, maxn in product(
    param_grid['dim'],
    param_grid['epoch'],
    param_grid['lr'],
    param_grid['wordNgrams'],
    param_grid['minCount'],
    param_grid['bucket'],
    param_grid['minn'],
    param_grid['maxn']
):
    combination_count += 1

    logger.info(f"\n[{combination_count}/{total_combinations}] 训练参数:")
    logger.info(f"  dim={dim}, epoch={epoch}, lr={lr}, ngrams={ngrams}")
    logger.info(f"  minCount={mincount}, bucket={bucket}, minn={minn}, maxn={maxn}")

    try:
        # 训练模型
        train_start = time.time()
        model = fasttext.train_supervised(
            input=config.process_train_datapath_word,  # ✅ 修复：使用词级别训练集
            dim=dim,
            epoch=epoch,
            lr=lr,
            wordNgrams=ngrams,
            minCount=mincount,
            bucket=bucket,
            minn=minn,
            maxn=maxn,
            thread=8,              # 多线程加速
            verbose=0              # 静默模式
        )
        train_time = time.time() - train_start

        # ✅ 修复：使用词级别验证集（原代码错误地使用了字符级别验证集）
        eval_start = time.time()
        num_samples, precision, recall = model.test(config.process_dev_datapath_word)
        eval_time = time.time() - eval_start

        # 计算 F1 分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        score = f1  # 以F1分数作为评价指标

        elapsed_time = time.time() - start_time

        logger.info(f"  ⏱️  训练耗时: {train_time:.1f}s | 评估耗时: {eval_time:.1f}s")
        logger.info(f"  📈 验证集结果 - 样本数: {num_samples}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        logger.info(f"  ⏰ 累计耗时: {elapsed_time/60:.1f}分钟")

        # 更新最佳参数
        if score > best_score:
            best_score = score
            best_params = {
                'dim': dim,
                'epoch': epoch,
                'lr': lr,
                'wordNgrams': ngrams,
                'minCount': mincount,
                'bucket': bucket,
                'minn': minn,
                'maxn': maxn
            }
            logger.info(f"  ✅ 🏆 新的最佳参数! F1={best_score:.4f}")

            # 保存最佳模型
            best_model_path = config.ft_model_save_path + "fasttext_model_word_best_autotune.bin"
            model.save_model(best_model_path)
            logger.info(f"  💾 最佳模型已保存: {best_model_path}")

    except Exception as e:
        logger.error(f"  ❌ 训练失败: {e}")
        continue

# 输出最终结果
total_time = time.time() - start_time
logger.info("\n" + "="*60)
logger.info("🎉 自动调参完成！")
logger.info("="*60)
logger.info(f"⏰ 总耗时: {total_time/60:.1f}分钟 ({total_time:.1f}秒)")
logger.info(f"🔢 完成组合数: {combination_count}/{total_combinations}")
logger.info(f"\n🏆 最终最佳参数:")
for key, value in best_params.items():
    logger.info(f"  {key:12s}: {value}")
logger.info(f"\n🏆 最佳F1分数: {best_score:.4f} ({best_score*100:.2f}%)")
logger.info(f"💾 最佳模型路径: {config.ft_model_save_path}fasttext_model_word_best_autotune.bin")
logger.info("="*60)

# 保存最佳参数到文件
import json
params_file = config.ft_model_save_path + "best_params_word.json"
with open(params_file, 'w', encoding='utf-8') as f:
    json.dump({
        'best_params': best_params,
        'best_f1_score': best_score,
        'total_time_minutes': total_time / 60,
        'total_combinations': total_combinations
    }, f, ensure_ascii=False, indent=2)
logger.info(f"📝 最佳参数已保存到: {params_file}")
