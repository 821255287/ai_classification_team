"""
随机森林基线模型 — 集中管理所有路径和参数
"""
import os
import pandas as pd


class Config():
    """随机森林基线模型 — 集中管理所有路径和参数"""
    def __init__(self):
        # ========== 项目根目录 ==========
        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ========== 原始数据路径 ==========
        self.train_datapath = os.path.join(self.root_path, '01-data', 'train.txt')
        self.test_datapath = os.path.join(self.root_path, '01-data', 'test.txt')
        self.dev_datapath = os.path.join(self.root_path, '01-data', 'dev.txt')
        self.class_path = os.path.join(self.root_path, '01-data', 'class.txt')
        self.stopwords_path = os.path.join(self.root_path, '01-data', 'stopwords.txt')

        # ========== 处理后数据保存路径 ==========
        self.process_dir = os.path.join(self.root_path, '02-rf', 'final_data')
        os.makedirs(self.process_dir, exist_ok=True)
        self.process_train_datapath = os.path.join(self.process_dir, 'train_process.csv')
        self.process_test_datapath = os.path.join(self.process_dir, 'test_process.csv')
        self.process_dev_datapath = os.path.join(self.process_dir, 'dev_process.csv')

        # ========== 模型保存路径 ==========
        self.model_save_dir = os.path.join(self.root_path, '02-rf', 'save_model')
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.rf_model_save_path = os.path.join(self.model_save_dir, 'rf_model.pkl')
        self.tfidf_model_save_path = os.path.join(self.model_save_dir, 'tfidf_model.pkl')

        # ========== 结果保存路径 ==========
        self.result_dir = os.path.join(self.root_path, '02-rf', 'result')
        os.makedirs(self.result_dir, exist_ok=True)
        self.model_predict_result = os.path.join(self.result_dir, 'predict_result.csv')
        self.train_metrics_path = os.path.join(self.result_dir, 'train_metrics.txt')
        self.dev_metrics_path = os.path.join(self.result_dir, 'dev_metrics.txt')
        self.test_metrics_path = os.path.join(self.result_dir, 'test_metrics.txt')

        # ========== 随机森林模型参数 ==========
        self.rf_params = {
            'n_estimators': 200,
            'max_depth': 50,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 666,
            'n_jobs': -1,
            'class_weight': 'balanced',
        }

        # ========== TF-IDF 参数 ==========
        self.tfidf_params = {
            'max_features': 50000,
            'ngram_range': (1, 2),
            'min_df': 3,
            'max_df': 0.8,
            'sublinear_tf': True,
        }

        # ========== 其他参数 ==========
        self.max_words = 50

    # ========== 工具方法 ==========
    def load_class_map(self):
        """加载类别映射 (每行一个类别名, 行号即 label_id)"""
        n2c = {}  # number → class_name
        c2n = {}  # class_name → number
        if os.path.exists(self.class_path):
            with open(self.class_path, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name:
                        n2c[i] = name
                        c2n[name] = i
        return n2c, c2n

    def load_stopwords(self):
        """加载停用词列表"""
        words = []
        if os.path.exists(self.stopwords_path):
            with open(self.stopwords_path, encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
        return words


# 测试代码
if __name__ == '__main__':
    # 创建配置类对象
    conf = Config()
    print('项目根目录: ', conf.root_path)
    print('训练集路径: ', conf.train_datapath)
    print('测试集路径: ', conf.test_datapath)
    print('验证集路径: ', conf.dev_datapath)
    print('分类文件路径: ', conf.class_path)
    print('停用词路径: ', conf.stopwords_path)
    print('模型保存路径: ', conf.rf_model_save_path)
    print('结果存放目录: ', conf.result_dir)

    # 测试读取数据
    data = pd.read_csv(conf.train_datapath, sep='\t', names=['text', 'label'])
    print('\n训练集样本数: ', len(data))
    print(data.head())
