# 创建配置类，集中管理json、停用词路径
class Config():
    def __init__(self):
        # 原始JSON数据源路径（项目只有json数据集）
        self.train_json_path = "./train.jsonl"
        self.test_json_path = "./test.jsonl"
        # 停用词文件保留不变
        self.stopwords_path = "./stopwords.txt"
        # 不再定义train_datapath/dev_datapath/test_datapath(方案二不用txt)

# 配置测试
if __name__ == '__main__':
    conf = Config()
    print("训练集json路径：", conf.train_json_path)
    print("测试集json路径：", conf.test_json_path)