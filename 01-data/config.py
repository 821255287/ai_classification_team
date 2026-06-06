# 创建配置类，用于集中管理数据文件路径等

class Config():
    #初始化
    def __init__(self):
        # 训练集文件路径
        self.train_datapath = './train.txt'
        # 测试集文件路径
        self.test_datapath = './test.txt'
        # 验证集文件路径
        self.dev_datapath = './dev.txt'
        # 分类文件路径
        self.class_path = './class.txt'
        # 停用词路径
        self.stopwords_path = './stopwords.txt'


# 测试代码
if __name__ == '__main__':
    # 创建配置类对象
    conf = Config()
    print(conf.train_datapath)
    print(conf.test_datapath)
