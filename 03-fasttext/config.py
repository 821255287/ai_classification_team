class Config:
    def __init__(self):
        self.root_path = '//'
        # 原始数据路
        self.train_datapath = self.root_path + '01-data/train.txt'
        self.test_datapath = self.root_path + '01-data/test.txt'
        self.dev_datapath = self.root_path + '01-data/dev.txt'
        # 类别文档
        self.class_datapath = self.root_path + "01-data/class.txt"

        # 数据处理保存路径
        # 字符级别fasttext
        self.process_train_datapath_char = self.root_path + "03-fasttext/final_data/train_process_char.txt"
        self.process_test_datapath_char = self.root_path + "03-fasttext/final_data/test_process_char.txt"
        self.process_dev_datapath_char = self.root_path + "03-fasttext/final_data/dev_process_char.txt"

        # 词级别fasttext
        self.process_train_datapath_word = self.root_path + "03-fasttext/final_data/train_process_word.txt"
        self.process_test_datapath_word = self.root_path + "03-fasttext/final_data/test_process_word.txt"
        self.process_dev_datapath_word = self.root_path + "03-fasttext/final_data/dev_process_word.txt"

        # 模型保存路径
        self.ft_model_save_path = self.root_path + '03-fasttext/save_model/'

        # 处理完的数据（用于训练）
        self.final_data = self.root_path + '03-fasttext/final_data'

        # 类别字典
        self.id2class_dict = {}
        with open(self.class_datapath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split('\t')
                    if len(parts) == 2:
                        class_id, class_name = parts
                        self.id2class_dict[int(class_id)] = class_name


if __name__ == '__main__':
    config = Config()
    print(config.root_path)
    print(config.train_datapath)
    print(config.class_datapath)
    print(config.id2class_dict)