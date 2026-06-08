# 导包
import jieba
from config import Config

# 创建文本配置对象
config = Config()


def cut_text(data_path, process_path, is_char=True):
    """
    :data_path:      待处理数据路径
    :process_path:   处理后数据保存路径
    :is_char:        是否进行字符级别分词
    """
    with open(data_path, encoding="utf-8") as f:
        with open(process_path, "w", encoding="utf-8") as wf:
            for line in f.readlines():
                line.strip()  # 去除首尾空
                if not line:
                    continue
                text, label = line.split("\t")
                if is_char:
                    words = " ".join(list(text))
                else:
                    words = " ".join(jieba.lcut(text))
                # 分类ID转换具体类别名称
                label_str = config.id2class_dict[int(label)]
                words_line = "__label__" + label_str + " " + words + "\n"
                wf.write(words_line)
    print(f"处理完成,保存路径: {process_path}")


if __name__ == '__main__':
    # 处理字符级别数据
    cut_text(config.train_datapath, config.process_train_datapath_char, is_char=True)
    cut_text(config.test_datapath, config.process_test_datapath_char, is_char=True)
    cut_text(config.dev_datapath, config.process_dev_datapath_char, is_char=True)

    # 处理词级别数据
    cut_text(config.train_datapath, config.process_train_datapath_word, is_char=False)
    cut_text(config.test_datapath, config.process_test_datapath_word, is_char=False)
    cut_text(config.dev_datapath, config.process_dev_datapath_word, is_char=False)