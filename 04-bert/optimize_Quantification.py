from config import Config
import torch
from utils import build_dataloader
from train import model2dev
from bert_classifer_model import BertClassifier
# 消除弃用警告
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from optimize_evalute import evaluate_quantization

# 初始化配置
conf = Config()
# 设备配置
device = 'cpu'
def quantification(model_path):
    # 加载模型
    model = BertClassifier()
    model.load_state_dict(torch.load(conf.model_save_path))
    print("模型加载完成！",flush=True)

    # TODO 1 模型量化
    print("开始量化...")
    model.to('cpu')
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    quantized_model.eval()
    # # ---------- 添加 torch.compile 加速推理----------
    # quantized_model = torch.compile(quantized_model, backend="inductor")  # 默认后端，也可选 "openvino" 等
    # # ----------------------------------------
    print("量化完成！")

    # TODO 2 保存整个量化模型
    torch.save(quantized_model, conf.quantization_bert_model_path)
    print("保存量化模型成功！地址为：", conf.quantization_bert_model_path)

    # TODO 3 量化评估
    evaluate_quantization()

if __name__ == '__main__':
    quantification(conf.model_save_path)
