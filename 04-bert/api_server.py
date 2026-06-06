# 模型部署
import fasttext
import jieba
from predict_fun import predict_fun

from flask import Flask, request,jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    # 获取请求数据
    data = request.get_json()
    #预测
    print("-------------预测结果------------")
    result = predict_fun(data)
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8003)