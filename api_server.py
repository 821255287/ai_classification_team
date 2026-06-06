# 导包
from flask import Flask, request, jsonify
from predict_func import predict


# 创建Flask对象
app = Flask("fasttext")

# 定义路由
@app.route("/predict", methods=["POST"])
def predict_app():
    data = request.get_json()       # 获取请求数据
    result = predict(data)          # 预测数据
    return jsonify(result)          # 返回结果

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)   # 启动服务,但是POST不能直接访问