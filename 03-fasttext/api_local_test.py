import requests


url = 'http://127.0.0.1:8080/predict'

if __name__ == '__main__':
    data = {"text": "T 7 G 滨 崎 小 熊 碗 仔 糖 果 三 合 "}

    res = requests.post(url, json=data)
    print(res.json())