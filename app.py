import streamlit as st
import requests


url = "http://127.0.0.1:8080/predict"

st.title("投满分分类项目")
st.write("这是一个新闻多分类项目")
test = st.text_input("请输入待预测的文本: ")
if st.button("预测分类"):
    try:
        with st.spinner("预测中..."):
            # 发送post请求
            response = requests.post(url, json={"text": test}, timeout=10)

            # 检查相应状态
            if response.status_code == 200:
                result = response.json()
                st.success(f'预测结果为: {result["pred_class"]}')
                st.json(result)
            else:
                st.error(f'API返回错误: {response.status_code}')
                st.code(response.text)
    except requests.exceptions.ConnectionError:
        st.error("无法连接到API服务，请确保 api_serve.py 已启动！")
    except requests.exceptions.Timeout:
        st.error("请求超时，请稍后重试！")
    except Exception as e:
        st.error(f'预测失败: {str(e)}')
