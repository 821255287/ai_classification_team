"""
启动入口 — 选择启动 Streamlit Web 应用 或 Flask API 服务
"""
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='随机森林商品分类 — 启动器')
    parser.add_argument('--mode', '-m', type=str, default='web',
                        choices=['web', 'api'],
                        help='启动模式: web=Streamlit界面, api=Flask接口 (默认: web)')
    parser.add_argument('--port', '-p', type=int, default=None,
                        help='服务端口 (web默认8501, api默认5000)')
    args = parser.parse_args()

    if args.mode == 'web':
        port = args.port or 8501
        print(f'启动 Streamlit Web 应用 (端口: {port})...')
        os.system(f'streamlit run app.py --server.port {port}')
    elif args.mode == 'api':
        port = args.port or 5000
        os.environ['PORT'] = str(port)
        print(f'启动 Flask API 服务 (端口: {port})...')
        from api_server import app
        app.run(host='0.0.0.0', port=port, debug=True)


if __name__ == '__main__':
    main()
