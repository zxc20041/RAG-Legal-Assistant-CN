from flask import Flask, render_template, request, jsonify, session
import requests
import json
import time

app = Flask(__name__)
app.secret_key = 'legal_assistant_secret_key'  # 用于session管理

# 后端API服务的完整地址
API_URL = "http://127.0.0.1:5000/predict"

# 模拟RAG数据 - 与您的后端代码保持一致
SIMULATED_RAG_DATA = [
    {"fact": "公诉机关起诉指控，被告人张某某秘密窃取他人财物，价值2210元...", "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 0, "criminals": ["张某某"], "term_of_imprisonment": {"death_penalty": False, "imprisonment": 2, "life_imprisonment": False}}},
    {"fact": "孝昌县人民检察院指控：2014年1月4日，被告人邬某在孝昌县城区2路公交车上扒窃被害人晏某白色VIVO手机一部。经鉴定，该手机价值为750元...", "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 1000, "criminals": ["邬某"], "term_of_imprisonment": {"death_penalty": False, "imprisonment": 4, "life_imprisonment": False}}}
]

@app.route('/')
def index():
    """主页面"""
    # 初始化对话历史
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    """接收用户消息并转发到后端API"""
    try:
        # 获取用户输入
        user_message = request.json.get('message', '')

        if not user_message:
            return jsonify({'error': '消息不能为空'}), 400

        # 检查退出指令
        if user_message.lower() in ['exit', 'quit', '退出']:
            session['conversation_history'] = []  # 清空对话历史
            return jsonify({
                'response': '感谢使用法律小助手，再见！',
                'should_exit': True
            })

        # 从session中获取对话历史
        conversation_history = session.get('conversation_history', [])

        # 构造发送给API的请求数据
        payload = {
            "user_question": user_message,
            "rag_data": SIMULATED_RAG_DATA,
            "chat_history": conversation_history
        }

        headers = {"Content-Type": "application/json"}

        # 发送HTTP POST请求到后端服务
        print(f"正在发送请求到后端: {API_URL}")
        print(f"请求数据: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()  # 如果状态码不是200，抛出异常

        # 解析服务器返回的JSON数据
        result = response.json()
        print(f"后端响应: {json.dumps(result, ensure_ascii=False, indent=2)}")

        # 检查返回的数据中是否有我们需要的'prediction'字段
        if "prediction" in result:
            # 成功获取到助手的回答
            assistant_response = result["prediction"]

            # 更新对话历史
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": assistant_response})

            # 保存更新后的对话历史到session
            session['conversation_history'] = conversation_history

            return jsonify({
                'response': assistant_response,
                'should_exit': False
            })
        else:
            # 如果返回的数据中没有'prediction'，说明可能出错了
            error_msg = result.get('error', '未知错误')
            return jsonify({'error': f'后端返回错误: {error_msg}'}), 500

    except requests.exceptions.Timeout:
        return jsonify({'error': '请求超时，请稍后重试'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': '无法连接到后端服务，请确保服务正在运行'}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'网络请求错误: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清空对话历史"""
    session['conversation_history'] = []
    return jsonify({'success': True, 'message': '对话历史已清空'})

@app.route('/get_history', methods=['GET'])
def get_history():
    """获取当前对话历史"""
    conversation_history = session.get('conversation_history', [])
    return jsonify({'history': conversation_history})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # 前端运行在5001端口，避免与后端冲突