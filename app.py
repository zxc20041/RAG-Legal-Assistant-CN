from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
import requests
import json

app = Flask(__name__)
app.secret_key = 'legal_assistant_secret_key233'

# 后端API服务的完整地址
API_URL = "http://127.0.0.1:5000/predict"

# 模拟RAG数据 - 与test_client.py保持一致
SIMULATED_RAG_DATA = [
    {"fact": "公诉机关起诉指控，被告人张某某秘密窃取他人财物，价值2210元...",
     "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 0,
              "criminals": ["张某G"], "term_of_imprisonment": {"death_penalty": False,
                                                               "imprisonment": 2, "life_imprisonment": False}}},
    {"fact": "孝昌县人民检察院指控：2014年1月4日，被告人邬某在孝昌县城区2路公交车上扒窃被害人晏某白色VIVO手机一部。经鉴定，该手机价值为750元...",
     "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 1000,
              "criminals": ["邬某"], "term_of_imprisonment": {"death_penalty": False,
                                                              "imprisonment": 4, "life_imprisonment": False}}}
]

@app.route('/')
def index():
    """主页面"""
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    if 'selected_model' not in session:
        session['selected_model'] = 'deepseek'  # 默认模型
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    """接收用户消息并转发到后端API - 支持流式响应"""
    try:
        # 获取用户输入和参数
        user_message = request.json.get('message', '')
        selected_model = request.json.get('model', session.get('selected_model', 'deepseek'))
        is_professional_mode = request.json.get('is_professional_mode', False)

        # 保存模型选择
        session['selected_model'] = selected_model

        if not user_message:
            return jsonify({'error': '消息不能为空'}), 400

        # 检查退出指令
        if user_message.lower() in ['exit', 'quit', '退出']:
            session['conversation_history'] = []
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
            "chat_history": conversation_history,
            "model_id": selected_model,
            "is_professional_mode": is_professional_mode
        }

        headers = {"Content-Type": "application/json"}

        # 对于judge模型，使用非流式请求
        if selected_model == 'judge':
            return handle_judge_request(payload, headers, conversation_history, user_message)
        else:
            # 对于其他模型，使用流式请求
            return handle_streaming_request(payload, headers, conversation_history, user_message)

    except Exception as e:
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

def handle_judge_request(payload, headers, conversation_history, user_message):
    """处理Judge模型的非流式请求"""
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        if "prediction" in result:
            assistant_response = result["prediction"]
            model_used = result.get("model_used", "未知")

            # 更新对话历史
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": assistant_response})
            session['conversation_history'] = conversation_history

            return jsonify({
                'response': assistant_response,
                'model_used': model_used,
                'judge_reasoning': result.get("judge_reasoning", ""),
                'should_exit': False
            })
        else:
            return jsonify({'error': f'后端返回错误: {result.get("error", "未知错误")}'}), 500

    except requests.exceptions.Timeout:
        return jsonify({'error': '请求超时 (Judge模式需要更长时间)'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'网络请求错误: {str(e)}'}), 500

def handle_streaming_request(payload, headers, conversation_history, user_message):
    """处理流式请求"""
    def generate():
        full_response = ""
        model_used = ""
        stream_had_error = False

        try:
            with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60) as response:
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    if line.startswith("event: model_info"):
                        # 忽略事件行，等待数据行
                        pass
                    elif line.startswith("event: end_of_stream"):
                        # 流结束
                        if full_response and not stream_had_error:
                            # 更新对话历史
                            conversation_history.append({"role": "user", "content": user_message})
                            conversation_history.append({"role": "assistant", "content": full_response})
                            session['conversation_history'] = conversation_history

                        yield f"data: {json.dumps({'event': 'end_of_stream', 'full_response': full_response})}\n\n"
                        break
                    elif line.startswith("event: error"):
                        stream_had_error = True
                    elif line.startswith("data: "):
                        try:
                            data_str = line.split("data: ", 1)[1]
                            data_json = json.loads(data_str)

                            if "model_used" in data_json:
                                model_used = data_json.get("model_used", "未知")
                                yield f"data: {json.dumps({'event': 'model_info', 'model_used': model_used})}\n\n"

                            elif "chunk" in data_json:
                                chunk = data_json.get("chunk", "")
                                full_response += chunk
                                yield f"data: {json.dumps({'event': 'chunk', 'chunk': chunk})}\n\n"

                            elif "error" in data_json:
                                stream_had_error = True
                                error_msg = data_json.get("error", "未知流错误")
                                yield f"data: {json.dumps({'event': 'error', 'error': error_msg})}\n\n"

                        except (json.JSONDecodeError, IndexError):
                            pass

        except requests.exceptions.RequestException as e:
            yield f"data: {json.dumps({'event': 'error', 'error': f'网络请求错误: {str(e)}'})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/plain')

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

@app.route('/set_model', methods=['POST'])
def set_model():
    """设置选择的模型"""
    model = request.json.get('model', 'deepseek')
    session['selected_model'] = model
    return jsonify({'success': True, 'model': model})

if __name__ == '__main__':
    app.run(debug=True, port=5001)