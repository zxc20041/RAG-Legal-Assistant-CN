from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
import requests
import json
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = 'legal_assistant_secret_key233'

# 后端API服务的完整地址
API_URL = "http://127.0.0.1:5000/predict"

# 全局变量，用于存储检索系统组件
retrieval_system = None

class LegalCaseRetriever:
    """法律案件检索器"""
    model_loaded = False
    def __init__(self, database_path, index_path, model_path, caseid_mapping_path):
        """初始化检索系统"""

        print("正在加载案件检索系统...")
        # 加载案件数据库
        with open(database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        self.case_database = database['legal_case_info']
        print(f"✓ 案件数据库已加载: {len(self.case_database)} 个案件")

        # 加载FAISS索引
        self.index = faiss.read_index(index_path)
        print(f"✓ FAISS索引已加载: {self.index.ntotal} 个向量")

        # 加载case_id到索引的映射
        with open(caseid_mapping_path, 'r', encoding='utf-8') as f:
            self.caseid_to_index = json.load(f)
        print(f"✓ 案件ID映射已加载: {len(self.caseid_to_index)} 个映射")

        # 创建索引到case_id的反向映射
        self.index_to_caseid = {v: k for k, v in self.caseid_to_index.items()}

        # 加载嵌入模型
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        # 加载到GPU（如果可用）
        device = 'cuda' if hasattr(self.model, 'device') else 'cpu'
        #device = 'cpu'
        self.model.to(device)
        print(f"✓ 嵌入模型已加载到: {device}")

        print("案件检索系统初始化完成！")

    def search_similar_cases(self, query_text, k=5, min_score=0.5):
        """
        搜索相似案件

        参数:
            query_text: 查询文本
            k: 返回最相似的案件数量
            min_score: 最小相似度阈值

        返回:
            List[Dict]: 相似案件列表，包含案件信息和相似度
        """
        # 生成查询向量
        query_vec = self.model.encode([query_text], normalize_embeddings=True)

        # 搜索最相似的k*2个案件（后续过滤）
        search_k = min(k * 2, self.index.ntotal)
        scores, indices = self.index.search(query_vec, search_k)

        # 过滤并格式化结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < min_score:
                continue

            # 获取案件ID
            case_id = self.index_to_caseid.get(idx)
            if not case_id:
                continue

            # 获取案件信息
            case_info = self.case_database.get(str(idx + 1))  # 注意：数据库键是从1开始的字符串
            if not case_info:
                continue

            # 转换为与SIMULATED_RAG_DATA相同的格式
            formatted_case = {
                "fact": case_info['fact'],
                "meta": {
                    "relevant_articles": case_info['articles'],
                    "accusation": case_info['accusation'],
                    "punish_of_money": case_info['fine'],
                    "criminals": case_info['criminals'],
                    "term_of_imprisonment": case_info['term']
                }
            }

            results.append({
                'similarity_score': float(score),
                'formatted_case': formatted_case
            })

            # 如果已经收集到足够的案件，提前结束
            if len(results) >= k:
                break
        print(results)
        return results

def initialize_retrieval_system():
    """初始化检索系统"""
    global retrieval_system
    try:
        retrieval_system = LegalCaseRetriever(
            database_path='./AutoSurvey-main/database/legal_case_db.json',
            index_path='./AutoSurvey-main/database/case_facts.index',
            model_path='./AutoSurvey-main/model/bge-large-zh-v1.5',
            caseid_mapping_path='./AutoSurvey-main/database/caseid_to_index.json'
        )
        return True
    except Exception as e:
        print(f"检索系统初始化失败: {e}")
        return False

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
        use_real_retrieval = True
        # = request.json.get('use_real_retrieval', False)  # 新增：是否使用真实检索

        # 保存模型选择
        session['selected_model'] = selected_model

        if not user_message:
            return jsonify({'error': '消息不能为空'}), 400

        # 从session中获取对话历史
        conversation_history = session.get('conversation_history', [])

        # 确定使用的RAG数据
        if use_real_retrieval and retrieval_system is not None:
            # 使用检索数据
            retrieval_results = retrieval_system.search_similar_cases(user_message, k=2, min_score=0.4)
            current_rag_data = [result['formatted_case'] for result in retrieval_results]
        else:
            # 使用空数据
            current_rag_data = []
            print("警告: 检索数据为空")

        # 构造发送给API的请求数据
        payload = {
            "user_question": user_message,
            "rag_data": current_rag_data,
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

@app.route('/retrieval_status', methods=['GET'])
def retrieval_status():
    """获取检索系统状态"""
    if retrieval_system is None:
        return jsonify({'status': '未初始化', 'initialized': False})
    else:
        return jsonify({
            'status': '已初始化',
            'initialized': True,
            'case_count': len(retrieval_system.case_database)
        })

if __name__ == '__main__':
    # 启动时初始化检索系统
    print("正在启动法律小助手Web应用...")
    if not LegalCaseRetriever.model_loaded:
        if initialize_retrieval_system():
            print("检索系统初始化成功！")
            LegalCaseRetriever.model_loaded = True
        else:
            print("警告: 检索系统初始化失败")

    app.run(debug=False, port=5001)