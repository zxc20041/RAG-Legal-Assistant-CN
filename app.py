from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from flask_session import Session
import requests
import json
import faiss
from sentence_transformers import SentenceTransformer
import os
import uuid
from datetime import datetime


app = Flask(__name__)
app.secret_key = 'legal_assistant_secret_key233'

# 配置服务器端会话存储
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'legal_assistant:'

# 初始化Flask-Session
Session(app)

# 后端API服务的完整地址
API_URL = "http://127.0.0.1:5000/predict"

# 全局变量，用于存储检索系统组件
retrieval_system = None

# 对话存储 - 使用字典模拟数据库
# 结构: {session_id: {conversations: {conversation_id: {title, history, created_at, updated_at}}, current_conversation: conversation_id}}
user_conversations = {}

def get_user_conversations(session_id):
    """获取用户的对话数据"""
    if session_id not in user_conversations:
        user_conversations[session_id] = {
            'conversations': {},
            'current_conversation': None
        }
    return user_conversations[session_id]

def create_new_conversation(session_id, title="新对话"):
    """创建新对话"""
    user_data = get_user_conversations(session_id)
    conversation_id = str(uuid.uuid4())

    user_data['conversations'][conversation_id] = {
        'title': title,
        'history': [],
        'rag_history': [],  # 新增：保存历史检索案例
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }

    # 设置为当前对话
    user_data['current_conversation'] = conversation_id

    print(f"创建新对话 - 会话ID: {session_id}, 对话ID: {conversation_id}, 标题: {title}")
    return conversation_id

def get_current_conversation(session_id):
    """获取当前对话"""
    user_data = get_user_conversations(session_id)
    current_id = user_data.get('current_conversation')

    if not current_id or current_id not in user_data['conversations']:
        # 如果没有当前对话，创建一个
        current_id = create_new_conversation(session_id)

    return user_data['conversations'][current_id]

def switch_conversation(session_id, conversation_id):
    """切换到指定对话"""
    user_data = get_user_conversations(session_id)

    if conversation_id in user_data['conversations']:
        user_data['current_conversation'] = conversation_id
        print(f"切换对话 - 会话ID: {session_id}, 对话ID: {conversation_id}")
        return True
    return False

def update_conversation_title(session_id, conversation_id, user_message):
    """更新对话标题（基于用户的第一条消息）"""
    user_data = get_user_conversations(session_id)

    if conversation_id in user_data['conversations']:
        conversation = user_data['conversations'][conversation_id]

        # 如果对话标题还是默认的"新对话"，并且这是第一条用户消息，则更新标题
        if conversation['title'] == "新对话" and user_message:
            # 取用户消息的前20个字符作为标题
            title = user_message[:20] + "..." if len(user_message) > 20 else user_message
            conversation['title'] = title
            conversation['updated_at'] = datetime.now().isoformat()
            print(f"更新对话标题 - 对话ID: {conversation_id}, 新标题: {title}")

def get_conversation_list(session_id):
    """获取用户的对话列表"""
    user_data = get_user_conversations(session_id)
    conversations = []

    for conv_id, conv_data in user_data['conversations'].items():
        conversations.append({
            'id': conv_id,
            'title': conv_data['title'],
            'created_at': conv_data['created_at'],
            'updated_at': conv_data['updated_at'],
            'is_current': conv_id == user_data.get('current_conversation')
        })

    # 按更新时间倒序排列
    conversations.sort(key=lambda x: x['updated_at'], reverse=True)
    return conversations

def save_conversation_history(session_id, conversation_id, history):
    """保存对话历史"""
    user_data = get_user_conversations(session_id)

    if conversation_id in user_data['conversations']:
        user_data['conversations'][conversation_id]['history'] = history
        user_data['conversations'][conversation_id]['updated_at'] = datetime.now().isoformat()
        print(f"保存对话历史 - 对话ID: {conversation_id}, 消息数: {len(history)}")

def delete_conversation(session_id, conversation_id):
    """删除对话"""
    user_data = get_user_conversations(session_id)

    if conversation_id in user_data['conversations']:
        # 如果删除的是当前对话，需要设置新的当前对话
        if user_data.get('current_conversation') == conversation_id:
            # 找另一个对话作为当前对话
            other_conversations = [cid for cid in user_data['conversations'].keys() if cid != conversation_id]
            if other_conversations:
                user_data['current_conversation'] = other_conversations[0]
            else:
                user_data['current_conversation'] = None

        del user_data['conversations'][conversation_id]
        print(f"删除对话 - 对话ID: {conversation_id}")
        return True
    return False

def add_rag_to_history(session_id, conversation_id, rag_data, user_message):
    """将检索案例添加到历史中"""
    user_data = get_user_conversations(session_id)

    if conversation_id in user_data['conversations']:
        conversation = user_data['conversations'][conversation_id]

        # 只保存有实际内容的检索结果
        valid_rag_data = []
        for item in rag_data:
            if item.get('fact') and item.get('meta'):
                # 简化案例信息，只保存关键内容
                simplified_case = {
                    'fact': item['fact'][:200] + '...' if len(item['fact']) > 200 else item['fact'],
                    'accusation': item['meta'].get('accusation', []),
                    'articles': item['meta'].get('relevant_articles', []),
                    'imprisonment': item['meta'].get('term_of_imprisonment', {}).get('imprisonment', '未知'),
                    'fine': item['meta'].get('punish_of_money', '未知'),
                    'timestamp': datetime.now().isoformat(),
                    'related_query': user_message[:100]  # 保存相关的查询内容
                }
                valid_rag_data.append(simplified_case)

        # 将新检索的案例添加到历史中
        conversation['rag_history'].extend(valid_rag_data)

        # 限制历史案例数量，避免过大
        if len(conversation['rag_history']) > 20:
            conversation['rag_history'] = conversation['rag_history'][-20:]

        print(f"保存检索案例到历史 - 对话ID: {conversation_id}, 新增案例数: {len(valid_rag_data)}, 总案例数: {len(conversation['rag_history'])}")

def get_relevant_rag_history(session_id, conversation_id, current_query):
    """获取与当前查询相关的历史检索案例"""
    user_data = get_user_conversations(session_id)

    if conversation_id not in user_data['conversations']:
        return []

    conversation = user_data['conversations'][conversation_id]
    rag_history = conversation.get('rag_history', [])

    # 这里可以添加更复杂的相关性判断逻辑
    # 目前简单返回最近的相关案例（排除当前查询直接相关的案例）
    relevant_cases = []

    for case in rag_history[-10:]:  # 只考虑最近10个案例
        # 简单的关键词匹配来判断相关性
        query_keywords = set(current_query.lower().split())
        case_text = f"{case.get('fact', '')} {' '.join(case.get('accusation', []))}".lower()

        # 如果案例与当前查询没有直接关联，且包含有用的量刑信息，则认为是相关历史案例
        if (any(keyword in case_text for keyword in ['月', '年', '刑期', '判决', '处罚']) and
                not any(keyword in case_text for keyword in query_keywords if len(keyword) > 2)):
            relevant_cases.append(case)

    return relevant_cases[:3]  # 最多返回3个相关历史案例

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
    # 生成或获取会话ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        print(f"创建新会话: {session['session_id']}")

    # 初始化会话数据
    if 'selected_model' not in session:
        session['selected_model'] = 'deepseek'
    if 'rag_enabled' not in session:
        session['rag_enabled'] = True
    if 'disclaimer_accepted' not in session:
        session['disclaimer_accepted'] = False

    # 确保用户有至少一个对话
    user_data = get_user_conversations(session['session_id'])
    if not user_data['conversations']:
        create_new_conversation(session['session_id'])

    return render_template('index.html')

@app.route('/accept_disclaimer', methods=['POST'])
def accept_disclaimer():
    """用户接受免责声明"""
    session['disclaimer_accepted'] = True
    return jsonify({'success': True, 'message': '免责声明已接受'})

# 新增的对话管理路由
@app.route('/conversations', methods=['GET'])
def get_conversations():
    """获取用户的对话列表"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': '会话不存在'}), 400

    conversations = get_conversation_list(session_id)
    return jsonify({'conversations': conversations})

@app.route('/conversations/new', methods=['POST'])
def create_conversation():
    """创建新对话"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': '会话不存在'}), 400

    data = request.get_json()
    title = data.get('title', '新对话')

    conversation_id = create_new_conversation(session_id, title)
    conversations = get_conversation_list(session_id)

    return jsonify({
        'success': True,
        'conversation_id': conversation_id,
        'conversations': conversations
    })

@app.route('/conversations/<conversation_id>/switch', methods=['POST'])
def switch_to_conversation(conversation_id):
    """切换到指定对话"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': '会话不存在'}), 400

    if switch_conversation(session_id, conversation_id):
        current_conv = get_current_conversation(session_id)
        conversations = get_conversation_list(session_id)

        return jsonify({
            'success': True,
            'current_conversation': {
                'id': conversation_id,
                'title': current_conv['title'],
                'history': current_conv['history']
            },
            'conversations': conversations
        })
    else:
        return jsonify({'error': '对话不存在'}), 404

@app.route('/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation_route(conversation_id):
    """删除对话"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': '会话不存在'}), 400

    if delete_conversation(session_id, conversation_id):
        conversations = get_conversation_list(session_id)
        return jsonify({
            'success': True,
            'conversations': conversations
        })
    else:
        return jsonify({'error': '对话不存在'}), 404

@app.route('/send_message', methods=['POST'])
def send_message():
    """接收用户消息并转发到后端API - 支持流式响应"""
    try:
        # 检查用户是否已接受免责声明
        if not session.get('disclaimer_accepted', False):
            return jsonify({'error': '请先阅读并接受免责声明'}), 403

        # 获取用户输入和参数
        user_message = request.json.get('message', '')
        selected_model = request.json.get('model', session.get('selected_model', 'deepseek'))
        is_professional_mode = request.json.get('is_professional_mode', False)
        rag_enabled = request.json.get('rag_enabled', session.get('rag_enabled', True))

        # 保存模型选择和RAG设置
        session['selected_model'] = selected_model
        session['rag_enabled'] = rag_enabled

        if not user_message:
            return jsonify({'error': '消息不能为空'}), 400

        # 获取当前对话
        session_id = session.get('session_id')
        current_conversation = get_current_conversation(session_id)
        conversation_history = current_conversation['history']
        conversation_id = user_conversations[session_id]['current_conversation']

        # 如果是第一条消息，更新对话标题
        if len(conversation_history) == 0:
            update_conversation_title(session_id, conversation_id, user_message)

        # 确定使用的RAG数据
        current_rag_data = []
        if rag_enabled and retrieval_system is not None:
            print(f"正在进行RAG检索，查询: {user_message}")
            retrieval_results = retrieval_system.search_similar_cases(user_message, k=2, min_score=0.4)
            current_rag_data = [result['formatted_case'] for result in retrieval_results]
            print(f"RAG检索完成，找到 {len(current_rag_data)} 个相关案例")

            # 保存检索案例到历史
            session_id = session.get('session_id')
            current_conversation = get_current_conversation(session_id)
            conversation_id = user_conversations[session_id]['current_conversation']
            add_rag_to_history(session_id, conversation_id, current_rag_data, user_message)
        else:
            print("RAG功能已关闭，不使用案例检索")

        # 获取相关历史检索案例
        historical_rag_data = []
        if rag_enabled:
            session_id = session.get('session_id')
            current_conversation = get_current_conversation(session_id)
            conversation_id = user_conversations[session_id]['current_conversation']
            historical_rag_data = get_relevant_rag_history(session_id, conversation_id, user_message)
            print(f"找到 {len(historical_rag_data)} 个相关历史检索案例")

        # 构造发送给API的请求数据，添加历史检索案例
        payload = {
            "user_question": user_message,
            "rag_data": current_rag_data,
            "historical_rag_data": historical_rag_data,  # 新增：历史检索案例
            "chat_history": conversation_history,
            "model_id": selected_model,
            "is_professional_mode": is_professional_mode
        }

        headers = {"Content-Type": "application/json"}

        # 对于judge模型，使用非流式请求
        if selected_model == 'judge':
            return handle_judge_request(payload, headers, conversation_history, user_message, session_id, conversation_id)
        else:
            # 对于其他模型，使用流式请求
            return handle_streaming_request(payload, headers, conversation_history, user_message, session_id, conversation_id)

    except Exception as e:
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

def handle_judge_request(payload, headers, conversation_history, user_message, session_id, conversation_id):
    """处理Judge模型的非流式请求"""
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        if "prediction" in result:
            assistant_response = result["prediction"]
            model_used = result.get("model_used", "未知")
            judge_reasoning = result.get("judge_reasoning", "")
            all_answers = result.get("all_answers", {})

            # 更新对话历史
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": assistant_response})
            save_conversation_history(session_id, conversation_id, conversation_history)

            return jsonify({
                'response': assistant_response,
                'model_used': model_used,
                'judge_reasoning': judge_reasoning,
                'all_answers': all_answers,
                'should_exit': False
            })
        else:
            return jsonify({'error': f'后端返回错误: {result.get("error", "未知错误")}'}), 500

    except requests.exceptions.Timeout:
        return jsonify({'error': '请求超时 (Judge模式需要更长时间)'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'网络请求错误: {str(e)}'}), 500

def handle_streaming_request(payload, headers, conversation_history, user_message, session_id, conversation_id):
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
                        pass
                    elif line.startswith("event: end_of_stream"):
                        if full_response and not stream_had_error:
                            # 更新对话历史
                            conversation_history.append({"role": "user", "content": user_message})
                            conversation_history.append({"role": "assistant", "content": full_response})
                            save_conversation_history(session_id, conversation_id, conversation_history)

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

# 其他路由保持不变...
@app.route('/toggle_rag', methods=['POST'])
def toggle_rag():
    """切换RAG开关状态"""
    data = request.get_json()
    rag_enabled = data.get('rag_enabled', True)
    session['rag_enabled'] = rag_enabled
    return jsonify({'success': True, 'rag_enabled': rag_enabled})

@app.route('/cancel_request', methods=['POST'])
def cancel_request():
    """取消当前请求"""
    return jsonify({'success': True, 'message': '取消请求已发送'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清空当前对话历史"""
    session_id = session.get('session_id')
    current_conversation = get_current_conversation(session_id)
    conversation_id = user_conversations[session_id]['current_conversation']

    current_conversation['history'] = []
    save_conversation_history(session_id, conversation_id, [])

    return jsonify({'success': True, 'message': '对话历史已清空'})

@app.route('/get_history', methods=['GET'])
def get_history():
    """获取当前对话历史"""
    session_id = session.get('session_id')
    current_conversation = get_current_conversation(session_id)
    return jsonify({'history': current_conversation['history']})

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
    # 确保会话目录存在
    if not os.path.exists('./flask_session'):
        os.makedirs('./flask_session')

    # 启动时初始化检索系统
    print("正在启动法律小助手Web应用...")
    if not LegalCaseRetriever.model_loaded:
        if initialize_retrieval_system():
            print("检索系统初始化成功！")
            LegalCaseRetriever.model_loaded = True
        else:
            print("警告: 检索系统初始化失败")

    app.run(debug=False, port=5001)

