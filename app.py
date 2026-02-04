from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
import requests
import json
import re
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import torch
import os
import uuid
import tempfile
from datetime import datetime
from multimodal_handler import process_multimodal_file
import config


app = Flask(__name__)
app.secret_key = config.APP_SECRET_KEY

# 文件上传配置
ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS
MAX_FILE_SIZE = config.MAX_FILE_SIZE
MAX_FILES_COUNT = config.MAX_FILES_COUNT

# 配置服务器端会话存储
app.config['SESSION_TYPE'] = config.SESSION_TYPE
app.config['SESSION_FILE_DIR'] = config.SESSION_FILE_DIR
app.config['SESSION_PERMANENT'] = config.SESSION_PERMANENT
app.config['SESSION_USE_SIGNER'] = config.SESSION_USE_SIGNER
app.config['SESSION_KEY_PREFIX'] = config.SESSION_KEY_PREFIX

# 配置SQLite数据库持久化存储
app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS

# 初始化Flask-Session和SQLAlchemy
Session(app)
db = SQLAlchemy(app)


# ============================================================================
# 数据库模型定义
# ============================================================================
class Conversation(db.Model):
    """对话模型 - 持久化存储对话历史"""
    __tablename__ = 'conversations'

    id = db.Column(db.String(36), primary_key=True)  # UUID
    session_id = db.Column(db.String(36), index=True, nullable=False)
    title = db.Column(db.String(200), default='新对话')
    history = db.Column(db.Text, default='[]')  # JSON字符串存储对话历史
    rag_history = db.Column(db.Text, default='[]')  # JSON字符串存储RAG历史
    is_current = db.Column(db.Boolean, default=False)  # 是否为当前对话
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def get_history(self):
        """获取对话历史列表"""
        try:
            return json.loads(self.history) if self.history else []
        except json.JSONDecodeError:
            return []

    def set_history(self, history_list):
        """设置对话历史"""
        self.history = json.dumps(history_list, ensure_ascii=False)

    def get_rag_history(self):
        """获取RAG历史列表"""
        try:
            return json.loads(self.rag_history) if self.rag_history else []
        except json.JSONDecodeError:
            return []

    def set_rag_history(self, rag_list):
        """设置RAG历史"""
        self.rag_history = json.dumps(rag_list, ensure_ascii=False)

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'title': self.title,
            'history': self.get_history(),
            'rag_history': self.get_rag_history(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_current': self.is_current
        }

# 后端API服务的完整地址
API_URL = config.API_URL

# 全局变量，用于存储检索系统组件
retrieval_system = None

# RAG 调试开关（环境变量 RAG_DEBUG=1/true/on）
RAG_DEBUG = config.RAG_DEBUG

# ============================================================================
# 数据库操作函数 - 替代原有的内存字典存储
# ============================================================================

def get_or_create_conversation(session_id, conversation_id=None):
    """获取或创建对话"""
    if conversation_id:
        conv = Conversation.query.filter_by(id=conversation_id, session_id=session_id).first()
        if conv:
            return conv

    # 获取当前对话
    conv = Conversation.query.filter_by(session_id=session_id, is_current=True).first()
    if conv:
        return conv

    # 没有当前对话，创建新对话
    return create_new_conversation(session_id)


def create_new_conversation(session_id, title="新对话"):
    """创建新对话"""
    # 将该用户的所有对话设为非当前
    Conversation.query.filter_by(session_id=session_id).update({'is_current': False})

    # 创建新对话
    conversation_id = str(uuid.uuid4())
    new_conv = Conversation(
        id=conversation_id,
        session_id=session_id,
        title=title,
        is_current=True
    )
    db.session.add(new_conv)
    db.session.commit()

    print(f"创建新对话 - 会话ID: {session_id}, 对话ID: {conversation_id}, 标题: {title}")
    return new_conv


def get_current_conversation(session_id):
    """获取当前对话"""
    conv = Conversation.query.filter_by(session_id=session_id, is_current=True).first()
    if not conv:
        conv = create_new_conversation(session_id)
    return conv


def switch_conversation(session_id, conversation_id):
    """切换到指定对话（不更新updated_at，仅查看时位置不变）"""
    conv = Conversation.query.filter_by(id=conversation_id, session_id=session_id).first()
    if not conv:
        return False

    # 保存目标对话的原始updated_at时间
    original_updated_at = conv.updated_at

    # 将所有对话设为非当前
    # 使用update()方法直接更新，避免触发ORM的onupdate
    Conversation.query.filter_by(session_id=session_id).update(
        {'is_current': False},
        synchronize_session='fetch'
    )

    # 设置目标对话为当前，并恢复原始的updated_at
    conv.is_current = True
    db.session.flush()  # 先提交is_current的变更

    # 恢复原始的updated_at时间（覆盖自动更新的值）
    conv.updated_at = original_updated_at
    db.session.commit()

    print(f"切换对话 - 会话ID: {session_id}, 对话ID: {conversation_id}")
    return True


def update_conversation_title(session_id, conversation_id, user_message):
    """更新对话标题（基于用户的第一条消息）"""
    conv = Conversation.query.filter_by(id=conversation_id, session_id=session_id).first()
    if conv and conv.title == "新对话" and user_message:
        title = user_message[:20] + "..." if len(user_message) > 20 else user_message
        conv.title = title
        db.session.commit()
        print(f"更新对话标题 - 对话ID: {conversation_id}, 新标题: {title}")


def get_conversation_list(session_id):
    """获取用户的对话列表"""
    conversations = Conversation.query.filter_by(session_id=session_id)\
        .order_by(Conversation.updated_at.desc()).all()

    return [{
        'id': conv.id,
        'title': conv.title,
        'created_at': conv.created_at.isoformat() if conv.created_at else None,
        'updated_at': conv.updated_at.isoformat() if conv.updated_at else None,
        'is_current': conv.is_current
    } for conv in conversations]


def save_conversation_history(session_id, conversation_id, history):
    """保存对话历史"""
    conv = Conversation.query.filter_by(id=conversation_id, session_id=session_id).first()
    if conv:
        conv.set_history(history)
        conv.updated_at = datetime.now()
        db.session.commit()
        print(f"保存对话历史 - 对话ID: {conversation_id}, 消息数: {len(history)}")


def delete_conversation(session_id, conversation_id):
    """删除对话"""
    conv = Conversation.query.filter_by(id=conversation_id, session_id=session_id).first()
    if not conv:
        return False

    was_current = conv.is_current
    db.session.delete(conv)
    db.session.commit()

    # 如果删除的是当前对话，设置另一个为当前
    if was_current:
        other_conv = Conversation.query.filter_by(session_id=session_id)\
            .order_by(Conversation.updated_at.desc()).first()
        if other_conv:
            other_conv.is_current = True
            db.session.commit()

    print(f"删除对话 - 对话ID: {conversation_id}")
    return True


def add_rag_to_history(session_id, conversation_id, rag_data, user_message):
    """将检索案例添加到历史中"""
    conv = Conversation.query.filter_by(id=conversation_id, session_id=session_id).first()
    if not conv:
        return

    rag_history = conv.get_rag_history()

    # 只保存有实际内容的检索结果
    for item in rag_data:
        if item.get('fact') and item.get('meta'):
            simplified_case = {
                'fact': item['fact'][:200] + '...' if len(item['fact']) > 200 else item['fact'],
                'accusation': item['meta'].get('accusation', []),
                'articles': item['meta'].get('relevant_articles', []),
                'imprisonment': item['meta'].get('term_of_imprisonment', {}).get('imprisonment', '未知'),
                'fine': item['meta'].get('punish_of_money', '未知'),
                'timestamp': datetime.now().isoformat(),
                'related_query': user_message[:100]
            }
            rag_history.append(simplified_case)

    # 限制历史案例数量
    if len(rag_history) > 20:
        rag_history = rag_history[-20:]

    conv.set_rag_history(rag_history)
    db.session.commit()
    print(f"保存检索案例到历史 - 对话ID: {conversation_id}, 总案例数: {len(rag_history)}")


def get_relevant_rag_history(session_id, conversation_id, current_query):
    """获取与当前查询相关的历史检索案例"""
    conv = Conversation.query.filter_by(id=conversation_id, session_id=session_id).first()
    if not conv:
        return []

    rag_history = conv.get_rag_history()

    # 简化逻辑：返回最近的历史检索案例作为参考
    # 排除与当前查询高度相似的案例（避免重复）
    relevant_cases = []

    for case in rag_history[-10:]:  # 只考虑最近10个案例
        case_text = f"{case.get('fact', '')} {' '.join(case.get('accusation', []))}".lower()

        # 检查案例是否包含有用的量刑信息
        has_sentencing_info = any(keyword in case_text for keyword in ['月', '年', '刑期', '判决', '处罚', '罚金'])

        # 简单的重复检测：如果案例的related_query与当前查询完全相同则跳过
        related_query = case.get('related_query', '')
        is_duplicate = related_query == current_query[:100]

        if has_sentencing_info and not is_duplicate:
            relevant_cases.append(case)

    return relevant_cases[:3]  # 最多返回3个相关历史案例


def truncate_chat_history(history, max_turns=10):
    """
    截断对话历史，防止token超限

    参数:
        history: 完整的对话历史列表
        max_turns: 保留的最大对话轮数（一轮 = 一个user + 一个assistant）

    返回:
        截断后的对话历史
    """
    if not history:
        return []

    # 每轮对话包含2条消息(user + assistant)
    max_messages = max_turns * 2

    if len(history) <= max_messages:
        return history

    # 保留最近的对话，丢弃过早的历史
    truncated = history[-max_messages:]

    print(f"对话历史已截断: {len(history)} -> {len(truncated)} 条消息")
    return truncated


def parse_rag_query(response_text):
    """
    解析LLM回复中的RAG查询指令

    参数:
        response_text: LLM的完整回复文本

    返回:
        tuple: (清理后的回复文本, RAG查询关键词或None)
    """
    # 匹配 [RAG_QUERY: xxx] 格式
    pattern = r'\[RAG_QUERY:\s*(.+?)\]'
    match = re.search(pattern, response_text)

    if match:
        query_keywords = match.group(1).strip()
        # 从回复中移除RAG_QUERY标记
        clean_response = re.sub(pattern, '', response_text).strip()
        print(f"检测到RAG查询请求: {query_keywords}")
        return clean_response, query_keywords

    return response_text, None

class LegalCaseRetriever:
    """法律案件检索器（Milvus 版）"""
    model_loaded = False
    def __init__(self, db_uri, model_path, collection_name="legal_cases"):
        print("正在加载案件检索系统...")
        self.collection_name = collection_name

        # 连接本地 Milvus SQLite（由 build_database.ipynb 生成的 legal_assistant.db）
        self.client = MilvusClient(db_uri)
        if not self.client.has_collection(self.collection_name):
            raise ValueError(f"Milvus 集合不存在: {self.collection_name}")

        # 获取数据量，用于状态上报
        stats = self.client.get_collection_stats(self.collection_name)
        self.case_count = int(stats.get("row_count", 0))
        print(f"✓ Milvus 已连接，集合包含 {self.case_count} 条记录")

        # 加载嵌入模型
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        print(f"✓ 嵌入模型已加载到: {device}")
        print("案件检索系统初始化完成！")

    def search_similar_cases(self, query_text, k=5, min_score=0.5):
        """使用 Milvus 检索相似案件"""
        query_vec = self.model.encode([query_text], normalize_embeddings=True)

        # Milvus 按距离/相似度排序，需与建库时的 metric_type 一致
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=query_vec,
            limit=k * 2,
            output_fields=["id", "fact", "summary", "accusation"],
            search_params={"metric_type": "COSINE"}
        )

        results = []
        for hit in search_res[0]:
            similarity = float(hit.get("distance", 0))
            if similarity < min_score:
                continue

            entity = hit.get("entity", {})
            formatted_case = {
                "fact": entity.get("fact", ""),
                "meta": {
                    "relevant_articles": entity.get("articles", []),
                    "accusation": entity.get("accusation", []),
                    "punish_of_money": entity.get("fine", "未知"),
                    "criminals": entity.get("criminals", []),
                    "term_of_imprisonment": entity.get("term", {})
                }
            }
            results.append({
                'similarity_score': similarity,
                'formatted_case': formatted_case
            })

            # 调试输出：打印命中的案件关键信息
            if RAG_DEBUG:
                fact_preview = formatted_case["fact"][:120] + ("..." if len(formatted_case["fact"]) > 120 else "")
                print(
                    f"[RAG] hit id={entity.get('id', 'N/A')} sim={similarity:.4f} "
                    f"accusation={formatted_case['meta'].get('accusation', [])} "
                    f"articles={formatted_case['meta'].get('relevant_articles', [])} "
                    f"fact='{fact_preview}'"
                )

            if len(results) >= k:
                break

        return results


def initialize_retrieval_system():
    """初始化检索系统（Milvus）"""
    global retrieval_system
    try:
        retrieval_system = LegalCaseRetriever(
            db_uri='./AutoSurvey-main/database/legal_assistant.db',
            model_path='./AutoSurvey-main/model/bge-large-zh-v1.5',
            collection_name='legal_cases'
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

    # 确保用户有至少一个对话（从数据库检查）
    existing_conv = Conversation.query.filter_by(session_id=session['session_id']).first()
    if not existing_conv:
        create_new_conversation(session['session_id'])

    return render_template('index.html')

@app.route('/accept_disclaimer', methods=['POST'])
def accept_disclaimer():
    """用户接受免责声明"""
    session['disclaimer_accepted'] = True
    return jsonify({'success': True, 'message': '免责声明已接受'})


# ============================================================================
# 文件上传路由
# ============================================================================

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_file', methods=['POST'])
def upload_file():
    """
    处理文件上传并识别内容
    返回: {success, file_id, filename, file_type, text, error}
    """
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有选择文件'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400

        # 检查文件格式
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'不支持的文件格式，仅支持: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # 检查文件大小
        file.seek(0, 2)  # 移到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置到文件开头

        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'文件过大，最大允许 {MAX_FILE_SIZE // (1024*1024)}MB'
            }), 400

        # 获取文件扩展名
        ext = file.filename.rsplit('.', 1)[1].lower()
        file_type = 'image' if ext in ['jpg', 'jpeg', 'png'] else 'audio'

        # 创建临时文件进行识别
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # 调用多模态识别
            print(f"开始识别文件: {file.filename}, 类型: {file_type}")
            recognized_text = process_multimodal_file(tmp_path)
            print(f"识别完成，文本长度: {len(recognized_text)}")

            # 检查是否识别失败
            if recognized_text.startswith('错误') or recognized_text.startswith('系统级异常'):
                return jsonify({
                    'success': False,
                    'error': recognized_text
                }), 500

            # 生成文件ID
            file_id = str(uuid.uuid4())[:8]

            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': file.filename,
                'file_type': file_type,
                'text': recognized_text
            })

        finally:
            # 删除临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        print(f"文件上传处理错误: {str(e)}")
        return jsonify({'success': False, 'error': f'处理失败: {str(e)}'}), 500


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

    new_conv = create_new_conversation(session_id, title)
    conversations = get_conversation_list(session_id)

    return jsonify({
        'success': True,
        'conversation_id': new_conv.id,
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
                'title': current_conv.title,
                'history': current_conv.get_history()
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
    """接收用户消息并转发到后端API - 支持流式响应和附件"""
    try:
        # 检查用户是否已接受免责声明
        if not session.get('disclaimer_accepted', False):
            return jsonify({'error': '请先阅读并接受免责声明'}), 403

        # 获取用户输入和参数
        user_message = request.json.get('message', '')
        selected_model = request.json.get('model', session.get('selected_model', 'deepseek'))
        is_professional_mode = request.json.get('is_professional_mode', False)
        rag_enabled = request.json.get('rag_enabled', session.get('rag_enabled', True))

        # 获取附件信息（多模态识别结果）
        attachments = request.json.get('attachments', [])

        # 保存模型选择和RAG设置
        session['selected_model'] = selected_model
        session['rag_enabled'] = rag_enabled

        if not user_message and not attachments:
            return jsonify({'error': '消息不能为空'}), 400

        # 如果有附件，将附件识别文本合并到用户消息中
        final_message = user_message
        if attachments:
            attachment_texts = []
            for att in attachments:
                file_type_name = "图片" if att.get('file_type') == 'image' else "音频"
                attachment_texts.append(f"【{file_type_name}附件: {att.get('filename', '未知文件')}】\n{att.get('text', '')}")

            attachments_content = "\n\n".join(attachment_texts)
            if user_message:
                final_message = f"【附件内容】\n{attachments_content}\n\n【用户提问】\n{user_message}"
            else:
                final_message = f"【附件内容】\n{attachments_content}\n\n【用户提问】\n请分析以上附件内容。"

            print(f"消息包含 {len(attachments)} 个附件，合并后消息长度: {len(final_message)}")

        # 获取当前对话（从数据库）
        session_id = session.get('session_id')
        current_conv = get_current_conversation(session_id)
        conversation_history = current_conv.get_history()
        conversation_id = current_conv.id

        # 截断对话历史，防止token超限（保留最近10轮对话）
        # 注意：完整历史仍保存在数据库中，仅传递给LLM时截断
        truncated_history = truncate_chat_history(conversation_history, max_turns=10)

        # 如果是第一条消息，更新对话标题（使用原始用户消息，不含附件前缀）
        if len(conversation_history) == 0:
            update_conversation_title(session_id, conversation_id, user_message if user_message else "附件分析")

        # 确定使用的RAG数据（使用原始用户消息进行检索，更精准）
        current_rag_data = []
        rag_query = user_message if user_message else "法律文件分析"
        if rag_enabled and retrieval_system is not None:
            print(f"正在进行RAG检索，查询: {rag_query}")
            retrieval_results = retrieval_system.search_similar_cases(rag_query, k=2, min_score=0.4)
            current_rag_data = [result['formatted_case'] for result in retrieval_results]
            print(f"RAG检索完成，找到 {len(current_rag_data)} 个相关案例")

            # 保存检索案例到历史
            add_rag_to_history(session_id, conversation_id, current_rag_data, rag_query)
        else:
            print("RAG功能已关闭，不使用案例检索")

        # 获取相关历史检索案例
        historical_rag_data = []
        if rag_enabled:
            historical_rag_data = get_relevant_rag_history(session_id, conversation_id, rag_query)
            print(f"找到 {len(historical_rag_data)} 个相关历史检索案例")

        # 构造发送给API的请求数据，使用合并后的消息
        payload = {
            "user_question": final_message,  # 使用包含附件的完整消息
            "rag_data": current_rag_data,
            "historical_rag_data": historical_rag_data,
            "chat_history": truncated_history,
            "model_id": selected_model,
            "is_professional_mode": is_professional_mode
        }

        headers = {"Content-Type": "application/json"}

        # 对于judge模型，使用非流式请求
        # 注意：保存到历史时使用final_message，同时传递附件信息
        if selected_model == 'judge':
            return handle_judge_request(payload, headers, conversation_history, final_message, session_id, conversation_id, attachments)
        else:
            # 对于其他模型，使用流式请求
            return handle_streaming_request(payload, headers, conversation_history, final_message, session_id, conversation_id, attachments)

    except Exception as e:
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

def handle_judge_request(payload, headers, conversation_history, user_message, session_id, conversation_id, attachments=None):
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
            # 用户消息：如果有附件，保存附件信息供前端查看
            user_msg = {"role": "user", "content": user_message}
            if attachments:
                user_msg["attachments"] = attachments  # 保存附件信息
            conversation_history.append(user_msg)

            # Judge模式的回答：存储最佳回答作为content，同时保存完整数据用于前端展示
            judge_message = {
                "role": "assistant",
                "content": assistant_response,  # 最佳回答，用于下次对话的上下文
                "is_judge_mode": True,  # 标记这是Judge模式的回答
                "judge_data": {  # Judge模式的完整数据，用于前端展示
                    "model_used": model_used,
                    "judge_reasoning": judge_reasoning,
                    "all_answers": all_answers,
                    "best_answer": assistant_response
                }
            }
            conversation_history.append(judge_message)
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

def handle_streaming_request(payload, headers, conversation_history, user_message, session_id, conversation_id, attachments=None):
    """处理流式请求，支持LLM主动触发RAG查询"""

    # 从payload中提取需要的参数，用于可能的二次请求
    rag_enabled = payload.get('rag_data') is not None or True

    def generate():
        full_response = ""
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
                            # 检查是否有RAG查询请求
                            clean_response, rag_query = parse_rag_query(full_response)

                            if rag_query and retrieval_system is not None:
                                # LLM请求了额外的RAG查询
                                yield f"data: {json.dumps({'event': 'rag_query_detected', 'query': rag_query})}\n\n"

                                # 执行RAG查询
                                print(f"执行LLM请求的RAG查询: {rag_query}")
                                retrieval_results = retrieval_system.search_similar_cases(rag_query, k=3, min_score=0.4)
                                new_rag_data = [result['formatted_case'] for result in retrieval_results]
                                print(f"RAG查询完成，找到 {len(new_rag_data)} 个案例")

                                if new_rag_data:
                                    # 保存新检索的案例到历史
                                    add_rag_to_history(session_id, conversation_id, new_rag_data, rag_query)

                                    # 通知前端找到了新案例
                                    yield f"data: {json.dumps({'event': 'rag_results_found', 'count': len(new_rag_data)})}\n\n"

                            # 更新对话历史（使用清理后的回复，不含RAG_QUERY标记）
                            # 用户消息：如果有附件，保存附件信息供前端查看
                            user_msg = {"role": "user", "content": user_message}
                            if attachments:
                                user_msg["attachments"] = attachments
                            conversation_history.append(user_msg)
                            conversation_history.append({"role": "assistant", "content": clean_response})
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
                                model_info = data_json.get("model_used", "未知")
                                yield f"data: {json.dumps({'event': 'model_info', 'model_used': model_info})}\n\n"

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
    current_conv = get_current_conversation(session_id)

    # 同时清空对话历史和RAG历史
    current_conv.set_history([])
    current_conv.set_rag_history([])
    db.session.commit()

    return jsonify({'success': True, 'message': '对话历史已清空'})

@app.route('/get_history', methods=['GET'])
def get_history():
    """获取当前对话历史"""
    session_id = session.get('session_id')
    current_conv = get_current_conversation(session_id)
    return jsonify({'history': current_conv.get_history()})

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
            'case_count': getattr(retrieval_system, 'case_count', 0),
            'rag_debug': RAG_DEBUG
        })

if __name__ == '__main__':
    # 确保会话目录存在
    if not os.path.exists('./flask_session'):
        os.makedirs('./flask_session')

    # 初始化数据库表（如果不存在则创建）
    with app.app_context():
        db.create_all()
        print("✓ 数据库初始化完成")

    # 启动时初始化检索系统
    print("正在启动法律小助手Web应用...")
    if not LegalCaseRetriever.model_loaded:
        if initialize_retrieval_system():
            print("检索系统初始化成功！")
            LegalCaseRetriever.model_loaded = True
        else:
            print("警告: 检索系统初始化失败")

    app.run(debug=False, port=5001)

