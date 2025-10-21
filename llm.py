# -*- coding: utf-8 -*-
"""
文件名: llm_api_handler.py
功  能: 刑事咨询助手的后端API服务。
描  述:
1. 使用Flask框架搭建一个Web服务。
2. 创建一个名为 /predict 的API接口，用于接收POST请求。
3. 接收客户端发来的数据，包括：新问题、RAG检索数据、历史对话。
4. 将这些信息整合成一个完整的prompt，发送给DeepSeek大模型。
5. 将大模型返回的最新回答，通过API返回给客户端。
6. 本服务是“无状态”的，它不保存任何对话历史，完全依赖客户端每次请求时提供。
"""

# 导入必要的库
import os
import openai  # 使用openai库来调用DeepSeek API，代码更简洁
from flask import Flask, request, jsonify

# --- 1. 全局配置 ---

# 从环境变量中获取DeepSeek的API密钥，如果找不到，则使用后面的默认值。
# 强烈建议使用环境变量，以避免将密钥直接写入代码。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-439b1754b1cc4c40be3e64d252031608")

# 初始化Flask Web应用
app = Flask(__name__)
# 设置JSON响应中的中文可以正常显示，而不是显示为Unicode编码
app.config['JSON_AS_ASCII'] = False


# --- 2. Prompt工程：定义大模型的角色和指令 ---

# 这是“系统指令”，用于设定大模型的身份、任务和行为准则。
# 针对多轮对话，我们特别强调了“结合上下文”的重要性。
SYSTEM_PROMPT = """
你是一个专业的刑事法律咨询助手。
你的任务是回答用户关于刑事法律的问题。

你需要遵循以下规则：
1.  **角色定位**：你不是法官，不能给出绝对的、确定的判决。请使用“可能”、“根据相似案例分析”等措_辞。
2.  **结合上下文**：你必须阅读并理解之前的对话历史，进行连贯的、有上下文的回答。
3.  **优先使用RAG数据**：当用户提问时，会提供【系统检索到的相似历史案例】。你必须优先基于这些案例进行分析和预测。
4.  **声明**：在回答的最后，声明这只是基于数据的预测，不构成正式的法律意见，建议咨询专业律师。
"""

def format_rag_data_for_prompt(rag_data):
    """
    一个辅助函数，用于将从RAG系统检索到的JSON数据列表，
    格式化成一段适合放入Prompt的、人类可读的文本。
    """
    # 如果RAG没有检索到任何数据，返回一个提示信息
    if not rag_data:
        return "本轮未检索到强相关的历史案例。"
        
    formatted_cases = []
    # 遍历每一条案例数据
    for i, item in enumerate(rag_data, 1):
        # 安全地获取每个字段，如果字段不存在则使用默认值
        fact = item.get("fact", "案情不详")
        meta = item.get("meta", {})
        accusation = "、".join(meta.get("accusation", ["不详"]))
        articles = "、".join(map(str, meta.get("relevant_articles", ["不详"])))
        imprisonment_months = meta.get("term_of_imprisonment", {}).get("imprisonment", "未披露")
        money_penalty = meta.get("punish_of_money", "未披露")

        # 构建单个案例的格式化字符串
        case_str = f"""
【相似案例{i}】
- 案情概要: {fact}
- 罪名: {accusation}
- 相关法条: 《中华人民共和国刑法》第{articles}条
- 判罚结果: 判处有期徒刑{imprisonment_months}个月，罚金{money_penalty}元。
"""
        formatted_cases.append(case_str)
    
    # 将所有格式化后的案例字符串拼接在一起
    return "\n".join(formatted_cases)


# --- 3. API接口定义 ---

# 定义API的访问路径是 /predict，并且只接受POST类型的HTTP请求
@app.route('/predict', methods=['POST'])
def predict():
    """
    这是核心的API处理函数。
    当客户端向 /predict 地址发送POST请求时，这个函数会被调用。
    """
    # --- 3.1. 接收和解析客户端数据 ---
    
    # 从POST请求中获取JSON数据
    data = request.json
    # 校验数据，确保至少有'user_question'字段
    if not data or 'user_question' not in data:
        return jsonify({"error": "请求格式错误，至少需要'user_question'字段"}), 400

    # 从JSON数据中提取各个部分，如果某些字段不存在，则使用默认值（如空列表）
    user_question = data['user_question']                # 本轮用户的新问题 (必需)
    rag_data = data.get('rag_data', [])                  # RAG为新问题检索的数据 (可选)
    chat_history = data.get('chat_history', [])         # 之前的对话历史 (可选)


    # --- 3.2. 构建发送给大模型的完整消息列表 ---
    
    # 大模型API需要一个消息列表(messages)，其中每个元素都是一个字典，包含'role'和'content'
    # 'role'可以是 'system', 'user', 'assistant'
    
    # 第一步：添加系统指令，设定模型的角色
    messages_for_llm = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # 第二步：添加历史对话记录，让模型知道“前情提要”
    messages_for_llm.extend(chat_history)
    
    # 第三步：构建本轮的用户输入，这部分非常关键
    # 我们将RAG数据和用户的新问题合并成一个更丰富的prompt
    formatted_rag_text = format_rag_data_for_prompt(rag_data)
    new_user_message_content = f"""
**【系统检索到的相似历史案例】**
{formatted_rag_text}

**【用户本轮提问】**
{user_question}

请根据以上信息、结合历史对话，回答我的问题。
"""
    
    # 第四步：将包含RAG数据的本轮新问题，作为最新的'user'消息添加到列表中
    messages_for_llm.append({"role": "user", "content": new_user_message_content})


    # --- 3.3. 调用大模型API ---
    
    try:
        # 初始化openai客户端，并指定API密钥和DeepSeek的服务器地址
        client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

        # 发送请求给大模型
        response = client.chat.completions.create(
            model="deepseek-chat",      # 指定使用的模型
            messages=messages_for_llm,  # 传入我们精心构建的完整消息列表
            temperature=0.1,            # 设置较低的温度，让回答更稳定、严谨
            stream=False                # 关闭流式输出
        )
        
        # 从返回结果中提取模型的回答
        if response.choices and len(response.choices) > 0:
            assistant_response = response.choices[0].message.content
            # 成功！将模型的回答包装成JSON格式，返回给客户端
            return jsonify({"prediction": assistant_response})
        else:
            # API调用成功，但返回的数据格式不正确
            return jsonify({"error": "调用大模型失败，未返回有效回答", "details": response.model_dump_json()}), 500

    except openai.OpenAIError as e:
        # 捕获由于网络问题、认证失败等导致的API调用错误
        app.logger.error(f"调用DeepSeek API时发生OpenAI库错误: {e}")
        return jsonify({"error": f"调用大模型API时发生错误: {e}"}), 503
    except Exception as e:
        # 捕获其他所有未预料到的服务器内部错误
        app.logger.error(f"处理请求时发生未知错误: {e}")
        return jsonify({"error": f"服务器内部错误: {e}"}), 500

# --- 4. 启动服务 ---

# 确保这个脚本是直接被运行时，才启动Flask服务
if __name__ == '__main__':
    # 打印启动信息
    print("刑事咨询小助手后端服务(多轮对话版)已启动，监听地址 http://0.0.0.0:5000")
    # 启动Web服务
    # host='0.0.0.0' 表示任何电脑都可以访问这个服务（如果防火墙允许）
    # port=5000 指定服务的端口号
    # debug=True 开启调试模式，修改代码后服务会自动重启，方便开发
    app.run(host='0.0.0.0', port=5000, debug=True)