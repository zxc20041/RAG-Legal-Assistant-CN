# -*- coding: utf-8 -*-
"""
文件名: llm_api_handler.py
功  能: 刑事咨询助手的后端API服务 (已增加Claude, Qwen, Doubao, Grok等模型)。
描  述:
1. 使用云雾API Key调用多个模型。
2. 增加了对客户端传来新模型ID的处理逻辑。
"""

import os
import openai
from flask import Flask, request, jsonify

# --- 1. 全局配置 (保持不变) ---
# 提醒：请在下方填入你自己的云雾API Key
YUNWU_API_KEY = os.getenv("YUNWU_API_KEY", "sk-x4uwyow5nBbDTiJZ61t79saFsgkAQnjaKFSvwQLO4aUotCD9")
YUNWU_BASE_URL = "https://yunwu.ai/v1" # 使用从新文档中获取的正确地址

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# --- 2. Prompt工程 (保持不变) ---
SYSTEM_PROMPT = """
你是一个专业的刑事法律咨询助手。
你的任务是回答用户关于刑事法律的问题。

你需要遵循以下规则：
1.  **角色定位**：你不是法官，不能给出绝对的、确定的判决。请使用“可能”、“根据相似案例分析”等措_辞。
2.  **结合上下文**：你必须阅读并理解之前的对话历史，进行连贯的、有上下文的回答。
3.  **优先使用RAG数据**：当用户提问时，会提供【系统检索到的相似历史案例】。你必须优先基于这些案例进行分析和预测。
4.  **声明**：在回答开始，告诉用户你是什么模型，在回答的最后，声明这只是基于数据的预测，不构成正式的法律意见，建议咨询专业律师。
"""
def format_rag_data_for_prompt(rag_data):
    # (这个函数完全不变，此处省略)
    if not rag_data:
        return "本轮未检索到强相关的历史案例。"
    formatted_cases = []
    for i, item in enumerate(rag_data, 1):
        fact = item.get("fact", "N/A")
        meta = item.get("meta", {})
        accusation = "、".join(meta.get("accusation", ["N/A"]))
        articles = "、".join(map(str, meta.get("relevant_articles", ["N/A"])))
        imprisonment_months = meta.get("term_of_imprisonment", {}).get("imprisonment", "未披露")
        money_penalty = meta.get("punish_of_money", "未披露")
        case_str = f"""
【相似案例{i}】
- 案情概要: {fact}
- 罪名: {accusation}
- 相关法条: 《中华人民共和国刑法》第{articles}条
- 判罚结果: 判处有期徒刑{imprisonment_months}个月，罚金{money_penalty}元。
"""
        formatted_cases.append(case_str)
    return "\n".join(formatted_cases)

# --- 3. API接口定义 (核心修改点) ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'user_question' not in data:
        return jsonify({"error": "请求格式错误"}), 400

    user_question = data['user_question']
    rag_data = data.get('rag_data', [])
    chat_history = data.get('chat_history', [])
    model_id = data.get('model_id', 'deepseek')

    messages_for_llm = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *chat_history,
        {"role": "user", "content": f"**【系统检索到的相似历史案例】**\n{format_rag_data_for_prompt(rag_data)}\n\n**【用户本轮提问】**\n{user_question}\n\n请根据以上信息、结合历史对话，回答我的问题。"}
    ]

    try:
        # ===================================================================
        # --- 核心修改：增加对新模型的处理 ---
        # ===================================================================
        selected_model_name = ""
        if model_id == 'deepseek':
            selected_model_name = "deepseek-chat"
        elif model_id == 'zhipu':
            selected_model_name = "glm-4"
        elif model_id == 'gpt4o':
            selected_model_name = "gpt-4o"
        elif model_id == 'claude': # <--- 新增
            selected_model_name = "claude-sonnet-4-5-20250929"
        elif model_id == 'qwen': # <--- 新增
            selected_model_name = "qwen3-max"
        elif model_id == 'doubao': # <--- 新增
            selected_model_name = "doubao-seed-1-6-251015-search"
        elif model_id == 'grok': # <--- 新增
            selected_model_name = "xl-grok-4"
        else:
            return jsonify({"error": f"未知的模型ID: '{model_id}'"}), 400
        # ===================================================================
        
        client = openai.OpenAI(api_key=YUNWU_API_KEY, base_url=YUNWU_BASE_URL)
        
        response = client.chat.completions.create(
            model=selected_model_name,
            messages=messages_for_llm,
            temperature=0.1,
            stream=False
        )
        
        if response.choices and len(response.choices) > 0:
            assistant_response = response.choices[0].message.content
            return jsonify({"prediction": assistant_response, "model_used": selected_model_name})
        else:
            return jsonify({"error": "调用大模型失败，未返回有效回答"}), 500

    except openai.OpenAIError as e:
        app.logger.error(f"调用云雾API ({selected_model_name}) 时发生错误: {e}")
        return jsonify({"error": f"调用大模型API时发生错误: {e}"}), 503
    except Exception as e:
        app.logger.error(f"处理请求时发生未知错误: {e}")
        return jsonify({"error": f"服务器内部错误: {e}"}), 500

# --- 4. 启动服务 (保持不变) ---
if __name__ == '__main__':
    print("刑事咨询小助手后端服务(云雾API版)已启动，监听地址 http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)