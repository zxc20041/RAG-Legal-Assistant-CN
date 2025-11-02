# -*- coding: utf-8 -*-
"""
文件名: llm_api_handler.py
功  能: 刑事咨询助手的后端API服务 (已增加流式输出 和 专业模式)。
描  述:
1. "judge" 模式保持不变 (非流式)。
2. "单个模型" 模式改为使用 stream=True，并以 event-stream 方式流式返回。
3. (新增) 支持 "is_professional_mode" 参数，用于切换不同的系统提示词。
"""

import os
import openai
from flask import Flask, request, jsonify, Response, stream_with_context
import threading
import json

# --- 1. 全局配置 ---
YUNWU_API_KEY = os.getenv("YUNWU_API_KEY", "sk-x4uwyow5nBbDTiJZ61t79saFsgkAQnjaKFSvwQLO4aUotCD9")
YUNWU_BASE_URL = "https://yunwu.ai/v1"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ===================================================================
# --- 2. Prompt工程 (已修改：增加专业版/普通版) ---
# ===================================================================

# 2.1 (新增) 普通模式 (非专业，口语化)
SYSTEM_PROMPT_NORMAL = """
**你是一个随和的法律咨询师。**
你的任务是**用最容易理解的大白话**，帮用户分析他们的法律问题。你要尽量避免难懂的“法律黑话”，让他们能轻松看懂。

你需要遵循以下规则：
1.  **角色定位**：你不是法官，所以不能“下判决”。你必须根据用户给的信息和相似案例，帮他们分析“**可能性**”。例如：“根据您的情况，**看起来可能会...**”
2.  **结合上下文**：你必须记住之前的对话，让沟通更顺畅。
3.  **优先使用案例**：系统会提供一些【相似历史案例】，这是你分析的主要依据。你必须**重点参考**这些案例来给用户回答。
4.  **!!! 格式要求 (非常重要) !!!**：
    * 你必须**用一段完整的、像聊天一样的口吻**来回答，把所有分析都融合在一段话里。
    * **禁止**使用任何形式的分点列表（比如 `1.`、`2.`、`* `）。
    * **禁止**使用任何标题（比如 `### 标题` 或 `**标题**`）。
    * **禁止**主动引用具体的法律条文（比如 `《刑法》第264条`），除非用户追问。
5.  **声明**：
    * **开头**：你必须先告诉用户你是什么模型。
    * **结尾**：**最后你必须提醒用户**：你的分析只是基于数据的预测，不构成正式的法律意见。如果事情比较重要，他们**仍需咨询专业的执业律师**。
"""
# 2.2 (新增) 专业模式 (严谨、冷酷)
SYSTEM_PROMPT_PROFESSIONAL = """
你是一个**严谨、专业、客观**的刑事法律咨询AI。
你的任务是**精准且冷静**地回答用户的刑事法律咨询。

你需要遵循以下规则：
1.  **角色定位**：你不是司法人员，禁止提供任何具有法律效力的判决或建议。所有回答必须基于提供的材料，并使用“根据所提供的案例分析”、“...具有...的可能性”等严谨措辞。
2.  **结合上下文**：你必须分析完整的对话历史，确保上下文的一致性。
3.  **基于RAG数据**：你必须适当基于【系统检索到的相似历史案例】进行分析和推断。如果RAG数据不足以支持回答，应明确指出。当没有合适的RAG数据或者RAG数据与用户输入关联不大，则无需提到有RAG数据，直接回答即可
4.  **格式化输出**：请使用清晰的 **Markdown** 格式。
5.  **声明**：在回答开始，表明你所使用的模型。在回答的最后进行总结，即对用户问题的整体回答，最后必须附上免责声明：大致意思是'本回答仅基于历史数据和模型推演，不构成任何形式的法律意见或建议。'但不要完全照搬，根据语言风格来选择回答方式
6.  **不规范输入**：当用户的输入与刑事咨询无关，你应该进行合理提醒
"""


# 2.3 "LLM as Judge" 模式的配置
CONTESTANT_MODELS = ['claude', 'qwen', 'zhipu', 'doubao', 'grok', 'deepseek']
JUDGE_MODEL_ID = 'gpt4o'

# (已修改) 裁判提示词模板，增加了 {system_instructions} 占位符
JUDGE_PROMPT_TEMPLATE = """
你是一个高级法律AI裁判。你的任务是评估多个AI助手对一个【用户问题】的回答，并选出最好的一个。

【评判标准】:
1.  **专业准确性**：回答是否在法律上准确无误？是否正确引用了RAG提供的案例？
2.  **完整性**：是否全面回答了用户的问题？有没有遗漏关键点？
3.  **易懂性**：(如果被要求)回答是否清晰、易于理解？(如果被要求严谨)回答是否足够严谨？
4.  **格式与遵循指示**：是否遵循了所有【系统指令】？

---
【原始用户问题】:
{user_question}
---
【RAG检索到的相似案例】:
{rag_data}
---
【系统指令】: (AI被要求遵循以下指示)
{system_instructions}
---
【所有AI的回答】:
{answers_text}
---

【你的任务】:
请根据上述【评判标准】和【系统指令】，评估所有AI的回答。
然后，以严格的JSON格式返回你的评判结果。

JSON格式必须如下:
{{
  "reasoning": "你的详细评判理由，说明为什么这个回答是最好的，其他回答有什么缺陷。",
  "best_answer": "在这里完整地、逐字不差地复制你认为最好的那个AI回答的全文。"
}}
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
- 判罚结果: 判处有期刑{imprisonment_months}个月，罚金{money_penalty}元。
"""
        formatted_cases.append(case_str)
    return "\n".join(formatted_cases)


# --- 3. 核心辅助函数 (保持不变) ---

def get_model_name(model_id):
    """根据客户端传来的model_id，返回云雾API实际接受的模型名称"""
    model_map = {
        'deepseek': "deepseek-chat",
        'zhipu': "glm-4",
        'gpt4o': "gpt-4o",
        'claude': "claude-sonnet-4-5-20250929",
        'qwen': "qwen3-max",
        'doubao': "doubao-seed-1-6-251015",
        'grok': "grok-4"
    }
    return model_map.get(model_id)

def call_model_sync(model_id, messages):
    """
    (重命名) 统一的、非流式(同步)的函数，用于Judge模式。
    """
    selected_model_name = get_model_name(model_id)
    if not selected_model_name:
        raise ValueError(f"未知的模型ID: '{model_id}'")
        
    try:
        client = openai.OpenAI(api_key=YUNWU_API_KEY, base_url=YUNWU_BASE_URL)
        response = client.chat.completions.create(
            model=selected_model_name,
            messages=messages,
            temperature=0.1,
            stream=False # 明确非流式
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            raise Exception("API未返回有效回答")
    except openai.OpenAIError as e:
        app.logger.error(f"调用云雾API ({selected_model_name}) 时发生错误: {e}")
        return f"模型 {model_id} 在回答时出错: {str(e)}"
    except Exception as e:
        app.logger.error(f"调用模型时发生未知错误: {e}")
        return f"模型 {model_id} 发生未知错误: {str(e)}"

# ===================================================================
# --- 4. API接口定义 (已重构，支持专业模式) ---
# ===================================================================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'user_question' not in data:
        return jsonify({"error": "请求格式错误"}), 400

    # 4.1 提取通用数据
    user_question = data['user_question']
    rag_data = data.get('rag_data', [])
    chat_history = data.get('chat_history', [])
    model_id = data.get('model_id', 'deepseek')
    
    # (新增) 获取专业模式开关，默认为 False (普通模式)
    is_professional = data.get('is_professional_mode', False)
    
    # (新增) 根据开关选择系统提示词
    selected_system_prompt = SYSTEM_PROMPT_PROFESSIONAL if is_professional else SYSTEM_PROMPT_NORMAL
    
    # 4.2 准备基础消息 (所有模型通用)
    rag_text = format_rag_data_for_prompt(rag_data)
    messages_for_llm = [
        # (修改) 使用选择的提示词
        {"role": "system", "content": selected_system_prompt},
        *chat_history,
        {"role": "user", "content": f"**【系统检索到的相似历史案例】**\n{rag_text}\n\n**【用户本轮提问】**\n{user_question}\n\n请根据以上信息、结合历史对话，回答我的问题。"}
    ]

    # ===================================================================
    # --- 4.3 核心修改：判断是否为 "Judge" 模式 ---
    # ===================================================================
    if model_id == 'judge':
        # --- JUDGE 模式 (保持非流式) ---
        try:
            # 1. 并行调用所有“参赛”模型
            # (注意: messages_for_llm 已经包含了正确的(专业或普通)系统提示词)
            threads = []
            results = {} 
            def thread_target(contestant_id):
                app.logger.info(f"Judge模式：开始调用 {contestant_id}...")
                answer = call_model_sync(contestant_id, messages_for_llm) # 使用非流式函数
                results[contestant_id] = answer
                app.logger.info(f"Judge模式：{contestant_id} 调用完成。")

            for contestant_id in CONTESTANT_MODELS:
                thread = threading.Thread(target=thread_target, args=(contestant_id,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

            app.logger.info("Judge模式：所有参赛模型调用完毕，准备调用裁判模型。")
            
            # 2. 准备裁判提示词
            answers_text_list = []
            for model_name, answer in results.items():
                answers_text_list.append(f"--- 来自模型 {model_name} 的回答 ---\n{answer}\n")
            
            # (修改) 将选择的系统提示词(selected_system_prompt)传给Judge
            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                user_question=user_question,
                rag_data=rag_text,
                system_instructions=selected_system_prompt, # <--- (新增) 告诉裁判使用了什么指令
                answers_text="\n".join(answers_text_list)
            )
            judge_messages = [{"role": "system", "content": judge_prompt}]
            
            # 3. 调用“裁判”模型
            judge_response_text = call_model_sync(JUDGE_MODEL_ID, judge_messages) # 使用非流式函数
            app.logger.info(f"Judge模式：裁判 ({JUDGE_MODEL_ID}) 评判完成。")

            # 4. 解析裁判的JSON输出
            try:
                if "```json" in judge_response_text:
                    judge_response_text = judge_response_text.split("```json")[1].split("```")[0].strip()
                judge_result = json.loads(judge_response_text)
                best_answer = judge_result.get("best_answer", "裁判未能选出最佳回答")
                reasoning = judge_result.get("reasoning", "裁判未提供理由")
                
                return jsonify({
                    "prediction": best_answer, 
                    "model_used": f"Judge ({JUDGE_MODEL_ID})",
                    "judge_reasoning": reasoning, 
                    "all_answers": results
                })
            except Exception as e:
                app.logger.error(f"Judge模式：裁判返回的JSON格式错误: {judge_response_text}。 错误: {e}")
                return jsonify({"error": "裁判返回结果格式错误，无法解析", "details": judge_response_text}), 500

        except Exception as e:
            app.logger.error(f"Judge模式处理时发生未知错误: {e}")
            return jsonify({"error": f"服务器内部错误: {e}"}), 500

    # ===================================================================
    # --- 4.4 单个模型模式：改为流式输出 ---
    # ===================================================================
    else:
        try:
            selected_model_name = get_model_name(model_id)
            if not selected_model_name:
                return jsonify({"error": f"未知的模型ID: '{model_id}'"}), 400

            # 定义流式响应生成器
            def stream_response():
                client = openai.OpenAI(api_key=YUNWU_API_KEY, base_url=YUNWU_BASE_URL)
                
                # 1. 告诉客户端模型名称 (自定义事件)
                model_name_data = json.dumps({"model_used": selected_model_name})
                yield f"event: model_info\ndata: {model_name_data}\n\n"

                # 2. 调用API (stream=True)
                # (注意: messages_for_llm 已经包含了正确的(专业或普通)系统提示词)
                stream = client.chat.completions.create(
                    model=selected_model_name,
                    messages=messages_for_llm,
                    temperature=0.1,
                    stream=True  # <--- 关键：开启流式
                )
                
                # 3. 迭代流，并将数据块转发给客户端
                try:
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            # 格式化为 Server-Sent Event (SSE)
                            chunk_data = json.dumps({"chunk": content})
                            yield f"data: {chunk_data}\n\n"
                            
                except Exception as e:
                    app.logger.error(f"流式传输中发生错误: {e}")
                    error_data = json.dumps({"error": str(e)})
                    yield f"event: error\ndata: {error_data}\n\n"
                
                # 4. 发送流结束信号 (自定义事件)
                yield "event: end_of_stream\ndata: {}\n\n"

            # 返回流式响应
            return Response(stream_with_context(stream_response()), mimetype='text/event-stream')

        except Exception as e:
            app.logger.error(f"流式模式启动时发生错误: {e}")
            return jsonify({"error": f"服务器内部错误: {e}"}), 500


# --- 5. 启动服务 (保持不变) ---
if __name__ == '__main__':
    print("刑事咨询小助手后端服务(流式版)已启动，监听地址 [http://0.0.0.0:5000](http://0.0.0.0:5000)")
    app.run(host='0.0.0.0', port=5000, debug=True)