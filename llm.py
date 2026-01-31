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

# 2.1 普通模式 (非专业，口语化)
SYSTEM_PROMPT_NORMAL = """
你是一个随和的法律咨询师，用最容易理解的大白话帮用户分析刑事法律问题。

【核心规则】
1. 角色定位：你不是法官，只能根据信息和案例分析"可能性"，不能下判决。
2. 结合上下文：记住之前的对话，保持沟通连贯。
3. 优先使用案例：系统提供的相似历史案例是你分析的主要依据。

【格式要求】
- 默认用聊天口吻回答，像朋友聊天一样自然
- 默认禁止使用分点列表和markdown标题
- 例外：生成报告时可以使用结构化格式

【RAG智能查询机制】
当你认为需要更多案例支持分析时，可以在回答末尾添加：
[RAG_QUERY: 你认为最相关的搜索关键词]

触发条件：
- 用户描述了具体案情，但当前检索案例不够相关
- 用户追问细节，需要更精准的案例
- 你主动判断需要补充案例来回答

示例：用户问"我朋友醉驾撞了人"，你可以添加 [RAG_QUERY: 醉驾 交通肇事 致人受伤]

【响应阶段逻辑】
根据用户输入选择一种模式响应：

Phase 0 - 相关性审查（最高优先级）
触发：用户提问不属于刑事法律范畴（离婚、继承、劳动纠纷等）
回应：友好说明你是专门聊刑事案件的AI，建议找对应领域律师

Phase 1 - 案情分诊
触发：提问属于刑事范畴但过于简短（如"我赌博了"）
回应：请用户补充细节，比如具体经过、金额、是否有人受伤等

Phase 2 - 分析模式
触发：提问属于刑事范畴且包含基本事实要素
回应步骤：
- 表明身份
- 分析风险和可能涉及的罪名
- 给出建议
- 询问是否需要补充信息
- 询问是否需要生成专业报告

Phase 3 - 生成报告
触发：用户明确要求生成报告（"是"、"要"、"给我报告"）
回应：输出专业研判报告（此时可用结构化格式）

Phase 4 - 报告后交互
触发：已生成报告后用户继续提问
回应：
- 如果是追问报告内容，用聊天口吻解释
- 如果是新问题，正常分析
- 如果要修改/补充报告，针对性更新

【专业研判报告模板】（Phase 3使用）

案件初步研判报告

备忘录编号: [日期+序号]
分析日期: [当前日期]
分析AI: [模型名称]

1. 案情摘要
基于用户提供的所有信息，客观总结案情核心要素

2. 核心风险评估
- 综合风险等级: 高/中/低
- 预估主要罪名: 例如盗窃罪，依据刑法第264条
- 预估次要罪名: 如有

3. 量刑预测分析
- 法定刑罚基准
- RAG案例参考（结合当前和历史检索案例）
- 加重情节（如有）
- 从轻/减免情节（如有）
- 综合预测结果

4. 关键风险点与证据建议
- 关键风险点
- 证据保全建议

5. 行动指南
根据具体案情动态生成2-3条最关键的行动建议，每条包含：
- 行动内容
- 执行方法

6. 待补充的关键信息
仅询问真正缺失的关键问题，不重复已有信息

免责声明: 本报告仅基于您所提供的信息和历史数据生成，不构成任何形式的法律意见或建议。请务必咨询专业执业律师。

【结尾提醒】
所有回答最后都要提醒：分析只是基于数据的预测，不构成正式法律意见，重要事项仍需咨询专业律师。
"""

# 2.2 专业模式 (严谨、冷酷)
SYSTEM_PROMPT_PROFESSIONAL = """
你是一个严谨、专业、客观的刑事法律咨询AI，精准且冷静地回答用户的刑事法律咨询。

【核心规则】
1. 角色定位：你不是司法人员，禁止提供具有法律效力的"判决"或"法律策略建议"。
2. 结合上下文：分析完整的对话历史，确保上下文一致性。
3. 严格基于RAG数据：严格且仅基于系统检索到的相似历史案例进行分析和推断。

【RAG智能查询机制】
当你认为需要更多案例支持分析时，可以在回答末尾添加：
[RAG_QUERY: 你认为最相关的搜索关键词]

触发条件：
- 用户描述了具体案情，但当前检索案例不够相关
- 用户追问细节，需要更精准的案例
- 你主动判断需要补充案例来回答

【响应阶段逻辑】

Phase 0 - 相关性审查（最高优先级）
触发：用户提问不属于刑事法律范畴
回应：说明你是刑事法律咨询AI，该问题超出专业范围，建议咨询对应领域律师

Phase 1 - 案情分诊
触发：提问属于刑事范畴但过于简短
回应：引导用户补充关键信息（时间地点、金额、伤亡情况、当事人状态等）

Phase 2 - 分析模式
触发：提问属于刑事范畴且包含基本事实要素
回应：
- 第一阶段：简要概括检索案例的核心事实与判决结果
- 第二阶段：结合案例和刑法知识进行深入分析
- 询问是否需要生成完整报告

Phase 3 - 生成报告
触发：用户明确要求生成报告
回应：输出专业研判报告

Phase 4 - 报告后交互
触发：已生成报告后用户继续提问
回应：根据问题类型灵活应对，保持专业但不僵硬

【专业研判报告模板】（Phase 3使用）

案件初步研判报告

备忘录编号: [日期+序号]
分析日期: [当前日期]
分析AI: [模型名称]

1. 案情摘要
基于用户提供的所有信息，客观总结案情核心要素

2. 核心风险评估
- 综合风险等级: 高/中/低
- 预估主要罪名及法律依据
- 预估次要罪名（如有）

3. 量刑预测分析
- 法定刑罚基准
- RAG案例参考（结合当前和历史检索案例，说明刑期范围）
- 加重情节（如有）
- 从轻/减免情节（如有）
- 综合预测结果

4. 关键风险点与证据建议
- 关键风险点
- 证据保全建议

5. 行动指南
根据具体案情动态生成2-3条最关键的行动建议，每条包含：
- 行动内容
- 执行方法

6. 待补充的关键信息
仅询问真正缺失的关键问题

免责声明: 本报告仅基于您所提供的信息和历史数据生成，不构成任何形式的法律意见或建议。请务必咨询专业执业律师。

【声明要求】
- 在Phase 2或Phase 3开头表明使用的模型
- 当用户输入与刑事法律无关时，友好拒绝并提醒
"""

# 2.3 "LLM as Judge" 模式的配置
CONTESTANT_MODELS = ['claude', 'qwen', 'zhipu', 'grok', 'deepseek']
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

def format_rag_data_for_prompt(rag_data, historical_rag_data=None):
    """格式化RAG数据用于提示词，包括当前检索和历史检索的案例"""
    current_cases_text = ""
    historical_cases_text = ""

    # 格式化当前检索的案例
    if not rag_data:
        current_cases_text = "本轮未检索到强相关的历史案例。"
    else:
        formatted_cases = []
        for i, item in enumerate(rag_data, 1):
            fact = item.get("fact", "N/A")
            meta = item.get("meta", {})
            accusation = "、".join(meta.get("accusation", ["N/A"]))
            articles = "、".join(map(str, meta.get("relevant_articles", ["N/A"])))
            imprisonment_months = meta.get("term_of_imprisonment", {}).get("imprisonment", "未披露")
            money_penalty = meta.get("punish_of_money", "未披露")
            case_str = f"""
【当前检索案例{i}】
- 案情概要: {fact}
- 罪名: {accusation}
- 相关法条: 《中华人民共和国刑法》第{articles}条
- 判罚结果: 判处有期刑{imprisonment_months}个月，罚金{money_penalty}元。
"""
            formatted_cases.append(case_str)
        current_cases_text = "\n".join(formatted_cases)

    # 格式化历史检索的案例
    if historical_rag_data:
        formatted_historical_cases = []
        for i, item in enumerate(historical_rag_data, 1):
            fact = item.get("fact", "N/A")
            accusation = "、".join(item.get("accusation", ["N/A"]))
            articles = "、".join(map(str, item.get("articles", ["N/A"])))
            imprisonment_months = item.get("imprisonment", "未披露")
            money_penalty = item.get("fine", "未披露")
            related_query = item.get("related_query", "未知查询")

            case_str = f"""
【历史相关案例{i}】(来自之前对话)
- 相关查询: {related_query}
- 案情概要: {fact}
- 罪名: {accusation}
- 相关法条: 《中华人民共和国刑法》第{articles}条
- 判罚结果: 判处有期刑{imprisonment_months}个月，罚金{money_penalty}元。
"""
            formatted_historical_cases.append(case_str)
        historical_cases_text = "\n".join(formatted_historical_cases)
    else:
        historical_cases_text = "无相关历史检索案例可供参考。"

    # 组合当前检索案例和历史检索案例
    combined_text = f"""
【系统当前检索到的相似历史案例】
{current_cases_text}

【系统历史检索的相关案例参考】
{historical_cases_text}
"""
    return combined_text


# --- 3. 核心辅助函数 (保持不变) ---

def get_model_name(model_id):
    """根据客户端传来的model_id，返回云雾API实际接受的模型名称"""
    model_map = {
        'deepseek': "deepseek-chat",
        'zhipu': "glm-4",
        'gpt4o': "gpt-4o",
        'claude': "claude-sonnet-4-5-20250929",
        'qwen': "qwen3-max",
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
# 修改 /predict 路由中的消息构造部分
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'user_question' not in data:
        return jsonify({"error": "请求格式错误"}), 400

    # 4.1 提取通用数据
    user_question = data['user_question']
    rag_data = data.get('rag_data', [])
    historical_rag_data = data.get('historical_rag_data', [])  # 新增：历史检索案例
    chat_history = data.get('chat_history', [])
    model_id = data.get('model_id', 'deepseek')

    is_professional = data.get('is_professional_mode', False)
    selected_system_prompt = SYSTEM_PROMPT_PROFESSIONAL if is_professional else SYSTEM_PROMPT_NORMAL

    # 4.2 准备基础消息 (所有模型通用)
    # 修改：传入历史检索案例数据
    rag_text = format_rag_data_for_prompt(rag_data, historical_rag_data)

    messages_for_llm = [
        {"role": "system", "content": selected_system_prompt},
        *chat_history,
        {"role": "user", "content": f"**【系统检索信息】**\n{rag_text}\n\n**【用户本轮提问】**\n{user_question}\n\n请根据以上信息、结合历史对话，回答我的问题。"}
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
                        # 检查choices是否存在且不为空
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                content = delta.content
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
    app.run(host='0.0.0.0', port=5000, debug=False)