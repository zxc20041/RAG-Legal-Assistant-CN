# -*- coding: utf-8 -*-
"""
文件名: test_client.py
功  能: 客户端脚本 (已支持流式输出 和 Markdown 渲染)。
描  述:
1. 需要先安装 'rich' 库: pip install rich
2. Judge模式: 保持不变，但最后会使用 rich.markdown 打印精美格式。
3. 单个模型模式:
    - 使用 requests.post(stream=True)
    - 实时解析 Server-Sent Events (SSE)
    - 实时打印流式输出的文字。
    - 将完整的 Markdown 回答存入历史。
4. (新增) 支持 --pro 参数，用于开启专业模式。
"""

import requests
import json
import argparse
import sys # 用于 flush 输出

# (新增) 导入 rich
try:
    from rich.console import Console
    from rich.markdown import Markdown
except ImportError:
    print("错误: 找不到 'rich' 库。")
    print("请先通过命令安装: pip install rich")
    sys.exit(1)

# --- 1. 全局配置与模拟数据 ---
API_URL = "http://127.0.0.1:5000/predict"
SIMULATED_RAG_DATA = [
    # (模拟数据和之前一样，此处省略)
    {"fact": "公诉机关起诉指控，被告人张某某秘密窃取他人财物，价值2210元...", "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 0, "criminals": ["张某G"], "term_of_imprisonment": {"death_penalty": False, "imprisonment": 2, "life_imprisonment": False}}},
    {"fact": "孝昌县人民检察院指控：2014年1月4日，被告人邬某在孝昌县城区2路公交车上扒窃被害人晏某白色VIVO手机一部。经鉴定，该手机价值为750元...", "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 1000, "criminals": ["邬某"], "term_of_imprisonment": {"death_penalty": False, "imprisonment": 4, "life_imprisonment": False}}}
]

# (新增) 初始化 rich Console
console = Console()

# ===================================================================
# --- 2. 主聊天循环 (已重构，增加 is_professional_mode 参数) ---
# ===================================================================
def main_chat_loop(selected_model_id, is_professional_mode=False):
    conversation_history = []
    
    # (修改) 打印当前模式
    mode_name = "专业模式" if is_professional_mode else "普通模式"
    print(f"\n--- 刑事咨询小助手已连接 (模型: {selected_model_id} | 模式: {mode_name}) ---")
    print("你好，请输入你的问题。输入 'exit' 或 '退出' 来结束对话。")

    while True:
        try:
            user_input = input("你: ")
        except EOFError:
            break
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("感谢使用，再见！")
            break

        print("--- (模拟RAG：正在为本轮问题检索相似案例...) ---")
        current_rag_data = SIMULATED_RAG_DATA 
        
        # (修改) 增加 "is_professional_mode" 字段
        payload = {
            "user_question": user_input,
            "rag_data": current_rag_data,
            "chat_history": conversation_history,
            "model_id": selected_model_id,
            "is_professional_mode": is_professional_mode # <--- (新增) 传递模式开关
        }
        headers = {"Content-Type": "application/json"}

        # ===================================================================
        # --- 核心修改：根据模式选择不同的请求方式 ---
        # ===================================================================

        if selected_model_id == 'judge':
            # --- Judge 模式 (非流式, 但使用 Rich 打印) ---
            try:
                # Judge模式超时时间更长
                response = requests.post(API_URL, headers=headers, json=payload, timeout=120) 
                response.raise_for_status()
                result = response.json()
                
                if "prediction" in result:
                    assistant_response = result["prediction"]
                    model_used = result.get("model_used", "未知")
                    
                    console.print(f"助手 (来自 {model_used}):")
                    # (修改) 使用 rich.markdown 打印排版后的回答
                    console.print(Markdown(assistant_response))
                    
                    if "judge_reasoning" in result:
                        console.print("\n--- (裁判评判理由) ---", style="yellow")
                        console.print(result["judge_reasoning"])
                        console.print("------------------------\n", style="yellow")
                    
                    conversation_history.append({"role": "user", "content": user_input})
                    conversation_history.append({"role": "assistant", "content": assistant_response})
                else:
                    console.print(f"助手(错误): {result.get('error', '未知错误')}", style="red")

            except requests.exceptions.Timeout:
                console.print("--- 请求超时 (Judge模式需要更长时间) ---", style="red")
            except requests.exceptions.RequestException as e:
                console.print(f"--- 请求失败: {e} ---", style="red")
                break
        
        else:
            # --- 单个模型模式 (流式) ---
            full_response = ""
            model_used = ""
            stream_had_error = False # <--- (新增) 错误标志
            try:
                # (修改) 增加 stream=True，使用 with 语句确保连接关闭
                with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    
                    print("助手 (正在思考...): ", end="")
                    
                    # ===================================================================
                    # --- (BUG 修复) 重写SSE解析逻辑 ---
                    # ===================================================================
                    for line in response.iter_lines(decode_unicode=True):
                        if not line: # 忽略空行
                            continue
                            
                        if line.startswith("event: model_info"):
                            # 这是 event 行, 我们忽略它, 等待
                            pass
                        
                        elif line.startswith("event: end_of_stream"):
                            # 接收流结束信号
                            print() # 换行
                            break
                        
                        elif line.startswith("event: error"):
                            stream_had_error = True # <--- (新增) 标记发生错误
                            pass # 忽略 event 行, 等待
                        
                        elif line.startswith("data: "):
                            # 接收 data: 事件 (聊天内容 或 model_info 或 error)
                            data_str = ""
                            try:
                                data_str = line.split("data: ", 1)[1] # 使用 split 1 次
                                data_json = json.loads(data_str)
                                
                                # 检查JSON内容是哪种类型的数据
                                if "model_used" in data_json:
                                    # --- 这是 model_info 数据 ---
                                    model_used = data_json.get("model_used", "未知")
                                    print(f"(来自 {model_used}): ", end="")
                                
                                elif "chunk" in data_json:
                                    # --- 这是聊天内容块 ---
                                    chunk = data_json.get("chunk", "")
                                    full_response += chunk
                                    # 实时打印文字流
                                    print(chunk, end="", flush=True) 
                                
                                elif "error" in data_json:
                                    # --- 这是错误信息 ---
                                    stream_had_error = True # <--- (新增) 标记发生错误
                                    error_msg = data_json.get("error", "未知流错误")
                                    console.print(f"\n--- 流传输中发生错误: {error_msg} ---", style="red")

                            except json.JSONDecodeError:
                                pass # 忽略空 data 或格式错误
                            except IndexError:
                                pass # 忽略空的 "data: " 行
                    # ===================================================================
                    # --- BUG 修复结束 ---
                    # ===================================================================
                            
                # ===================================================================
                # --- (重点修改处) 检查流结束后是否有内容 ---
                # ===================================================================
                if full_response:
                    # 流结束后，将完整的回答存入历史
                    conversation_history.append({"role": "user", "content": user_input})
                    conversation_history.append({"role": "assistant", "content": full_response})
                
                elif not stream_had_error:
                    # (新增) 如果流正常结束，但 full_response 为空，并且流中未报告错误
                    # 这通常意味着触发了内容安全策略
                    console.print(f"(助手未返回任何有效信息。这可能是因为提问触发了安全策略，或模型无法回答该问题。)", style="yellow")
                
                # (注意) 如果 stream_had_error 为 True 且 full_response 为空，
                # 我们就不再打印消息了，因为上面的 "error" in data_json 部分已经打印过错误了。


            except requests.exceptions.RequestException as e:
                console.print(f"--- 请求失败: {e} ---", style="red")
                break

# ===================================================================
# --- 3. 启动脚本 (已修改，增加 '--pro' 选项) ---
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="刑事咨询小助手客户端")
    
    parser.add_argument(
        "--model",
        type=str,
        choices=['deepseek', 'zhipu', 'gpt4o', 'claude', 'qwen', 'doubao', 'grok', 'judge'],
        default='deepseek',
        help="选择要使用的大模型 (可选项: deepseek, zhipu, gpt4o, claude, qwen, doubao, grok 或 'judge' 模式)"
    )
    
    # (新增) 专业模式开关
    parser.add_argument(
        "--pro",
        action="store_true",  # store_true 意味着如果带上 --pro，则值为 True，否则为 False
        help="开启'专业模式' (回答更严谨冷酷)"
    )
    
    args = parser.parse_args()
    
    # (修改) 将 is_professional 状态 (args.pro) 传递给主循环
    main_chat_loop(args.model, args.pro)