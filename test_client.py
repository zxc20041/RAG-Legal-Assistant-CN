"""
文件名: test_client.py
功  能: 客户端脚本 (已增加Claude, Qwen, Doubao, Grok等模型的命令行支持)。
描  述:
1. 在 --model 参数的 choices 列表中增加了新模型。
"""

import requests
import json
import argparse

# --- 1. 全局配置与模拟数据 (这部分完全没动) ---
API_URL = "http://127.0.0.1:5000/predict"
SIMULATED_RAG_DATA = [
    {"fact": "公诉机关起诉指控，被告人张某某秘密窃取他人财物，价值2210元...", "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 0, "criminals": ["张某G"], "term_of_imprisonment": {"death_penalty": False, "imprisonment": 2, "life_imprisonment": False}}},
    {"fact": "孝昌县人民检察院指控：2014年1月4日，被告人邬某在孝昌县城区2路公交车上扒窃被害人晏某白色VIVO手机一部。经鉴定，该手机价值为750元...", "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 1000, "criminals": ["邬某"], "term_of_imprisonment": {"death_penalty": False, "imprisonment": 4, "life_imprisonment": False}}}
]

# --- 2. 主聊天循环 (这部分完全没动) ---
def main_chat_loop(selected_model_id):
    conversation_history = []
    
    print(f"\n--- 刑事咨询小助手已连接 (当前模型: {selected_model_id}) ---")
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
        
        payload = {
            "user_question": user_input,
            "rag_data": current_rag_data,
            "chat_history": conversation_history,
            "model_id": selected_model_id 
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if "prediction" in result:
                assistant_response = result["prediction"]
                model_used = result.get("model_used", "未知")
                print(f"助手 (来自 {model_used}): {assistant_response}")
                
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": assistant_response})
            else:
                print(f"助手(错误): {result.get('error', '未知错误')}")

        except requests.exceptions.RequestException as e:
            print(f"\n--- 请求失败 ---")
            print(f"错误: {e}")
            print("请确保 llm_api_handler.py 服务正在运行中。")
            break

# --- 3. 启动脚本 (这里是唯一改动的地方) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="刑事咨询小助手客户端")
    
    # ===================================================================
    # --- 核心修改：在choices列表中增加新模型 ---
    # ===================================================================
    parser.add_argument(
        "--model",
        type=str,
        # 新增了 'claude', 'qwen', 'doubao', 'grok'
        choices=['deepseek', 'zhipu', 'gpt4o', 'claude', 'qwen', 'doubao', 'grok'],
        default='deepseek',
        help="选择要使用的大模型 (可选项: deepseek, zhipu, gpt4o, claude, qwen, doubao, grok)"
    )
    # ===================================================================
    
    args = parser.parse_args()
    main_chat_loop(args.model)

