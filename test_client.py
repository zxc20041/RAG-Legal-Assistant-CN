# -*- coding: utf-8 -*-
"""
文件名: test_client.py
功  能: 用于测试后端API的客户端脚本。
描  述:
1. 模拟一个真实的用户/前端应用。
2. 负责维护完整的对话历史（上下文）。
3. 进入一个循环，不断接收用户的输入。
4. 每次都将【新问题】+【RAG数据】+【全部历史对话】打包发送给后端API。
5. 接收并打印后端返回的回答。
6. 将新的一轮问答追加到对话历史中，为下一次提问做准备。
7. 本脚本是“有状态”的，它必须记住对话的全过程。
"""

# 导入必要的库
import requests
import json

# --- 1. 全局配置与模拟数据 ---

# 后端API服务的完整地址
API_URL = "http://127.0.0.1:5000/predict"

# 在这个测试脚本中，我们用固定的数据来“模拟”RAG系统的检索结果。
# 在您的真实软件中，这一步应该是根据用户的输入，去数据库中动态检索得到的。
SIMULATED_RAG_DATA = [
    {"fact": "公诉机关起诉指控，被告人张某某秘密窃取他人财物，价值2210元...", "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 0, "criminals": ["张某某"], "term_of_imprisonment": {"death_penalty": False, "imprisonment": 2, "life_imprisonment": False}}},
    {"fact": "孝昌县人民检察院指控：2014年1月4日，被告人邬某在孝昌县城区2路公交车上扒窃被害人晏某白色VIVO手机一部。经鉴定，该手机价值为750元...", "meta": {"relevant_articles": [264], "accusation": ["盗窃"], "punish_of_money": 1000, "criminals": ["邬某"], "term_of_imprisonment": {"death_penalty": False, "imprisonment": 4, "life_imprisonment": False}}}
]


# --- 2. 主聊天循环 ---

def main_chat_loop():
    """
    这是脚本的核心，一个循环，让对话可以持续进行。
    """
    
    # 关键点 1: 客户端在这里定义一个列表，用于存储和管理整个对话的历史。
    # 列表中的每个元素都是一个字典，格式为 {"role": "...", "content": "..."}
    conversation_history = []
    
    # 打印欢迎信息
    print("--- 刑事咨询小助手已连接 ---")
    print("你好，请输入你的问题。输入 'exit' 或 '退出' 来结束对话。")

    # 关键点 2: 使用无限循环，让用户可以一轮一轮地提问。
    while True:
        # --- 2.1. 获取用户本轮的输入 ---
        try:
            user_input = input("你: ")
        except EOFError:
            # 如果输入结束（例如按Ctrl+D），则退出
            break
            
        # 如果用户输入退出指令，则结束对话
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("感谢使用，再见！")
            break

        # --- 2.2. 模拟RAG系统工作 ---
        # 在真实应用中，这一步会把 user_input 发给RAG模块去检索
        print("--- (模拟RAG：正在为本轮问题检索相似案例...) ---")
        current_rag_data = SIMULATED_RAG_DATA 
        
        # --- 2.3. 构造发送给API的请求数据 ---
        # 这是一个字典，包含了后端API需要的所有信息
        payload = {
            "user_question": user_input,          # 用户的【新问题】
            "rag_data": current_rag_data,         # 本轮检索到的【RAG数据】
            "chat_history": conversation_history  # 之前【所有】的对话历史
        }

        # 定义HTTP请求头，告诉服务器我们发送的是JSON格式的数据
        headers = {"Content-Type": "application/json"}

        # --- 2.4. 发送HTTP POST请求到后端服务 ---
        try:
            # 发送请求，并设置一个60秒的超时时间
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            # 如果服务器返回了错误状态码（如404, 500），则会在这里抛出异常
            response.raise_for_status()
            
            # 解析服务器返回的JSON数据
            result = response.json()
            
            # 检查返回的数据中是否有我们需要的'prediction'字段
            if "prediction" in result:
                # 成功获取到助手的回答
                assistant_response = result["prediction"]
                print(f"助手: {assistant_response}")
                
                # 关键点 3: 更新本地的对话历史，为下一次对话做准备。
                # 必须同时保存用户的提问和助手的回答，这样上下文才是完整的。
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": assistant_response})
            else:
                # 如果返回的数据中没有'prediction'，说明可能出错了
                print(f"助手(错误): {result.get('error', '未知错误')}")

        except requests.exceptions.RequestException as e:
            # 捕获所有网络相关的错误，例如连接超时、服务器无法访问等
            print(f"\n--- 请求失败 ---")
            print(f"错误: {e}")
            print("请确保 llm_api_handler.py 服务正在运行中。")
            break # 连接失败，直接退出聊天循环

# --- 3. 启动脚本 ---

# 确保这个脚本是直接被运行时，才执行主聊天循环
if __name__ == "__main__":
    main_chat_loop()

