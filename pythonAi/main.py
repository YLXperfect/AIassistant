# main.py
from agentCore import add_to_memory, clear_memory, get_api_key, create_ai_agent, get_memory, run_chat_loop

momery_list = []


# 初始化记忆列表
def initMomerList(momery_list):
    
    clear_memory(momery_list)
    # 添加一条明确的系统指令，赋予模型“记忆”的角色和能力
    system_prompt = """你是一个拥有完整对话记忆的AI助手。我们的整个对话历史都将被提供给你。你必须仔细阅读整个历史，并基于历史中的信息来回答用户的问题。当用户问到关于历史的问题时，你要根据历史给出明确的答案。"""
    add_to_memory(momery_list,'system', system_prompt)
    

def main():
    try:
        # 1. 获取密钥
        api_key = get_api_key()
        # 2. 创建Agent大脑
        agent_brain = create_ai_agent(api_key)
        # 3. 启动对话循环，并传入创建好的“大脑”
        initMomerList(momery_list)
        run_chat_loop(agent_brain,momery_list)
    except ValueError as e:
        
        print(e)
        print("程序启动失败。")

#程序入口 
if __name__ == "__main__":
    main()