# main.py
from agentCore import  get_api_key, create_ai_agent, run_chat_loop
from memory import ConversationMemory #用对话管理类

momery_list = []


# 初始化记忆列表
def initMomerList(momery_obj):
    
    #清除类中消息列表， 确保重新开始
    momery_obj.clearList()
    # 添加一条明确的系统指令，赋予模型“记忆”的角色和能力
    system_prompt = """你是一个拥有完整对话记忆的AI助手。我们的整个对话历史都将被提供给你。你必须仔细阅读整个历史，并基于历史中的信息来回答用户的问题。当用户问到关于历史的问题时，你要根据历史给出明确的答案。"""
    momery_obj.add_to_memory('system', system_prompt)
    

def main():
    try:
        # 初始化对话管理类
        con1 = ConversationMemory()
        initMomerList(con1)
        # 1. 获取密钥
        api_key = get_api_key()
        # 2. 创建Agent大脑
        agent_brain = create_ai_agent(api_key)
        # 3. 启动对话循环，并传入创建好的“大脑”
        run_chat_loop(agent_brain,con1)

        x = len(con1.getAllMemoryList())
        print(f"最近有几条消息,{x-1} 他们是{con1.get_Recent_messages(x-1)}")
        print(f"用户消息内容,{con1.get_Recent_user(x-1)}")

    except ValueError as e:
        
        print(e)
        print("程序启动失败。")

#程序入口 
if __name__ == "__main__":
    main()