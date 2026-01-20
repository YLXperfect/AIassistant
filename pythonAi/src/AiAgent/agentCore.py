'''
# 导入必要的库
# 使用新版模块结构
'''

from langchain_community.chat_models import ChatZhipuAI

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os 
from memory import ConversationMemory  #  倒入对话管理类


    
# 1. 设置你的API Key (这里是唯一需要修改的地方)

#day2 把步骤封装成函数，在main里调用
def get_api_key():
    """安全地获取API密钥。如果未设置，则抛出异常。"""
    zhipu_api_key = os.getenv("ZHIPUAI_API_KEY")
    if not zhipu_api_key:
        # 改为抛出异常，而非直接退出
        raise ValueError("❌ 未找到环境变量 ZHIPUAI_API_KEY。请在终端执行: export ZHIPUAI_API_KEY='你的密钥'")
    return zhipu_api_key


def create_ai_agent(api_key):
    """根据给定的API密钥，创建并返回一个AI Agent实例（模型）。"""
    print("🧠 正在初始化AI Agent...")
    llm = ChatZhipuAI(
        model="glm-4-flash",
        temperature=0.1,
        api_key=api_key,
    )
    return llm

# 在记忆模块部分，添加以下函数（放在 get_memory 函数后面即可）  添加参数memory_obj， 用他来管理消息操作
def get_memory_as_langchain_messages(memory_obj):
    """将内部记忆格式转换为LangChain的Message对象列表"""
    langchain_messages = []
    for msg in memory_obj.getAllMemoryList():  
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            langchain_messages.append(SystemMessage(content=msg["content"]))
    return langchain_messages
    #返回的是一个全是langchian对象的消息列表，将整个对话内容发送给模型， 使得模型有记忆

def run_chat_loop(agent_brain,memory_obj): #添加参数memory_obj， 用他来管理消息操作

    print("\n🤖 你的AI Agent已上线！请输入您的问题或者输入'NO' or '退出' 结束对话。")
    
    
    while True:
        user_input = input("\n💬 你: ").strip()
        if user_input.lower() in ['NO', '退出', 'exit', 'q']:
             print("👋 Agent期待与你再次对话！")
             break

        if not user_input:
            continue
    # 构造消息并调用模型

        try:
            memory_obj.add_to_memory('user', user_input)
            # 2. 【关键】获取转换后的完整消息历史（此时包含刚存的用户输入）
            langchain_messages = get_memory_as_langchain_messages(memory_obj)
            print(f"（调试）发送给模型的消息：{(langchain_messages)} ")  # 调试行
                # 3. 调用模型
            response = agent_brain.invoke(langchain_messages)
    
             # 4. 将AI回复存入记忆
            memory_obj.add_to_memory('assistant', response.content)
        
        # 打印Agent回复
            print(f"\n🤖 💬 机器人回复: {response.content}")
            print("-" * 40)
        
        except Exception as e:
            print(f"⚠️  出错了: {e}")
            

# # 3. 构造一个简单的用户消息
# messages = [HumanMessage(content="我要学ai agent开发，请帮我写一个学习计划")]

# # 4. 调用模型并打印回复

if __name__ == "__main__":

    print("111111111")
# response = llm.invoke(messages)
# print("💬 机器人回复：", response.content)


