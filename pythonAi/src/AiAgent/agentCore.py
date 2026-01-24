'''
# 导入必要的库
# 使用新版模块结构
'''

from langchain_community.chat_models import ChatZhipuAI

from langchain.agents import create_agent


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage


import os 
from memory import ConversationMemory  #  导入对话管理类

from tools import calculator, search_Weather,get_current_time,query_document #导入工具
import time


    
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
    
    """根据给定的API密钥，创建并返回一个AI Agent实例（模型）。需要手动管理ReAct流程"""
    print("🧠 正在初始化AI Agent...")
    llm = ChatZhipuAI(
        model="glm-4",
        temperature=0.1,
        streaming=True,
        api_key=api_key,
    )
    
    tools = [calculator, search_Weather,get_current_time,query_document]  
    # llm_with_tools = llm.bind_tools(tools)  #添加并绑定工具给模型
    # return llm_with_tools
    '''
    使用create_agent创建自动管理ReAct的agent
    '''

   # Prompt 作为 state_modifier 传（字符串或 PromptTemplate 都行）
    system_prompt = """你是一个有帮助的AI助手，能使用工具解决问题。
请严格按照 ReAct 格式思考：并且每一步都打印出来
Thought: 先思考下一步该做什么
Action: 如果需要，调用工具
Observation: 观察工具结果
Final Answer: 给出用户最终回答"""
    agent = create_agent(
        model=llm,
        tools=tools,
       system_prompt =system_prompt, 
    )
    return agent
    

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
        elif msg["role"] == "tool":
            # 注意：你的内存中存储的是字典，需要提取信息构造ToolMessage 
            # 假设你存储时格式是：{"role": "tool", "content": "...", "name": "...", "tool_call_id": "..."}
            # 你需要根据实际存储的字段来调整
            tool_message = ToolMessage(
                content=msg.get("content", ""),
                name=msg.get("name", ""),  
                tool_call_id=msg.get("tool_call_id", "")  
            )
            langchain_messages.append(tool_message)
    return langchain_messages
    #返回的是一个全是langchian对象的消息列表，将整个对话内容发送给模型， 使得模型有记忆


#手动管理ReAct  需要手动存储消息，工具消息， 用户消息， AI消息
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
            print(f"（调试）转换后的消息数：{len(langchain_messages)}，角色分布：")

            for msg in langchain_messages:
                print(f"  - {type(msg).__name__}")

            # 显示“正在思考”动画（掩盖第一轮延迟）
            print("🤖 正在思考", end="", flush=True)
            for _ in range(3):
                time.sleep(0.1)
                print(".", end="", flush=True)
            print("\r", end="")  # 清掉动画行，准备打印回复

            invoke_start = time.time()
            # 4. 用 invoke（非流式）获取完整响应，便于检查 消息中是否有tool_calls
            first_response = agent_brain.invoke(langchain_messages)
            # 把第一轮模型输出（可能包含 tool_calls）加入历史
            
            
            full_response = ""  # 用来累积最终回复内容（后面存记忆）
            # print(f"【调试】第一轮invoke耗时: {time.time() - invoke_start:.2f}s")

            
            #判断第一轮回答中是否有工具调用
            if hasattr(first_response,"tool_calls") and first_response.tool_calls:

                langchain_messages.append(first_response)  
                
                for tool_call in first_response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    print(f"\n🛠️  正在调用工具: {tool_name}({tool_args})")
                     # 执行对应工具
                    if tool_name == "domath":
                        result = calculator.invoke(tool_args)
                    elif tool_name == "search_Weather":
                        result = search_Weather.invoke(tool_args)
                    elif tool_name == "get_current_time":
                        result = get_current_time.invoke(tool_args)
                    else:
                        result = "未知工具"
                    print(f"✅ 工具结果: {result}")
                  
                    # 创建标准的ToolMessage对象
                    tool_message = ToolMessage(
                        content=str(result),          # 工具执行结果
                        tool_call_id=tool_call["id"], # 必须与调用的id对应
                        name=tool_name                # 可选，但建议提供
                    )
                    langchain_messages.append(tool_message)
                    memory_obj.add_to_memory('tool', f"{tool_name} 工具结果: {result}")
                # 6. 第二轮调用：把工具结果塞回，用 stream 流式输出最终回复
                print("🤖 机器人回复: ", end="", flush=True)
                for chunk in agent_brain.stream(langchain_messages):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_response += chunk.content
                print()  # 结束时换行
            else:

            #修改为流式打印
                print("\n🤖 机器人回复: ", end="", flush=True)  # 开始打印，不换行            
            # 3. 【关键修改】使用流式调用
                for chunk in agent_brain.stream(langchain_messages):  # 改成 .stream()
                    if chunk.content:  # 有些chunk可能为空
                        print(chunk.content, end="", flush=True)  # 实时打印
                        full_response += chunk.content
            
            # 4. 流式结束，换行 + 分隔线
                print("\n" + "-" * 40)
            
            # 5. 将完整回复存入记忆
            memory_obj.add_to_memory('assistant', full_response)
        
        
        
        except Exception as e:
            print(f"⚠️  出错了: {e}")


#用agent管理ReAct 不需要再存储工具消息， 消息列表里只有 用户 跟 Ai助手消息
def newRun_chat_loop(memory_obj,agent):
    print("\n🤖 LangGraph Agent 已上线！")
    
    while True:
        user_input = input("\n💬 你: ").strip()
        if user_input.lower() in ['退出', 'q']:
            break
        
        memory_obj.add_to_memory('user', user_input)
        messages = get_memory_as_langchain_messages(memory_obj)
        
        # 显示“正在思考”动画（掩盖第一轮延迟）
        print("🤖 正在思考", end="", flush=True)
        for _ in range(3):
            time.sleep(0.1)
            print(".", end="", flush=True)
        print("\r", end="")  # 清掉动画行，准备打印回复
        full_response = ""

        print("\n【ReAct 思考链开始】")
            
        
        for chunk in agent.stream({"messages": messages},stream_mode="updates",):
            for step, data in chunk.items():
                #每一步的思考过程
                # print(f"step: {step}")
                content_blocks = data['messages'][-1].content_blocks if data['messages'] else []
                # print(f"content: {content_blocks}")

                #每一步的回复
                for block in content_blocks:
                    if block['type'] == 'text':
                        text = block['text']
                        print(text, end="", flush=True)  # 流式打印
                        #拼接回复，最后存入memoryList  只存最终答案 明天优化
                        full_response += text

            
        print("\n【ReAct 思考链结束】")
        print("-" * 40)
            
        memory_obj.add_to_memory('assistant', full_response)


# # 3. 构造一个简单的用户消息
# messages = [HumanMessage(content="我要学ai agent开发，请帮我写一个学习计划")]

# # 4. 调用模型并打印回复

if __name__ == "__main__":

    print("111111111")
# response = llm.invoke(messages)
# print("💬 机器人回复：", response.content)


