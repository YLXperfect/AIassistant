#!/usr/bin/env python3
# 简单的 MCP 测试脚本

import sys
import os
import asyncio

# 添加当前目录到 Python 路径
sys.path.append(os.path.abspath('.'))

# 导入必要的库
try:
    from langchain_community.chat_models import ChatZhipuAI
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from src.AiAgentDeep.memory import ConversationMemory
    print("✅ 导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 获取 API Key
def get_api_key():
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("❌ 未找到环境变量 ZHIPUAI_API_KEY")
    return api_key

# 异步加载 MCP 工具
async def load_mcp_tools():
    """异步加载 MCP 工具"""
    try:
        client = MultiServerMCPClient(
            {
                "mock-news": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["src/AiAgentDeep/graph/mockMcp.py"],
                }
            }
        )
        mcp_tools = await client.get_tools()
        print(f"✅ MCP 工具加载成功: {[tool.name for tool in mcp_tools]}")
        return mcp_tools
    except Exception as e:
        print(f"❌ MCP 工具加载失败: {e}")
        import traceback
        traceback.print_exc()
        return []

# 主测试函数
async def main():
    print("🧠 开始测试 MCP 工具调用...")
    
    try:
        # 获取 API Key
        api_key = get_api_key()
        print("✅ API Key 获取成功")
        
        # 初始化 LLM
        llm = ChatZhipuAI(
            model="glm-4.6",
            temperature=0.1,
            streaming=True,
            api_key=api_key,
        )
        print("✅ LLM 初始化成功")
        
        # 加载 MCP 工具
        mcp_tools = await load_mcp_tools()
        
        if not mcp_tools:
            print("⚠️  没有加载到 MCP 工具，测试结束")
            return
        
        # 创建 agent
        system_prompt = """你是一个智能助手，使用ReAct框架（Thought-Action-Observation）回答问题。
        Thought: 先思考下一步该做什么
        Action: 如果需要，调用工具
        Observation: 观察工具结果
        Final Answer: 给出用户最终回答
        """
        
        agent = create_agent(
            model=llm,
            tools=mcp_tools,
            system_prompt=system_prompt,
        )
        print("✅ Agent 创建成功")
        
        # 测试 MCP 工具调用
        print("\n🤖 测试 MCP 工具调用...")
        user_input = "今天股市有什么新闻？"
        print(f"💬 你: {user_input}")
        
        # 调用 agent
        messages = [{"role": "user", "content": user_input}]
        print("📋 准备调用 agent...")
        response = await agent.ainvoke({"messages": messages})
        print("📋 Agent 调用完成")
        
        # 打印结果
        print(f"📋 响应类型: {type(response)}")
        if isinstance(response, dict) and 'messages' in response:
            print(f"📋 消息数量: {len(response['messages'])}")
            for i, msg in enumerate(response['messages']):
                print(f"📋 消息 {i} 类型: {type(msg)}")
                if hasattr(msg, 'content'):
                    print(f"📋 消息 {i} 内容: {msg.content}")
        else:
            print(f"📋 响应内容: {response}")
            
        print(f"🤖 助手: {response['messages'][-1].content}")
        
        print("\n🎉 MCP 工具调用测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())