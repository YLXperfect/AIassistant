# test_import.py
import langchain
print("LangChain版本:", langchain.__version__)

# 尝试常见路径
import_paths = [
    "langchain.agents.AgentExecutor",
    "langchain_experimental.agents.AgentExecutor", 
    "langchain.agents.agent_toolkits.AgentExecutor",
    "langchain.agents.agent_toolkits.base.AgentExecutor"
]

for path in import_paths:
    try:
        # 动态导入测试
        exec(f"from {path} import AgentExecutor")
        print(f"✅ 成功导入: {path}")
        break
    except ImportError:
        print(f"❌ 无法导入: {path}")