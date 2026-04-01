"""
长桥MCP工具集成脚本

本脚本展示了如何在现有的agentCore.py中快速集成长桥MCP工具
使用步骤：
1. 复制本文件中的代码到agentCore.py的相应位置
2. 设置环境变量 LONGBRIDGE_CLIENT_ID 和 LONGBRIDGE_CLIENT_SECRET
3. 完成OAuth授权流程
4. 运行Agent即可使用长桥工具
"""

# ==================== 第1步：导入长桥工具 ====================
"""
在agentCore.py的导入部分添加：

# 原有导入
from src.AiAgentDeep.tools import calculator, search_Weather, get_current_time, query_document, smart_document_qa

# 添加长桥工具导入
try:
    from src.AiAgentDeep.longbridge_tools import LONGBRIDGE_TOOLS
    LONG bridge_AVAILABLE = True
except ImportError:
    LONG bridge_AVAILABLE = False
    print("⚠️ 长桥工具未安装，跳过加载")
"""

# ==================== 第2步：修改create_ai_agent函数 ====================
"""
修改create_ai_agent函数，添加长桥工具：

def create_ai_agent(api_key):
    \"\"\"根据给定的API密钥，创建并返回一个AI Agent实例\"\"\"
    print("🧠 正在初始化AI Agent...")
    llm = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.1,
        streaming=True,
        api_key=api_key,
    )
    
    # 原有工具
    tools = [calculator, search_Weather, get_current_time, query_document, smart_document_qa]
    
    # 添加长桥工具（如果可用）
    if LONG bridge_AVAILABLE:
        tools.extend(LONGBRIDGE_TOOLS)
        print("📈 长桥MCP工具已加载")
    
    # 系统提示词（添加长桥相关说明）
    system_prompt = \"\"\"你是一个智能助手，使用ReAct框架回答问题。
    
你具备以下能力：
1. 基础工具：计算、天气查询、时间查询、文档查询
2. 股票交易：查询行情、查看持仓、执行交易（如已配置长桥MCP）

对于股票相关查询，你可以：
- 查询实时行情：使用query_stock_quote工具
- 查看K线数据：使用query_candlesticks工具
- 搜索股票：使用search_stock工具
- 查看账户资产：使用query_account_assets工具
- 查看持仓：使用query_positions工具
- 执行交易：使用place_stock_order工具（需要用户确认）

⚠️ 交易安全提示：
- 执行交易前必须获得用户明确确认
- 建议先在模拟账户测试
- 股市有风险，投资需谨慎
\"\"\"
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent
"""

# ==================== 第3步：快速测试函数 ====================

def test_longbridge_tools():
    """
    快速测试长桥工具是否正常工作
    
    运行此函数可以验证：
    1. 环境变量是否正确设置
    2. 长桥工具是否可以正常导入
    3. OAuth授权是否完成
    """
    print("=" * 50)
    print("🧪 长桥MCP工具测试")
    print("=" * 50)
    
    # 测试1: 检查环境变量
    print("\n1️⃣ 检查环境变量...")
    import os
    client_id = os.getenv("LONGBRIDGE_CLIENT_ID")
    client_secret = os.getenv("LONGBRIDGE_CLIENT_SECRET")
    
    if client_id and client_secret:
        print(f"✅ Client ID: {client_id[:10]}...")
        print(f"✅ Client Secret: {client_secret[:10]}...")
    else:
        print("❌ 环境变量未设置")
        print("请运行：")
        print("export LONGBRIDGE_CLIENT_ID='your_client_id'")
        print("export LONGBRIDGE_CLIENT_SECRET='your_client_secret'")
        return
    
    # 测试2: 导入工具
    print("\n2️⃣ 测试工具导入...")
    try:
        from longbridge_tools import (
            query_stock_quote,
            query_candlesticks,
            search_stock,
            query_account_assets,
            query_positions,
            place_stock_order,
            query_orders,
            cancel_stock_order
        )
        print("✅ 所有工具导入成功")
    except ImportError as e:
        print(f"❌ 工具导入失败: {e}")
        return
    
    # 测试3: 创建客户端
    print("\n3️⃣ 测试创建客户端...")
    try:
        from longbridge_mcp import create_longbridge_client
        client = create_longbridge_client()
        print("✅ 客户端创建成功")
        
        # 检查是否已有访问令牌
        if client.config.access_token:
            print("✅ 发现已有访问令牌")
        else:
            print("⚠️ 未找到访问令牌，需要完成OAuth授权")
            print("请运行 longbridge_config_example.py 中的授权流程")
    except Exception as e:
        print(f"❌ 客户端创建失败: {e}")
        return
    
    # 测试4: 查询股票行情（需要授权）
    print("\n4️⃣ 测试查询股票行情...")
    try:
        result = query_stock_quote("AAPL")
        if "❌" not in result:
            print("✅ 行情查询成功")
            print(result[:200] + "...")  # 只显示前200字符
        else:
            print("⚠️ 行情查询失败，可能需要重新授权")
    except Exception as e:
        print(f"❌ 行情查询失败: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)


# ==================== 第4步：一键集成函数 ====================

def quick_integrate():
    """
    一键集成长桥工具到agentCore.py
    
    此函数会：
    1. 检查当前agentCore.py的内容
    2. 在适当位置添加长桥工具的导入和集成代码
    3. 备份原始文件
    
    ⚠️ 注意：这是一个辅助函数，实际修改前请确保已备份代码
    """
    import shutil
    from datetime import datetime
    
    # 文件路径
    agent_core_path = "agentCore.py"
    backup_path = f"agentCore_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    print("🚀 开始一键集成长桥工具...")
    print(f"📄 目标文件: {agent_core_path}")
    
    # 1. 备份原文件
    try:
        shutil.copy(agent_core_path, backup_path)
        print(f"✅ 已备份原文件到: {backup_path}")
    except Exception as e:
        print(f"❌ 备份失败: {e}")
        return
    
    # 2. 读取原文件
    try:
        with open(agent_core_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("✅ 已读取原文件")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 3. 添加长桥工具导入（在原有导入后）
    import_section = """
# 添加长桥工具导入
try:
    from src.AiAgentDeep.longbridge_tools import LONGBRIDGE_TOOLS
    LONGBRIDGE_AVAILABLE = True
    print("📈 长桥MCP工具已加载")
except ImportError as e:
    LONGBRIDGE_AVAILABLE = False
    print(f"⚠️ 长桥工具加载失败: {e}")
"""
    
    # 在tools导入后添加长桥导入
    if "from src.AiAgentDeep.tools import" in content:
        content = content.replace(
            "from src.AiAgentDeep.tools import",
            f"from src.AiAgentDeep.tools import\n{import_section}\n# 原有tools导入继续"
        )
    
    # 4. 修改create_ai_agent函数中的工具列表
    if "tools = [calculator" in content:
        content = content.replace(
            "tools = [calculator",
            """tools = [calculator
    
    # 添加长桥工具（如果可用）
    if LONGBRIDGE_AVAILABLE:
        tools.extend(LONGBRIDGE_TOOLS)
"""
        )
    
    # 5. 保存修改后的文件
    try:
        with open(agent_core_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ 文件修改完成")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 集成完成！")
    print("=" * 50)
    print("\n下一步：")
    print("1. 设置环境变量：")
    print("   export LONGBRIDGE_CLIENT_ID='your_client_id'")
    print("   export LONGBRIDGE_CLIENT_SECRET='your_client_secret'")
    print("\n2. 完成OAuth授权：")
    print("   运行 longbridge_config_example.py 中的授权流程")
    print("\n3. 测试工具：")
    print("   运行 test_longbridge_tools()")
    print("\n4. 启动Agent：")
    print("   运行 agentCore.py 中的对话循环")


# ==================== 使用说明 ====================
"""
📖 完整使用流程：

1. 准备工作：
   - 注册长桥账户并完成实名认证
   - 在长桥开放平台申请开发者权限
   - 获取 Client ID 和 Client Secret

2. 环境配置：
   export LONGBRIDGE_CLIENT_ID='your_client_id'
   export LONGBRIDGE_CLIENT_SECRET='your_client_secret'

3. 完成授权：
   - 运行 longbridge_config_example.py
   - 按提示完成OAuth授权流程
   - 获取访问令牌

4. 集成到Agent：
   - 方法A：手动复制本文件中的代码到agentCore.py
   - 方法B：运行 quick_integrate() 自动集成

5. 启动测试：
   - 运行 test_longbridge_tools() 验证工具
   - 启动Agent，开始对话

6. 使用示例对话：
   用户：查询苹果股票行情
   Agent：调用 query_stock_quote("AAPL")
   
   用户：查看我的持仓
   Agent：调用 query_positions()
   
   用户：买入100股特斯拉
   Agent：确认后调用 place_stock_order("TSLA", "BUY", 100, "MARKET")
"""

if __name__ == "__main__":
    print("长桥MCP工具集成脚本")
    print("=" * 50)
    print("\n可用功能：")
    print("1. test_longbridge_tools() - 测试工具")
    print("2. quick_integrate() - 一键集成到agentCore.py")
    print("\n请根据需要使用相应功能")