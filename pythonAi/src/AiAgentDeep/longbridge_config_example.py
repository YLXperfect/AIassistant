"""
长桥MCP配置示例和使用说明

本文件提供了长桥MCP工具的完整配置示例和使用指南
"""

import os

# ==================== 环境变量配置 ====================
"""
在使用长桥MCP工具之前，需要配置以下环境变量：

1. LONGBRIDGE_CLIENT_ID: 长桥开放平台的客户端ID
2. LONGBRIDGE_CLIENT_SECRET: 长桥开放平台的客户端密钥
3. LONGBRIDGE_ACCESS_TOKEN: 访问令牌（授权后自动填充，可选）

配置方法：

方法1 - 临时配置（当前终端会话有效）：
    export LONGBRIDGE_CLIENT_ID="your_client_id"
    export LONGBRIDGE_CLIENT_SECRET="your_client_secret"

方法2 - 永久配置（添加到 ~/.zshrc 或 ~/.bashrc）：
    echo 'export LONGBRIDGE_CLIENT_ID="your_client_id"' >> ~/.zshrc
    echo 'export LONGBRIDGE_CLIENT_SECRET="your_client_secret"' >> ~/.zshrc
    source ~/.zshrc

方法3 - 使用 .env 文件（推荐用于开发环境）：
    在项目根目录创建 .env 文件，内容如下：
    LONGBRIDGE_CLIENT_ID=your_client_id
    LONGBRIDGE_CLIENT_SECRET=your_client_secret
"""

# ==================== 获取长桥API凭证的步骤 ====================
"""
步骤1: 注册长桥账户
    - 访问 https://longbridge.com 注册账户
    - 完成实名认证

步骤2: 申请开放平台权限
    - 访问 https://open.longbridge.com
    - 申请成为开发者
    - 创建应用获取 Client ID 和 Client Secret

步骤3: 完成OAuth授权
    - 运行授权脚本（见下方示例）
    - 在浏览器中登录长桥账户并授权
    - 获取访问令牌
"""

# ==================== 授权流程示例代码 ====================

def authorization_example():
    """
    OAuth授权流程示例
    
    这是完成长桥MCP授权的标准流程
    """
    from longbridge_mcp import LongbridgeMCPClient, LongbridgeConfig
    
    # 1. 创建配置（从环境变量自动加载）
    config = LongbridgeConfig()
    
    # 2. 创建客户端
    client = LongbridgeMCPClient(config)
    
    # 3. 获取授权URL
    auth_url = client.get_authorization_url()
    print(f"请访问以下URL进行授权：\n{auth_url}")
    
    # 4. 用户授权后，从回调URL获取授权码
    # 回调URL格式：http://localhost:8000/callback?code=AUTHORIZATION_CODE
    auth_code = input("请输入授权码：")
    
    # 5. 用授权码交换访问令牌
    success = client.exchange_code_for_token(auth_code)
    
    if success:
        print("✅ 授权成功！")
        print(f"访问令牌：{client.config.access_token}")
        # 建议将令牌保存到环境变量或安全存储
    else:
        print("❌ 授权失败")


# ==================== 在Agent中使用长桥工具 ====================

def setup_agent_with_longbridge_tools():
    """
    在LangChain Agent中集成长桥工具的示例
    
    展示了如何将长桥工具添加到您的Agent中
    """
    from langchain.agents import create_agent
    from langchain_community.chat_models import ChatZhipuAI
    from longbridge_tools import LONGBRIDGE_TOOLS
    from tools import calculator, search_Weather  # 您的其他工具
    
    # 1. 初始化模型
    llm = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.1,
        api_key=os.getenv("ZHIPUAI_API_KEY")
    )
    
    # 2. 合并所有工具
    all_tools = [
        calculator,
        search_Weather,
        # 添加长桥工具
        *LONGBRIDGE_TOOLS
    ]
    
    # 3. 创建系统提示词（添加长桥相关说明）
    system_prompt = """你是一个智能投资助手，可以帮助用户查询股市信息和执行交易。

你可以使用以下工具：
1. 股票行情查询 - 获取实时股价、涨跌幅等信息
2. K线数据查询 - 获取历史价格数据
3. 股票搜索 - 根据名称或代码搜索股票
4. 账户资产查询 - 查看账户总资产和资金
5. 持仓查询 - 查看当前持有的股票
6. 下单交易 - 买入或卖出股票（需要用户确认）
7. 订单查询 - 查看订单状态
8. 撤单 - 取消未成交订单

⚠️ 重要提示：
- 交易操作前必须获得用户明确确认
- 建议先在模拟账户中测试交易功能
- 股市有风险，投资需谨慎
"""
    
    # 4. 创建Agent
    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_prompt
    )
    
    return agent


# ==================== 使用示例 ====================

def usage_examples():
    """
    长桥工具的使用示例
    """
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
    
    # 示例1: 查询股票行情
    print("=" * 50)
    print("示例1: 查询苹果股票行情")
    result = query_stock_quote("AAPL")
    print(result)
    
    # 示例2: 搜索股票
    print("\n" + "=" * 50)
    print("示例2: 搜索腾讯股票")
    result = search_stock("腾讯")
    print(result)
    
    # 示例3: 查询K线数据
    print("\n" + "=" * 50)
    print("示例3: 查询特斯拉最近30天日线")
    result = query_candlesticks("TSLA", period="day", count=30)
    print(result)
    
    # 示例4: 查询账户资产
    print("\n" + "=" * 50)
    print("示例4: 查询账户资产")
    result = query_account_assets()
    print(result)
    
    # 示例5: 查询持仓
    print("\n" + "=" * 50)
    print("示例5: 查询当前持仓")
    result = query_positions()
    print(result)
    
    # 示例6: 查询订单
    print("\n" + "=" * 50)
    print("示例6: 查询待成交订单")
    result = query_orders("PENDING")
    print(result)
    
    # 示例7: 下单（需要用户确认）
    print("\n" + "=" * 50)
    print("示例7: 市价买入100股苹果股票")
    # ⚠️ 注意：实际使用时需要用户确认
    # result = place_stock_order("AAPL", "BUY", 100, "MARKET")
    # print(result)
    print("⚠️ 此示例仅作演示，实际交易需要用户确认")
    
    # 示例8: 撤单
    print("\n" + "=" * 50)
    print("示例8: 撤销订单")
    # result = cancel_stock_order("ORDER123456")
    # print(result)
    print("⚠️ 此示例仅作演示，需要提供实际订单ID")


# ==================== 安全建议 ====================
"""
🔒 安全建议：

1. 凭证管理：
   - 不要将 Client ID 和 Client Secret 硬编码在代码中
   - 使用环境变量或安全的密钥管理服务
   - 定期轮换密钥

2. 授权管理：
   - 访问令牌有过期时间，需要定期刷新
   - 在不需要时撤销授权
   - 使用最小权限原则，只申请需要的权限

3. 交易安全：
   - 始终要求用户确认交易操作
   - 设置交易限额和风控规则
   - 记录所有交易日志
   - 建议先在模拟账户中测试

4. 错误处理：
   - 实现完善的错误处理机制
   - 记录异常日志以便排查问题
   - 为用户提供清晰的错误信息

5. 数据保护：
   - 不要记录敏感信息（如密码、密钥）
   - 使用HTTPS进行所有API通信
   - 定期清理不必要的日志数据
"""


if __name__ == "__main__":
    print("长桥MCP配置示例")
    print("=" * 50)
    print("\n请按照以下步骤配置：")
    print("1. 设置环境变量 LONGBRIDGE_CLIENT_ID 和 LONGBRIDGE_CLIENT_SECRET")
    print("2. 运行 authorization_example() 完成OAuth授权")
    print("3. 使用 setup_agent_with_longbridge_tools() 在Agent中集成工具")
    print("4. 参考 usage_examples() 了解如何使用各个工具")