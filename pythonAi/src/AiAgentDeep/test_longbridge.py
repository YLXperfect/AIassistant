"""
长桥MCP工具测试脚本

用于验证长桥MCP配置是否正确，以及API调用是否正常
"""

import os
import sys
from pathlib import Path

# 添加项目路径到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 自动加载.env文件（如果存在）
env_file = os.path.join(project_root, "..", ".env")
if os.path.exists(env_file):
    print(f"📄 发现.env文件: {env_file}")
    from dotenv import load_dotenv
    load_dotenv(env_file)
    print("✅ .env文件加载成功")
else:
    print(f"⚠️  未找到.env文件: {env_file}")

def test_environment_variables():
    """测试1: 检查环境变量是否正确设置"""
    print("=" * 60)
    print("🔍 测试1: 检查环境变量")
    print("=" * 60)
    
    client_id = os.getenv("LONGBRIDGE_CLIENT_ID")
    client_secret = os.getenv("LONGBRIDGE_CLIENT_SECRET")
    access_token = os.getenv("LONGBRIDGE_ACCESS_TOKEN")
    
    all_set = True
    
    if client_id:
        print(f"✅ LONGBRIDGE_CLIENT_ID: {client_id[:10]}...{client_id[-4:]}")
    else:
        print("❌ LONGBRIDGE_CLIENT_ID 未设置")
        all_set = False
    
    if client_secret:
        print(f"✅ LONGBRIDGE_CLIENT_SECRET: {client_secret[:10]}...{client_secret[-4:]}")
    else:
        print("❌ LONGBRIDGE_CLIENT_SECRET 未设置")
        all_set = False
    
    if access_token:
        print(f"✅ LONGBRIDGE_ACCESS_TOKEN: {access_token[:20]}...{access_token[-10:]}")
    else:
        print("❌ LONGBRIDGE_ACCESS_TOKEN 未设置")
        all_set = False
    
    if all_set:
        print("\n✅ 所有环境变量已正确设置")
        return True
    else:
        print("\n❌ 部分环境变量未设置")
        print("\n请运行以下命令设置环境变量：")
        print("export LONGBRIDGE_CLIENT_ID='your_app_key'")
        print("export LONGBRIDGE_CLIENT_SECRET='your_app_secret'")
        print("export LONGBRIDGE_ACCESS_TOKEN='your_access_token'")
        return False

def test_import_tools():
    """测试2: 测试长桥工具导入"""
    print("\n" + "=" * 60)
    print("🔍 测试2: 导入长桥工具")
    print("=" * 60)
    
    try:
        # 使用相对导入路径（因为已经在src/AiAgentDeep目录下）
        from longbridge_mcp import LongbridgeMCPClient, create_longbridge_client
        print("✅ longbridge_mcp 模块导入成功")
        
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
        print("✅ longbridge_tools 模块导入成功")
        print(f"✅ 共加载 {len([query_stock_quote, query_candlesticks, search_stock, query_account_assets, query_positions, place_stock_order, query_orders, cancel_stock_order])} 个工具")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("\n请确保以下文件存在：")
        print("- longbridge_mcp.py")
        print("- longbridge_tools.py")
        return False
    except Exception as e:
        print(f"❌ 导入时出错: {e}")
        return False

def test_client_creation():
    """测试3: 测试创建客户端"""
    print("\n" + "=" * 60)
    print("🔍 测试3: 创建长桥客户端")
    print("=" * 60)
    
    try:
        from longbridge_mcp import create_longbridge_client
        client = create_longbridge_client()
        
        print("✅ 客户端创建成功")
        print(f"✅ MCP端点: {client.config.mcp_endpoint}")
        print(f"✅ 授权状态: {'已授权' if client._authorized else '未授权'}")
        
        if client._authorized:
            print("✅ Access Token 认证成功，可以直接调用API")
        else:
            print("⚠️  未授权，需要完成OAuth流程")
        
        return client
        
    except Exception as e:
        print(f"❌ 客户端创建失败: {e}")
        return None

def test_quote_query(client):
    """测试4: 测试查询股票行情"""
    print("\n" + "=" * 60)
    print("🔍 测试4: 查询股票行情")
    print("=" * 60)
    
    if not client:
        print("❌ 客户端未创建，跳过此测试")
        return False
    
    try:
        # 测试查询苹果股票
        print("正在查询苹果(AAPL)股票行情...")
        result = client.get_quote("AAPL")
        
        if result:
            print("✅ 行情查询成功")
            print(f"📈 股票: {result.get('name', 'N/A')} ({result.get('symbol', 'N/A')})")
            print(f"💰 价格: ${result.get('last_price', 'N/A')}")
            print(f"📊 涨跌: {result.get('change', 'N/A')} ({result.get('change_percent', 'N/A')}%)")
            print(f"📊 成交量: {result.get('volume', 'N/A'):,}")
            return True
        else:
            print("❌ 行情查询失败，返回结果为空")
            return False
            
    except Exception as e:
        print(f"❌ 行情查询失败: {e}")
        print("\n可能的原因：")
        print("1. Access Token 无效或已过期")
        print("2. 网络连接问题")
        print("3. API端点地址错误")
        return False

def test_tool_functions():
    """测试5: 测试工具函数调用"""
    print("\n" + "=" * 60)
    print("🔍 测试5: 测试工具函数调用")
    print("=" * 60)
    
    try:
        from longbridge_tools import query_stock_quote
        
        print("正在调用 query_stock_quote 工具...")
        # LangChain 工具需要使用 .invoke() 方法调用
        result = query_stock_quote.invoke({"symbol": "AAPL"})
        
        if result and "❌" not in result:
            print("✅ 工具函数调用成功")
            print(f"📊 返回结果预览: {result[:150]}...")
            return True
        else:
            print("❌ 工具函数调用失败")
            print(f"📊 返回结果: {result}")
            return False
            
    except Exception as e:
        print(f"❌ 工具函数调用失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "🚀" * 30)
    print("长桥MCP工具测试")
    print("🚀" * 30)
    
    # 测试1: 环境变量
    env_ok = test_environment_variables()
    if not env_ok:
        print("\n❌ 环境变量未正确设置，测试终止")
        return
    
    # 测试2: 导入工具
    import_ok = test_import_tools()
    if not import_ok:
        print("\n❌ 工具导入失败，测试终止")
        return
    
    # 测试3: 创建客户端
    client = test_client_creation()
    if not client:
        print("\n❌ 客户端创建失败，测试终止")
        return
    
    # 测试4: API调用
    api_ok = test_quote_query(client)
    
    # 测试5: 工具函数
    tool_ok = test_tool_functions()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    results = {
        "环境变量": env_ok,
        "工具导入": import_ok,
        "客户端创建": client is not None,
        "API调用": api_ok,
        "工具函数": tool_ok
    }
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！长桥MCP工具可以正常使用")
    else:
        print("⚠️  部分测试失败，请检查配置")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    main()
