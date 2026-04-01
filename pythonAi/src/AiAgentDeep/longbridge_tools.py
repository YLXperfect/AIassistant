"""
长桥MCP工具模块
将长桥MCP客户端的功能封装为LangChain工具，供Agent调用

本模块提供了以下工具：
1. 查询股票实时行情
2. 查询K线历史数据
3. 搜索股票
4. 查询账户资产
5. 查询持仓信息
6. 下单交易
7. 查询订单
8. 撤单

使用说明：
- 需要先完成长桥OAuth授权
- 建议在模拟账户中测试后再进行实盘交易
- 交易操作有风险，请谨慎使用
"""

from langchain.tools import tool
from typing import Optional
import logging

# 导入长桥MCP客户端（支持相对导入和绝对导入）
try:
    from .longbridge_mcp import LongbridgeMCPClient, create_longbridge_client
except ImportError:
    from longbridge_mcp import LongbridgeMCPClient, create_longbridge_client

# 配置日志
logger = logging.getLogger(__name__)

# 全局客户端实例（延迟初始化）
_longbridge_client: Optional[LongbridgeMCPClient] = None


def get_longbridge_client() -> LongbridgeMCPClient:
    """
    获取长桥MCP客户端实例（单例模式）
    
    使用单例模式确保整个应用中只有一个客户端实例，
    避免重复创建和多次授权
    
    Returns:
        LongbridgeMCPClient实例
    """
    global _longbridge_client
    if _longbridge_client is None:
        _longbridge_client = create_longbridge_client()
        logger.info("创建长桥MCP客户端实例")
    return _longbridge_client


# ==================== 市场数据查询工具 ====================

@tool
def query_stock_quote(symbol: str) -> str:
    """
    查询股票实时行情数据
    
    获取指定股票的最新价格、涨跌幅、成交量等实时行情信息
    
    Args:
        symbol: 股票代码，例如：
               - 美股："AAPL"（苹果）、"TSLA"（特斯拉）
               - 港股："00700.HK"（腾讯）、"09988.HK"（阿里）
               - A股："600519.SH"（茅台）
    
    Returns:
        格式化的行情信息字符串
        
    Example:
        >>> query_stock_quote("AAPL")
        '苹果(AAPL)最新行情：\n当前价格: $150.25\n涨跌: +$2.50 (+1.69%)\n成交量: 50,000,000股\n...'
    """
    try:
        client = get_longbridge_client()
        quote = client.get_quote(symbol)
        
        if not quote:
            return f"❌ 未能获取 {symbol} 的行情数据，请检查股票代码是否正确"
        
        # 格式化行情数据
        result = f"📈 {quote.get('name', symbol)}({symbol}) 实时行情\n"
        result += "=" * 40 + "\n"
        result += f"💰 最新价格: ${quote.get('last_price', 'N/A')}\n"
        result += f"📊 涨跌额: {quote.get('change', 'N/A')}\n"
        result += f"📈 涨跌幅: {quote.get('change_percent', 'N/A')}%\n"
        result += f"📊 成交量: {quote.get('volume', 'N/A'):,}股\n"
        result += f"🔼 最高价: ${quote.get('high', 'N/A')}\n"
        result += f"🔽 最低价: ${quote.get('low', 'N/A')}\n"
        result += f"💵 开盘价: ${quote.get('open', 'N/A')}\n"
        result += f"📅 昨收价: ${quote.get('prev_close', 'N/A')}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"查询行情失败: {e}")
        return f"❌ 查询行情时出错: {str(e)}"


@tool
def query_candlesticks(symbol: str, period: str = "day", count: int = 30) -> str:
    """
    查询股票K线历史数据
    
    获取指定股票的历史价格数据，用于技术分析
    
    Args:
        symbol: 股票代码，例如 "AAPL"、"00700.HK"
        period: K线周期，可选值：
               - "min": 分钟线
               - "day": 日线（默认）
               - "week": 周线
               - "month": 月线
        count: 获取的K线数量，默认30条
    
    Returns:
        格式化的K线数据字符串
        
    Example:
        >>> query_candlesticks("AAPL", period="day", count=5)
        'AAPL 最近5天K线数据：\n2024-01-15: 开150.00 高155.00 低149.00 收152.50 量10000000\n...'
    """
    try:
        client = get_longbridge_client()
        candles = client.get_candlesticks(symbol, period, count)
        
        if not candles:
            return f"❌ 未能获取 {symbol} 的K线数据"
        
        # 格式化K线数据
        period_name = {"min": "分钟", "day": "日", "week": "周", "month": "月"}
        result = f'📊 {symbol} 最近{len(candles)}个{period_name.get(period, period)}K线\n'
        result += "=" * 50 + "\n"
        
        # 只显示最近10条数据，避免过长
        display_candles = candles[-10:] if len(candles) > 10 else candles
        
        for candle in display_candles:
            date = candle.get('timestamp', 'N/A')
            open_price = candle.get('open', 'N/A')
            high = candle.get('high', 'N/A')
            low = candle.get('low', 'N/A')
            close = candle.get('close', 'N/A')
            volume = candle.get('volume', 0)
            
            result += f"📅 {date}\n"
            result += f"   开: ${open_price} | 高: ${high} | 低: ${low} | 收: ${close}\n"
            result += f"   成交量: {volume:,}股\n"
            result += "-" * 40 + "\n"
        
        if len(candles) > 10:
            result += f"... (共{len(candles)}条数据，显示最近10条)\n"
        
        return result
        
    except Exception as e:
        logger.error(f"查询K线失败: {e}")
        return f"❌ 查询K线数据时出错: {str(e)}"


@tool
def search_stock(keyword: str) -> str:
    """
    搜索股票
    
    根据关键词搜索股票，支持股票名称或代码搜索
    
    Args:
        keyword: 搜索关键词，例如：
                - 股票名称："苹果"、"腾讯"、"阿里巴巴"
                - 股票代码："AAPL"、"00700"
    
    Returns:
        搜索结果列表字符串
        
    Example:
        >>> search_stock("苹果")
        '搜索 "苹果" 的结果：\n1. AAPL - Apple Inc. (美股)\n2. ...'
    """
    try:
        client = get_longbridge_client()
        results = client.search_stocks(keyword)
        
        if not results:
            return f'❌ 未找到与 "{keyword}" 相关的股票'
        
        # 格式化搜索结果
        result = f'🔍 搜索 "{keyword}" 的结果\n'
        result += "=" * 40 + "\n"
        
        for i, stock in enumerate(results[:10], 1):  # 最多显示10条
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')
            exchange = stock.get('exchange', 'N/A')
            
            result += f"{i}. {symbol} - {name} ({exchange})\n"
        
        if len(results) > 10:
            result += f"\n... (共找到{len(results)}条结果，显示前10条)\n"
        
        return result
        
    except Exception as e:
        logger.error(f"搜索股票失败: {e}")
        return f"❌ 搜索股票时出错: {str(e)}"


# ==================== 账户信息查询工具 ====================

@tool
def query_account_assets() -> str:
    """
    查询账户资产信息
    
    获取当前账户的资产概况，包括总资产、可用资金、持仓市值等
    
    Returns:
        格式化的账户资产信息字符串
        
    Example:
        >>> query_account_assets()
        '💼 账户资产概况\n总资产: $100,000.00\n可用资金: $50,000.00\n...'
    """
    try:
        client = get_longbridge_client()
        assets = client.get_account_assets()
        
        if not assets:
            return "❌ 未能获取账户资产信息，请检查是否已完成授权"
        
        # 格式化资产信息
        result = "💼 账户资产概况\n"
        result += "=" * 40 + "\n"
        result += f'💰 总资产: ${assets.get("total_assets", 0):,.2f}\n'
        result += f'💵 可用资金: ${assets.get("available_cash", 0):,.2f}\n'
        result += f'📊 持仓市值: ${assets.get("position_market_value", 0):,.2f}\n'
        result += f'❄️ 冻结资金: ${assets.get("frozen_cash", 0):,.2f}\n'
        
        # 显示各币种资产
        currencies = assets.get('currencies', [])
        if currencies:
            result += "\n🌍 各币种资产:\n"
            for curr in currencies:
                currency = curr.get('currency', 'N/A')
                total = curr.get('total', 0)
                available = curr.get('available', 0)
                result += f'   {currency}: 总额 {total:,.2f}, 可用 {available:,.2f}\n'
        
        return result
        
    except Exception as e:
        logger.error(f"查询账户资产失败: {e}")
        return f"❌ 查询账户资产时出错: {str(e)}"


@tool
def query_positions() -> str:
    """
    查询持仓信息
    
    获取当前账户的所有持仓股票信息
    
    Returns:
        格式化的持仓列表字符串
        
    Example:
        >>> query_positions()
        '📈 当前持仓\n1. AAPL - 100股 @ $150.00\n   市值: $15,000.00 | 盈亏: +$500.00\n...'
    """
    try:
        client = get_longbridge_client()
        positions = client.get_positions()
        
        if not positions:
            return "📭 当前没有持仓"
        
        # 格式化持仓信息
        result = f'📈 当前持仓 ({len(positions)}只股票)\n'
        result += "=" * 50 + "\n"
        
        total_value = 0
        total_profit = 0
        
        for i, pos in enumerate(positions, 1):
            symbol = pos.get('symbol', 'N/A')
            name = pos.get('name', symbol)
            quantity = pos.get('quantity', 0)
            cost_price = pos.get('cost_price', 0)
            current_price = pos.get('current_price', 0)
            market_value = pos.get('market_value', 0)
            profit = pos.get('profit_loss', 0)
            profit_pct = pos.get('profit_loss_percent', 0)
            
            total_value += market_value
            total_profit += profit
            
            # 盈亏表情符号
            profit_emoji = "🟢" if profit >= 0 else "🔴"
            
            result += f'{i}. {name} ({symbol})\n'
            result += f'   📊 持仓: {quantity}股 @ ${cost_price:.2f}\n'
            result += f'   💰 现价: ${current_price:.2f}\n'
            result += f'   📈 市值: ${market_value:,.2f}\n'
            result += f'   {profit_emoji} 盈亏: ${profit:,.2f} ({profit_pct:+.2f}%)\n'
            result += "-" * 40 + "\n"
        
        # 汇总信息
        result += f'\n💼 持仓汇总\n'
        result += f'   总市值: ${total_value:,.2f}\n'
        profit_emoji = "🟢" if total_profit >= 0 else "🔴"
        result += f'   {profit_emoji} 总盈亏: ${total_profit:,.2f}\n'
        
        return result
        
    except Exception as e:
        logger.error(f"查询持仓失败: {e}")
        return f"❌ 查询持仓时出错: {str(e)}"


# ==================== 交易工具 ====================

@tool
def place_stock_order(symbol: str, side: str, quantity: int, 
                     order_type: str = "MARKET", price: Optional[float] = None) -> str:
    """
    股票交易下单
    
    买入或卖出指定数量的股票
    
    ⚠️ 风险提示：交易操作有风险，请谨慎使用！
    ⚠️ 建议先在模拟账户中测试
    
    Args:
        symbol: 股票代码，例如 "AAPL"、"00700.HK"
        side: 交易方向，"BUY"(买入) 或 "SELL"(卖出)
        quantity: 交易数量（股数）
        order_type: 订单类型，可选值：
                   - "MARKET": 市价单（按当前市场价格成交）
                   - "LIMIT": 限价单（按指定价格成交）
        price: 限价单的价格（仅当order_type为LIMIT时需要）
    
    Returns:
        订单提交结果字符串
        
    Example:
        >>> place_stock_order("AAPL", "BUY", 100, "MARKET")
        '✅ 订单提交成功\n订单ID: ORDER123456\n股票: AAPL\n方向: 买入\n数量: 100股\n类型: 市价单'
        
        >>> place_stock_order("AAPL", "BUY", 100, "LIMIT", 150.00)
        '✅ 订单提交成功\n订单ID: ORDER123457\n股票: AAPL\n方向: 买入\n数量: 100股\n类型: 限价单\n价格: $150.00'
    """
    try:
        # 参数验证
        if side not in ["BUY", "SELL"]:
            return "❌ 交易方向错误，请使用 'BUY'(买入) 或 'SELL'(卖出)"
        
        if quantity <= 0:
            return "❌ 交易数量必须大于0"
        
        if order_type not in ["MARKET", "LIMIT"]:
            return "❌ 订单类型错误，请使用 'MARKET'(市价) 或 'LIMIT'(限价)"
        
        if order_type == "LIMIT" and price is None:
            return "❌ 限价单必须指定价格"
        
        client = get_longbridge_client()
        order = client.place_order(symbol, side, quantity, order_type, price)
        
        if not order:
            return "❌ 订单提交失败，请检查账户状态和资金"
        
        # 格式化订单结果
        side_name = "买入" if side == "BUY" else "卖出"
        type_name = "市价单" if order_type == "MARKET" else f"限价单(${price})"
        
        result = "✅ 订单提交成功\n"
        result += "=" * 40 + "\n"
        result += f"📝 订单ID: {order.get('order_id', 'N/A')}\n"
        result += f"📈 股票: {symbol}\n"
        result += f"🔄 方向: {side_name}\n"
        result += f"📊 数量: {quantity}股\n"
        result += f"💰 类型: {type_name}\n"
        result += f"⏱️ 状态: {order.get('status', 'N/A')}\n"
        result += f"📅 时间: {order.get('create_time', 'N/A')}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"下单失败: {e}")
        return f"❌ 下单时出错: {str(e)}"


@tool
def query_orders(status: Optional[str] = None) -> str:
    """
    查询订单列表
    
    获取当前账户的订单信息
    
    Args:
        status: 订单状态过滤，可选值：
               - "PENDING": 待成交
               - "FILLED": 已成交
               - "CANCELLED": 已取消
               - None: 所有订单（默认）
    
    Returns:
        格式化的订单列表字符串
        
    Example:
        >>> query_orders("PENDING")
        '📋 待成交订单 (2条)\n1. ORDER123 - AAPL BUY 100股 @ MARKET\n...'
    """
    try:
        client = get_longbridge_client()
        orders = client.get_orders(status)
        
        if not orders:
            status_name = status if status else "所有"
            return f"📭 没有找到{status_name}订单"
        
        # 状态名称映射
        status_map = {
            "PENDING": "待成交",
            "FILLED": "已成交",
            "CANCELLED": "已取消"
        }
        
        status_name = status_map.get(status, "所有")
        result = f"📋 {status_name}订单 ({len(orders)}条)\n"
        result += "=" * 50 + "\n"
        
        for i, order in enumerate(orders[:20], 1):  # 最多显示20条
            order_id = order.get('order_id', 'N/A')
            symbol = order.get('symbol', 'N/A')
            side = order.get('side', 'N/A')
            quantity = order.get('quantity', 0)
            order_type = order.get('order_type', 'N/A')
            price = order.get('price', 'N/A')
            order_status = order.get('status', 'N/A')
            create_time = order.get('create_time', 'N/A')
            
            side_name = "买入" if side == "BUY" else "卖出"
            
            result += f"{i}. {order_id}\n"
            result += f"   📈 {symbol} {side_name} {quantity}股\n"
            if order_type == "LIMIT":
                result += f"   💰 限价: ${price}\n"
            else:
                result += f"   💰 市价单\n"
            result += f"   ⏱️ 状态: {status_map.get(order_status, order_status)}\n"
            result += f"   📅 时间: {create_time}\n"
            result += "-" * 40 + "\n"
        
        if len(orders) > 20:
            result += f"... (共{len(orders)}条订单，显示前20条)\n"
        
        return result
        
    except Exception as e:
        logger.error(f"查询订单失败: {e}")
        return f"❌ 查询订单时出错: {str(e)}"


@tool
def cancel_stock_order(order_id: str) -> str:
    """
    撤销订单
    
    取消指定ID的未成交订单
    
    Args:
        order_id: 订单ID
    
    Returns:
        撤单结果字符串
        
    Example:
        >>> cancel_stock_order("ORDER123456")
        '✅ 订单撤销成功\n订单ID: ORDER123456\n状态: 已取消'
    """
    try:
        if not order_id:
            return "❌ 订单ID不能为空"
        
        client = get_longbridge_client()
        result = client.cancel_order(order_id)
        
        if not result:
            return f"❌ 撤销订单 {order_id} 失败"
        
        return f"✅ 订单撤销成功\n订单ID: {order_id}\n状态: 已取消"
        
    except Exception as e:
        logger.error(f"撤单失败: {e}")
        return f"❌ 撤单时出错: {str(e)}"


# ==================== 工具列表导出 ====================

# 所有长桥工具的列表，供agent使用
LONGBRIDGE_TOOLS = [
    query_stock_quote,      # 查询股票行情
    query_candlesticks,     # 查询K线数据
    search_stock,           # 搜索股票
    query_account_assets,   # 查询账户资产
    query_positions,        # 查询持仓
    place_stock_order,      # 下单交易
    query_orders,           # 查询订单
    cancel_stock_order,     # 撤单
]