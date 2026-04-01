"""
长桥MCP (Model Context Protocol) 客户端模块
用于与长桥证券的MCP服务进行交互，实现股市数据查询和交易功能

功能说明：
1. 支持OAuth 2.1授权流程，无需手动管理API密钥
2. 提供实时行情查询、历史数据获取
3. 支持账户信息查询和模拟交易
4. 集成到LangChain工具体系中

使用前提：
- 需要长桥证券账户并完成实名认证
- 需要在长桥开放平台完成OAuth授权
- 需要安装相关依赖：pip install requests

文档参考：https://open.longbridge.com/docs/mcp
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# 配置日志记录器
logger = logging.getLogger(__name__)


@dataclass
class LongbridgeConfig:
    """
    长桥MCP配置类
    用于存储和管理长桥MCP的连接配置信息
    
    环境变量对应关系（长桥开放平台面板 -> 环境变量名）：
    - App Key -> LONGBRIDGE_CLIENT_ID
    - App Secret -> LONGBRIDGE_CLIENT_SECRET  
    - Access Token -> LONGBRIDGE_ACCESS_TOKEN
    
    Attributes:
        mcp_endpoint: MCP服务端点URL
        client_id: OAuth客户端ID（对应长桥App Key）
        client_secret: OAuth客户端密钥（对应长桥App Secret）
        redirect_uri: OAuth回调地址
        access_token: 访问令牌（对应长桥Access Token）
        refresh_token: 刷新令牌（授权后自动填充）
    """
    mcp_endpoint: str = "https://openapi.longbridge.com/mcp"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    redirect_uri: str = "http://localhost:8000/callback"
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理：从环境变量加载配置"""
        # 如果未提供配置，尝试从环境变量加载
        # 环境变量名对应长桥开放平台的凭证名称
        if not self.client_id:
            self.client_id = os.getenv("LONGBRIDGE_CLIENT_ID")  # 对应 App Key
        if not self.client_secret:
            self.client_secret = os.getenv("LONGBRIDGE_CLIENT_SECRET")  # 对应 App Secret
        if not self.access_token:
            self.access_token = os.getenv("LONGBRIDGE_ACCESS_TOKEN")  # 对应 Access Token


class LongbridgeMCPClient:
    """
    长桥MCP客户端类
    负责与长桥MCP服务进行通信，处理OAuth授权和API调用
    
    主要功能：
    1. OAuth 2.1授权流程管理
    2. 股市数据查询（实时行情、历史数据）
    3. 账户信息查询
    4. 交易操作（查询、下单、改单、撤单）
    
    使用示例：
        client = LongbridgeMCPClient()
        # 授权流程
        auth_url = client.get_authorization_url()
        # ... 用户授权后 ...
        client.exchange_code_for_token(code)
        # 查询行情
        quote = client.get_quote("AAPL")
    """
    
    def __init__(self, config: Optional[LongbridgeConfig] = None):
        """
        初始化长桥MCP客户端
        
        Args:
            config: 长桥配置对象，如果为None则使用默认配置
        """
        self.config = config or LongbridgeConfig()
        self.session = requests.Session()
        
        # 如果配置了Access Token，自动设置为已授权状态
        # 这样可以直接使用长桥开放平台提供的Access Token，无需走OAuth流程
        if self.config.access_token:
            self._authorized = True
            logger.info("✅ 使用Access Token认证，已授权")
        else:
            self._authorized = False
            logger.info("⚠️ 未配置Access Token，需要完成OAuth授权")
        
        logger.info("长桥MCP客户端初始化完成")
    
    def get_authorization_url(self) -> str:
        """
        获取OAuth授权URL
        用户需要访问此URL进行授权
        
        Returns:
            完整的授权URL字符串
            
        Note:
            用户访问此URL后，会跳转到长桥登录页面
            授权完成后，会重定向到配置的redirect_uri
        """
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": "quote trade account"  # 请求的权限范围
        }
        
        # 构建授权URL
        auth_url = f"{self.config.mcp_endpoint}/oauth/authorize"
        query_string = "&".join([f"{k}={v}" for k, v in params.items() if v])
        full_url = f"{auth_url}?{query_string}"
        
        logger.info(f"生成授权URL: {full_url}")
        return full_url
    
    def exchange_code_for_token(self, code: str) -> bool:
        """
        用授权码交换访问令牌
        
        Args:
            code: OAuth授权码（用户授权后从回调URL获取）
            
        Returns:
            是否成功获取令牌
            
        Note:
            成功后会自动保存access_token和refresh_token到配置中
        """
        try:
            token_url = f"{self.config.mcp_endpoint}/oauth/token"
            
            data = {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "redirect_uri": self.config.redirect_uri
            }
            
            response = self.session.post(token_url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.config.access_token = token_data.get("access_token")
            self.config.refresh_token = token_data.get("refresh_token")
            self._authorized = True
            
            logger.info("成功获取访问令牌")
            return True
            
        except Exception as e:
            logger.error(f"获取访问令牌失败: {e}")
            return False
    
    def _make_request(self, method: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        发送JSON-RPC 2.0请求到MCP服务
        
        Args:
            method: 要调用的MCP方法名
            params: 方法参数
            
        Returns:
            API响应的结果数据，失败返回None
            
        Note:
            自动添加Authorization头，使用Bearer Token认证
            使用JSON-RPC 2.0格式进行通信
        """
        if not self._authorized and not self.config.access_token:
            logger.error("未授权，请先完成OAuth授权流程")
            return None
        
        url = f"{self.config.mcp_endpoint}"
        
        headers = {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json"
        }
        
        # 构建JSON-RPC 2.0请求
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1  # 简单使用固定ID
        }
        
        try:
            response = self.session.post(url, headers=headers, json=request_data)
            response.raise_for_status()
            
            response_data = response.json()
            
            # 检查响应是否包含错误
            if "error" in response_data:
                logger.error(f"MCP调用失败: {response_data['error']}")
                return None
            
            # 返回结果
            return response_data.get("result")
            
        except Exception as e:
            logger.error(f"请求失败 {method}: {e}")
            return None
    
    # ==================== 市场数据查询功能 ====================
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取股票实时行情
        
        Args:
            symbol: 股票代码，如 "AAPL"、"00700.HK"
            
        Returns:
            包含行情数据的字典，失败返回None
            
        Example:
            quote = client.get_quote("AAPL")
            # 返回示例：
            # {
            #     "symbol": "AAPL",
            #     "name": "Apple Inc.",
            #     "last_price": 150.25,
            #     "change": 2.5,
            #     "change_percent": 1.69,
            #     "volume": 50000000,
            #     ...
            # }
        """
        logger.info(f"查询股票行情: {symbol}")
        # 使用正确的MCP方法名
        return self._make_request("quote.get", {"symbols": [symbol]})
    
    def get_candlesticks(self, symbol: str, period: str = "day", 
                        count: int = 100) -> Optional[List[Dict]]:
        """
        获取股票K线数据（历史价格）
        
        Args:
            symbol: 股票代码
            period: K线周期，可选值："min"(分钟)、"day"(日)、"week"(周)、"month"(月)
            count: 获取的K线数量
            
        Returns:
            K线数据列表，每个元素包含open/high/low/close/volume等数据
            
        Example:
            candles = client.get_candlesticks("AAPL", period="day", count=30)
        """
        logger.info(f"查询K线数据: {symbol}, 周期: {period}, 数量: {count}")
        params = {"period": period, "count": count}
        return self._make_request(f"/v1/candlesticks/{symbol}", params=params)
    
    def search_stocks(self, keyword: str) -> Optional[List[Dict]]:
        """
        搜索股票
        
        Args:
            keyword: 搜索关键词（股票名称或代码）
            
        Returns:
            匹配的股票列表
            
        Example:
            results = client.search_stocks("苹果")
        """
        logger.info(f"搜索股票: {keyword}")
        params = {"keyword": keyword}
        return self._make_request("/v1/search", params=params)
    
    # ==================== 账户信息查询功能 ====================
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        获取账户信息
        
        Returns:
            账户基本信息，包括账户ID、类型、状态等
        """
        logger.info("查询账户信息")
        return self._make_request("/v1/account")
    
    def get_account_assets(self) -> Optional[Dict[str, Any]]:
        """
        获取账户资产信息
        
        Returns:
            账户资产数据，包括：
            - 总资产 (total_assets)
            - 可用资金 (available_cash)
            - 持仓市值 (position_market_value)
            - 冻结资金 (frozen_cash)
            - 各币种资产详情
        """
        logger.info("查询账户资产")
        return self._make_request("/v1/account/assets")
    
    def get_positions(self) -> Optional[List[Dict]]:
        """
        获取持仓信息
        
        Returns:
            持仓列表，每个持仓包含：
            - 股票代码 (symbol)
            - 持仓数量 (quantity)
            - 成本价 (cost_price)
            - 当前市值 (market_value)
            - 盈亏 (profit_loss)
        """
        logger.info("查询持仓信息")
        return self._make_request("/v1/account/positions")
    
    # ==================== 交易功能 ====================
    
    def place_order(self, symbol: str, side: str, quantity: int, 
                   order_type: str = "MARKET", price: Optional[float] = None) -> Optional[Dict]:
        """
        下单（买入或卖出）
        
        Args:
            symbol: 股票代码
            side: 交易方向，"BUY"(买入) 或 "SELL"(卖出)
            quantity: 交易数量
            order_type: 订单类型，"MARKET"(市价) 或 "LIMIT"(限价)
            price: 限价单的价格（仅当order_type为LIMIT时需要）
            
        Returns:
            订单信息，包含订单ID、状态等
            
        Example:
            # 市价买入100股AAPL
            order = client.place_order("AAPL", "BUY", 100, "MARKET")
            
            # 限价买入100股AAPL，价格150.00
            order = client.place_order("AAPL", "BUY", 100, "LIMIT", 150.00)
            
        Note:
            实盘交易前建议先用模拟账户测试
        """
        logger.info(f"下单: {symbol} {side} {quantity} {order_type}")
        
        data = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type
        }
        
        if order_type == "LIMIT" and price:
            data["price"] = price
        
        return self._make_request("/v1/orders", method="POST", data=data)
    
    def get_orders(self, status: Optional[str] = None) -> Optional[List[Dict]]:
        """
        查询订单列表
        
        Args:
            status: 订单状态过滤，可选值：
                   "PENDING"(待成交)、"FILLED"(已成交)、"CANCELLED"(已取消)
                   为None时返回所有订单
            
        Returns:
            订单列表
        """
        logger.info(f"查询订单列表, 状态: {status}")
        params = {}
        if status:
            params["status"] = status
        return self._make_request("/v1/orders", params=params)
    
    def cancel_order(self, order_id: str) -> Optional[Dict]:
        """
        撤单
        
        Args:
            order_id: 订单ID
            
        Returns:
            撤单结果
        """
        logger.info(f"撤单: {order_id}")
        return self._make_request(f"/v1/orders/{order_id}/cancel", method="POST")
    
    def modify_order(self, order_id: str, quantity: Optional[int] = None, 
                    price: Optional[float] = None) -> Optional[Dict]:
        """
        修改订单
        
        Args:
            order_id: 订单ID
            quantity: 新的数量（不修改则传None）
            price: 新的价格（不修改则传None）
            
        Returns:
            修改后的订单信息
        """
        logger.info(f"修改订单: {order_id}")
        data = {}
        if quantity:
            data["quantity"] = quantity
        if price:
            data["price"] = price
        
        return self._make_request(f"/v1/orders/{order_id}", method="PUT", data=data)


# ==================== 便捷函数 ====================

def create_longbridge_client() -> LongbridgeMCPClient:
    """
    创建长桥MCP客户端的便捷函数
    
    自动从环境变量加载配置：
    - LONGBRIDGE_CLIENT_ID: 客户端ID（对应长桥App Key）
    - LONGBRIDGE_CLIENT_SECRET: 客户端密钥（对应长桥App Secret）
    - LONGBRIDGE_ACCESS_TOKEN: 访问令牌（对应长桥Access Token）
    
    使用方法：
    1. 设置环境变量：
       export LONGBRIDGE_CLIENT_ID="your_app_key"
       export LONGBRIDGE_CLIENT_SECRET="your_app_secret"
       export LONGBRIDGE_ACCESS_TOKEN="your_access_token"
    
    2. 创建客户端：
       client = create_longbridge_client()
       # 如果配置了Access Token，客户端会自动处于已授权状态
       # 可以直接调用API，无需再走OAuth授权流程
    
    Returns:
        配置好的LongbridgeMCPClient实例
        
    Example:
        client = create_longbridge_client()
        # 查询股票行情
        quote = client.get_quote("AAPL")
    """
    config = LongbridgeConfig()
    return LongbridgeMCPClient(config)