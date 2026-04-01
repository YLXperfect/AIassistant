#!/usr/bin/env python3
"""
简单的 MCP 服务器（stdio 传输）
提供 get_news 工具，返回模拟新闻。
"""

import asyncio
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# 创建服务器实例
server = Server("mock-news-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """列出可用的工具"""
    return [
        types.Tool(
            name="get_news",
            description="获取最新的新闻摘要（模拟）",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "新闻主题，如 '科技', '股市', '国际'"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """执行工具调用"""
    if name == "get_news":
        topic = arguments.get("topic", "综合") if arguments else "综合"
        # 模拟不同的新闻内容
        if "股市" in topic:
            news = "今日美股三大指数小幅上涨，道指涨0.3%，纳指涨0.5%。"
        elif "国际" in topic:
            news = "国际局势方面，多国举行会谈，呼吁加强合作。"
        else:
            news = f"【{topic}】最新消息：技术发展迅速，创新不断。"
        return [types.TextContent(type="text", text=news)]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mock-news-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())