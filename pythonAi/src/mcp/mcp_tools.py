# src/mcp/mcp_tools.py
import os
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

class MCPClient:
    """
    MCP (Model Context Protocol) 客户端
    负责与模型进行交互，处理请求和响应
    """
    
    def __init__(self, mcp_manager):
        self.logger = logging.getLogger(__name__)
        self.mcp_manager = mcp_manager
        self.llm_cache = {}
    
    def _get_llm(self, model_name: str = None) -> Optional[ChatOpenAI]:
        """
        获取语言模型实例
        """
        if model_name in self.llm_cache:
            return self.llm_cache[model_name]
        
        model_config = self.mcp_manager.get_model_config(model_name)
        api_key = os.getenv("OPENAI_API_KEY") or self.mcp_manager.config.get("api_keys", {}).get("openai", "")
        
        if not api_key:
            self.logger.error("未找到OpenAI API密钥")
            return None
        
        try:
            llm = ChatOpenAI(
                model_name=model_config.get("default_model", "gpt-4o-mini"),
                temperature=model_config.get("temperature", 0.1),
                max_tokens=model_config.get("max_tokens", 4096),
                api_key=api_key
            )
            self.llm_cache[model_name] = llm
            return llm
        except Exception as e:
            self.logger.error(f"初始化语言模型失败: {e}")
            return None
    
    def generate_response(self, session_id: str, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        生成模型响应
        """
        try:
            # 获取或创建上下文
            context = self.mcp_manager.get_context(session_id)
            if not context:
                context = self.mcp_manager.create_context(session_id)
            
            # 构建消息列表
            messages = []
            
            # 添加系统提示
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            # 添加历史消息
            for msg in context.get("messages", []):
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            # 添加当前提示
            messages.append(HumanMessage(content=prompt))
            
            # 获取语言模型
            llm = self._get_llm()
            if not llm:
                return "模型初始化失败，请检查API密钥"
            
            # 生成响应
            response = llm.invoke(messages)
            answer = response.content
            
            # 更新上下文
            self.mcp_manager.add_message(session_id, "user", prompt)
            self.mcp_manager.add_message(session_id, "assistant", answer)
            
            return answer
        except Exception as e:
            self.logger.error(f"生成响应失败: {e}")
            return f"生成响应时出错: {str(e)}"
    
    async def generate_response_stream(self, session_id: str, prompt: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        流式生成模型响应
        """
        try:
            # 获取或创建上下文
            context = self.mcp_manager.get_context(session_id)
            if not context:
                context = self.mcp_manager.create_context(session_id)
            
            # 构建消息列表
            messages = []
            
            # 添加系统提示
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            # 添加历史消息
            for msg in context.get("messages", []):
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            # 添加当前提示
            messages.append(HumanMessage(content=prompt))
            
            # 获取语言模型
            llm = self._get_llm()
            if not llm:
                yield "模型初始化失败，请检查API密钥"
                return
            
            # 流式生成响应
            full_response = ""
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    full_response += chunk.content
            
            # 更新上下文
            self.mcp_manager.add_message(session_id, "user", prompt)
            self.mcp_manager.add_message(session_id, "assistant", full_response)
            
        except Exception as e:
            self.logger.error(f"流式生成响应失败: {e}")
            yield f"生成响应时出错: {str(e)}"
    
    def generate_with_tools(self, session_id: str, prompt: str, tools: list = None, system_prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        使用工具生成响应
        """
        try:
            # 这里可以实现与工具的集成
            # 目前返回基本响应
            response = self.generate_response(session_id, prompt, system_prompt)
            return {"response": response, "tool_calls": []}
        except Exception as e:
            self.logger.error(f"使用工具生成响应失败: {e}")
            return {"response": f"生成响应时出错: {str(e)}", "tool_calls": []}
    
    def get_session_summary(self, session_id: str) -> Optional[str]:
        """
        获取会话摘要
        """
        try:
            context = self.mcp_manager.get_context(session_id)
            if not context:
                return "会话不存在"
            
            messages = context.get("messages", [])
            if not messages:
                return "会话为空"
            
            # 构建摘要提示
            summary_prompt = "请为以下对话生成一个简短的摘要：\n"
            for msg in messages:
                role = "用户" if msg["role"] == "user" else "助手"
                summary_prompt += f"{role}: {msg['content']}\n"
            
            # 生成摘要
            llm = self._get_llm()
            if not llm:
                return "模型初始化失败"
            
            summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
            return summary
        except Exception as e:
            self.logger.error(f"获取会话摘要失败: {e}")
            return f"获取摘要时出错: {str(e)}"
