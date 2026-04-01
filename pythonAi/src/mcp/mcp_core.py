# src/mcp/mcp_core.py
import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

class MCPManager:
    """
    MCP (Model Context Protocol) 管理器
    负责管理模型上下文、状态和配置
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.json")
        self.config = self._load_config()
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.state = {"active_session": None, "last_activity": None}
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载MCP配置
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        """
        return {
            "api_keys": {
                "openai": "",
                "zhipu": ""
            },
            "model_settings": {
                "default_model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 4096
            },
            "context_settings": {
                "max_history": 50,
                "memory_seconds": 3600
            }
        }
    
    def save_config(self) -> bool:
        """
        保存配置到文件
        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            return False
    
    def create_context(self, session_id: str, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建新的上下文
        """
        self.contexts[session_id] = initial_context or {
            "messages": [],
            "metadata": {},
            "last_updated": None
        }
        self.state["active_session"] = session_id
        self.logger.info(f"创建新的MCP上下文: {session_id}")
        return self.contexts[session_id]
    
    def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定会话的上下文
        """
        return self.contexts.get(session_id)
    
    def update_context(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新上下文
        """
        if session_id not in self.contexts:
            self.logger.error(f"会话不存在: {session_id}")
            return False
        
        self.contexts[session_id].update(updates)
        return True
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        向上下文添加消息
        """
        if session_id not in self.contexts:
            self.create_context(session_id)
        
        message = {"role": role, "content": content}
        self.contexts[session_id]["messages"].append(message)
        
        # 限制消息数量
        max_history = self.config.get("context_settings", {}).get("max_history", 50)
        if len(self.contexts[session_id]["messages"]) > max_history:
            self.contexts[session_id]["messages"] = self.contexts[session_id]["messages"][-max_history:]
        
        return True
    
    def clear_context(self, session_id: str) -> bool:
        """
        清除指定会话的上下文
        """
        if session_id in self.contexts:
            del self.contexts[session_id]
            self.logger.info(f"清除MCP上下文: {session_id}")
            return True
        return False
    
    def get_all_sessions(self) -> List[str]:
        """
        获取所有会话ID
        """
        return list(self.contexts.keys())
    
    def get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取模型配置
        """
        settings = self.config.get("model_settings", {})
        if model_name:
            # 可以为特定模型提供配置
            model_specific = settings.get(model_name, {})
            return {**settings, **model_specific}
        return settings
    
    def update_model_config(self, model_name: str, config: Dict[str, Any]) -> bool:
        """
        更新模型配置
        """
        try:
            self.config["model_settings"][model_name] = config
            return self.save_config()
        except Exception as e:
            self.logger.error(f"更新模型配置失败: {e}")
            return False
