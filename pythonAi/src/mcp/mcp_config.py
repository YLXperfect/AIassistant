# src/mcp/mcp_config.py
import os
import json
import logging
from typing import Dict, Any, Optional

class MCPConfig:
    """
    MCP (Model Context Protocol) 配置管理
    负责管理MCP的配置信息
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.json")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
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
                "openai": os.getenv("OPENAI_API_KEY", ""),
                "zhipu": os.getenv("ZHIPU_API_KEY", "")
            },
            "model_settings": {
                "default_model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "context_settings": {
                "max_history": 50,
                "memory_seconds": 3600,
                "auto_save": True
            },
            "api_settings": {
                "timeout": 30,
                "retry_attempts": 3,
                "base_url": None
            },
            "tools": {
                "enabled": True,
                "list": ["search", "calculator", "file_reader"]
            }
        }
    
    def save(self) -> bool:
        """
        保存配置到文件
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """
        设置配置值
        """
        try:
            keys = key.split(".")
            config = self.config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            return True
        except Exception as e:
            self.logger.error(f"设置配置失败: {e}")
            return False
    
    def get_api_key(self, provider: str) -> str:
        """
        获取API密钥
        """
        return self.get(f"api_keys.{provider}", "") or os.getenv(f"{provider.upper()}_API_KEY", "")
    
    def set_api_key(self, provider: str, key: str) -> bool:
        """
        设置API密钥
        """
        return self.set(f"api_keys.{provider}", key)
    
    def get_model_setting(self, setting: str, default: Any = None) -> Any:
        """
        获取模型设置
        """
        return self.get(f"model_settings.{setting}", default)
    
    def set_model_setting(self, setting: str, value: Any) -> bool:
        """
        设置模型设置
        """
        return self.set(f"model_settings.{setting}", value)
    
    def get_context_setting(self, setting: str, default: Any = None) -> Any:
        """
        获取上下文设置
        """
        return self.get(f"context_settings.{setting}", default)
    
    def set_context_setting(self, setting: str, value: Any) -> bool:
        """
        设置上下文设置
        """
        return self.set(f"context_settings.{setting}", value)
    
    def get_api_setting(self, setting: str, default: Any = None) -> Any:
        """
        获取API设置
        """
        return self.get(f"api_settings.{setting}", default)
    
    def set_api_setting(self, setting: str, value: Any) -> bool:
        """
        设置API设置
        """
        return self.set(f"api_settings.{setting}", value)
    
    def get_tool_list(self) -> list:
        """
        获取工具列表
        """
        return self.get("tools.list", [])
    
    def add_tool(self, tool_name: str) -> bool:
        """
        添加工具
        """
        tools = self.get_tool_list()
        if tool_name not in tools:
            tools.append(tool_name)
            return self.set("tools.list", tools)
        return True
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        移除工具
        """
        tools = self.get_tool_list()
        if tool_name in tools:
            tools.remove(tool_name)
            return self.set("tools.list", tools)
        return True
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """
        检查工具是否启用
        """
        return tool_name in self.get_tool_list() and self.get("tools.enabled", True)
