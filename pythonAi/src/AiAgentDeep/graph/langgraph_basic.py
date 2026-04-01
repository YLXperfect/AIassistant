"""
Day4 最终版 + MCP 集成（智谱联网搜索服务）
- 对话式多 Agent 简历助手
- MCP 集成（智谱联网搜索服务）
- 自动降级：MCP 不可用时仅使用本地工具
"""
import os
import re
import sqlite3
import time
import asyncio
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.chat_models import ChatZhipuAI
from langchain.tools import tool
import math
import requests
from datetime import datetime
import pytz
import traceback

# ==================== 1. 辅助函数：带重试的 LLM 调用 ====================
def invoke_llm_with_retry(llm, messages, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"⚠️ LLM 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(delay)
    raise Exception("LLM 调用多次失败，请稍后再试")

# ==================== 2. 定义基础工具 ====================
@tool
def search_Weather(city: str) -> str:
    """查询指定城市的天气"""
    try:
        url = f"http://wttr.in/{city}?format=%C+%t"
        response = requests.get(url)
        return response.text.strip() if response.status_code == 200 else "无法获取天气"
    except:
        return "天气查询失败"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持幂运算（^或**）"""
    try:
        safe_expression = expression.replace("^", "**")
        result = eval(safe_expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except:
        return "计算错误，请检查表达式"

@tool
def get_current_time(city: str = "北京") -> str:
    """获取指定城市的当前时间"""
    try:
        tz = pytz.timezone({
            "北京": "Asia/Shanghai",
            "香港": "Asia/Hong_Kong",
            "纽约": "America/New_York"
        }.get(city, "Asia/Shanghai"))
        return datetime.now(tz).strftime("%Y年%m月%d日 %H:%M:%S %Z")
    except:
        return "城市不支持"

# ==================== 3. 润色工具（封装子图）====================
polish_subgraph_instance = None

@tool
def polish_resume(resume_text: str) -> str:
    """对用户提供的简历进行润色或评估。"""
    if not resume_text or not resume_text.strip():
        return "请提供您的简历内容，例如：'润色简历：教育经历...'"

    global polish_subgraph_instance
    if not polish_subgraph_instance:
        return "润色服务暂时不可用，请稍后再试。"

    sub_state = {
        "messages": [],
        "user_resume": resume_text,
        "rules": [],
        "polished_resume": "",
        "score": 0,
        "evaluation": "",
        "suggestion": "",
        "human_decision": "continue",
        "human_feedback": "",
        "polish_subgraph_completed": False,
    }
    try:
        result = polish_subgraph_instance.invoke(sub_state)
        if result.get("score", 0) >= 7:
            return f"润色完成：\n{result.get('polished_resume', '无结果')}"
        else:
            return f"改进建议：\n{result.get('suggestion', '无建议')}"
    except Exception as e:
        return f"润色过程中出错：{str(e)}"

base_tools = [search_Weather, calculator, get_current_time]

# ==================== 4. 初始化 LLM ====================
try:
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请先设置 ZHIPUAI_API_KEY 环境变量")

    llm = ChatZhipuAI(
        model="glm-4.6V",
        temperature=0.7,
        max_tokens=1024,
        timeout=120.0,
        api_key=api_key,
    )
    print("✅ LLM 初始化成功")
except Exception as e:
    print("❌ LLM 初始化失败:", e)
    exit(1)

# ==================== 5. MCP 工具加载（智谱服务）====================
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP 库未安装，跳过 MCP 工具加载")


# def get_mcp_tools():
#     if not MCP_AVAILABLE:
#         return []
#     async def _load():
#         api_key = os.getenv("ZHIPUAI_API_KEY")
#         if not api_key:
#             print("⚠️ 未设置 ZHIPUAI_API_KEY，无法加载智谱 MCP")
#             return []

#         # 尝试两种传输方式
#         for transport in ["sse", "http"]:
#             try:
#                 servers = {
#                     "web-search-prime": {
#                         "url": "https://open.bigmodel.cn/api/mcp/web_search_prime/mcp",
#                         "transport": transport,
#                         "headers": {"Authorization": f"Bearer {api_key}"}
#                     }
#                 }
#                 client = MultiServerMCPClient(servers)
#                 # 增加超时并捕获连接错误
#                 tools = await asyncio.wait_for(client.get_tools(), timeout=15.0)
#                 print(f"✅ 使用 {transport} 传输成功，加载 {len(tools)} 个 MCP 工具")
#                 return tools
#             except asyncio.TimeoutError:
#                 print(f"⚠️ {transport} 传输超时")
#             except Exception as e:
#                 print(f"⚠️ {transport} 传输失败: {e}")
#         return []

#     # 使用新事件循环运行异步函数
#     try:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         return loop.run_until_complete(_load())
#     except Exception as e:
#         print(f"❌ MCP 加载整体失败: {e}")
#         return []
#     finally:
#         loop.close()

# 本地mcp 模拟
def get_mcp_tools():
    if not MCP_AVAILABLE:
        return []
    async def _load():
        # 本地 Mock MCP 服务器（通过 command 启动子进程）
        servers = {
            "mock-news": {
                "command": "python",
                "args": ["mockMcp.py"],   # 确保文件路径正确
                "transport": "stdio",
            }
        }
        try:
            client = MultiServerMCPClient(servers)
            tools = await asyncio.wait_for(client.get_tools(), timeout=10.0)
            print(f"✅ 本地 MCP 服务器加载成功，工具: {[t.name for t in tools]}")
            return tools
        except Exception as e:
            print(f"⚠️ 本地 MCP 加载失败: {e}")
            return []
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_load())
    except Exception as e:
        print(f"❌ MCP 加载整体失败: {e}")
        return []
    finally:
        loop.close()

# 加载 MCP 工具
mcp_tools = get_mcp_tools()

if mcp_tools:
    print(f"✅ 已加载 {len(mcp_tools)} 个 MCP 工具: {[tool.name for tool in mcp_tools]}")
    all_tools = base_tools + [polish_resume] + mcp_tools
else:
    all_tools = base_tools + [polish_resume]

# 绑定工具
llm_with_tools = llm.bind_tools(all_tools)
print("✅ LLM 已绑定工具（含 MCP）")

# ==================== 6. 状态定义 ====================
class ResumeState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_resume: str
    rules: List[str]
    polished_resume: str
    score: int
    evaluation: str
    suggestion: str
    human_decision: str
    human_feedback: str
    polish_subgraph_completed: bool

# ==================== 7. 润色子图 ====================
def check_resume_node(state: ResumeState):
    if not state.get("user_resume", "").strip():
        return {
            "messages": [AIMessage(content="请提供您的简历内容，格式如：'请润色简历：您的简历文本...'")],
            "polish_subgraph_completed": False
        }
    return {}

def plan_node(state: ResumeState):
    print("\n→ 规划润色方向")
    prompt = f"用户简历：{state['user_resume'][:300]}...\n请简要规划本次润色需要重点使用的规则。"
    response = invoke_llm_with_retry(llm, [HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)], "rules": []}

def retrieve_node(state: ResumeState):
    print("\n→ 检索规则")
    sample_rules = [
        "使用 STAR 方法描述经历",
        "所有成就必须量化",
        "使用强行动动词开头"
    ]
    return {"rules": sample_rules, "messages": [AIMessage(content=f"已检索到 {len(sample_rules)} 条规则")]}

def evaluate_node(state: ResumeState):
    print("\n→ 评估简历")
    prompt = f"""你是一位资深HR。请根据以下简历和简历写作规则，对简历质量进行评分（1-10分），并给出简短评价。
规则：{state['rules'][:3] if state['rules'] else '无规则'}
简历：{state['user_resume'][:1000]}

输出格式：
评分：X
评价：..."""
    response = invoke_llm_with_retry(llm, [HumanMessage(content=prompt)])
    score_match = re.search(r'评分[：:]\s*(\d+)', response.content)
    score = int(score_match.group(1)) if score_match else 5
    evaluation = response.content.strip()
    return {"score": score, "evaluation": evaluation, "messages": [AIMessage(content=f"评估结果：{evaluation}")]}

def suggest_node(state: ResumeState):
    print("\n→ 提供改进建议（支持工具调用）")
    prompt = f"""你是简历专家。这份简历得分 {state['score']} 分。请给出具体的改进建议，帮助提升质量。
原始简历：{state['user_resume'][:1000]}
需遵循的规则：{state['rules']}
如果用户提出了要求：{state.get('human_feedback', '无')}，必须按照用户的要求给出具体的改进建议。

同时，你可以在建议中引用当前时间、天气等实时信息来让建议更生动（如果需要，可以调用工具）。
输出详细建议。"""
    response = invoke_llm_with_retry(llm_with_tools, [HumanMessage(content=prompt)])
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"   检测到工具调用: {[tc['name'] for tc in response.tool_calls]}")
        tool_messages = []
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            # 注意：这里工具范围包含所有工具（包括 MCP），使用异步调用
            for t in all_tools:
                if t.name == tool_name:
                    # 使用异步调用，传递空 config 参数
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            # 同步工具的 _arun 内部会调用 _run，传递 config 不会影响结果
                            # 异步工具则需要 config 参数（LangChain MCP 适配器要求）
                            result = loop.run_until_complete(t._arun(config={}, **tool_args))
                        finally:
                            loop.close()
                    except Exception as e:
                        # 直接捕获异常，提供友好的错误提示
                        result = f"工具调用失败: {str(e)}"
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                    break
        final_response = invoke_llm_with_retry(llm_with_tools, [response] + tool_messages)
        suggestion_text = final_response.content
    else:
        suggestion_text = response.content
    return {"suggestion": suggestion_text, "polished_resume": suggestion_text,
            "messages": [AIMessage(content=f"改进建议：{suggestion_text[:200]}...")]}

def refine_node(state: ResumeState):
    print("\n→ 润色简历")
    rules_str = "\n".join(state.get("rules", []))
    prompt = f"""你是一个专业简历润色专家。
规则：
{rules_str}

原始简历：
{state['user_resume']}

请严格按照规则，输出优化后的完整简历版本。"""
    response = invoke_llm_with_retry(llm, [HumanMessage(content=prompt)])
    return {"polished_resume": response.content, "messages": [AIMessage(content=response.content)]}

def human_review_node(state: ResumeState):
    print("\n🔔 人机协同节点")
    decision = state.get("human_decision", "continue")
    return {"human_decision": decision, "human_feedback": state.get("human_feedback", "")}

def route_after_human(state: ResumeState):
    decision = state.get("human_decision", "continue")
    if decision == "continue":
        return "refine" if state.get("score", 0) >= 7 else "suggest"
    else:
        return "suggest"

def build_polish_subgraph():
    subgraph = StateGraph(ResumeState)
    subgraph.add_node("check_resume", check_resume_node)
    subgraph.add_node("plan", plan_node)
    subgraph.add_node("retrieve", retrieve_node)
    subgraph.add_node("evaluate", evaluate_node)
    subgraph.add_node("human_review", human_review_node)
    subgraph.add_node("refine", refine_node)
    subgraph.add_node("suggest", suggest_node)

    def route_after_check(state: ResumeState):
        if not state.get("user_resume", "").strip():
            return END
        return "plan"
    subgraph.add_edge(START, "check_resume")
    subgraph.add_conditional_edges("check_resume", route_after_check, {END: END, "plan": "plan"})
    subgraph.add_edge("plan", "retrieve")
    subgraph.add_edge("retrieve", "evaluate")
    subgraph.add_edge("evaluate", "human_review")
    subgraph.add_conditional_edges("human_review", route_after_human,
                                   {"refine": "refine", "suggest": "suggest"})
    subgraph.add_edge("refine", END)
    subgraph.add_edge("suggest", END)

    return subgraph.compile()

polish_subgraph_instance = build_polish_subgraph()

# ==================== 8. 主图 ====================
class MainState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_resume: str
    rules: List[str]
    polished_resume: str
    score: int
    evaluation: str
    suggestion: str
    human_decision: str
    human_feedback: str
    polish_subgraph_completed: bool
    next: str

def router_node(state: MainState):
    last_msg = state["messages"][-1].content
    if "简历：" in last_msg or "简历：" in last_msg:
        resume_text = last_msg.split("简历：", 1)[1].strip() if "简历：" in last_msg else ""
        return {"next": "polish_subgraph", "user_resume": resume_text}
    else:
        return {"next": "chat"}

def chat_node(state: MainState):
    print("\n💬 聊天模式（支持工具调用，包括润色简历和 MCP 工具）")
    try:
        response = invoke_llm_with_retry(llm_with_tools, state["messages"])
    except Exception as e:
        error_msg = f"抱歉，网络连接出现问题，请稍后再试。错误详情：{str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"   检测到工具调用: {[tc['name'] for tc in response.tool_calls]}")
        tool_messages = []
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            for t in all_tools:
                if t.name == tool_name:
                    # 统一使用异步调用，传递空 config 参数
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            # 同步工具的 _arun 内部会调用 _run，传递 config 不会影响结果
                            # 异步工具则需要 config 参数（LangChain MCP 适配器要求）
                            result = loop.run_until_complete(t._arun(config={}, **tool_args))
                            # 解析工具调用结果，提取实际文本内容
                            if isinstance(result, tuple) and len(result) > 0:
                                # 处理 MCP 工具返回的复杂结构
                                content_list = result[0]
                                if content_list and isinstance(content_list, list):
                                    # 提取文本内容
                                    text_parts = []
                                    for item in content_list:
                                        if isinstance(item, dict) and 'text' in item:
                                            text_parts.append(item['text'])
                                        elif hasattr(item, 'text'):
                                            text_parts.append(item.text)
                                    result = ' '.join(text_parts)
                                else:
                                    result = str(result)
                            elif not isinstance(result, str):
                                result = str(result)
                        finally:
                            loop.close()
                    except Exception as e:
                        # 直接捕获异常，提供友好的错误提示
                        result = f"工具调用失败: {str(e)}"
                    print(f"   处理后的工具调用结果: {result}")
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                    break
        try:
            # 确保消息格式正确
            messages_for_llm = [response] + tool_messages
            # 打印消息格式，用于调试
            print(f"   向 LLM 发送的消息数量: {len(messages_for_llm)}")
            final_response = invoke_llm_with_retry(llm_with_tools, messages_for_llm)
        except Exception as e:
            print(f"   LLM 调用详细错误: {e}")
            final_response = AIMessage(content=f"工具调用后生成回复时出错：{str(e)}")
        return {"messages": [final_response]}
    else:
        return {"messages": [response]}


def polish_subgraph_node(state: MainState):
    print("\n🔄 进入简历润色子图（直接模式）")
    sub_state = {
        "messages": state["messages"],
        "user_resume": state.get("user_resume", ""),
        "rules": [],
        "polished_resume": "",
        "score": 0,
        "evaluation": "",
        "suggestion": "",
        "human_decision": "continue",
        "human_feedback": "",
        "polish_subgraph_completed": False,
    }
    try:
        result = polish_subgraph_instance.invoke(sub_state)
    except Exception as e:
        result = {"messages": [AIMessage(content=f"润色过程中出错：{str(e)}")]}
    return {
        "messages": result["messages"],
        "user_resume": result.get("user_resume", state.get("user_resume", "")),
        "polished_resume": result.get("polished_resume", ""),
        "score": result.get("score", 0),
        "evaluation": result.get("evaluation", ""),
        "suggestion": result.get("suggestion", ""),
        "polish_subgraph_completed": True,
    }

main_workflow = StateGraph(MainState)
main_workflow.add_node("router", router_node)
main_workflow.add_node("chat", chat_node)
main_workflow.add_node("polish_subgraph", polish_subgraph_node)

main_workflow.add_edge(START, "router")
main_workflow.add_conditional_edges(
    "router",
    lambda state: state.get("next", "chat"),
    {"chat": "chat", "polish_subgraph": "polish_subgraph"}
)
main_workflow.add_edge("chat", END)
main_workflow.add_edge("polish_subgraph", END)

conn = sqlite3.connect("main_checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = main_workflow.compile(checkpointer=memory)

# ==================== 9. 命令行交互 ====================
def run_cli():
    print("\n" + "="*60)
    print("简历润色助手 V2.0（多 Agent + 工具调用 + MCP 智谱服务 + 状态持久化）")
    print("支持功能：")
    print("  - 普通聊天 & 工具调用：查询天气、计算数学表达式、获取时间")
    print("  - 简历润色：您可以直接说'帮我润色简历：...'，或通过聊天让助手调用润色工具")
    print("  - 多轮对话记忆：退出后再次运行，输入相同 thread_id 可恢复历史")
    print("="*60)

    thread_id = input("\n请输入会话 ID（默认 user_001）: ").strip() or "user_001"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        snapshot = app.get_state(config)
        current_state = snapshot.values if snapshot else {}
        updated_state = dict(current_state)
        updated_state["messages"] = current_state.get("messages", []) + [HumanMessage(content=user_input)]
        updated_state.setdefault("next", "")

        try:
            result = app.invoke(updated_state, config=config)
            last_ai = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
            print(f"助手: {last_ai.content}")
        except Exception as e:
            print(f"执行出错: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    run_cli()