"""
Day4 最终版：对话式多 Agent 简历助手（增加重试和错误处理）
- 增加 LLM 调用重试机制，应对网络波动
- 优化润色工具，空简历时友好提示
- 其他功能不变
"""
import os
import re
import sqlite3
import time
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
    """调用 LLM，失败时自动重试"""
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
    """
    对用户提供的简历进行润色或评估。
    参数:
        resume_text: 用户简历的文本内容（必须包含完整的简历信息）
    返回:
        润色后的简历或改进建议。
    """
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
all_tools = base_tools + [polish_resume]

# ==================== 4. 初始化 LLM ====================
try:
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请先设置 ZHIPUAI_API_KEY 环境变量")

    llm = ChatZhipuAI(
        model="glm-4-flash",
        temperature=0.7,
        max_tokens=1024,
        timeout=120.0,
        api_key=api_key,
    )
    llm_with_tools = llm.bind_tools(all_tools)
    print("✅ LLM 初始化成功，已绑定工具")
except Exception as e:
    print("❌ LLM 初始化失败:", e)
    exit(1)

# ==================== 5. 状态定义 ====================
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

# ==================== 6. 润色子图 ====================
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
    # 使用绑定了基础工具的 LLM（避免子图中调用润色工具）
    response = invoke_llm_with_retry(llm_with_tools, [HumanMessage(content=prompt)])
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"   检测到工具调用: {[tc['name'] for tc in response.tool_calls]}")
        tool_messages = []
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            for t in base_tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
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

# ==================== 7. 主图 ====================
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
    print("\n💬 聊天模式（支持工具调用，包括润色简历）")
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
                    result = t.invoke(tool_args)
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                    break
        try:
            final_response = invoke_llm_with_retry(llm_with_tools, [response] + tool_messages)
        except Exception as e:
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

# ==================== 8. 命令行交互 ====================
def run_cli():
    print("\n" + "="*60)
    print("简历润色助手 V2.0（多 Agent + 工具调用 + 状态持久化）")
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

        current_state = app.get_state(config).values if app.get_state(config) else {}
        if current_state is None:
            current_state = {}

        updated_state = {
            "messages": current_state.get("messages", []) + [HumanMessage(content=user_input)],
            "user_resume": current_state.get("user_resume", ""),
            "polish_subgraph_completed": current_state.get("polish_subgraph_completed", False),
            "next": "",
        }
        try:
            result = app.invoke(updated_state, config=config)
            last_ai = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
            print(f"助手: {last_ai.content}")
        except Exception as e:
            print(f"执行出错: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    run_cli()