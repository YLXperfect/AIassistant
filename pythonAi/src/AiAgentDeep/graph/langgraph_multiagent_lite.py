"""
Day 22: LangGraph 多 Agent Lite - 状态持久化 + 条件路由（测试优化版）
目标：更容易看到循环优化效果
"""

from typing import TypedDict, Annotated, List
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatZhipuAI
import os

print("=== Day 22: LangGraph 持久化 + 条件路由（测试优化版） ===")

class ResumeState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_resume: str
    rules: List[str]
    polished_resume: str
    review_feedback: str
    next_step: str

# ====================== LLM ======================
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置 ZHIPU_API_KEY 环境变量")

llm = ChatZhipuAI(model="glm-4-flash", temperature=0.7, max_tokens=1024, timeout=120.0, api_key=api_key)
print("✅ LLM 初始化成功")

# ====================== Nodes ======================
def plan_node(state: ResumeState) -> ResumeState:
    print("→ [1] plan_node")
    return {"messages": [AIMessage(content="规划：使用STAR方法 + 量化成就 + 强动词")], "next_step": "retrieve"}

def retrieve_node(state: ResumeState) -> ResumeState:
    print("→ [2] retrieve_node")
    rules = ["使用 STAR 方法（Situation, Task, Action, Result）", 
             "所有成就必须量化", 
             "使用强行动动词开头"]
    return {"rules": rules, "messages": [AIMessage(content=f"检索到 {len(rules)} 条核心规则")]}

def generate_node(state: ResumeState) -> ResumeState:
    print("→ [3] generate_node")
    rules_str = "\n".join(state.get("rules", []))
    prompt = f"""你是一位专业的简历优化专家。请严格按照以下规则润色简历：

{rules_str}

用户原始简历：
{state['user_resume']}

请输出优化后的完整简历（使用Markdown格式）。"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "polished_resume": response.content,
        "messages": [AIMessage(content=response.content)],
        "next_step": "review"
    }

def review_node(state: ResumeState) -> ResumeState:
    print("→ [4] review_node")
    prompt = f"""你是一个非常挑剔的简历评审专家。请严格评审以下润色结果：

{state.get('polished_resume', '')}

请给出详细反馈。最后**必须**明确写出结论：
- 如果已经足够优秀，请写：【通过】
- 如果还有明显提升空间，请写：【需要优化】并指出至少2个具体改进点"""

    response = llm.invoke([HumanMessage(content=prompt)])
    feedback = response.content
    
    next_step = "END" if "【通过】" in feedback or "通过" in feedback else "generate"
    print(f"   review 决策 → {next_step}")
    
    return {
        "review_feedback": feedback,
        "messages": [AIMessage(content=feedback)],
        "next_step": next_step
    }

def should_continue(state: ResumeState) -> str:
    decision = state.get("next_step", "END")
    print(f"   → 条件路由决策: {decision}")
    return decision

# ====================== Graph 构建 ======================
workflow = StateGraph(ResumeState)

workflow.add_node("plan", plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("review", review_node)

# 固定流程
workflow.add_edge(START, "plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "review")

# 条件路由（核心）
workflow.add_conditional_edges(
    "review",
    should_continue,
    {"generate": "generate", "END": END}
)

# ====================== 持久化 ======================
DB_PATH = "checkpointer.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
print(f"✅ SQLite Checkpointer 初始化成功")

app = workflow.compile(checkpointer=checkpointer)

# ====================== 测试（使用更长的简历） ======================
if __name__ == "__main__":
    test_resume = """
    教育经历：香港理工大学 计算机科学本科 2022-2026
    项目经验：参与过机器学习图像识别项目，主要负责数据处理部分。
    实习经历：在一家科技公司做过三个月实习，主要写代码。
    """

    initial_state = {
        "messages": [HumanMessage(content="请帮我润色这份简历")],
        "user_resume": test_resume,
        "rules": [],
        "polished_resume": "",
        "review_feedback": "",
        "next_step": ""
    }

    config = {"configurable": {"thread_id": "resume_session_001"}}

    print("\n=== 开始第1次运行 ===\n")
    result = app.invoke(initial_state, config=config)
    
    print("\n=== 第1次运行最终结果 ===")
    print("润色简历：\n", result.get("polished_resume", "")[:800])
    print("\n审核反馈：\n", result.get("review_feedback", "")[-600:])
    
    # 可选：再跑一次看看是否触发循环
    print("\n=== 第2次运行（继续优化） ===")
    result2 = app.invoke(
        {"messages": [HumanMessage(content="请根据审核反馈继续优化")]},
        config=config
    )
    
    print("最终润色版本：\n", result2.get("polished_resume", "")[:600])