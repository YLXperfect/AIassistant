# src/AiAgentDeep/langgraph_basic.py
"""
第1天：LangGraph 状态图基础
- 定义状态结构
- 构建顺序执行图
- 观察状态流转
"""
from typing import TypedDict, Annotated,List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatZhipuAI
import os
import traceback




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
    print("✅ LLM 初始化成功")
except Exception as e:
    print("❌ LLM 初始化失败:")
    print(traceback.format_exc())
    exit(1)

# 1. 定义状态结构（可以在节点间共享）
class ResumeState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_resume: str
    rules: List[str]
    polished_resume: str
    review_feedback: str
    next_step: str
    # 新增字段
    score: int               # 评估分数 1-10
    evaluation: str          # 评估意见
    suggestion: str          # 改进建议（评分低时使用）

# ====================== Nodes（节点函数） ======================
def plan_node(state: ResumeState) -> ResumeState:
    print("→ 执行 plan_node")
    prompt = f"用户简历片段：{state['user_resume'][:300]}...\n请简要规划本次润色需要重点使用的规则。"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content=response.content)],
        "next_step": "retrieve"
    }

def retrieve_node(state: ResumeState) -> ResumeState:
    print("→ 执行 retrieve_node")
    sample_rules = [
        "使用 STAR 方法描述经历",
        "所有成就必须量化",
        "使用强行动动词开头"
    ]
    return {
        "rules": sample_rules,
        "messages": [AIMessage(content=f"已检索到 {len(sample_rules)} 条规则")]
    }

#评估节点
def evaluate_node(state: ResumeState) -> ResumeState:
    """评估节点：根据原始简历和规则打分"""
    prompt = f"""你是一位资深HR。请根据以下简历和简历写作规则，对简历质量进行评分（1-10分），并给出简短评价。
规则：{state['rules'][:3] if state['rules'] else '无规则'}
简历：{state['user_resume'][:1000]}

输出格式：
评分：X
评价：..."""
    response = llm.invoke([HumanMessage(content=prompt)])
    # 简单解析评分
    import re
    score_match = re.search(r'评分[：:]\s*(\d+)', response.content)
    score = int(score_match.group(1)) if score_match else 5
    evaluation = response.content.strip()
    return {
        "score": score,
        "evaluation": evaluation,
        "messages": [AIMessage(content=f"评估结果：{evaluation}")]
    }

def suggest_node(state: ResumeState) -> ResumeState:
    """改进建议节点"""
    prompt = f"""你是简历专家。这份简历得分 {state['score']} 分。请给出具体的改进建议，帮助提升质量。
原始简历：{state['user_resume'][:1000]}
需遵循的规则：{state['rules']}
输出详细建议。"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "suggestion": response.content,
        "polished_resume": response.content,  # 复用字段存储建议
        "messages": [AIMessage(content=f"改进建议：{response.content[:200]}...")]
    }

def refine_node(state: ResumeState) -> ResumeState:
    """润色节点（高分时）"""
    rules_str = "\n".join(state.get("rules", []))
    prompt = f"""你是一个专业简历润色专家。
规则：
{rules_str}

原始简历：
{state['user_resume']}

请严格按照规则，输出优化后的完整简历版本。"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "polished_resume": response.content,
        "messages": [AIMessage(content=response.content)]
    }

#构建图
workflow = StateGraph(ResumeState)

# 添加节点
workflow.add_node("plan", plan_node)          # 规划（保留）
workflow.add_node("retrieve", retrieve_node)  # 检索
workflow.add_node("evaluate", evaluate_node)  # 评估
workflow.add_node("refine", refine_node)      # 润色
workflow.add_node("suggest", suggest_node)    # 建议

# 添加边
workflow.add_edge(START, "plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "evaluate")

# 条件边：根据分数决定下一步
def route_after_evaluate(state: ResumeState) -> str:
    if state.get("score", 0) >= 7:
        return "refine"
    else:
        return "suggest"

workflow.add_conditional_edges(
    "evaluate",
    route_after_evaluate,
    {
        "refine": "refine",
        "suggest": "suggest"
    }
)

# 最终输出
workflow.add_edge("refine", END)
workflow.add_edge("suggest", END)

app = workflow.compile()


if __name__ == "__main__":
    test_resume = """
    教育经历：
    香港理工大学 本科 计算机科学 2022-2026
    参与过一些项目，负责部分开发工作。
    """
    initial_state = {
        "messages": [HumanMessage(content="请帮我评估并润色简历")],
        "user_resume": test_resume,
        "rules": [],
        "polished_resume": "",
        "review_feedback": "",
        "next_step": "",
        "score": 0,
        "evaluation": "",
        "suggestion": ""
    }

    result = app.invoke(initial_state)
    print("最终输出：")
    if result.get("score", 0) >= 7:
        print("润色后简历：\n", result.get("polished_resume", ""))
    else:
        print("改进建议：\n", result.get("suggestion", ""))
    print("评估详情：", result.get("evaluation", ""))