"""
Day3: Human-in-the-Loop（人机协同）
- 在评估后增加中断点，让用户确认或修改建议
- 使用 SqliteSaver 持久化状态，支持恢复
"""
import os
import re
import sqlite3
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatZhipuAI
import traceback

# ---------- 初始化 LLM ----------
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

# ---------- 状态定义 ----------
class ResumeState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_resume: str
    rules: List[str]
    polished_resume: str
    review_feedback: str
    next_step: str
    score: int
    evaluation: str
    suggestion: str
    human_decision: str          # 新增：用户决策（"continue"/"modify"/"re-evaluate"）
    human_feedback: str          # 新增：用户修改意见（当决策为 modify 时）

# ---------- 节点函数 ----------
def plan_node(state: ResumeState):
    """规划节点：分析简历，输出润色方向（这里简化，直接返回）"""
    print("\n→ 执行 plan_node")
    prompt = f"用户简历：{state['user_resume'][:300]}...\n请简要规划本次润色需要重点使用的规则。"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content=response.content)],
        "next_step": "retrieve"
    }

def retrieve_node(state: ResumeState):
    """检索节点：从知识库获取规则（模拟）"""
    print("\n→ 执行 retrieve_node")
    sample_rules = [
        "使用 STAR 方法描述经历",
        "所有成就必须量化",
        "使用强行动动词开头"
    ]
    return {
        "rules": sample_rules,
        "messages": [AIMessage(content=f"已检索到 {len(sample_rules)} 条规则")]
    }

def evaluate_node(state: ResumeState):
    """评估节点：根据简历和规则打分"""
    print("\n→ 执行 evaluate_node")
    prompt = f"""你是一位资深HR。请根据以下简历和简历写作规则，对简历质量进行评分（1-10分），并给出简短评价。
规则：{state['rules'][:3] if state['rules'] else '无规则'}
简历：{state['user_resume'][:1000]}

输出格式：
评分：X
评价：..."""
    response = llm.invoke([HumanMessage(content=prompt)])
    score_match = re.search(r'评分[：:]\s*(\d+)', response.content)
    score = int(score_match.group(1)) if score_match else 5
    evaluation = response.content.strip()
    return {
        "score": score,
        "evaluation": evaluation,
        "messages": [AIMessage(content=f"评估结果：{evaluation}")]
    }

def suggest_node(state: ResumeState):
    """建议节点：低分时生成改进建议"""
    print("\n→ 执行 suggest_node")
    prompt = f"""你是简历专家。这份简历得分 {state['score']} 分。请给出具体的改进建议，帮助提升质量。
原始简历：{state['user_resume'][:1000]}
需遵循的规则：{state['rules']}
如果用户提出了要求：{state.get('human_feedback', '无')} ,必须按照用户的要求给出具体的改进建议
输出详细建议。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "suggestion": response.content,
        "polished_resume": response.content,  # 复用字段存储建议
        "messages": [AIMessage(content=f"改进建议：{response.content[:200]}...")]
    }

def refine_node(state: ResumeState):
    """润色节点：高分时生成润色版本"""
    print("\n→ 执行 refine_node")
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

# ---------- 【核心知识点】Human-in-the-Loop 节点 ----------
def human_review_node(state: ResumeState):
    """
    中断节点：暂停执行，等待用户输入。
    用户可以选择：
        - continue: 继续执行原来的建议（润色或建议）
        - modify: 提供修改意见，然后重新生成
        - re-evaluate: 重新评估
    """
    print("\n🔔 进入人机协同节点")
    print(f"当前评估得分：{state.get('score', 'N/A')}")
    print(f"评估意见：{state.get('evaluation', '无')}")
    if state.get('score', 0) >= 7:
        print("系统将自动进行润色。")
    else:
        print("系统将提供改进建议。")

    # 模拟用户输入（实际生产环境中应通过API接收）
    # 这里我们使用 input() 进行交互式演示
    print("\n请选择操作：")
    print("1. 继续执行")
    print("2. 修改建议（输入你的修改意见）")
    print("3. 重新评估")
    choice = input("请输入数字（1/2/3）：").strip()
    
    if choice == "1":
        decision = "continue"
        feedback = ""
    elif choice == "2":
        decision = "modify"
        feedback = input("请输入你的修改意见：").strip()
    elif choice == "3":
        decision = "re-evaluate"
        feedback = ""
    else:
        decision = "continue"
        feedback = ""

    return {
        "human_decision": decision,
        "human_feedback": feedback,
        "messages": [HumanMessage(content=f"用户决策：{decision}，意见：{feedback}")]
    }

# ---------- 路由函数 ----------
def route_after_evaluate(state: ResumeState) -> str:
    """评估后先进入人机节点"""
    return "human_review"

def route_after_human(state: ResumeState) -> str:
    """根据人机决策决定下一步"""
    decision = state.get("human_decision", "continue")
    if decision == "continue":
        # 继续原流程：根据评分选择润色或建议
        if state.get("score", 0) >= 7:
            return "refine"
        else:
            return "suggest"
    elif decision == "modify":
        # 用户要求修改：根据反馈重新生成（这里简化：直接进入修改建议节点）
        # 实际可设计一个专门的处理节点
        # 我们复用 suggest_node，但传入用户反馈
        # 因为 state 已经包含 human_feedback，可以在 suggest_node 中读取
        return "suggest"
    elif decision == "re-evaluate":
        # 重新评估：回到 evaluate 节点
        return "evaluate"
    else:
        return END

# ---------- 构建图 ----------
# 使用 SqliteSaver 持久化状态（支持中断恢复）
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

workflow = StateGraph(ResumeState)

# 添加节点
workflow.add_node("plan", plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("refine", refine_node)
workflow.add_node("suggest", suggest_node)

# 添加边
workflow.add_edge(START, "plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "evaluate")
workflow.add_edge("evaluate", "human_review")   # 评估后先进入人机节点

# 条件边：根据人机决策分支
workflow.add_conditional_edges(
    "human_review",
    route_after_human,
    {
        "refine": "refine",
        "suggest": "suggest",
        "evaluate": "evaluate",   # 重新评估
        END: END
    }
)

workflow.add_edge("refine", END)
workflow.add_edge("suggest", END)

# 编译图，传入 checkpointer
app = workflow.compile(checkpointer=memory)

# ---------- 运行示例 ----------
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
        "suggestion": "",
        "human_decision": "",
        "human_feedback": ""
    }

    # 配置：thread_id 用于区分不同会话
    config = {"configurable": {"thread_id": "user_123"}}

    print("=== 开始执行（将进入人机协同节点）===")
    try:
        # 第一次执行，会停在 human_review 节点
        result = app.invoke(initial_state, config=config)
        # 注意：第一次 invoke 并不会真正执行完，因为我们在 human_review 中使用了 input()
        # 所以上面实际上会执行到 human_review 并等待用户输入，然后继续执行到结束。
        # 为了让示例更清晰，我们可以单独控制执行流程（见下文注释）
        # 这里简单展示最终结果
        if result.get("score", 0) >= 7:
            print("\n最终输出（润色结果）：\n", result.get("polished_resume", ""))
        else:
            print("\n最终输出（改进建议）：\n", result.get("suggestion", ""))
        print("\n评估详情：", result.get("evaluation", ""))
    except Exception as e:
        print("执行出错：", e)

    # 如果需要演示中断恢复，可以分两次 invoke：
    # 第一次只执行到中断前，第二次提供输入后继续
    # 但这里我们直接在一个 invoke 里完成了（因为 input() 阻塞了）