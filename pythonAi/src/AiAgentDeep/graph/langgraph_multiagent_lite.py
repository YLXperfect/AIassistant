"""
Day 23: LangGraph 多 Agent Lite - RAG 子 Agent 集成（最终版）
"""

from typing import TypedDict, Annotated, List
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatZhipuAI
import os

# ====================== 导入 RAG 子 Agent ======================
from rag_sub_agent import RAGSubAgent

print("=== Day 23: LangGraph 多 Agent + RAG 子 Agent 集成 ===")

class ResumeState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_resume: str
    rules: List[str]
    polished_resume: str
    review_feedback: str
    next_step: str

# ====================== LLM 初始化（差异化配置） ======================
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置 ZHIPU_API_KEY 环境变量")



# 严格模式 LLM（用于生成和审核，降低创造力）
llm_strict = ChatZhipuAI(
    model="glm-4-flash",
    temperature=0.25,      # 显著降低温度，提高忠实度
    max_tokens=1024,
    timeout=120.0,
    api_key=api_key
)

# 普通模式 LLM（用于规划，允许一定创造力）
llm_normal = ChatZhipuAI(
    model="glm-4-flash",
    temperature=0.6,
    max_tokens=1024,
    timeout=120.0,
    api_key=api_key
)

print("✅ LLM 初始化完成（strict + normal 两种配置）")


# ====================== 初始化 RAG 子 Agent ======================
rag_sub_agent = RAGSubAgent()

# ====================== Nodes ======================
def plan_node(state: ResumeState) -> ResumeState:
    print("→ [1] plan_node")
    prompt = f"用户简历：{state['user_resume'][:300]}...\n请简要规划本次简历润色的重点方向。"
    response = llm_normal.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)], "next_step": "retrieve"}

def retrieve_node(state: ResumeState) -> ResumeState:
    """Day 23 核心：使用 RAG 子 Agent 检索真实规则"""
    print("→ [2] retrieve_node (RAG 子 Agent)")
    
    # 构建检索查询（可结合历史消息优化）
    last_msg = state.get("messages", [])[-1].content if state.get("messages") else "简历优化规则"
    query = f"简历润色相关规则：{last_msg}"
    
    # 调用 RAG 子 Agent
    retrieved_rules = rag_sub_agent.retrieve_rules(query, k=10)
    
    print(f"   共检索到 {len(retrieved_rules)} 条规则")
    
    return {
        "rules": retrieved_rules,
        "messages": [AIMessage(content=f"已从知识库检索到 {len(retrieved_rules)} 条相关简历优化规则")]
    }

def generate_node(state: ResumeState) -> ResumeState:
    print("→ [3] generate_node")
    
    rules_str = "\n\n".join(state.get("rules", []))
    
    prompt = f"""你是一位**极其严格**的简历优化专家，只做润色，不做内容创作。

**铁律**（必须绝对遵守）：
1. **只能**使用用户提供的原始简历内容，**绝对禁止**编造任何新经历、新公司、新项目、新数据。
2. **只能**基于下面提供的规则进行语言优化、结构调整和亮点突出。
3. 严格应用 STAR 方法和量化原则，但必须建立在用户已有内容基础上。
4. 不要添加任何用户简历中没有提到的信息（包括姓名、邮箱、电话、GPA 等）。

=== 检索到的优化规则 ===
{rules_str}

=== 用户原始简历（这是唯一允许使用的内容） ===
{state['user_resume']}

请直接输出优化后的简历（Markdown 格式），保持简洁、专业。"""

    response = llm_strict.invoke([HumanMessage(content=prompt)])
    
    return {
        "polished_resume": response.content,
        "messages": [AIMessage(content=response.content)],
        "next_step": "review"
    }

def review_node(state: ResumeState) -> ResumeState:
    print("→ [4] review_node")
    prompt = f"""你是一个严格的简历审核专家。请审核以下润色结果：

{state.get('polished_resume', '')[:800]}

请给出具体反馈。最后必须明确写出：
- 如果质量优秀：【通过】
- 如果需要改进：【需要优化】并指出至少2个具体问题"""

    response = llm_strict.invoke([HumanMessage(content=prompt)])
    feedback = response.content
    next_step = "END" if "【通过】" in feedback or "通过" in feedback else "generate"
    print(f"   review 决策 → {next_step}")
    
    return {
        "review_feedback": feedback,
        "messages": [AIMessage(content=feedback)],
        "next_step": next_step
    }

def should_continue(state: ResumeState) -> str:
    return state.get("next_step", "END")

# ====================== 构建 Graph ======================
workflow = StateGraph(ResumeState)

workflow.add_node("plan", plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("review", review_node)

workflow.add_edge(START, "plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "review")

workflow.add_conditional_edges("review", should_continue, {"generate": "generate", "END": END})

# ====================== 持久化 ======================
DB_PATH = "checkpointer.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
print(f"✅ SQLite Checkpointer 初始化成功")

app = workflow.compile(checkpointer=checkpointer)

# ====================== 测试运行 ======================
if __name__ == "__main__":
    test_resume = """
    姓名：张三
    年龄：24
    教育经历：香港理工大学 计算机科学本科 2022-2026
    项目经验：参与过机器学习图像识别项目，主要负责数据处理部分，提升了模型准确率。
    实习经历：在一家科技公司做过三个月实习，主要负责编写和维护代码。
    技能：python,langchain, langgraph, RAG
    """

    initial_state = {
        "messages": [HumanMessage(content="请帮我润色这份简历，重点使用 STAR 方法和量化成就")],
        "user_resume": test_resume,
        "rules": [],
        "polished_resume": "",
        "review_feedback": "",
        "next_step": ""
    }

    config = {"configurable": {"thread_id": "resume_session_001"}}

    print("\n=== Day 23 多 Agent + RAG 测试运行 ===")
    result = app.invoke(initial_state, config=config)
    
    print("\n=== 最终润色简历 ===")
    print(result.get("polished_resume", "")[:1000])
    
    print("\n=== 审核反馈 ===")
    print(result.get("review_feedback", "")[-600:])