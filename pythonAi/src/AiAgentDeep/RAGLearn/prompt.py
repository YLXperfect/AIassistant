# src/AiAgentDeep/prompts.py
# Day10: Prompt 工程 - 兼容 LangChain 1.2.x 的 few-shot 写法
# 关键：prefix/suffix 不再直接传给 FewShotChatMessagePromptTemplate

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

# ------------------ 基础系统提示 ------------------
BASE_SYSTEM_PROMPT = """你是一个专业的简历润色专家，必须严格遵守以下规则：
1. 所有输出必须基于提供的上下文（本地知识库中的规则、STAR 方法、量化表达等），**禁止编造、添加外部知识或主观臆断**。
2. 优先使用 STAR 结构：Situation/Task 融合为开头一句，Action 聚焦“我做了什么”（强行动词），Result 前置并必须量化（数字、百分比、金额、时间等），说明对业务的影响。
3. 语言精炼、每条不超过 2 行，使用强行动词开头，避免模糊形容词（如“很大提升”“负责”）。
4. 如果上下文不足以完整优化，直接回复“根据现有规则无法优化，请提供更多细节”。
5. 只输出优化后的单条简历描述，不要额外解释或列步骤（除非明确要求 CoT）。"""

# ------------------ few-shot 示例数据 ------------------
FEW_SHOT_EXAMPLES = [
    {
        "input": "负责公司网站维护，提升用户体验",
        "output": "优化官网用户体验：通过用户调研与热图分析，精简注册流程从 5 步减至 3 步，转化率从 12% 提升至 28%，月新增注册用户 +4500"
    },
    {
        "input": "参与市场推广活动",
        "output": "主导校园路演推广：策划执行 8 场高校活动，覆盖 12000+ 目标用户，收集 2800+ 潜在线索，活动后转化率 18%，贡献季度销售额 45 万元"
    },
    {
        "input": "开发内部管理系统",
        "output": "重构内部审批系统：主导需求分析与架构设计，使用 Python + Django 重写核心模块，审批时效从 3 天缩短至 4 小时，处理效率提升 85%"
    },
]

# 单条示例的提示模板（human → ai）
example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessagePromptTemplate.from_template("{output}"),
])

# Few-shot 模板本体（1.2 版本不传 prefix/suffix）
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=FEW_SHOT_EXAMPLES,
    example_prompt=example_prompt,
)

# ------------------  Prompt ------------------
PROMPT_FEW_SHOT_WITH_CONTEXT = ChatPromptTemplate.from_messages([
    # 系统提示
    SystemMessagePromptTemplate.from_template(
        BASE_SYSTEM_PROMPT + "\n\n参考上下文规则：\n{context}\n\n"
        "以下是使用 STAR + 量化方法优化简历经历的优秀示例，请严格模仿风格："
    ),
    
    # 插入 few-shot 示例（这里不带前缀后缀）
    few_shot_prompt,
    
    #  用户输入
    HumanMessagePromptTemplate.from_template(
        "现在请对以下原始经历进行优化：\n{user_experience}"
    ),
])

PROMPT_COT = ChatPromptTemplate.from_messages([
    ("system", BASE_SYSTEM_PROMPT + "\n\n参考上下文：\n{context}"),
    ("human", """请严格按照以下步骤思考并优化这段简历经历（思考过程可见，但最终只输出优化结果）：

步骤1：从上下文规则中提取本次最适用的 2~3 条关键规则。
步骤2：分析用户原始经历：识别 S/T/A/R 元素，找出可量化的点和模糊表述。
步骤3：应用规则重写：融合 S/T 开头，前置量化 R，使用强行动词。
步骤4：输出最终优化版（只输出单条描述）。

用户经历：
{user_experience}

现在开始多步思考："""),
])

# ------------------ 基础版（对比基准） ------------------
PROMPT_BASIC = ChatPromptTemplate.from_messages([
    ("system", BASE_SYSTEM_PROMPT + "\n\n参考上下文：\n{context}"),
    ("human", "请优化以下简历经历：{user_experience}"),
])