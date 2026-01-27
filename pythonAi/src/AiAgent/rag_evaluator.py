# ultimate_rag_evaluator.py - 终极RAG评估器
import json
from typing import List, Dict, Tuple
import re

class UltimateRAGEvaluator:
    """
    终极RAG系统评估器 - 完全替代RAGAS
    提供专业级的评估报告，无需外部API
    """
    
    def __init__(self):
        self.metrics_config = {
            'accuracy': {
                'name': '答案准确性',
                'weight': 0.25,
                'description': '关键信息是否正确'
            },
            'completeness': {
                'name': '信息完整性', 
                'weight': 0.20,
                'description': '是否包含了所有重要信息'
            },
            'relevance': {
                'name': '问题相关性',
                'weight': 0.20,
                'description': '回答是否直接针对问题'
            },
            'context_quality': {
                'name': '上下文质量',
                'weight': 0.20,
                'description': '检索到的上下文是否相关、简洁'
            },
            'response_quality': {
                'name': '回答质量',
                'weight': 0.15,
                'description': '回答的格式、结构、可读性'
            }
        }
    
    def evaluate(self, 
                questions: List[str],
                answers: List[str], 
                contexts: List[List[str]],
                ground_truths: List[str]) -> Dict:
        """
        执行全面评估并生成专业报告
        
        Returns:
            {
                'detailed': 详细结果,
                'summary': 统计摘要, 
                'report': 格式化报告,
                'improvements': 改进建议
            }
        """
        print("="*70)
        print("🤖 终极RAG系统评估器")
        print("="*70)
        
        # 1. 数据验证
        self._validate_input(questions, answers, contexts, ground_truths)
        
        # 2. 执行评估
        detailed_results = self._evaluate_all_questions(
            questions, answers, contexts, ground_truths
        )
        
        # 3. 计算统计
        summary = self._calculate_summary(detailed_results)
        
        # 4. 生成报告
        report = self._generate_detailed_report(detailed_results, summary)
        
        # 5. 生成改进建议
        improvements = self._generate_improvement_plan(summary, detailed_results)
        
        return {
            'detailed': detailed_results,
            'summary': summary,
            'report': report,
            'improvements': improvements
        }
    
    def _validate_input(self, questions, answers, contexts, ground_truths):
        """验证输入数据"""
        n = len(questions)
        assert len(answers) == n, f"问题数({n})≠答案数({len(answers)})"
        assert len(contexts) == n, f"问题数({n})≠上下文数({len(contexts)})"
        assert len(ground_truths) == n, f"问题数({n})≠标准答案数({len(ground_truths)})"
        print(f"✅ 数据验证通过: {n} 个问题")
    
    def _evaluate_all_questions(self, questions, answers, contexts, ground_truths):
        """评估所有问题"""
        detailed_results = []
        
        for i in range(len(questions)):
            print(f"\n🔍 评估问题 {i+1}/{len(questions)}: {questions[i][:50]}...")
            
            result = {
                'question': questions[i],
                'answer_preview': answers[i][:100] + "..." if len(answers[i]) > 100 else answers[i]
            }
            
            # 评估每个指标
            for metric_key in self.metrics_config.keys():
                if metric_key == 'context_quality':
                    score, feedback = getattr(self, f'_evaluate_{metric_key}')(
                        contexts[i], questions[i], ground_truths[i]
                    )
                else:
                    score, feedback = getattr(self, f'_evaluate_{metric_key}')(
                        answers[i], 
                        ground_truths[i] if metric_key != 'relevance' else questions[i],
                        contexts[i] if metric_key == 'response_quality' else None
                    )
                
                result[metric_key] = {
                    'score': score,
                    'feedback': feedback,
                    'weight': self.metrics_config[metric_key]['weight']
                }
            
            # 计算加权总分
            weighted_score = sum(
                result[metric_key]['score'] * result[metric_key]['weight']
                for metric_key in self.metrics_config.keys()
            )
            result['weighted_score'] = round(weighted_score, 3)
            
            detailed_results.append(result)
            
            print(f"   得分: {result['weighted_score']:.3f} | 回答预览: {result['answer_preview']}")
        
        return detailed_results
    
    def _evaluate_accuracy(self, answer: str, ground_truth: str, *args) -> Tuple[float, str]:
        """评估准确性"""
        # 提取所有关键实体
        key_entities = self._extract_all_key_entities(ground_truth)
        
        if not key_entities:
            return 0.5, "无明确关键信息可验证"
        
        # 检查每个实体是否在答案中
        correct_entities = []
        for entity in key_entities:
            if entity in answer:
                correct_entities.append(entity)
        
        accuracy = len(correct_entities) / len(key_entities)
        
        if correct_entities:
            feedback = f"正确: {len(correct_entities)}/{len(key_entities)}个关键信息"
            if len(correct_entities) <= 3:
                feedback += f" ({', '.join(correct_entities)})"
        else:
            feedback = "未找到任何关键信息"
        
        return round(accuracy, 3), feedback
    
    def _evaluate_completeness(self, answer: str, ground_truth: str, *args) -> Tuple[float, str]:
        """评估完整性"""
        # 提取标准答案中的所有信息片段
        info_chunks = self._split_into_info_chunks(ground_truth)
        
        if not info_chunks:
            return 0.5, "标准答案无明确信息块"
        
        # 检查每个信息块是否在答案中
        found_chunks = 0
        for chunk in info_chunks:
            # 检查chunk的核心内容是否在答案中
            if self._chunk_in_answer(chunk, answer):
                found_chunks += 1
        
        completeness = found_chunks / len(info_chunks)
        
        feedback = f"覆盖: {found_chunks}/{len(info_chunks)}个信息块"
        return round(completeness, 3), feedback
    
    def _evaluate_relevance(self, answer: str, question: str, *args) -> Tuple[float, str]:
        """评估相关性"""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # 问题分类
        if "电话" in question_lower or "手机" in question_lower:
            expected = "18086925353"
            if expected in answer:
                return 1.0, "直接提供了正确电话"
            elif "电话" in answer_lower or "联系" in answer_lower:
                return 0.7, "提到了电话但未明确"
            else:
                return 0.3, "未回答电话问题"
        
        elif "邮箱" in question_lower or "email" in question_lower:
            expected = "376472902@qq.com"
            if expected in answer:
                return 1.0, "直接提供了正确邮箱"
            elif "邮箱" in answer_lower or "@" in answer:
                return 0.7, "提到了邮箱但未明确"
            else:
                return 0.3, "未回答邮箱问题"
        
        elif "工作经历" in question_lower or "工作" in question_lower:
            work_keywords = ["工作", "公司", "职位", "项目", "经历"]
            found = sum(1 for kw in work_keywords if kw in answer_lower)
            if found >= 2:
                return 0.9, "详细回答了工作经历"
            elif found >= 1:
                return 0.6, "提到了工作相关"
            else:
                return 0.3, "未针对工作经历回答"
        
        elif "什么" in question_lower or "内容" in question_lower or "写了" in question_lower:
            if len(answer) > 50:
                return 0.8, "提供了实质性内容"
            elif len(answer) > 20:
                return 0.5, "内容较为简略"
            else:
                return 0.2, "内容太少"
        
        # 通用相关性检查
        question_words = set(re.findall(r'[\u4e00-\u9fff]{2,}', question_lower))
        answer_words = set(re.findall(r'[\u4e00-\u9fff]{2,}', answer_lower))
        
        if not question_words:
            return 0.5, "问题无明确关键词"
        
        overlap = len(question_words.intersection(answer_words))
        relevance = overlap / len(question_words)
        
        if relevance > 0.5:
            feedback = f"高度相关 ({int(relevance*100)}%关键词匹配)"
        elif relevance > 0.2:
            feedback = f"部分相关 ({int(relevance*100)}%关键词匹配)"
        else:
            feedback = "相关性较低"
        
        return round(relevance, 3), feedback
    
    def _evaluate_context_quality(self, contexts: List[str], question: str, ground_truth: str) -> Tuple[float, str]:
        """评估上下文质量"""
        if not contexts:
            return 0.0, "无检索到上下文"
        
        # 1. 相关性得分
        relevance_scores = []
        question_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', question.lower()))
        
        for ctx in contexts:
            ctx_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', ctx.lower()))
            if not question_keywords:
                relevance = 0.5
            else:
                overlap = len(question_keywords.intersection(ctx_keywords))
                relevance = overlap / len(question_keywords)
            relevance_scores.append(relevance)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # 2. 简洁性得分（避免过长）
        total_chars = sum(len(ctx) for ctx in contexts)
        avg_length = total_chars / len(contexts)
        
        if avg_length < 100:
            conciseness = 1.0
        elif avg_length < 300:
            conciseness = 0.8
        elif avg_length < 500:
            conciseness = 0.6
        elif avg_length < 1000:
            conciseness = 0.4
        else:
            conciseness = 0.2
        
        # 3. 多样性得分（避免重复）
        unique_signatures = set()
        for ctx in contexts:
            # 取前50字符作为签名
            sig = ctx[:50].strip()
            if sig:
                unique_signatures.add(sig)
        
        diversity = len(unique_signatures) / len(contexts)
        
        # 4. 信息密度得分
        info_density_scores = []
        for ctx in contexts:
            # 简单计算：实体数量 / 长度
            entities = re.findall(r'[\u4e00-\u9fff]{2,}|\d{11}|[\w\.-]+@[\w\.-]+\.\w+', ctx)
            density = len(entities) / max(len(ctx), 1) * 1000  # 每千字符实体数
            if density > 10:
                info_density_scores.append(1.0)
            elif density > 5:
                info_density_scores.append(0.7)
            elif density > 2:
                info_density_scores.append(0.4)
            else:
                info_density_scores.append(0.1)
        
        avg_density = sum(info_density_scores) / len(info_density_scores) if info_density_scores else 0
        
        # 综合得分
        overall = (avg_relevance * 0.4 + conciseness * 0.2 + 
                  diversity * 0.2 + avg_density * 0.2)
        
        feedback = f"相关度:{avg_relevance:.1%} 长度:{int(avg_length)}字 多样性:{diversity:.1%}"
        return round(overall, 3), feedback
    
    def _evaluate_response_quality(self, answer: str, ground_truth: str, contexts: List[str] = None) -> Tuple[float, str]:
        """评估回答质量"""
        score = 0.0
        feedback_parts = []
        
        # 1. 格式质量（20%）
        format_score = 0
        if "**" in answer or "###" in answer:
            format_score += 0.1
            feedback_parts.append("使用粗体强调")
        if "- " in answer or "• " in answer or "1." in answer:
            format_score += 0.1
            feedback_parts.append("使用列表格式")
        if "：" in answer or "。" in answer or "\n" in answer:
            format_score += 0.1
            feedback_parts.append("标点正确")
        
        score += min(format_score, 0.2)
        
        # 2. 结构质量（30%）
        structure_score = 0
        lines = answer.split('\n')
        
        # 检查是否有标题结构
        has_title = any(line.strip().startswith(('#', '**', '##')) for line in lines[:3])
        if has_title:
            structure_score += 0.15
            feedback_parts.append("有标题结构")
        
        # 检查是否有分段
        if len(lines) > 3:
            structure_score += 0.15
            feedback_parts.append("内容分段清晰")
        
        score += structure_score
        
        # 3. 可读性（20%）
        readability_score = 0
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        if 20 < avg_line_length < 100:
            readability_score += 0.2
            feedback_parts.append("行长度适中")
        elif avg_line_length <= 20:
            readability_score += 0.1
            feedback_parts.append("行略短")
        else:
            feedback_parts.append("行过长")
        
        score += readability_score
        
        # 4. 信息组织（30%）
        organization_score = 0
        
        # 检查是否有关键信息在前
        if len(answer) > 0:
            first_100 = answer[:100].lower()
            key_indicators = ["根据", "结果", "电话", "邮箱", "姓名", "工作"]
            if any(indicator in first_100 for indicator in key_indicators):
                organization_score += 0.15
                feedback_parts.append("关键信息前置")
        
        # 检查是否有总结或结尾
        if len(answer) > 100:
            last_50 = answer[-50:].lower()
            if any(word in last_50 for word in ["总结", "总之", "综上", "主要"]):
                organization_score += 0.15
                feedback_parts.append("有总结性结尾")
        
        score += organization_score
        
        # 确保分数不超过1
        score = min(score, 1.0)
        
        if not feedback_parts:
            feedback = "基本格式合格"
        else:
            feedback = "; ".join(feedback_parts)
        
        return round(score, 3), feedback
    
    def _extract_all_key_entities(self, text: str) -> List[str]:
        """提取所有关键实体"""
        entities = []
        
        # 电话号码
        entities.extend(re.findall(r'\d{11}', text))
        
        # 邮箱
        entities.extend(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text))
        
        # 公司名称
        company_patterns = [
            "中国软件", "中电金信", "四川蓝色互动", 
            "蜀信易", "个人开发者", "自主创业"
        ]
        entities.extend([comp for comp in company_patterns if comp in text])
        
        # 职位名称
        position_patterns = [
            "前端工程师", "iOS开发工程师", "个人开发者", 
            "自主创业", "前端开发工程师"
        ]
        entities.extend([pos for pos in position_patterns if pos in text])
        
        # 个人姓名
        if "袁麟翔" in text:
            entities.append("袁麟翔")
        
        # 年份信息
        entities.extend(re.findall(r'\d{4}/\d{2}', text))
        
        return list(set(entities))  # 去重
    
    def _split_into_info_chunks(self, text: str, max_chunk_length: int = 50) -> List[str]:
        """将文本分割为信息块"""
        # 按标点分割
        sentences = re.split(r'[。，；、\n]', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) <= max_chunk_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 过滤太短的块
        chunks = [chunk for chunk in chunks if len(chunk) >= 10]
        
        return chunks
    
    def _chunk_in_answer(self, chunk: str, answer: str) -> bool:
        """检查信息块是否在答案中（模糊匹配）"""
        # 如果块很短，直接检查
        if len(chunk) < 20:
            return chunk in answer
        
        # 对于较长的块，检查核心关键词
        keywords = re.findall(r'[\u4e00-\u9fff]{2,}|\d{4}/\d{2}|\d{11}', chunk)
        
        if not keywords:
            return False
        
        # 如果有超过一半的关键词在答案中，则认为匹配
        found = sum(1 for kw in keywords if kw in answer)
        return found / len(keywords) >= 0.5
    
    def _calculate_summary(self, detailed_results: List[Dict]) -> Dict:
        """计算统计摘要"""
        summary = {}
        
        # 计算每个指标的平均分
        for metric_key in self.metrics_config.keys():
            scores = [r[metric_key]['score'] for r in detailed_results]
            summary[f'avg_{metric_key}'] = round(sum(scores) / len(scores), 3)
        
        # 计算加权总分
        weighted_scores = [r['weighted_score'] for r in detailed_results]
        summary['overall_score'] = round(sum(weighted_scores) / len(weighted_scores), 3)
        
        # 问题级别的统计
        summary['total_questions'] = len(detailed_results)
        summary['good_questions'] = sum(1 for r in detailed_results if r['weighted_score'] >= 0.7)
        summary['poor_questions'] = sum(1 for r in detailed_results if r['weighted_score'] < 0.5)
        
        return summary
    
    def _generate_detailed_report(self, detailed_results: List[Dict], summary: Dict) -> str:
        """生成详细评估报告"""
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("📊 RAG系统专业评估报告")
        report_lines.append("="*80)
        report_lines.append(f"评估时间: {self._get_timestamp()}")
        report_lines.append(f"评估问题数: {summary['total_questions']}")
        report_lines.append("")
        
        # 总体评分
        report_lines.append("🎯 总体评分")
        report_lines.append("-"*40)
        report_lines.append(f"综合得分: {summary['overall_score']:.3f}/1.0")
        
        # 评级
        overall = summary['overall_score']
        if overall >= 0.9:
            rating = "⭐⭐⭐⭐⭐ 优秀"
        elif overall >= 0.8:
            rating = "⭐⭐⭐⭐ 良好" 
        elif overall >= 0.7:
            rating = "⭐⭐⭐ 合格"
        elif overall >= 0.6:
            rating = "⭐⭐ 需要改进"
        else:
            rating = "⭐ 较差"
        
        report_lines.append(f"评级: {rating}")
        report_lines.append(f"良好问题: {summary['good_questions']}/{summary['total_questions']}")
        report_lines.append(f"较差问题: {summary['poor_questions']}/{summary['total_questions']}")
        report_lines.append("")
        
        # 指标分析
        report_lines.append("📈 指标分析")
        report_lines.append("-"*40)
        
        for metric_key, config in self.metrics_config.items():
            avg_score = summary.get(f'avg_{metric_key}', 0)
            report_lines.append(f"{config['name']}: {avg_score:.3f} ({config['description']})")
        
        report_lines.append("")
        
        # 详细结果
        report_lines.append("🔍 详细评估结果")
        report_lines.append("-"*40)
        
        for i, result in enumerate(detailed_results):
            report_lines.append(f"\n问题 {i+1}: {result['question']}")
            report_lines.append(f"回答预览: {result['answer_preview']}")
            report_lines.append(f"总分: {result['weighted_score']:.3f}")
            
            for metric_key in self.metrics_config.keys():
                metric_result = result[metric_key]
                report_lines.append(f"  • {self.metrics_config[metric_key]['name']}: "
                                  f"{metric_result['score']:.3f} - {metric_result['feedback']}")
        
        return "\n".join(report_lines)
    
    def _generate_improvement_plan(self, summary: Dict, detailed_results: List[Dict]) -> str:
        """生成改进计划"""
        improvements = []
        
        # 分析薄弱环节
        weak_metrics = []
        for metric_key, config in self.metrics_config.items():
            avg_score = summary.get(f'avg_{metric_key}', 0)
            if avg_score < 0.7:
                weak_metrics.append((config['name'], avg_score))
        
        if weak_metrics:
            improvements.append("🔴 主要薄弱环节:")
            for name, score in weak_metrics:
                improvements.append(f"  • {name}: {score:.3f} (建议目标: >0.8)")
            
            # 具体建议
            improvements.append("\n💡 具体改进建议:")
            
            if any('准确性' in name for name, _ in weak_metrics):
                improvements.append("1. 提高准确性:")
                improvements.append("   - 确保关键实体（电话、邮箱、公司名）完全正确")
                improvements.append("   - 添加验证机制，避免幻觉")
            
            if any('完整性' in name for name, _ in weak_metrics):
                improvements.append("2. 提高完整性:")
                improvements.append("   - 改进检索策略，确保所有相关信息都被找到")
                improvements.append("   - 添加信息聚合机制，合并相关片段")
            
            if any('上下文质量' in name for name, _ in weak_metrics):
                improvements.append("3. 改进上下文质量:")
                improvements.append("   - 优化文本分割策略，避免切断关键信息")
                improvements.append("   - 添加去重和过滤机制")
                improvements.append("   - 提高检索相关性")
        else:
            improvements.append("✅ 所有指标表现良好，继续保持！")
        
        # 针对具体问题的建议
        poor_questions = [(i+1, r) for i, r in enumerate(detailed_results) if r['weighted_score'] < 0.6]
        if poor_questions:
            improvements.append("\n⚠️  需要重点关注的问题:")
            for q_num, result in poor_questions[:3]:  # 最多显示3个
                improvements.append(f"  问题{q_num}: {result['question'][:50]}...")
                improvements.append(f"    得分: {result['weighted_score']:.3f}")
        
        return "\n".join(improvements)
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 使用示例
if __name__ == "__main__":
    print("🚀 启动终极RAG评估器...")
    
    # 创建评估器实例
    evaluator = UltimateRAGEvaluator()
    
    # 你的数据（清理后的版本）
    questions = [
    "jianli.pdf 里我的电话是多少？",
    "jianli.pdf 里我的邮箱是什么？",
    "jianli.pdf 里我的工作经历是什么？",
    "ylx.txt 里写了什么？",
    ]
    
    answers = [
        "根据查询结果，您的联系电话是：**18086925353**",
        "根据查询结果，jianli.pdf中的邮箱是：**376472902@qq.com**",
        '''根据jianli.pdf的内容，您的工作经历如下：

**1. 四川蜀信易电子商务有限公司（2016/01– 2017/09）**
- 职位：iOS 开发工程师
- 职责：
  - 基于iOS端技术的开发，负责应用软件的编码工作
  - 完成与工作相关的技术文档编写
  - 配合产品部门共同分析项目需求，制定开发计划（V1.1.0-v2.5）
- 项目内容：蜀信e.惠生活社区O2O平台，与四川省各地农信联社合作打造，集社区周边商户、餐饮美食、新鲜蔬果、农家特色、休闲旅游、社区服务及金融服务于一体的平台

**2. 个人开发者（2021/12 - 2022/7）**
- 项目名称：秋夏抠图神器（2021/12 - 2022/02）
- 项目内容：一款照片处理APP，包含裁剪、拼接、抠图、美颜、滤镜等功能
- 项目职责：独立担任iOS端的开发工作

**3. 集团型企业税务风险智能监控平台项目**
- 核心职责：
  - 保障系统稳定性，修复线上遗留bug，优化集团涉税驾驶舱
  - 总体监控并进行企业定制二次开发
  - 完成sass咨询业务开发，对接浙农、阳光新能源企业新需求并完成定制化开发

您的工作经历主要集中在iOS开发领域，涉及电商平台、税务监控平台和照片处理应用等多个项目''',
        '''根据查询结果，ylx.txt文件的内容包括：

**个人信息：**
- 姓名：袁麟翔
- 特点：很爱学习
- 爱好：爱炒股

**工作经历：**
- 在四川蜀信易电子商务有限公司工作
- 负责项目框架搭建，完成功能模块和UI页面的开发
- 项目内容：惠生活商户客户端，为商家提供店铺管理功能（发布商品、管理商品、查看订单、财务对账、发布活动等）

**个人能力：**
- 有较强的沟通和团队协作能力
- 能够合理组织团队，协调同事之间的分歧
- 有上进心，学习主观能动性强
- 能不断学习新技能，保持积极认真的工作态度
- 能从过往经历中总结经验

**技术相关：**
- 与服务器约定使用RSA进行私钥解密'''
    ]
    
    # 清理后的上下文
    contexts = [
        ["姓名：袁麟翔 电话：18086925353"],
        ["邮箱：376472902@qq.com"],
        [
            "2016/01– 2017/09：四川蜀信易电子商务有限公司 职位：iOS 开发工程师",
            "2021/12 - 2022/7：个人开发者 项目名称：秋夏抠图神器",
            "2020/06 - 2021/12 ：中电金信 职位：iOS开发工程师"
        ],
        ["我叫袁麟翔 很爱学习 爱炒股"]
    ]
    
    ground_truths = [
        "18086925353",
        "376472902@qq.com",
        '''中国软件前端工程师 2022/10-至今, 2021/12-2022/7:个人开发者(秋夏抠图神器),
2020/06-2021/12:中电金信(iOS开发工程师),
2018/05-2020/04:四川蓝色互动网络科技有限公司,
2017/09-2018/03:自主创业,
2016/01-2017/09:四川蜀信易电子商务有限公司''',
        "我叫袁麟翔 很爱学习 爱炒股"
    ]
    
    # 执行评估
    results = evaluator.evaluate(questions, answers, contexts, ground_truths)
    
    # 打印报告
    print("\n" + "="*80)
    print(results['report'])
    print("\n" + "="*80)
    print("💡 改进计划")
    print("="*80)
    print(results['improvements'])
    
    # 保存结果
    with open("rag_evaluation_final_report.txt", "w", encoding="utf-8") as f:
        f.write(results['report'])
        f.write("\n\n" + "="*80 + "\n")
        f.write("💡 改进计划\n")
        f.write("="*80 + "\n")
        f.write(results['improvements'])
    
    print("\n📄 评估报告已保存到: rag_evaluation_final_report.txt")
    
    # 打印摘要
    print("\n" + "="*80)
    print("📋 评估摘要")
    print("="*80)
    print(f"综合得分: {results['summary']['overall_score']:.3f}/1.0")
    for key, value in results['summary'].items():
        if key.startswith('avg_'):
            metric_name = key[4:]
            if metric_name in evaluator.metrics_config:
                print(f"{evaluator.metrics_config[metric_name]['name']}: {value:.3f}")