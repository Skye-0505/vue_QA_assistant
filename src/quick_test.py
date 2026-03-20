"""
5_test_memory.py - 测试多轮记忆功能（带上下文查询优化）
功能：整合检索+重排序+记忆，提供命令行交互界面
"""

import os
import sys
from typing import List, Dict

# 导入已有模块
from retriever import HybridRetriever
from reranker import Reranker
from memory import ConversationMemory

# 模拟LLM（实际使用时替换为真正的LLM）
class MockLLM:
    """模拟LLM，基于检索结果生成回答"""
    
    def generate(self, query: str, context: List[str], history: str) -> str:
        """
        模拟生成回答
        实际项目中这里应该调用真正的LLM API
        """
        print("\n   🤖 [LLM接收到以下信息]")
        print(f"     历史: {history[:100]}..." if len(history) > 100 else f"     历史: {history}")
        print(f"     当前问题: {query}")
        print(f"     参考文档数: {len(context)}")
        
        # 模拟回答
        if not context:
            return "抱歉，我没有找到相关文档。"
        
        # 简单拼接一个回答（实际应该用LLM）
        response = f"基于文档，我来回答：{query}\n\n"
        response += f"找到 {len(context)} 个相关文档：\n"
        for i, doc in enumerate(context[:2]):  # 只用前2个
            response += f"{i+1}. {doc[:100]}...\n"
        
        return response


class ChatAssistant:
    """整合所有模块的聊天助手"""
    
    def __init__(self):
        print("=" * 60)
        print("🚀 初始化智能问答助手")
        print("=" * 60)
        
        # 1. 初始化检索器
        print("\n📚 加载检索器...")
        self.retriever = HybridRetriever(data_dir="./data")
        
        # 2. 初始化重排序器
        print("\n🎯 加载重排序器...")
        self.reranker = Reranker()
        
        # 3. 初始化记忆管理器
        print("\n💬 加载记忆管理器...")
        self.memory = ConversationMemory(max_rounds=5)
        
        # 4. 初始化LLM（这里用模拟的）
        self.llm = MockLLM()
        
        print("\n" + "=" * 60)
        print("✅ 初始化完成！输入 'exit' 退出，输入 'clear' 清空历史")
        print("=" * 60)
    
    # ===== 新增：上下文查询优化 =====
    def _optimize_query_with_context(self, query: str) -> str:
        """
        根据历史上下文优化查询词
        解决"这两个"、"他们"等代词指代问题
        """
        # 获取最近3轮对话
        history = self.memory.get_last_n_rounds(3)
        
        # 提取最近的问题
        last_questions = []
        for msg in history:
            if msg["role"] == "user":
                last_questions.append(msg["content"])
        
        print(f"   📝 最近问题: {last_questions}")
        
        # 规则1：处理"这两个"、"它们"、"他们"
        if any(word in query for word in ["这两个", "它们", "他们", "二者"]):
            if len(last_questions) >= 2:
                # 取最近两个问题
                q1 = last_questions[-2] if len(last_questions) >= 2 else ""
                q2 = last_questions[-1] if len(last_questions) >= 1 else ""
                
                # 提取关键词
                topics = []
                for q in [q1, q2]:
                    if "v-for" in q:
                        topics.append("v-for")
                    elif "v-if" in q:
                        topics.append("v-if")
                    elif "ref" in q:
                        topics.append("ref")
                    elif "reactive" in q:
                        topics.append("reactive")
                    elif "computed" in q:
                        topics.append("computed")
                    elif "watch" in q:
                        topics.append("watch")
                    elif "生命周期" in q:
                        topics.append("生命周期")
                
                if len(topics) >= 2:
                    return f"{topics[0]}和{topics[1]}的区别"
                elif len(topics) == 1:
                    return f"{topics[0]}的详细说明"
        
        # 规则2：处理"性能"、"效率"相关问题
        if any(word in query for word in ["性能", "效率", "速度", "哪个快"]):
            # 找最近讨论的技术
            for q in reversed(last_questions):
                if "ref" in q and "reactive" in q:
                    return "ref和reactive的性能对比"
                if "v-for" in q and "v-if" in q:
                    return "v-for和v-if的性能对比"
                if "computed" in q and "watch" in q:
                    return "computed和watch的性能对比"
                if "ref" in q:
                    return "ref的性能特点"
                if "reactive" in q:
                    return "reactive的性能特点"
        
        # 规则3：处理"举个例子"、"详细说说"
        if any(word in query for word in ["例子", "示例", "详细", "展开"]):
            if last_questions:
                last_topic = last_questions[-1]
                # 提取主题词
                if "v-for" in last_topic:
                    return "v-for的使用示例"
                if "v-if" in last_topic:
                    return "v-if的使用示例"
                if "ref" in last_topic:
                    return "ref的使用示例"
                if "reactive" in last_topic:
                    return "reactive的使用示例"
        
        return query
    # =================================
    
    def answer(self, query: str) -> str:
        """
        处理用户问题，返回回答
        """
        print(f"\n🔍 [用户] {query}")
        
        # 1. 添加用户问题到记忆
        self.memory.add_user_message(query)
        
        # 2. 上下文查询优化
        optimized_query = self._optimize_query_with_context(query)
        if optimized_query != query:
            print(f"   🔄 查询优化: '{query}' → '{optimized_query}'")
        
        # 3. 检索相关文档（用优化后的查询）
        print("   📚 正在检索...")
        retrieved_docs = self.retriever.retrieve(
            optimized_query, 
            top_k=15,
            method="weighted",
            alpha=0.8
        )
        
        # 4. 重排序
        print("   🔄 正在重排序...")
        reranked_docs = self.reranker.rerank(
            optimized_query,
            retrieved_docs,
            top_k=5
        )
        
        # 5. 获取历史
        history = self.memory.get_history(format="text")
        
        # 6. 准备上下文（文档内容）
        context = [doc.page_content for doc in reranked_docs]
        
        # 7. 调用LLM生成回答
        print("   🤖 正在生成回答...")
        response = self.llm.generate(optimized_query, context, history)
        
        # 8. 添加回答到记忆
        self.memory.add_assistant_message(response)
        
        return response
    
    def chat_loop(self):
        """交互式对话循环"""
        print("\n" + "=" * 60)
        print("💬 开始对话（输入 'exit' 退出，输入 'clear' 清空历史）")
        print("=" * 60)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 你: ").strip()
                
                # 检查退出命令
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 再见！")
                    break
                
                # 检查清空命令
                if user_input.lower() in ['clear', 'new']:
                    self.memory.clear()
                    print("💫 对话历史已清空，开始新对话")
                    continue
                
                if not user_input:
                    continue
                
                # 生成回答
                response = self.answer(user_input)
                
                # 显示回答
                print(f"\n🤖 助手: {response}")
                
                # 显示当前记忆状态
                if len(self.memory.history) > 0:
                    print(f"\n📊 [当前记忆轮数: {len(self.memory.history)//2}]")
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 出错了: {e}")
    
    def show_memory(self):
        """显示当前记忆"""
        print("\n" + "=" * 60)
        print("📋 当前对话历史")
        print("=" * 60)
        print(self.memory.get_history(format="text"))


# ========== 测试代码 ==========
if __name__ == "__main__":
    
    # 创建助手实例
    assistant = ChatAssistant()
    
    # 如果有命令行参数，执行单次查询
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        response = assistant.answer(query)
        print(f"\n🤖 {response}")
        print("\n📋 当前记忆:")
        assistant.show_memory()
    
    # 否则进入交互模式
    else:
        assistant.chat_loop()