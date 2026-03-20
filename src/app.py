"""
app.py - Vue文档智能问答助手（最终稳定版）
适配：智谱官方zai-sdk + glm-3-turbo + 限流防护 + 跨境提速 + 多轮记忆
"""
import os
import sys
import time
from typing import List, Dict, Tuple
import gradio as gr
# 官方zai-sdk正确导入（测试通过的方式）
from zai import ZhipuAiClient

# ========== 全局配置（限流防护+警告关闭）==========
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRADIO_TEMP_DIR"] = "/tmp/gradio"

# 添加父目录路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模块
from retriever import HybridRetriever
from reranker import Reranker
from memory import ConversationMemory

# ========== 导入配置 ==========
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from config import ZHIPU_API_KEY
    # 强制使用测试通过的glm-3-turbo
    MODEL_NAME = "glm-3-turbo"
    print(f"✅ 已加载配置文件: {parent_dir}/config.py")
    print(f"✅ 已切换模型为：{MODEL_NAME}（限流更宽松）")
except ImportError:
    print("=" * 60)
    print("❌ 未找到配置文件")
    print("=" * 60)
    sys.exit(1)

# ========== LLM 客户端（官方zai-sdk+限流防护）==========
class LLMClient:
    """智谱LLM客户端（适配zai-sdk+glm-3-turbo+限流防护）"""
    def __init__(self):
        print("🤖 初始化智谱 LLM（官方zai-sdk）...")
        # 测试通过的客户端初始化方式
        self.client = ZhipuAiClient(
            api_key=ZHIPU_API_KEY,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            timeout=20.0  # 跨境适配的超时时间
        )
        self.model = MODEL_NAME
        # 限流防护：记录上次请求时间，强制间隔≥10秒
        self.last_request_time = 0
        self.min_request_interval = 10  # 免费账户安全间隔

    def generate(self, query: str, context: List[str], history: str) -> str:
        """生成回答（限流防护+极简参数+跨境提速）"""
        # 1. 限流防护：强制间隔
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            wait_time = self.min_request_interval - (current_time - self.last_request_time)
            print(f"⚠️  触发限流防护，等待{wait_time:.1f}秒...")
            time.sleep(wait_time)
        self.last_request_time = time.time()

        # 2. 极简提示词（跨境提速：减少请求体积）
        context_text = "".join(context)[:300] if context else "无参考文档"
        
        # 构建带历史的提示词
        if history:
            prompt = f"""
历史对话：
{history}

参考文档：{context_text}

当前问题：{query}

要求：
1. 基于参考文档回答，如果参考文档有相关内容就使用，没有就基于你的知识
2. 如果历史对话中有上下文，请结合上下文理解问题
3. 简洁明了，有代码示例的话只保留核心代码
4. 纯中文回答，不要多余内容
"""
        else:
            prompt = f"""
参考文档：{context_text}

当前问题：{query}

要求：
1. 基于参考文档回答，简洁明了
2. 有代码示例的话只保留核心代码
3. 纯中文回答，不要多余内容
"""

        # 3. 官方zai-sdk调用（测试通过的逻辑）
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,       # 稍微增加，因为可能有历史
                temperature=0.1,
                stream=False,
                timeout=20.0
            )
            return response.choices[0].message.content
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"❌ LLM调用失败【{error_type}】: {str(e)}\n\n"
            if "1302" in error_msg or "429" in error_msg:
                error_msg += "🔴 触发限流！请10分钟后再使用，或降低请求频率\n"
            elif "Timeout" in error_msg:
                error_msg += "🔴 跨境超时！请检查网络，或稍等再试\n"
            elif "Authentication" in error_msg:
                error_msg += "🔴 API Key错误！请检查config.py中的配置\n"
            return error_msg


# ========== 聊天助手（完整版+多轮记忆）==========
class ChatAssistant:
    """整合所有模块的聊天助手"""
    
    def __init__(self):
        print("=" * 60)
        print("🚀 初始化智能问答助手（完整版）")
        print("=" * 60)
        
        # 1. 初始化检索器
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        print(f"\n📁 数据目录: {data_dir}")
        
        print("\n📚 加载检索器...")
        self.retriever = HybridRetriever(data_dir=data_dir)
        
        # 2. 初始化重排序器
        print("\n🎯 加载重排序器...")
        self.reranker = Reranker()
        
        # 3. 初始化记忆管理器 ← 关键！
        print("\n💬 加载记忆管理器...")
        self.memory = ConversationMemory(max_rounds=5)
        
        # 4. 初始化LLM
        print("\n🤖 加载LLM客户端（glm-3-turbo）...")
        self.llm = LLMClient()
        
        print("\n✅ 初始化完成！")
        print("=" * 60)
    
    def _optimize_query_with_context(self, query: str) -> str:
        """根据历史上下文优化查询词"""
        
        # 获取最近3轮对话
        history = self.memory.get_last_n_rounds(3)
        
        # 提取最近的问题
        last_questions = []
        for msg in history:
            if msg["role"] == "user":
                last_questions.append(msg["content"])
        
        print(f"   📝 最近问题: {last_questions}")
        
        # ===== 规则1：处理"和xxx的区别" =====
        if "和" in query and "区别" in query:
            if len(last_questions) >= 1:
                last_topic = last_questions[-1]
                if "ref" in last_topic or "reactive" in last_topic:
                    return "ref和reactive的区别"
                if "v-for" in last_topic or "v-if" in last_topic:
                    return "v-for和v-if的区别"
                if "computed" in last_topic or "watch" in last_topic:
                    return "computed和watch的区别"
        
        # ===== 规则2：处理"它的"、"他们的"等代词 =====
        if any(word in query for word in ["它的", "他们的", "它的用法", "他们的区别"]):
            if len(last_questions) >= 1:
                last_topic = last_questions[-1]
                if "ref" in last_topic:
                    return "ref的详细说明"
                if "reactive" in last_topic:
                    return "reactive的详细说明"
                if "v-for" in last_topic:
                    return "v-for的详细说明"
                if "v-if" in last_topic:
                    return "v-if的详细说明"
        
        # ===== 规则3：处理查询历史 =====
        if any(word in query for word in ["我刚问", "之前问了", "刚才说了", "我问了", "我说了", "问过什么"]):
            return "查询历史记录"
        
        return query
    
    def answer(self, query: str, history: List[Dict[str, str]]) -> str:
        """处理用户问题，返回回答"""
        print(f"\n🔍 [用户] {query}")
        
        # 1. 添加用户问题到记忆
        self.memory.add_user_message(query)
        
        # 2. 获取历史文本
        history_text = self.memory.get_history(format="text")
        print(f"   📜 当前历史:\n{history_text}")
        
        # 3. 上下文查询优化
        optimized_query = self._optimize_query_with_context(query)
        
        # 4. 特殊处理：查询历史
        if optimized_query == "查询历史记录":
            print("   📜 用户查询历史，直接返回")
            if history_text:
                response = f"你刚才问了：\n{history_text}"
            else:
                response = "还没有任何对话历史，请先提问吧！"
            self.memory.add_assistant_message(response)
            return response
        
        # 5. 正常流程：检索
        if optimized_query != query:
            print(f"   🔄 查询优化: '{query}' → '{optimized_query}'")
        
        print("   📚 正在检索...")
        retrieved_docs = self.retriever.retrieve(
            optimized_query, top_k=5, method="weighted", alpha=0.8
        )
        
        # 6. 重排序
        print("   🔄 正在重排序...")
        reranked_docs = self.reranker.rerank(optimized_query, retrieved_docs, top_k=3)
        context = [doc.page_content for doc in reranked_docs]
        
        # 7. 生成回答（传入历史）
        print("   🤖 正在生成回答...")
        response = self.llm.generate(optimized_query, context, history_text)
        
        # 8. 添加回答到记忆
        self.memory.add_assistant_message(response)
        
        return response
    
    def clear_memory(self):
        """清空记忆"""
        self.memory.clear()
        return "对话历史已清空，开始新对话"


# ========== Gradio界面（适配6.x+确保渲染）==========
def create_interface(assistant: ChatAssistant):
    with gr.Blocks(title="Vue文档智能问答助手") as demo:
        gr.Markdown("""
        # 📚 Vue文档智能问答助手
        适配：智谱glm-3-turbo（限流宽松版）| 支持多轮对话记忆
        """)
        
        # 极简Chatbot（确保渲染）
        chatbot = gr.Chatbot(
            label="对话", height=500, avatar_images=(None, "🤖"), value=[]
        )
        msg = gr.Textbox(label="输入问题", placeholder="v-for怎么用？")
        
        with gr.Row():
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

        # 核心响应函数
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
            
            try:
                response = assistant.answer(message, chat_history)
            except Exception as e:
                response = f"❌ 处理失败：{str(e)}"
            
            # 确保Gradio格式正确
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            return "", chat_history

        # 清空函数
        def clear_chat():
            assistant.clear_memory()
            return []

        # 绑定事件
        send_btn.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(clear_chat, None, chatbot)

    return demo


# ========== 主程序 ==========
if __name__ == "__main__":
    # 检查API Key
    if not ZHIPU_API_KEY or ZHIPU_API_KEY == "你的智谱API Key":
        print("=" * 60)
        print("⚠️  请先在config.py中配置正确的ZHIPU_API_KEY")
        print("=" * 60)
        print("\n获取方式：")
        print("1. 访问 https://bigmodel.cn 注册")
        print("2. 进入控制台 → API Keys")
        print("3. 创建新 Key，复制到 config.py")
        sys.exit(1)
    
    # 初始化并启动
    assistant = ChatAssistant()
    demo = create_interface(assistant)
    
    print("\n" + "=" * 60)
    print("🚀 启动成功！访问地址：http://127.0.0.1:7860")
    print("⚠️  免费账户限流提示：单次请求间隔≥10秒，每分钟最多5次请求")
    print("💬 支持多轮对话：可以追问、对比、查询历史")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )