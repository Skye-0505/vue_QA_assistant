"""
3_reranker.py - 重排序模块
功能：对检索结果进行重排序，提升准确性
核心：用 CrossEncoder 模型对 (问题, 文档) 对重新打分
"""

import os
import pickle
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import torch


class Reranker:
    """重排序器：用 CrossEncoder 对检索结果重新打分"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = None):
        """
        初始化重排序器
        
        Args:
            model_name: CrossEncoder 模型名称
            device: 运行设备 ('cuda', 'cpu', None 自动选择)
        """
        print("=" * 60)
        print("🎯 初始化重排序器")
        print("=" * 60)
        
        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"   使用设备: {self.device}")
        print(f"   加载模型: {model_name}")
        
        # 加载 CrossEncoder 模型
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=512  # 限制输入长度，避免过长文档
        )
        
        print(f"✅ 重排序器初始化完成")
        print("=" * 60)
    
    def rerank(self, 
               query: str, 
               documents: List[Document], 
               top_k: int = 5,
               batch_size: int = 32) -> List[Document]:
        """
        对文档列表进行重排序
        
        Args:
            query: 原始查询问题
            documents: 待重排序的文档列表（通常来自检索器）
            top_k: 返回前k个结果
            batch_size: 批处理大小
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        print(f"\n🔄 开始重排序...")
        print(f"   输入文档数: {len(documents)}")
        print(f"   目标返回: {top_k}")
        
        # 1. 准备模型输入对 (query, document)
        pairs = []
        for doc in documents:
            # 截断文档内容，避免过长（模型有最大长度限制）
            content = doc.page_content[:1000]  # 取前1000字符就够了
            pairs.append([query, content])
        
        # 2. 模型打分（批处理加速）
        print(f"   模型打分中...")
        scores = self.model.predict(
            pairs, 
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        # 3. 将分数与文档组合
        scored_docs = list(zip(documents, scores))
        
        # 4. 按分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 5. 返回前top_k个文档
        reranked = [doc for doc, score in scored_docs[:top_k]]
        
        print(f"   完成重排序，返回 {len(reranked)} 个文档")
        
        # 6. 打印分数信息（调试用）
        print("\n📊 重排序分数：")
        for i, (doc, score) in enumerate(scored_docs[:top_k]):
            source = doc.metadata.get('source', 'unknown')
            print(f"   {i+1}. {score:.4f} - {os.path.basename(source)}")
        
        return reranked
    
    def rerank_with_scores(self, 
                          query: str, 
                          documents: List[Document], 
                          top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        重排序并返回分数（用于调试）
        """
        if not documents:
            return []
        
        # 准备输入对
        pairs = [[query, doc.page_content[:1000]] for doc in documents]
        
        # 模型打分
        scores = self.model.predict(pairs)
        
        # 组合并排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]


# ========== 测试代码 ==========
if __name__ == "__main__":
    
    # 导入检索器（用于获取待重排序的文档）
    import sys
    sys.path.append(os.path.dirname(__file__))
    from retriever import HybridRetriever
    
    # 1. 初始化检索器
    print("\n" + "=" * 60)
    print("🔍 初始化检索器")
    print("=" * 60)
    retriever = HybridRetriever(data_dir="./data")
    
    # 2. 初始化重排序器
    reranker = Reranker()
    
    # 3. 测试问题列表
    test_questions = [
        "v-for怎么用",
        "ref和reactive区别",
        "生命周期有哪些",
        "computed和watch区别",
        "组件通信方式"
    ]
    
    # 4. 对比有/无重排序的效果
    for query in test_questions:
        print("\n" + "=" * 80)
        print(f"📝 测试问题: \"{query}\"")
        print("=" * 80)
        
        # 先用检索器获取 Top-10
        initial_docs = retriever.retrieve(
            query, 
            top_k=10, 
            method="weighted", 
            alpha=0.7
        )
        
        # 打印原始结果
        print("\n【原始检索结果】")
        for i, doc in enumerate(initial_docs[:5], 1):
            source = doc.metadata.get('source', 'unknown')
            print(f"  {i}. {os.path.basename(source)}")
            print(f"     {doc.page_content[:100]}...")
        
        # 重排序
        reranked_docs = reranker.rerank(query, initial_docs, top_k=5)
        
        # 打印重排序结果
        print("\n【重排序结果】")
        for i, doc in enumerate(reranked_docs, 1):
            source = doc.metadata.get('source', 'unknown')
            print(f"  {i}. {os.path.basename(source)}")
            print(f"     {doc.page_content[:100]}...")
    
    # 5. 详细对比测试（选一个问题）
    print("\n" + "=" * 80)
    print("🔬 详细分数对比")
    print("=" * 80)
    
    test_query = "v-for和v-if优先级"
    print(f"\n问题: {test_query}")
    
    # 获取初始结果
    initial = retriever.retrieve(test_query, top_k=30, method="weighted", alpha=0.7)
    
    # 获取带分数的重排序结果
    reranked_with_scores = reranker.rerank_with_scores(test_query, initial, top_k=10)
    
    print("\n【重排序前后对比】")
    print(f"{'排名':<4} {'原始来源':<30} {'重排后来源':<30} {'分数':<10}")
    print("-" * 80)
    
    for i in range(10):
        orig_source = os.path.basename(initial[i].metadata.get('source', 'unknown')) if i < len(initial) else '-'
        rerank_doc, score = reranked_with_scores[i] if i < len(reranked_with_scores) else (None, 0)
        rerank_source = os.path.basename(rerank_doc.metadata.get('source', 'unknown')) if rerank_doc else '-'
        
        print(f"{i+1:<4} {orig_source:<30} {rerank_source:<30} {score:.4f}")