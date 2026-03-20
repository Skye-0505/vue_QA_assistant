"""
2_retriever.py - 手写混合检索模块
功能：加载向量库和BM25索引，实现混合检索（向量+关键词）
特点：手写融合算法，不依赖LangChain的EnsembleRetriever
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class HybridRetriever:
    """手写混合检索器"""
    
    def __init__(self, data_dir: str = "./data"):
        """
        初始化检索器，加载向量库和BM25索引
        
        Args:
            data_dir: 数据目录，包含 chroma_db/ 和 bm25.pkl
        """
        print("=" * 60)
        print("🔍 初始化混合检索器")
        print("=" * 60)
        
        self.data_dir = data_dir
        self.vector_store_dir = f"{data_dir}/chroma_db"
        self.bm25_path = f"{data_dir}/bm25.pkl"
        
        # 1. 加载向量库
        self._load_vectorstore()
        
        # 2. 加载BM25索引
        self._load_bm25()
        
        # 3. 配置参数
        self.alpha = 0.5  # 向量检索的权重，BM25权重为1-alpha
        self.rrf_k = 60    # RRF算法的常数
        
        print(f"✅ 检索器初始化完成")
        print(f"   向量库: {self.vector_store_dir}")
        print(f"   BM25索引: {self.bm25_path}")
        print(f"   融合权重: α={self.alpha}")
        print("=" * 60)
    
    def _load_vectorstore(self):
        """加载向量库"""
        print("\n📦 加载向量库...")
        
        # 初始化嵌入模型（必须和构建时一致）
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 加载已有向量库
        self.vectorstore = Chroma(
            persist_directory=self.vector_store_dir,
            embedding_function=embeddings
        )
        
        # 测试一下
        test_result = self.vectorstore.similarity_search("测试", k=1)
        print(f"   向量库加载成功，包含文档数量: {self.vectorstore._collection.count()}")
    
    def _load_bm25(self):
        """加载BM25索引"""
        print("\n📄 加载BM25索引...")
        
        with open(self.bm25_path, "rb") as f:
            self.bm25_retriever = pickle.load(f)
        
        print(f"   BM25索引加载成功")
    
    # ========== 核心检索方法 ==========
    
    def vector_search(self, query: str, k: int = 100) -> List[Document]:
        """
        纯向量检索
        
        Args:
            query: 查询问题
            k: 返回数量
        
        Returns:
            文档列表
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    # ===== 修改1：BM25检索添加噪音过滤 =====
    def bm25_search(self, query: str, k: int = 100) -> List[Document]:
        """
        纯BM25关键词检索（带噪音过滤）
        
        Args:
            query: 查询问题
            k: 返回数量
        
        Returns:
            过滤后的文档列表
        """
        # 获取BM25结果
        results = self.bm25_retriever.invoke(query)
        
        # 要过滤的噪音文件关键词
        noise_patterns = [
            'README.md',
            'translations/index',
            'pull_request_template',
            'writing-guide',
            '贡献指南',
            'LICENSE',
            'DO_NOT_TRANSLATE',
            'PLEASE_DO_NOT_TRANSLATE'
        ]
        
        # 过滤噪音
        filtered = []
        for doc in results:
            source = doc.metadata.get('source', '')
            # 如果包含噪音关键词，跳过
            if any(pattern in source for pattern in noise_patterns):
                continue
            filtered.append(doc)
            if len(filtered) >= k:
                break
        
        # 如果过滤后不够，再从原结果补充（但要确保不是噪音）
        if len(filtered) < k:
            for doc in results:
                source = doc.metadata.get('source', '')
                if doc not in filtered and not any(pattern in source for pattern in noise_patterns):
                    filtered.append(doc)
                if len(filtered) >= k:
                    break
        
        return filtered[:k]
    # ====================================
    
    # ========== 融合算法1：加权平均融合 ==========
    
    def _normalize_scores(self, docs_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        分数归一化到[0,1]区间
        
        Args:
            docs_with_scores: (文档, 原始分数) 列表
        
        Returns:
            归一化后的 (文档, 分数) 列表
        """
        if not docs_with_scores:
            return []
        
        # 提取分数
        scores = [score for _, score in docs_with_scores]
        min_score = min(scores)
        max_score = max(scores)
        
        # 避免除零
        if max_score == min_score:
            return [(doc, 1.0) for doc, _ in docs_with_scores]
        
        # 归一化
        normalized = []
        for doc, score in docs_with_scores:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((doc, norm_score))
        
        return normalized
    
    def _weighted_fusion(self, 
                        vector_results: List[Document], 
                        bm25_results: List[Document],
                        vector_scores: List[float] = None,
                        bm25_scores: List[float] = None,
                        alpha: float = None,
                        top_k: int = 10) -> List[Document]:
        """
        加权平均融合算法
        
        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            vector_scores: 向量检索的原始分数（如果有）
            bm25_scores: BM25检索的原始分数（如果有）
            alpha: 向量权重，None时用self.alpha
            top_k: 最终返回数量
        
        Returns:
            融合后的文档列表
        """
        if alpha is None:
            alpha = self.alpha
        
        # 如果没有分数，则用位置倒推分数
        if vector_scores is None:
            vector_scores = [1.0 - i/len(vector_results) for i in range(len(vector_results))]
        if bm25_scores is None:
            bm25_scores = [1.0 - i/len(bm25_results) for i in range(len(bm25_results))]
        
        # 组合文档和分数
        vector_pairs = list(zip(vector_results, vector_scores))
        bm25_pairs = list(zip(bm25_results, bm25_scores))
        
        # 分别归一化
        vector_pairs = self._normalize_scores(vector_pairs)
        bm25_pairs = self._normalize_scores(bm25_pairs)
        
        # 融合分数
        doc_score_dict = {}
        
        # 处理向量结果
        for doc, score in vector_pairs:
            doc_id = doc.metadata.get('source', '') + doc.page_content[:50]
            doc_score_dict[doc_id] = {
                'doc': doc,
                'vector_score': score,
                'bm25_score': 0,
                'count': 1
            }
        
        # 处理BM25结果（合并相同文档）
        for doc, score in bm25_pairs:
            doc_id = doc.metadata.get('source', '') + doc.page_content[:50]
            if doc_id in doc_score_dict:
                doc_score_dict[doc_id]['bm25_score'] = score
                doc_score_dict[doc_id]['count'] += 1
            else:
                doc_score_dict[doc_id] = {
                    'doc': doc,
                    'vector_score': 0,
                    'bm25_score': score,
                    'count': 1
                }
        
        # 计算最终分数
        final_scores = []
        for doc_id, item in doc_score_dict.items():
            final_score = alpha * item['vector_score'] + (1 - alpha) * item['bm25_score']
            final_scores.append((item['doc'], final_score))
        
        # 按最终分数排序
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in final_scores[:top_k]]
    
    # ========== 融合算法2：RRF (Reciprocal Rank Fusion) ==========
    
    def _rrf_fusion(self, 
                   vector_results: List[Document],
                   bm25_results: List[Document],
                   k: int = None,
                   top_k: int = 10) -> List[Document]:
        """
        RRF融合算法（不依赖分数，只用排序位置）
        
        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            k: RRF常数，默认self.rrf_k
            top_k: 最终返回数量
        
        Returns:
            融合后的文档列表
        """
        if k is None:
            k = self.rrf_k
        
        # 构建位置映射
        doc_score_dict = {}
        
        # 处理向量结果（位置从1开始）
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc.metadata.get('source', '') + doc.page_content[:50]
            rrf_score = 1 / (k + rank)
            doc_score_dict[doc_id] = {
                'doc': doc,
                'score': rrf_score
            }
        
        # 处理BM25结果
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.metadata.get('source', '') + doc.page_content[:50]
            rrf_score = 1 / (k + rank)
            if doc_id in doc_score_dict:
                doc_score_dict[doc_id]['score'] += rrf_score
            else:
                doc_score_dict[doc_id] = {
                    'doc': doc,
                    'score': rrf_score
                }
        
        # 按分数排序
        final_scores = [(item['doc'], item['score']) for item in doc_score_dict.values()]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in final_scores[:top_k]]
    
    # ========== 主检索接口（修改2：完整重构） ==========
    def retrieve(self, 
                query: str, 
                top_k: int = 10,
                method: str = "weighted",
                alpha: float = None) -> List[Document]:
        """
        混合检索主接口（优化版）
        
        Args:
            query: 查询问题
            top_k: 返回文档数量
            method: 融合方法，"weighted"或"rrf"
            alpha: 加权融合的向量权重（仅weighted方法有效）
        
        Returns:
            检索到的文档列表
        """
        print(f"\n🔎 检索: \"{query}\"")
        print(f"   方法: {method}, top_k={top_k}")
        
        # ===== 修改2.1：查询词扩展 =====
        query_expansions = {
            "生命周期": "生命周期 lifecycle hooks onMounted mounted onUpdated updated onUnmounted unmounted 钩子",
            "组件通信": "组件通信 props emit events 父子组件 传值 事件 通信",
            "通信方式": "通信方式 props emit events 父子组件 传值",
            "computed": "computed 计算属性 缓存 依赖 响应式 watch 侦听器 区别",  # 加强
            "watch": "watch 侦听器 副作用 响应式 computed 计算属性 区别",        # 加强
            "ref": "ref reactive 响应式 区别",
            "reactive": "reactive ref 响应式 区别",
            "v-for": "v-for 列表渲染 循环",
            "v-if": "v-if 条件渲染 优先级",
            "优先级": "优先级 同时使用 v-if v-for 条件渲染 列表渲染",  # 新增
        }
        
        expanded_query = query
        for key, expansion in query_expansions.items():
            if key in query:
                expanded_query = expansion
                print(f"   🔍 查询扩展: '{query}' → '{expansion}'")
                break
        # =============================
        
        # ===== 修改2.2：大幅增加召回数量 =====
        vector_results = self.vector_search(expanded_query, k=top_k*5)  # 从2倍改成5倍
        bm25_results = self.bm25_search(query, k=top_k*2)  # BM25用原词
        # ====================================
        
        print(f"   向量检索返回: {len(vector_results)} 个")
        print(f"   BM25检索返回: {len(bm25_results)} 个")
        
        # ===== 修改2.3：添加调试信息 =====
        core_files = ['lifecycle.md', 'computed.md', 'props.md', 'events.md', 'list.md']
        print(f"\n   📊 核心文档召回情况：")
        for core in core_files:
            in_vector = any(core in d.metadata.get('source','') for d in vector_results)
            in_bm25 = any(core in d.metadata.get('source','') for d in bm25_results)
            print(f"      {core}: 向量={'✅' if in_vector else '❌'}, BM25={'✅' if in_bm25 else '❌'}")
        # ================================
        
        # ===== 修改2.4：提高默认向量权重 =====
        if alpha is None:
            alpha = 0.8  # 从0.5提高到0.8
        # ====================================
        
        # 2. 融合
        if method == "weighted":
            fused_results = self._weighted_fusion(
                vector_results, 
                bm25_results,
                alpha=alpha,
                top_k=top_k
            )
        elif method == "rrf":
            fused_results = self._rrf_fusion(
                vector_results,
                bm25_results,
                top_k=top_k
            )
        else:
            raise ValueError(f"未知的融合方法: {method}")
        
        print(f"   融合后返回: {len(fused_results)} 个")
        
        return fused_results
    
    # ========== 检索并打印详情 ==========
    
    def retrieve_with_details(self, 
                             query: str, 
                             top_k: int = 5,
                             method: str = "weighted") -> List[Document]:
        """
        检索并打印详细信息（用于测试）
        """
        results = self.retrieve(query, top_k=top_k, method=method)
        
        print("\n" + "=" * 80)
        print(f"📋 检索结果 for: \"{query}\"")
        print("=" * 80)
        
        for i, doc in enumerate(results, 1):
            print(f"\n【结果 {i}】")
            print(f"来源: {doc.metadata.get('source', 'unknown')}")
            print(f"内容: {doc.page_content[:200]}...")
            print("-" * 60)
        
        return results
    
    # ========== 对比不同融合方法 ==========
    
    def compare_methods(self, query: str, top_k: int = 5):
        """
        对比不同融合方法的效果
        """
        print("\n" + "=" * 80)
        print(f"🔬 融合方法对比 for: \"{query}\"")
        print("=" * 80)
        
        # 1. 纯向量
        vector_results = self.vector_search(query, k=top_k)
        print("\n【纯向量检索】")
        for i, doc in enumerate(vector_results[:3], 1):
            print(f"  {i}. {doc.page_content[:100]}...")
        
        # 2. 纯BM25
        bm25_results = self.bm25_search(query, k=top_k)
        print("\n【纯BM25检索】")
        for i, doc in enumerate(bm25_results[:3], 1):
            print(f"  {i}. {doc.page_content[:100]}...")
        
        # 3. 加权融合 (alpha=0.3)
        weighted_03 = self.retrieve(query, top_k=top_k, method="weighted", alpha=0.3)
        print("\n【加权融合 α=0.3】")
        for i, doc in enumerate(weighted_03[:3], 1):
            print(f"  {i}. {doc.page_content[:100]}...")
        
        # 4. 加权融合 (alpha=0.8)
        weighted_08 = self.retrieve(query, top_k=top_k, method="weighted", alpha=0.8)
        print("\n【加权融合 α=0.8】")
        for i, doc in enumerate(weighted_08[:3], 1):
            print(f"  {i}. {doc.page_content[:100]}...")
        
        # 5. RRF融合
        rrf_results = self.retrieve(query, top_k=top_k, method="rrf")
        print("\n【RRF融合】")
        for i, doc in enumerate(rrf_results[:3], 1):
            print(f"  {i}. {doc.page_content[:100]}...")


# ========== 测试代码 ==========
if __name__ == "__main__":
    
    # 初始化检索器
    retriever = HybridRetriever(data_dir="./data")
    
    # 测试问题列表
    test_questions = [
        "v-for怎么用",
        "ref和reactive区别",
        "组件通信方式",  # 简化，去掉"vue"
        "生命周期有哪些",
        "computed和watch区别"
    ]
    
    # 测试每个问题
    for q in test_questions:
        retriever.retrieve_with_details(q, top_k=3, method="weighted")
    
    # 对比不同融合方法（选一个问题）
    print("\n" + "=" * 80)
    print("🔬 开始方法对比测试")
    print("=" * 80)
    retriever.compare_methods("v-for和v-if优先级", top_k=3)