# debug_retriever.py
from retriever import HybridRetriever

retriever = HybridRetriever(data_dir="./data")

# 测试问题
test_queries = [
    "生命周期有哪些",
    "computed和watch区别", 
    "组件通信方式"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"🔍 测试: {query}")
    print('='*60)
    
    # 分别查看向量和BM25的结果
    vector_results = retriever.vector_search(query, k=20)
    bm25_results = retriever.bm25_search(query, k=20)
    
    print(f"\n【向量检索前5名】")
    for i, doc in enumerate(vector_results[:5]):
        source = doc.metadata.get('source', 'unknown')
        print(f"{i+1}. {source}")
    
    print(f"\n【BM25检索前5名】")
    for i, doc in enumerate(bm25_results[:5]):
        source = doc.metadata.get('source', 'unknown')
        print(f"{i+1}. {source}")
    
    # 检查核心文档是否出现
    core_docs = ['lifecycle.md', 'computed.md', 'props.md', 'events.md']
    print(f"\n【核心文档出现情况】")
    for core in core_docs:
        in_vector = any(core in d.metadata.get('source','') for d in vector_results)
        in_bm25 = any(core in d.metadata.get('source','') for d in bm25_results)
        print(f"{core}: 向量={'✅' if in_vector else '❌'}, BM25={'✅' if in_bm25 else '❌'}")