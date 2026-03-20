"""
build_kb.py - 知识库构建脚本
功能：自动下载Vue文档 -> 加载文档 -> 分割 -> 向量化 -> 存储
"""

import os
import pickle
import subprocess
import shutil
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

# ========== 1. 配置参数 ==========
CURRENT_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../data'))
RAW_DIR = f"{DATA_DIR}/raw"            # 原始文档存放位置
VECTOR_STORE_DIR = f"{DATA_DIR}/chroma_db"  # 向量库位置
BM25_PATH = f"{DATA_DIR}/bm25.pkl"     # BM25索引文件
CHUNK_SIZE = 500                        # 每块大小
CHUNK_OVERLAP = 50                      # 重叠大小
EMBEDDING_MODEL = "BAAI/bge-large-zh"   # 嵌入模型

# 创建目录
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

print("=" * 60)
print("🚀 开始构建知识库")
print(f"📁 原始文档目录: {RAW_DIR}")
print(f"📁 向量库目录: {VECTOR_STORE_DIR}")
print(f"📄 BM25索引: {BM25_PATH}")
print("=" * 60)

# ========== 2. 下载Vue文档（如果不存在） ==========
def download_vue_docs():
    """如果raw目录为空，则下载Vue文档"""
    
    # 检查raw目录是否已有文件
    existing_files = []
    if os.path.exists(RAW_DIR):
        for root, dirs, files in os.walk(RAW_DIR):
            existing_files.extend([f for f in files if f.endswith('.md')])
    
    if existing_files:
        print(f"\n✅ 文档已存在，跳过下载（共 {len(existing_files)} 个文件）")
        return True
    
    print("\n📥 开始下载Vue中文文档...")
    
    # 临时克隆目录
    temp_dir = "/tmp/vue_docs_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    try:
        # 执行git clone
        result = subprocess.run(
            ["git", "clone", "https://github.com/vuejs-translations/docs-zh-cn.git", temp_dir],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ git clone失败: {result.stderr}")
            return False
        
        # 复制所有.md文件到raw目录（保持目录结构）
        md_count = 0
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.md'):
                    # 计算相对路径
                    rel_path = os.path.relpath(root, temp_dir)
                    target_dir = os.path.join(RAW_DIR, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # 复制文件
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, file)
                    shutil.copy2(src_file, dst_file)
                    md_count += 1
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        print(f"✅ 下载完成，共 {md_count} 个Markdown文件")
        return True
        
    except FileNotFoundError:
        print("❌ 未找到git命令，请先安装git")
        print("或手动下载: https://github.com/vuejs-translations/docs-zh-cn/archive/refs/heads/main.zip")
        print("解压后将内容放入 data/raw 目录")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

# ========== 3. 加载文档 ==========
def load_documents():
    """从raw目录加载所有md文件（使用TextLoader，无unstructured依赖）"""
    print("\n📖 正在加载文档...")
    
    # 关键修改：指定loader_cls为TextLoader，纯文本加载MD文件
    loader = DirectoryLoader(
        RAW_DIR,
        glob="**/*.md",           # 加载所有.md文件
        loader_cls=TextLoader,    # 替换默认加载器为TextLoader，避开unstructured
        loader_kwargs={"encoding": "utf-8"},  # 指定编码，避免中文乱码
        show_progress=True,
        use_multithreading=True,   # 多线程加速
        silent_errors=True         # 忽略错误文件
    )
    
    docs = loader.load()
    print(f"   加载了 {len(docs)} 个文档")

    # ===== 新增：检查核心文件是否被加载 =====
    core_files = ['lifecycle.md', 'computed.md', 'props.md', 'events.md']
    print("\n🔍 检查核心文件是否被加载：")
    for core_file in core_files:
        found = False
        for doc in docs:
            if core_file in doc.metadata.get('source', ''):
                found = True
                print(f"   ✅ {core_file} 已加载")
                break
        if not found:
            print(f"   ❌ {core_file} 未被加载！")
    
    # ===== 监控1：统计文档来源 =====
    from collections import Counter
    sources = []
    for doc in docs:
        source = doc.metadata.get('source', '')
        rel_path = os.path.relpath(source, RAW_DIR)
        top_dir = rel_path.split(os.sep)[0] if os.sep in rel_path else 'root'
        sources.append(top_dir)
    
    print("\n📊 文档来源统计：")
    for dir_name, count in Counter(sources).most_common():
        print(f"   {dir_name}: {count} 个文件")
    
    # ===== 监控2：检查关键文档是否存在 =====
    key_docs = [
        "src/guide/essentials/lifecycle.md",
        "src/guide/essentials/computed.md",
        "src/guide/essentials/list.md",
        "src/guide/essentials/reactivity-fundamentals.md",
        "src/guide/components/props.md",
        "src/guide/components/events.md"
    ]
    
    print("\n🔍 关键文档检查：")
    found_docs = []
    for doc_path in key_docs:
        full_path = os.path.join(RAW_DIR, doc_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"   ✅ {doc_path} ({size} 字节)")
            found_docs.append(doc_path)
        else:
            print(f"   ❌ {doc_path} 不存在")
    
    # ===== 监控3：检查文件完整性（看最后几行） =====
    print("\n📝 文件完整性抽查：")
    for doc_path in found_docs[:3]:  # 抽查前3个
        full_path = os.path.join(RAW_DIR, doc_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"\n   {doc_path}:")
                print(f"     总行数: {len(lines)}")
                print(f"     开头: {lines[0][:50].strip()}...")
                if len(lines) > 5:
                    print(f"     结尾: {lines[-5].strip()[:50]}...")
                    print(f"           {lines[-4].strip()[:50]}...")
                    print(f"           {lines[-3].strip()[:50]}...")
        except Exception as e:
            print(f"     读取失败: {e}")
    
    # 显示一些统计信息
    if docs:
        print(f"   第一个文档: {docs[0].metadata.get('source', 'unknown')}")
        print(f"   总字符数: {sum(len(doc.page_content) for doc in docs):,}")
    
    return docs

# ========== 4. 分割文本 ==========
def split_documents(docs):
    """将文档分割成chunks"""
    print("\n✂️ 正在分割文本...")

    # 分割前先检查
    core_files = ['lifecycle.md', 'computed.md', 'props.md', 'events.md']
    for core_file in core_files:
        for doc in docs:
            if core_file in doc.metadata.get('source', ''):
                print(f"   📄 {core_file} 即将被分割，长度: {len(doc.page_content)}")
    
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n```", "\n\n", "\n", " ", ""],
        keep_separator=True,
        length_function=len
    )
    
    chunks = splitter.split_documents(docs)
    print(f"   分割成 {len(chunks)} 个chunks")
    
    # ===== 监控4：chunks统计 =====
    if chunks:
        avg_chars = sum(len(c.page_content) for c in chunks) // len(chunks)
        print(f"   平均每个chunk约 {avg_chars} 字符")
        
        # 按来源统计chunks
        from collections import Counter
        source_counter = Counter()
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            source_counter[os.path.basename(source)] += 1
        
        print("\n   📊 chunks分布（前10个文件）：")
        for source, count in source_counter.most_common(10):
            print(f"     {source}: {count} chunks")
    

    # 显示几个chunk示例
    print("\n   chunk示例：")
    for i, chunk in enumerate(chunks[:2]):
        print(f"   [{i+1}] {chunk.page_content[:100]}...")
        print(f"       来源: {chunk.metadata.get('source', 'unknown')}")
    
    # 分割后检查
    print("\n🔍 分割后检查：")
    for core_file in core_files:
        count = 0
        for chunk in chunks:
            if core_file in chunk.metadata.get('source', ''):
                count += 1
        print(f"   {core_file}: {count} 个chunks")
    return chunks

# ========== 5. 初始化嵌入模型 ==========
def init_embeddings():
    """初始化嵌入模型"""
    print("\n🧠 初始化嵌入模型...")
    print(f"   模型: {EMBEDDING_MODEL}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},      # 可以用'cuda'如果有GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 测试一下
    test_vector = embeddings.embed_query("测试")
    print(f"   向量维度: {len(test_vector)}")
    
    return embeddings

# ========== 6. 构建向量库 ==========
def build_vectorstore(chunks, embeddings):
    """构建并保存向量库"""
    print("\n💾 构建向量库...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    
    vectorstore.persist()
    print(f"   向量库已保存到: {VECTOR_STORE_DIR}")

    # ===== 监控5：向量库统计 =====
    collection = vectorstore._collection
    print(f"   向量库包含 {collection.count()} 个向量")

    return vectorstore

# ========== 7. 构建BM25索引 ==========
def build_bm25_index(chunks):
    """构建并保存BM25索引"""
    print("\n🔍 构建BM25索引...")
    
    # BM25需要纯文本
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    bm25_retriever = BM25Retriever.from_texts(
        texts=texts,
        metadatas=metadatas,
        k=100  # 检索时返回前100个
    )
    
    # 保存BM25索引
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    
    print(f"   BM25索引已保存到: {BM25_PATH}")
    
    return bm25_retriever

# ========== 8. 简单测试 ==========
def test_retrieval(vectorstore, bm25_retriever):
    """简单测试检索效果（适配新版langchain API）"""
    print("\n🧪 运行简单测试...")
    
    test_questions = [
        "v-for怎么用",
        "ref和reactive区别",
        "vue组件通信"
    ]
    
    for q in test_questions:
        print(f"\n问题: {q}")
        
        # 向量检索
        vector_results = vectorstore.similarity_search(q, k=2)
        print(f"  向量检索 Top1: {vector_results[0].page_content[:50]}...")
        
        # BM25检索
        try:
            bm25_results = bm25_retriever.invoke(q)
            print(f"  BM25检索 Top1: {bm25_results[0].page_content[:50]}...")
        except Exception as e:
            bm25_results = bm25_retriever._get_relevant_documents(q)
            print(f"  BM25检索 Top1: {bm25_results[0].page_content[:50]}...")

# ========== 主程序 ==========
if __name__ == "__main__":
    # 1. 下载文档
    if not download_vue_docs():
        exit(1)
    
    # 2. 加载文档
    docs = load_documents()
    if not docs:
        print("❌ 没有加载到任何文档")
        exit(1)
    
    # 3. 分割文档
    chunks = split_documents(docs)
    
    # 4. 初始化嵌入模型
    embeddings = init_embeddings()
    
    # 5. 构建向量库
    vectorstore = build_vectorstore(chunks, embeddings)
    
    # 6. 构建BM25索引
    bm25_retriever = build_bm25_index(chunks)
    
    # 7. 简单测试
    test_retrieval(vectorstore, bm25_retriever)
    
    # 8. 完成
    print("\n" + "=" * 60)
    print("✅ 知识库构建完成！")
    print(f"📁 数据位置: {DATA_DIR}")
    print("=" * 60)