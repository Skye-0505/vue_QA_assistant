# Vue 文档智能问答助手

基于 RAG（检索增强生成）架构的 Vue.js 智能问答系统，支持多轮对话、混合检索、重排序优化。

---

## 项目简介

这是一个专门针对 Vue.js 官方文档的智能问答助手。用户可以用自然语言提问，系统会从 Vue 文档中检索相关内容，并结合大模型生成准确的回答。

**核心特性**：
- 混合检索：向量检索 + BM25 关键词检索，兼顾语义理解与精确匹配
- 重排序优化：使用 CrossEncoder 对检索结果重新打分，提升准确率
- 多轮对话记忆：记住上下文，支持追问和指代消解
- Web 界面：基于 Gradio 的友好交互界面
- 免费大模型：支持智谱 GLM-3-Turbo、阿里云百炼等免费 API

---

## 文件结构

```
vue-rag-assistant/
├── src/
│   ├── build_kb.py          # 知识库构建脚本（下载文档、分割、向量化）
│   ├── retriever.py         # 混合检索模块（向量+BM25）
│   ├── reranker.py          # 重排序模块（CrossEncoder）
│   ├── memory.py            # 多轮记忆模块（对话历史管理）
│   └── app.py               # 主程序（Gradio Web 界面）
├── data/                    # 知识库数据目录（自动生成）
│   ├── raw/                 # 原始 Vue 文档
│   ├── chroma_db/           # 向量数据库
│   └── bm25.pkl             # BM25 索引文件
├── config.py                # API 配置文件（不提交到 git）
├── requirements.txt         # 依赖包列表
└── README.md                # 项目说明
```

---

## 各模块功能说明

### 1. build_kb.py - 知识库构建

**功能**：下载 Vue 文档 -> 分割文本 -> 向量化 -> 存储

**做了什么**：
- 自动从 GitHub clone Vue 中文文档
- 将文档按 500 字切分成小块（chunks）
- 用 `BAAI/bge-large-zh` 模型将文本转为向量
- 存入 Chroma 向量库，同时构建 BM25 关键词索引

**运行方式**：
```bash
cd src
python build_kb.py   # 只运行一次，生成 data/ 目录
```

---

### 2. retriever.py - 混合检索模块

**功能**：实现向量检索 + BM25 检索的混合融合

**核心算法**：
- 加权平均融合：score = α × 向量分数 + (1-α) × BM25分数
- RRF 融合：score = Σ 1/(k + rank)（只用排序位置）

**优化策略**：
- 查询扩展（如"生命周期" -> "生命周期 lifecycle hooks onMounted"）
- BM25 噪音过滤（排除 README.md 等无关文档）
- 召回数量扩充（从 top_k*2 增加到 top_k*5）

---

### 3. reranker.py - 重排序模块

**功能**：用 CrossEncoder 对检索结果重新打分

**原理**：
- 第一阶段：检索器快速召回 Top-50
- 第二阶段：CrossEncoder 逐对计算（问题, 文档）相关性分数
- 输出：重新排序后的 Top-5

**模型**：`BAAI/bge-reranker-large`（中文效果好的重排序模型）

---

### 4. memory.py - 多轮记忆模块

**功能**：管理对话历史，支持上下文理解

**核心能力**：
- 自动存储最近 N 轮对话（问+答）
- 格式化历史文本用于 prompt
- 自动截断过长的历史
- 支持清空历史、保存/加载

---

### 5. app.py - 主程序（Gradio 界面）

**功能**：整合所有模块，提供 Web 交互界面

**特点**：
- 多轮对话记忆
- 限流防护（免费 API 需要间隔 10 秒）
- 错误处理和友好提示

---

## 快速开始

### 1. 配置 API Key

创建 `config.py` 文件（与 src 目录同级）：

```python
# config.py
ZHIPU_API_KEY = "你的智谱API Key"  # 从 https://bigmodel.cn 获取
```

### 2. 构建知识库

```bash
cd src
python build_kb.py
```

等待下载 Vue 文档并构建向量库（约 5-10 分钟）。

### 3. 启动应用

```bash
python app.py
```

访问 http://127.0.0.1:7860 即可使用。

---

## 运行示例

```
你: v-for怎么用
助手: v-for 用于基于数组渲染列表。基本用法如下：
    <li v-for="todo in todos" :key="todo.id">{{ todo.text }}</li>

你: 和v-if有什么区别
助手: v-for 和 v-if 的主要区别在于用途和执行顺序：
    1. 用途不同：v-for 用于渲染列表，v-if 用于条件渲染
    2. 执行顺序：当同时使用时，v-for 优先级更高...

你: 我刚都问了你什么问题
助手: 你刚才问了：
    - v-for怎么用
    - 和v-if有什么区别
```

---

## 技术难点与解决方案

### 问题1：检索结果不准确（核心文档召不回）

**现象**：问"生命周期有哪些"，返回的是 README.md 等无关文档。

**原因**：
- 向量检索的查询词太短（"生命周期"）
- BM25 被 README.md 等噪音文档污染

**解决方案**：
1. 查询扩展：将"生命周期"扩展为"生命周期 lifecycle hooks onMounted mounted onUpdated..."
2. BM25 过滤：排除 README.md、translations/index.md 等噪音文档
3. 增加召回数：从 top_k*2 增加到 top_k*5

---

### 问题2：重排序效果差

**现象**：相关文档分数低，无关文档分数高。

**原因**：检索器没有把正确文档召回来，重排序无法弥补。

**解决方案**：先优化检索器，再运行重排序。重排序只能在给定的候选集里排序，不能凭空生成。

---

### 问题3：多轮记忆不工作

**现象**：第二轮问"和v-if有什么区别"，模型不知道"和"指的是什么。

**原因**：历史没有传递给 LLM，也没有做查询优化。

**解决方案**：
1. 在 app.py 中把历史传给 llm.generate()
2. 添加 _optimize_query_with_context() 方法处理代词
3. 特殊处理"我刚问了什么"这类查询，直接返回历史

---

### 问题4：免费 API 限流

**现象**：调用 API 时返回 429 错误。

**原因**：免费账户有 QPS 限制。

**解决方案**：
- 强制请求间隔 ≥10 秒
- 降低 max_tokens（从 1024 降到 500）
- 使用温度参数 0.1 加快响应

---

## 常见问题

### Q: 如何切换 LLM？
编辑 app.py 中的 MODEL_NAME，支持：
- glm-3-turbo（免费，限流宽松）
- glm-4-flash（免费，能力更强但可能限流）

### Q: 构建知识库失败？
检查网络，手动下载文档：
```bash
git clone https://github.com/vuejs-translations/docs-zh-cn.git
mv docs-zh-cn/* data/raw/
```

### Q: 检索效果不好？
调整 retriever.py 中的参数：
- alpha：向量权重（0.5-0.9，推荐 0.8）
- top_k：召回数量（默认 5）
- CHUNK_SIZE：文档分割大小（默认 500）

---

## 后续优化方向

1. 文档分割优化：增大 chunk_overlap（从 50 到 200），让每个 chunk 包含更多上下文
2. 更多文档源：支持 Vue 2 文档、官方博客、GitHub Issues
3. 代码示例执行：集成 CodePen 或 Playground
4. 用户反馈系统：收集回答质量数据

---

## 致谢

- Vue.js 官方文档
- LangChain
- 智谱 AI
- BAAI/bge 系列模型