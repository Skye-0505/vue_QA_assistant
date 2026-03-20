"""
4_memory.py - 多轮记忆模块
功能：管理对话历史，支持上下文理解
设计：支持最近N轮记忆，自动截断，格式化输出
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(self, max_rounds: int = 5, max_token_length: int = 2000):
        """
        初始化记忆管理器
        
        Args:
            max_rounds: 最多记住几轮对话（1轮 = 1问 + 1答）
            max_token_length: 历史文本最大长度（防止超出模型限制）
        """
        self.max_rounds = max_rounds
        self.max_token_length = max_token_length
        
        # 存储对话历史
        # 格式: [
        #   {"role": "user", "content": "问题", "timestamp": "2024-01-01 12:00:00"},
        #   {"role": "assistant", "content": "回答", "timestamp": "2024-01-01 12:00:05"},
        # ]
        self.history = []
        
        print("=" * 60)
        print("💬 初始化对话记忆管理器")
        print("=" * 60)
        print(f"   最多记忆轮数: {max_rounds} 轮")
        print(f"   最大历史长度: {max_token_length} 字符")
        print("=" * 60)
    
    def add_user_message(self, message: str) -> None:
        """
        添加用户消息
        
        Args:
            message: 用户输入的问题
        """
        self.history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 自动截断，只保留最近 max_rounds*2 条消息（因为每轮有问和答）
        if len(self.history) > self.max_rounds * 2:
            self.history = self.history[-(self.max_rounds * 2):]
    
    def add_assistant_message(self, message: str) -> None:
        """
        添加助手消息
        
        Args:
            message: 助手的回答
        """
        self.history.append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 自动截断
        if len(self.history) > self.max_rounds * 2:
            self.history = self.history[-(self.max_rounds * 2):]
    
    def get_history(self, format: str = "text") -> Any:
        """
        获取历史记录
        
        Args:
            format: 返回格式
                - "text": 纯文本格式（用于拼接到prompt）
                - "list": 消息列表格式
                - "json": JSON格式（用于调试）
        
        Returns:
            指定格式的历史记录
        """
        if format == "text":
            return self._format_as_text()
        elif format == "list":
            return self.history
        elif format == "json":
            return json.dumps(self.history, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"未知的格式: {format}")
    
    def _format_as_text(self) -> str:
        """
        将历史格式化为文本（用于拼接到prompt）
        
        Returns:
            格式化的历史文本
        """
        if not self.history:
            return ""
        
        lines = []
        for msg in self.history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}：{msg['content']}")
        
        history_text = "\n".join(lines)
        
        # 截断过长的历史
        if len(history_text) > self.max_token_length:
            history_text = history_text[-self.max_token_length:]
            history_text = "...(历史过长已截断)\n" + history_text
        
        return history_text
    
    def get_last_n_rounds(self, n: int) -> List[Dict]:
        """
        获取最近N轮对话
        
        Args:
            n: 轮数（1轮 = 1问 + 1答）
        
        Returns:
            最近N轮的消息列表
        """
        return self.history[-(n * 2):]
    
    def clear(self) -> None:
        """清空历史"""
        self.history = []
        print("💫 对话历史已清空")
    
    def get_summary(self) -> Dict:
        """
        获取记忆统计信息
        
        Returns:
            包含统计信息的字典
        """
        user_msgs = [m for m in self.history if m["role"] == "user"]
        assistant_msgs = [m for m in self.history if m["role"] == "assistant"]
        
        return {
            "总消息数": len(self.history),
            "用户消息数": len(user_msgs),
            "助手消息数": len(assistant_msgs),
            "对话轮数": len(user_msgs),  # 用户消息数 = 对话轮数
            "最早消息": self.history[0]["timestamp"] if self.history else None,
            "最晚消息": self.history[-1]["timestamp"] if self.history else None,
        }
    
    def search_history(self, keyword: str) -> List[Dict]:
        """
        在历史中搜索关键词（用于调试）
        
        Args:
            keyword: 搜索关键词
        
        Returns:
            包含关键词的消息列表
        """
        results = []
        for msg in self.history:
            if keyword.lower() in msg["content"].lower():
                results.append({
                    "role": msg["role"],
                    "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"],
                    "timestamp": msg["timestamp"]
                })
        return results
    
    def save_to_file(self, filepath: str) -> None:
        """
        保存历史到文件（用于调试/分析）
        
        Args:
            filepath: 保存路径
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"💾 对话历史已保存到: {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """
        从文件加载历史
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.history = json.load(f)
        print(f"📂 已加载 {len(self.history)} 条历史记录")


# ========== 测试代码 ==========
if __name__ == "__main__":
    
    # 1. 初始化记忆管理器
    memory = ConversationMemory(max_rounds=3)
    
    # 2. 模拟对话
    print("\n📝 开始模拟对话...")
    
    # 第一轮
    memory.add_user_message("v-for怎么用？")
    memory.add_assistant_message("v-for 是 Vue 中用于列表渲染的指令，可以遍历数组或对象。")
    
    # 第二轮
    memory.add_user_message("那v-if呢？")
    memory.add_assistant_message("v-if 是条件渲染指令，根据条件决定是否渲染元素。")
    
    # 第三轮
    memory.add_user_message("能举个例子吗？")
    memory.add_assistant_message("比如 v-for='item in items' 渲染列表，v-if='show' 控制显示。")
    
    # 第四轮（会超过 max_rounds=3，最早的一轮会被丢弃）
    memory.add_user_message("它们可以一起用吗？")
    memory.add_assistant_message("v-for 和 v-if 不推荐一起使用，因为 v-if 优先级更高。")
    
    # 3. 查看历史
    print("\n" + "=" * 60)
    print("📋 当前历史记录")
    print("=" * 60)
    print(memory.get_history(format="text"))
    
    # 4. 查看统计
    print("\n" + "=" * 60)
    print("📊 记忆统计")
    print("=" * 60)
    summary = memory.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # 5. 演示清空功能
    print("\n" + "=" * 60)
    print("🧹 测试清空功能")
    print("=" * 60)
    memory.clear()
    print(f"清空后历史: {memory.get_history(format='text') or '空'}")
    
    # 6. 演示保存/加载（可选）
    print("\n" + "=" * 60)
    print("💾 测试保存/加载")
    print("=" * 60)
    
    # 重新添加一些对话
    memory.add_user_message("ref和reactive区别？")
    memory.add_assistant_message("ref 用于基本类型，reactive 用于对象。")
    
    # 保存
    memory.save_to_file("memory_backup.json")
    
    # 清空
    memory.clear()
    print(f"清空后: {memory.get_history(format='text') or '空'}")
    
    # 加载
    memory.load_from_file("memory_backup.json")
    print(f"加载后: {memory.get_history(format='text')}")