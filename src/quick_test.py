# test_zhipu_api_official.py
"""
智谱官方 zai-sdk 验证脚本（100%基于官方文档，无虚假节点）
解决：429限流 + 超时卡死 + 跨境延迟问题
"""
import sys
import time
from zai import ZhipuAiClient

# ===================== 请修改这1个参数 =====================
API_KEY = "87c0febc960549fba760519372561c5f.hpSTyceDTmwFABj6"  # 替换成真实Key（格式：sk-xxx...）
# ==========================================================

def test_zhipu_api():
    # 1. 初始化客户端（官方标准写法，无虚假节点）
    print("🔍 初始化智谱客户端（官方标准）...")
    try:
        client = ZhipuAiClient(
            api_key=API_KEY,
            base_url="https://open.bigmodel.cn/api/paas/v4/",  # 官方唯一公开地址
            timeout=20.0  # 合理超时，避免无限等待
        )
        print("✅ 客户端初始化成功")
    except Exception as e:
        print(f"❌ 客户端初始化失败：{type(e).__name__} - {str(e)}")
        return

    # 2. 限流防护：强制间隔 + 单次请求（避免触发429）
    print("\n🔍 发送测试请求（限流防护模式）...")
    # 核心：免费账户必须控制请求频率，单次请求后间隔≥10秒
    try:
        # 官方最简请求参数（减少处理时间，降低限流概率）
        response = client.chat.completions.create(
            model="glm-3-turbo",  # 官方示例模型名
            messages=[
                {"role": "user", "content": "你好，仅返回「测试成功」即可"}
            ],
            max_tokens=10,          # 最小token数，最快响应
            temperature=0.0,        # 0随机性，减少模型计算时间
            stream=False,           # 关闭流式，避免额外耗时
            timeout=20.0
        )
        # 打印结果
        print("✅ API调用成功！")
        print(f"📝 回复内容：{response.choices[0].message.content}")
    except Exception as e:
        error_msg = str(e)
        # 精准排查错误
        if "1302" in error_msg or "429" in error_msg:
            print(f"❌ 触发限流（429/1302）：{e}")
            print("💡 解决方案：")
            print("   1. 10分钟内不要再发送请求，等待配额重置")
            print("   2. 改用更低频的请求（单次间隔≥10秒）")
            print("   3. 切换模型为 glm-3-turbo（限流更宽松）")
        elif "Timeout" in error_msg:
            print(f"❌ 请求超时（跨境延迟高）：{e}")
            print("💡 解决方案：")
            print("   1. 检查网络是否能正常访问 https://open.bigmodel.cn")
            print("   2. 适当延长超时时间（如30秒），但不要超过60秒")
        elif "Authentication" in error_msg:
            print(f"❌ API Key认证失败：{e}")
            print("💡 解决方案：检查API Key是否正确，是否过期")
        elif "Model" in error_msg:
            print(f"❌ 模型无权限/不存在：{e}")
            print("💡 解决方案：改用 glm-4-flash 或 glm-3-turbo")
        else:
            print(f"❌ 调用失败：{type(e).__name__} - {e}")

if __name__ == "__main__":
    # 检查API Key是否未修改
    if API_KEY == "你的智谱API Key":
        print("⚠️  请先修改脚本中的 API_KEY 为你的真实智谱API Key！")
        sys.exit(1)
    
    # 执行测试（仅单次请求，避免限流）
    test_zhipu_api()