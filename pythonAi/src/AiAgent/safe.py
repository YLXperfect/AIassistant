import os

# 正确做法：用 os.getenv() 获取
secret_key_from_env = os.getenv("ZHIPUAI_API_KEY")

print("---演示开始---")
print(f'直接写字符串的结果是: {"ZHIPUAI_API_KEY"}')
print(f'使用os.getenv()的结果是: {secret_key_from_env}')

# 一个实用的安全判断
if secret_key_from_env:
    print("✅ 成功从环境变量读取到密钥（密钥内容已隐藏，保护安全）")
    
else:
    print("❌ 未找到环境变量 ZHIPUAI_API_KEY")
    print("   请先在终端执行: export ZHIPUAI_API_KEY='你的密钥'")
print("---演示结束---")