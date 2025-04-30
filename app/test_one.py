from langchain_ollama import ChatOllama

# 设置 Ollama 服务地址（如果你不是默认 127.0.0.1:11434）
# 可选：也可通过设置环境变量 OLLAMA_HOST 来全局指定
llm = ChatOllama(
    base_url="http://192.168.31.163:11500",  # 改为你 Windows Ollama 的 IP+端口
    model="qwen3:0.6b",
    temperature=0.7,
    num_predict=256
)

# 构建对话消息
messages = [
    ("system", "你是一位乐于助人的 AI 助手"),
    ("human", "请介绍一下你自己")
]

# 调用模型
response = llm.invoke(messages)

# 打印结果
print("AI 回复：", response.content)
