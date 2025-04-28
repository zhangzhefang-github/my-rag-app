from openai import OpenAI

client = OpenAI(
    api_key = "sk-71DSnBsuDyfSPICxW1Vn4ANe4WHwPWsGqpU1v3LC5dRC2uh8",
    base_url = "https://api.fe8.cn/v1"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "讲个笑话",
        }
    ],
    model="gpt-4o-mini", #此处更换其它模型,请参考模型列表 eg: google/gemma-7b-it
)
print(chat_completion.choices[0].message.content)