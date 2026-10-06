from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="qwen2.5:7b-instruct-q4_K_M",
    messages=[
        {"role": "system", "content": "你是AI开发助教"},
        {"role": "user", "content": "一句话解释Function Calling"}
    ]
)
print(response.choices[0].message.content)

