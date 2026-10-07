import ollama

stream = ollama.chat(
    model="qwen2.5:7b-instruct-q4_K_M",
    messages=[
        {"role": "system", "content": "你是技术助手，回答简洁清晰"},
        {"role": "user", "content": "解释什么是function calling"}
    ],
    stream=True,
    # ========= 在这里调参数做实验 =========
    options={
        "temperature": 0.3,
        "max_tokens": 512
    }
)

for chunk in stream:
    if chunk["message"]["content"]:
        print(chunk["message"]["content"], end="", flush=True)
print("\n")
