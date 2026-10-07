from openai import OpenAI

# 连接本地Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# ========== 对话历史（这就是"记忆"）==========
# 大模型本身没有记忆！所谓记忆，就是你每次把历史消息都塞回去
messages = [
    {"role": "system", "content": "你是一个友好的本地聊天助手，回答简洁，每次不超过3句话。"}
]

print("=" * 50)
print("本地聊天机器人已启动，输入你的问题，输入 exit 退出")
print("=" * 50)

while True:
    # 1. 读取用户输入
    user_input = input("\n你：")
    
    # 2. 退出判断
    if user_input.strip().lower() == "exit":
        print("再见！")
        break
    
    if not user_input.strip():
        continue

    # 3. 把用户说的话，追加到消息列表里
    messages.append({"role": "user", "content": user_input})

    # 4. 调用模型，把完整历史都传过去
    response = client.chat.completions.create(
        model="qwen2.5:7b-instruct-q4_K_M",
        messages=messages,
        temperature=0.7,
        stream=True  # 流式输出，打字机效果
    )

    # 5. 流式接收回复
    print("AI：", end="", flush=True)
    ai_reply = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            ai_reply += text
            print(text, end="", flush=True)
    print()  # 换行

    # 6. 关键！把AI的回复也追加进历史
    # 这样下一轮对话，模型才知道自己刚才说了什么
    messages.append({"role": "assistant", "content": ai_reply})

    # 打印一下当前历史长度，观察记忆增长（可选）
    print(f"  [当前对话历史：{len(messages)} 条消息]")
