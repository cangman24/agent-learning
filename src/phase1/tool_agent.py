"""
阶段1：Function Calling 工具调用 Agent
目标：模型自主判断何时调用工具 → 输出 tool_call 参数 → 代码执行函数 → 结果回传 → 模型整合答案
工具：四则计算器（加/减/乘/除）
"""
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# 加载根目录.env环境变量
load_dotenv("../../.env")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# 定义工具函数，@tool装饰器自动生成工具JSON Schema
@tool
def add(a: float, b: float) -> float:
    """计算两个数的加法，返回 a + b 的结果。"""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """计算两个数的减法，返回 a - b 的结果。"""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """计算两个数的乘法，返回 a * b 的结果。"""
    return a * b

@tool
def divide(a: float, b: float) -> str | float:
    """计算两个数的除法，返回 a / b 的结果。b 不能为 0。"""
    if b == 0:
        return "错误：除数不能为零"
    return a / b

# 工具注册表：工具名字映射到函数对象
TOOLS = {"add": add, "subtract": subtract, "multiply": multiply, "divide": divide}

# 初始化Ollama模型
llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE,
    temperature=0.3,
    num_predict=512,
)
# 将工具绑定到大模型，模型就具备输出tool_calls的能力
llm_with_tools = llm.bind_tools(list(TOOLS.values()))

SYSTEM_PROMPT = """你是一个数学计算助手。
规则：
1. 当用户要求计算数学题时，你必须调用对应的工具函数，不要自己心算。
2. 如果用户问题不需要计算（比如聊天、打招呼），直接用文字回答。
3. 工具返回结果后，用简洁的中文把最终答案告诉用户。
4. 禁止重复输出相同文字，禁止无意义刷屏。
"""

def run_agent():
    print("=" * 50)
    print("四则计算器 Agent 已就绪（输入 quit 退出）")
    print(f"模型：{MODEL_NAME}")
    print("=" * 50)
    # 消息列表 = Agent的上下文记忆，全程携带
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() == "quit":
            print("再见！")
            break
        if not user_input:
            continue

        # 1. 用户消息入上下文，调用带工具能力的LLM
        messages.append(HumanMessage(content=user_input))
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # 判断：模型是否选择调用工具
        if not ai_msg.tool_calls:
            # 无工具调用：直接输出文字回答
            print(f"Agent：{ai_msg.content}")
            continue

        # 2. 解析模型输出的tool_call，执行对应Python函数
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  [调用工具] {tool_name}({tool_args})")
            result = TOOLS[tool_name].invoke(tool_args)
            print(f"  [工具返回] {result}")
            # 把工具执行结果封装成ToolMessage放回消息列表，携带tool_call_id做匹配
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            ))

        # 3. 二次调用LLM：模型读取工具返回值，整理最终自然语言答案
        final_msg = llm_with_tools.invoke(messages)
        messages.append(final_msg)
        print(f"Agent：{final_msg.content}")

if __name__ == "__main__":
    run_agent()
