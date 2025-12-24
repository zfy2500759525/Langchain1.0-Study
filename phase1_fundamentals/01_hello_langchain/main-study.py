import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from main import *

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# print(f'GROQ_API_KEY: {GROQ_API_KEY}')

# 示例 1：最简单的 LLM 调用
def simple_invoke():
    model = init_chat_model(
        model="groq:llama-3.3-70b-versatile",
        # api_key=GROQ_API_KEY
    )

    response = model.invoke("你好！请用一句话介绍什么是人工智能。")

    print(type(response))
    print(response.content)


# 示例 2：使用消息列表进行对话
def massages_2():
    model = init_chat_model(
        model="groq:llama-3.3-70b-versatile"
    )

    messages = [
        SystemMessage(content="你是一个友好的 Python 编程助手，擅长用简单易懂的方式解释编程概念。 回答字数不超过100字。"),
        HumanMessage(content="什么是 python 装饰器")
    ]

    print(f'messages-->{messages}')
    print(f'messages-->{messages[0].content}')

    response = model.invoke(messages)
    print(f'response-->{response}')
    print(f'response-->{response.content}')

    messages.append(response)
    print(f'messages-->{messages}')


# 示例 5：理解 invoke 方法的返回值
def response_structure():
    model = init_chat_model(
        "groq:llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )

    response = model.invoke("解释一下什么是递归？用一句话。")
    print(f"response-->{type(response)}")
    print(f"response-->{response}")

    print(f"content-->{type(response.content)}")
    print(f"additional_kwargs-->{type(response.additional_kwargs)}")
    print(f"response_metadata-->{type(response.response_metadata)}")
    # # 不加items Python 会尝试迭代字典的键（默认行为）
    # for key, value in response.response_metadata.items():
    #     print(f"   {key}: {value}")
    print(f"id-->{type(response.id)}")
    print(f"usage_metadata-->{type(response.usage_metadata)}")

    # 检查 token 使用情况（如果可用）
    if "token_usage" in response.response_metadata:
        usage = response.response_metadata["token_usage"]
        print("\n5. Token 使用情况:")
        print(f"   提示 tokens: {usage.get('prompt_tokens', 'N/A')}")
        print(f"   完成 tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"   总计 tokens: {usage.get('total_tokens', 'N/A')}")


# 示例 6：错误处理
def error_handling():
    ...


if __name__ == '__main__':
    # simple_invoke()
    # massages_2()

    # response_structure()
    example_7_multiple_models()


    # import requests
    #
    # # 从环境变量获取 API Key
    # api_key = os.getenv("GROQ_API_KEY")
    #
    # # 在请求头中配置 Authorization
    # headers = {
    #     "Authorization": f"Bearer {api_key}",
    #     "Content-Type": "application/json"
    # }
    #
    # response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
    # print(response.json())