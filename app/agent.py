"""
Agent 工具调用：让大模型自主决定调用什么工具来回答问题
支持工具：知识库查询、计算器、获取当前时间
"""
import json
import datetime
from app.rag import llm_client, hybrid_search, rerank, logger


# ===== 工具定义 =====
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "查询知识库，根据用户问题检索相关文档内容，回答文档相关问题时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询词"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "数学计算器，支持加减乘除、括号等数学运算，回答数学问题时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '2 + 3 * 4' 或 '(10 + 5) / 3'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间，回答时间相关问题时使用",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


# ===== 工具实现 =====
def search_knowledge_base(query):
    """知识库查询工具"""
    raw_results = hybrid_search(query, top_k=10)
    results = rerank(query, raw_results, top_k=5)
    if not results:
        return "知识库中没有找到相关内容。"
    context_parts = []
    for i, item in enumerate(results):
        context_parts.append(
            f"[资料{i+1}]（来自 {item['metadata']['source']} 第{item['metadata']['page']}页）\n{item['text']}"
        )
    return "\n\n".join(context_parts)


def calculator(expression):
    """计算器工具（安全计算，只允许数字和运算符）"""
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含非法字符"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


def get_current_time():
    """获取当前时间工具"""
    now = datetime.datetime.now()
    return f"当前时间是：{now.strftime('%Y年%m月%d日 %H:%M:%S')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}"


# 工具名称 → 函数映射
TOOL_FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
    "calculator": calculator,
    "get_current_time": get_current_time,
}


# ===== Agent 主函数 =====
def agent_ask(question, max_iterations=5):
    """
    Agent 问答：让大模型自主决定调用工具
    1. 调用 LLM，传入工具列表
    2. 如果 LLM 要调用工具，执行工具，把结果加回消息
    3. 再次调用 LLM，直到不再调用工具或达到最大轮次
    """
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以使用工具来回答问题。如果问题需要查询文档、计算数学题或获取时间，请调用对应的工具。如果不需要工具，直接回答。"},
        {"role": "user", "content": question}
    ]

    for i in range(max_iterations):
        resp = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = resp.choices[0].message
        messages.append(message.model_dump())

        # 如果没有工具调用，说明 LLM 已经给出最终答案
        if not message.tool_calls:
            logger.info(f"Agent 完成：第{i+1}轮，无工具调用，直接回答")
            return {
                "answer": message.content,
                "tools_used": []
            }

        # 执行所有工具调用
        tools_used = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            logger.info(f"Agent 调用工具：{tool_name}，参数：{tool_args}")

            # 执行工具
            tool_func = TOOL_FUNCTIONS.get(tool_name)
            if tool_func:
                tool_result = tool_func(**tool_args)
            else:
                tool_result = f"未知工具：{tool_name}"

            tools_used.append(tool_name)

            # 把工具结果加回消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

        logger.info(f"Agent 第{i+1}轮，调用了 {len(tools_used)} 个工具：{tools_used}")

    # 达到最大轮次，最后再调一次 LLM 生成答案
    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    return {
        "answer": resp.choices[0].message.content,
        "tools_used": tools_used
    }