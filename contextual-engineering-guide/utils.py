
"""
用于格式化和显示对话消息的实用函数模块。

该模块提供函数，使用 `rich` 库在控制台中以结构化且美观的方式呈现消息对象。
"""

# 导入必要的标准库
import json
# 导入类型提示以获得清晰的函数签名
from typing import Any, List

# 从 rich 库导入组件以增强控制台输出
from rich.console import Console
from rich.panel import Panel

# 初始化一个全局的 rich 库 Console 对象。
# 该对象将用于所有到终端的样式化输出。
console = Console()


def format_message_content(message: Any) -> str:
    """
    将消息对象的内容转换为可显示的字符串。

    此函数处理简单的字符串内容以及复杂的基于列表的内容（如工具调用），
    通过适当地解析和格式化它们。

    参数:
        message: 一个具有 'content' 属性的消息对象。

    返回:
        消息内容的格式化字符串表示。
    """
    # 从消息对象中检索内容。
    content = message.content

    # 检查内容是否为简单字符串。
    if isinstance(content, str):
        # 如果是，则直接返回。
        return content
    # 检查内容是否为列表，这通常表示复杂数据，如工具调用。
    elif isinstance(content, list):
        # 初始化一个空列表以保存内容的格式化部分。
        parts = []
        # 遍历内容列表中的每一项。
        for item in content:
            # 如果项目是简单的文本块。
            if item.get("type") == "text":
                # 直接将文本附加到我们的 parts 列表中。
                parts.append(item["text"])
            # 如果项目表示正在使用的工具。
            elif item.get("type") == "tool_use":
                # 格式化一个字符串以指示工具调用，包括工具的名称。
                tool_call_str = f"\n🔧 工具调用: {item.get('name')}"
                # 将工具的输入参数格式化为美观的 JSON 字符串。
                tool_args_str = f"   参数: {json.dumps(item.get('input', {}), indent=2)}"
                # 将格式化的工具调用字符串添加到我们的 parts 列表中。
                parts.extend([tool_call_str, tool_args_str])
        # 将所有格式化的部分连接成一个单一的字符串，用换行符分隔。
        return "\n".join(parts)
    # 对于任何其他类型的内容。
    else:
        # 将内容转换为字符串作为后备方案。
        return str(content)


def format_messages(messages: List[Any]) -> None:
    """
    使用 rich Panel 格式化并显示消息列表。

    每条消息都在一个带样式的面板内呈现，其标题和边框颜色
    与其角色（例如，人类、AI、工具）相对应。

    参数:
        messages: 要显示的消息对象列表。
    """
    # 遍历提供的列表中的每个消息对象。
    for m in messages:
        # 通过获取类名并删除“Message”来确定消息类型。
        msg_type = m.__class__.__name__.replace("Message", "")
        # 使用我们的辅助函数获取消息的格式化字符串内容。
        content = format_message_content(m)

        # 定义 rich Panel 的默认参数。
        panel_args = {"title": f"{msg_type}", "border_style": "white"}

        # 根据消息类型自定义面板外观。
        # 如果消息来自人类用户。
        if msg_type == "Human":
            # 更新标题并将边框颜色设置为蓝色。
            panel_args.update(title="人类", border_style="blue")
        # 如果消息来自 AI 助手。
        elif msg_type == "Ai":
            # 更新标题并将边框颜色设置为绿色。
            panel_args.update(title="助手", border_style="green")
        # 如果消息是工具的输出。
        elif msg_type == "Tool":
            # 更新标题并将边框颜色设置为黄色。
            panel_args.update(title="工具输出", border_style="yellow")

        # 使用格式化的内容和自定义参数创建一个 Panel。
        # 然后，将面板打印到控制台。
        console.print(Panel(content, **panel_args))


def format_message(messages: List[Any]) -> None:
    """
    format_messages 函数的别名。

    这为任何可能仍在使用单数名称 `format_message` 的代码
    提供了向后兼容性。

    参数:
        messages: 要显示的消息对象列表。
    """
    # 调用主 format_messages 函数来执行渲染。
    format_messages(messages)
