'''
langchain工具模版
@tool
def 工具名(参数1: 类型, 参数2: 类型) -> 返回类型:
    """
    清晰描述工具功能的文档字符串。
    模型完全依赖这个描述来决定是否调用此工具。
    
    Args:
        参数1: 参数说明。
        参数2: 参数说明。
        
    Returns:
        返回结果的说明。
        
    示例:
        可以给出调用示例。
    """
    # 1. 在这里编写核心逻辑（计算、查询API等）
    # 2. 处理可能发生的错误
    # 3. 关键：用 return 返回一个字符串结果
    return "格式良好的结果字符串"
'''


from langchain.tools import tool

import math
import requests

# @tool 把函数变成了一个具有标准化接口的“工具对象”
@tool
def search_Weather(city:str )->str:
    #描述， 告诉大模型这个工具是做什么的
    """在网上查询天气
    Args:
        city: 城市
        
    """
    try:
        url = f"http://wttr.in/{city}?format=%C+%t"  # 免费API，无需key
        response = requests.get(url)
        return response.text.strip() if response.status_code == 200 else "无法获取天气"
    except:
        return "天气查询失败"
    





# 自定义工具    自定义工具名  可省略  ;  description :工具描述  args_schema : 参数描述  用到哪个写哪个
#eval() python内置函数， 把字符串当成Python代码执行，并返回计算结果
'''
第二个参数：{"builtins": {}}（globals字典）最关键的安全措施！
默认eval能访问所有内置函数（如open、import、exec）。
这里把__builtins__设成空字典{}：完全禁用内置函数。
效果：恶意代码如__import__('os')会直接报NameError，无法执行。
这叫“禁用内置”沙盒，防止代码注入攻击。

第三个参数：{"math": math}（locals字典）提供有限的本地变量/模块。
这里只允许用math模块   提前（import math）。
所以表达式能用math.sqrt、math.pi、math.sin等，但不能用其他（如os、sys）。
'''

@tool("domath",description="计算输入的数学表达式")
def calculator(expression: str) -> str:
    # 当让模型进行幂运算时出错，Python的运算符陷阱：^ 在Python是按位异或（XOR），不是幂运算
    """精确计算数学表达式，支持幂运算（如2^5或2**5）。输入纯表达式字符串。"""
    safe_expression = expression.replace("^","**")
    result = eval(safe_expression, {"__builtins__": {}}, {"math": math})


    try:
        
        return str(result)
    except:
        return "计算错误，请检查表达式"




@tool("sayHello",description="say hello to you!")
def greeting(name:str)->str:
    
    return print(f"你好{name}")


if __name__ == "__main__":
    print(greeting.invoke("张三"))
    print(search_Weather.name)
    print(search_Weather.args)
    print(search_Weather.description)

    print(calculator.name)
#测试调用， 用invoke
    print(search_Weather.invoke({'city':"成都"}))