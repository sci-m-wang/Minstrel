# Generate the constraints module needed for the task

import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

def gen_constraints(client,messages):
    messages=[
        {"role": "system", "content": "用户正在编写提示词，你需要帮助用户完善提示词的约束部分。首先分析用户给出的任务，确定执行任务需要的约束。然后直接指定关于约束的具体细节，例如对于长度约束，你应该指定如“长度不要超过20词”这种具体的约束，而不是“预期字数范围”这种模糊的约束方向。如果用户没有给出具体的约束，根据你的经验与知识帮用户补全内容。以无序列表的格式输出，不要输出任何交互信息。例如，当用户需要LLM为论文构思一个题目时，约束信息可能为：\n- 题目长度不超过20字\n- 不能出现侮辱性的词汇\n- 要使用专业术语"},
    ] + messages
    response = client.generate_response(messages)
    return response

