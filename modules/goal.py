# Generate the goal module needed for the task

import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

def gen_goal(generator,messages):
    messages=[
        {"role": "system", "content": "你需要分析用户给出的任务，确定任务的目标，以无序列表的格式输出，不要输出任何交互信息。请注意，目标应当尽量简单，在任务没有明显的多种行为的形况下，通常为1条，一般不超过2条。例如，当用户需要LLM计算方程的解时，目标信息可能为：\n- 计算出正确的方程解"},
    ] + messages
    response = generator.generate_response(messages)
    return response
