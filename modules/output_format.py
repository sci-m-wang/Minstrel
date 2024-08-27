# Generate the output_format module needed for the task

import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

def gen_output_format(generator,messages):
    messages=[
        {"role": "system", "content": "你需要分析用户给出的任务，确定任务的输出格式，不要输出任何交互信息。请注意，输出格式应当尽量简单，优先考虑json、xml等标准格式，如果标准格式不适用于这个任务，用一句话简要描述规定的格式，不要包含过多的细节。例如，当用户需要LLM计算方程的解时，输出格式信息可能为：\n- 输出仅为一个数字，表示方程的解"},
    ] + messages
    response = generator.generate_response(messages)
    return response
