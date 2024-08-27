# Generate the initialization module needed for the task

import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

def gen_initialization(generator,messages):
    messages=[
        {"role": "system", "content": "你需要分析用户给出的任务，确定任务的初始化部分，即任务开始前和用户的一句话交流问候。例如，当用户需要LLM提供营养规划时，初始化信息可能为：\n作为一名营养规划师，我将根据您的情况给出营养建议。"},
    ] + messages
    response = generator.generate_response(messages)
    return response
