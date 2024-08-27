# Generate the background module of the task

import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

def gen_background(client,messages):
    messages=[
        {"role": "system", "content": "你需要分析用户给出的任务，生成任务的背景信息，以无序列表的格式输出。例如，当用户需要LLM玩谁是卧底游戏时，背景信息可能为：\n- 你正在参与一场谁是卧底游戏\n- 你的身份词是“黄桃”。"},
    ] + messages
    response = client.generate_response(messages)
    return response
