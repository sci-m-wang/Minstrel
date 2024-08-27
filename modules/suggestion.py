# Generate the suggestion module of the task

import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

def gen_suggestion(client,messages):
    messages=[
        {"role": "system", "content": "你需要分析用户给出的任务，生成任务的建议信息，以无序列表的格式输出，不要输出任何交互信息。例如，当用户需要LLM玩谁是卧底游戏时，建议信息可能为：\n- 当无法确定你的阵营时，你的描述应该尽量模糊"},
    ] + messages
    response = client.generate_response(messages)
    return response
