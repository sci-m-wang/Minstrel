# Generate the command module of the task

import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

def gen_command(client,messages):
    messages=[
        {"role": "system", "content": "你需要分析用户给出的任务，根据任务可能需要的动作为LLM创建命令提示词，以无序列表的格式输出，每个命令需要以“/”开头，然后连接命令符，之后是关于命令的解释。例如，当用户需要LLM玩谁是卧底游戏时，命令信息可能为：\n- /describe 请描述你的身份词\n- /vote 请投票给你认为是敌对阵营的玩家"},
    ] + messages
    response = client.generate_response(messages)
    return response
