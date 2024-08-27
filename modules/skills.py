# Generate the skills module needed for the task

import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

def gen_skills(generator,messages):
    messages=[
        {"role": "system", "content": "你需要分析用户给出的任务，确定任务所需的技能，以无序列表的格式输出。请注意，技能描述应当尽量简单，不要包含过多的细节。例如，当用户需要LLM点评时事热点时，所需技能可能为：\n- 对各类社会热点事件了如指掌,能快速把握事件的来龙去脉\n- 善于从多角度分析事件,给出独特犀利的评论观点"},
    ] + messages
    response = generator.generate_response(messages)
    return response
