# Generate the workflow module needed for the task

import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

def gen_workflow(generator,messages):
    messages=[
        {"role": "system", "content": "你需要分析用户给出的任务，拆解执行这个任务需要完成的工作流程，以有序列表的格式输出，序号表示先后顺序，不要输出任何交互信息。对于需要判断的分支流程，可以用类似“1.1”和“1.2”这样的次级流程表示；对于需要迭代的循环流程，可以用“跳转至第2步”或“重复上一步”之类的流程表示。例如，当用户需要LLM求解一元二次方程时，工作流程信息可能为：\n1. 将方程化为标准形式，确定方程中的系数a、b和c。\n2. 计算方程的判别式，分析方程根的情况\n2.1 判别式＞0，方程有两个实数跟；\n2.2 判别式=0，方程有一个实数根；\n2.3 判别式<0，方程没有实数根\n3. 根据解的情况给出解的形式\n4. 求解方程的根\n5. 检验解的正确性"},
    ] + messages
    response = generator.generate_response(messages)
    return response
