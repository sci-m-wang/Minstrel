# Description: Get the modules needed for the task
import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.
import json

def get_modules(generator,messages):
    '''
    Get the modules needed for the task
    :param client: OpenAI client
    :param text: The task description
    :return: The modules needed for the task
    The Correct format of the modules is:
    {
        background: bool,
        command: bool,
        suggesstion: bool,
        goal: bool,
        examples: bool,
        constraints: bool,
        workflow: bool,
        output_format: bool,
        skills: bool,
        style: bool,
        initialization: bool
    }
    '''
    default_modules = {
        "background": True,
        "command": False,
        "suggesstion": False,
        "goal": True,
        "examples": False,
        "constraints": True,
        "workflow": True,
        "output_format": True,
        "skills": False,
        "style": False,
        "initialization": True
    }
    ## Generate the modules needed for the task
    messages=[
        {"role": "system", "content": "你需要分析用户给出的任务类型，分析完整描述该任务所需的提示词需要的模块，例如：背景、目标、约束、命令、建议、任务样例、工作流程、输出格式、技能、风格、初始化等。按照json的格式输出，表示某个类是否需要，需要的类为True，不需要的类为False。例如，当需要背景、技能、工作流程、输出格式和初始化时，具体格式如下：{\"background\": True, \"command\": False, \"suggesstion\": False, \"goal\": False, \"examples\": False, \"constraints\": False, \"workflow\": True, \"output_format\": True, \"skills\": True, \"style\": False, \"initialization\": True}"},
    ] + messages
    response = generator.generate_response(messages).replace("```", "").replace("\n", "").replace("json", "").replace(" ", "").replace("True", "true").replace("False", "false")

    for i in range(5):
        ## Verify if the format of the modules is correct
        try:
            ## Load the modules
            print(response)
            modules = json.loads(response)
            ## Check if there are missing modules or extra modules
            for key in ["background", "command", "suggesstion", "goal", "examples", "constraints", "workflow", "output_format", "skills", "style", "initialization"]:
                if key not in modules:
                    modules[key] = False
                    pass
                pass
            extra_keys = []
            for key in modules.keys():
                if key not in ["background", "command", "suggesstion", "goal", "examples", "constraints", "workflow", "output_format", "skills", "style", "initialization"]:
                    extra_keys.append(key)
                    pass
                pass
            for key in extra_keys:
                del modules[key]
                pass
            return modules
        except Exception as e:
            print(e)
            continue
        pass
    ## Return the default modules if the format is incorrect
    return default_modules
