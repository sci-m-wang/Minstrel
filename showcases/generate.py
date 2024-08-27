import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

import streamlit as st
from modules.get_modules import get_modules
from modules.background import gen_background
from modules.command import gen_command
from modules.constraints import gen_constraints
from modules.goal import gen_goal
from modules.initialization import gen_initialization
from modules.output_format import gen_output_format
from modules.skills import gen_skills
from modules.suggestion import gen_suggestion
from modules.workflow import gen_workflow

module_name_dict = {
    "background": "背景",
    "command": "命令",
    "suggesstion": "建议",
    "goal": "目标",
    "examples": "任务样例",
    "constraints": "约束",
    "workflow": "工作流程",
    "output_format": "输出格式",
    "skills": "技能",
    "style": "风格",
    "initialization": "初始化"
}

module_func_dict = {
    "background": gen_background,
    "command": gen_command,
    "suggesstion": gen_suggestion,
    "goal": gen_goal,
    "examples": None,
    "constraints": gen_constraints,
    "workflow": gen_workflow,
    "output_format": gen_output_format,
    "skills": gen_skills,
    "style": None,
    "initialization": gen_initialization
}

## The page to generate the LangGPT prompt
def generate():
    state = st.session_state
    ## A text input for the user to input the basic description of the task
    col1, col2 = st.columns([8, 2])
    with col1:
        task = st.text_input("任务描述","撰写科幻小说",label_visibility="collapsed")
        pass
    ## A button to analyze the task and generate the modules
    with col2:
        if st.button("分析任务",type="primary"):
            ## Get the modules
            state.module_messages = [{"role": "user", "content": f"我希望LLM帮我执行的任务是：{task}"}]
            state.modules = get_modules(state.generator, state.module_messages)
            pass
    with st.sidebar:
        st.subheader("基本信息")
        state.role_name = st.text_input("助手名称","",help="例如：大模型、助手等")
        state.author = st.text_input("作者","LangGPT")
        state.version = st.number_input("版本",min_value=0.1,value=0.1,step=0.1)
        state.description = st.text_area("描述","这是一个LangGPT生成的助手",height=100)
        st.subheader("模块控制")
        if "modules" not in state:
            state.modules = {
                "background": False,
                "command": False,
                "suggesstion": False,
                "goal": False,
                "examples": False,
                "constraints": False,
                "workflow": False,
                "output_format": False,
                "skills": False,
                "style": False,
                "initialization": False
            }
        ## Some toggles to show the modules
        if "on_modules" not in state:
            state.on_modules = {}
            pass
        for key in state.modules.keys():
            if key in module_name_dict:
                state.on_modules[key] = st.toggle(module_name_dict[key],state.modules[key])
                pass
            pass
        pass
    if "modules" in state:
        if state.on_modules["examples"]:
            st.subheader("请提供任务样例：")
            input_area, output_area = st.columns(2)
            with input_area:
                input_example = st.text_area("样例输入","")
                pass
            with output_area:
                output_example = st.text_area("样例输出","")
                pass
            state.examples = {
                "input": input_example,
                "output": output_example
            }
            pass
        if state.on_modules["style"]:
            st.subheader("请指定回复的风格：")
            style = st.text_input("风格","",help="例如：正式、幽默、严肃等",label_visibility="collapsed")
            state.style = style
            pass
        ## A button to control the generation of the modules
        for key in state.modules.keys():
            if key in state:
                if state.on_modules[key]:
                    with st.expander(module_name_dict[key]):
                        st.text_area(module_name_dict[key],state[key],label_visibility="collapsed")
                        pass
            pass
        g,c = st.columns([1,1])
        with g:
            generate_button = st.button("生成模块")
            pass
        with c:
            compose_button = st.button("合成提示")
            pass
        if generate_button:
            for key in state.modules.keys():
                if key == "examples" or key == "style":
                    continue
                else:
                    if state.on_modules[key]:
                        if key not in state:
                            state[key] = module_func_dict[key](state.generator,state.module_messages)
                    pass
                pass
            st.rerun()
            pass
        if compose_button:
            if "prompt" not in state:
                state.prompt = ""
                pass
            if state.role_name:
                state.prompt += f"# Role: {state.role_name}\n"
                pass
            state.prompt += f"## profile\n"
            if state.author:
                state.prompt += f"- Author: {state.author}\n"
                pass
            if state.version:
                state.prompt += f"- Version: {state.version}\n"
                pass
            if state.description:
                state.prompt += f"- Description: {state.description}\n"
                pass
            ## Check if all the checked modules are generated
            for key in state.modules.keys():
                if state.on_modules[key]:
                    if key not in state:
                        st.error(f"请先生成{module_name_dict[key]}")
                        return
                    else:
                        if key == "examples":
                            state.prompt += f"## {module_name_dict[key]}\n"
                            state.prompt += f"### 输入\n"
                            state.prompt += state.examples["input"]
                            state.prompt += "\n"
                            state.prompt += f"### 输出\n"
                            state.prompt += state.examples["output"]
                            state.prompt += "\n\n"
                        state.prompt += f"## {key}\n"
                        state.prompt += state[key]
                        state.prompt += "\n\n"
                        state.page = "test"
                pass
            st.rerun()

