import sys
import os
abs_path = os.getcwd()
sys.path.append(abs_path) # Adds higher directory to python modules path.

from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import models, generate

from pydantic import BaseModel

schema = """
{
    "title": "Modules",
    "type": "object",
    "properties": {
        "background": {"type": "boolean"},
        "command": {"type": "boolean"},
        "suggesstion": {"type": "boolean"},
        "goal": {"type": "boolean"},
        "examples": {"type": "boolean"},
        "constraints": {"type": "boolean"},
        "workflow": {"type": "boolean"},
        "output_format": {"type": "boolean"},
        "skills": {"type": "boolean"},
        "style": {"type": "boolean"},
        "initialization": {"type": "boolean"}
    },
    "required": ["background", "command", "suggesstion", "goal", "examples", "constraints", "workflow", "output_format", "skills", "style", "initialization"]
}
"""

class Generator:
    def __init__(self, model_path, device):
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)
        self.llm = self.llm.eval()
        self.model = models.Transformers(self.llm, self.tokenizer)
        pass
    def generate_response(self, messages):
        g = generate.text(self.model)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        response = g(prompt)
        return response
    def json_response(self, messages):
        g = generate.json(self.model, schema)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        response = g(prompt)
        return response