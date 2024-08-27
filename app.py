from showcases.generate import generate
from showcases.test import test
from models.openai import Generator
import streamlit as st

if __name__ == "__main__":
    state = st.session_state
    if "generator" not in state:
        state.generator = Generator(
            api_key = "YOUR_API_KEY",
            base_url = "BASE_URL"
        )
        state.generator.set_model("MODEL_NAME")
        pass
    if "page" not in state:
        state.page = "generate"
        pass
    if state.page == "generate":
        generate()
        pass
    elif state.page == "test":
        test()
        pass
