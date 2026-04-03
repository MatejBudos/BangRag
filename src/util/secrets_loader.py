import os

from dotenv import load_dotenv


def get_secret_value(name: str) -> str:
    load_dotenv()

    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError

        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except StreamlitSecretNotFoundError:
            pass
    except Exception:
        pass

    return os.getenv(name, "")
