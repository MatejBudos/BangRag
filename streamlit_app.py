import json
import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from main import DEFAULT_SOURCE_PATH, create_pipeline, get_files_in_directory


LOG_DIR = Path("logs")

st.set_page_config(page_title="Bang! RAG", page_icon="🤠", layout="wide")


@st.cache_resource
def get_pipeline():
    return create_pipeline()


def get_secret_value(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except StreamlitSecretNotFoundError:
        pass
    return os.getenv(name, "")


def ensure_index(pipeline) -> int:
    try:
        return pipeline.datastore.table.count_rows()
    except Exception:
        return 0


def reindex_rules(pipeline) -> int:
    document_paths = get_files_in_directory(DEFAULT_SOURCE_PATH)
    if not document_paths:
        return 0
    pipeline.reset()
    pipeline.add_documents(document_paths)
    return pipeline.datastore.table.count_rows()


def list_log_files() -> list[Path]:
    if not LOG_DIR.exists():
        return []
    return sorted(LOG_DIR.glob("evaluation-*.json"), reverse=True)


def load_log_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_log_file(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_effective_result(result: dict) -> bool:
    manual_value = result.get("manual_is_correct")
    if manual_value is None:
        return bool(result.get("is_correct"))
    return bool(manual_value)


def render_card(title: str, content: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(content if content else "-")


def render_user_gate() -> bool:
    app_password = get_secret_value("APP_PASSWORD")
    if not app_password:
        return True

    if st.session_state.get("app_authenticated"):
        return True

    st.title("Bang! RAG")
    st.caption("Zadaj heslo pre pristup k chatbotovi")

    with st.form("app_login_form", clear_on_submit=False):
        password = st.text_input("Heslo", type="password")
        submitted = st.form_submit_button("Prihlasit")
        if submitted:
            if password == app_password:
                st.session_state["app_authenticated"] = True
                st.rerun()
            else:
                st.error("Nespravne heslo.")

    st.stop()


def render_developer_access() -> bool:
    dev_password = get_secret_value("DEV_PASSWORD")
    if not dev_password:
        return True

    if st.session_state.get("dev_authenticated"):
        return True

    st.markdown("**Developer Access**")
    with st.form("dev_login_form", clear_on_submit=False):
        password = st.text_input("Developer heslo", type="password")
        submitted = st.form_submit_button("Odomknut developer mode")
        if submitted:
            if password == dev_password:
                st.session_state["dev_authenticated"] = True
                st.rerun()
            else:
                st.error("Nespravne developer heslo.")

    return False


render_user_gate()

pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Som Bang! chatbot. Pytaj sa na pravidla, karty, postavy a interakcie.",
        }
    ]


st.title("Bang! RAG")
st.caption("Chat nad lokalnou Bang RAG pipeline")

with st.sidebar:
    developer_mode = False
    if render_developer_access():
        developer_mode = st.toggle("Developer Mode", value=False)

        if developer_mode:
            st.subheader("Databaza pravidiel")

            current_rows = ensure_index(pipeline)
            st.write(f"Indexovane chunky: `{current_rows}`")
            source_paths = get_files_in_directory(DEFAULT_SOURCE_PATH)
            st.write(f"Najdene source subory: `{len(source_paths)}`")
            if not source_paths:
                st.warning(
                    "V deployi sa nenasli ziadne .tex subory v BangRules/. Skontroluj, ci je BangRules/ naozaj v repozitari a dostal sa do deploya."
                )

            if st.button("Reset a znovu naindexovat pravidla", use_container_width=True):
                with st.spinner("Indexujem Bang pravidla..."):
                    current_rows = reindex_rules(pipeline)
                if current_rows == 0:
                    st.error(
                        "Reindex neprebehol, lebo sa nenasli ziadne zdrojove pravidla alebo sa z nich nic nevyparsovalo."
                    )
                else:
                    st.success(f"Hotovo. Indexovanych chunkov: {current_rows}")

            st.markdown(
                "Spustenie:\n```powershell\nstreamlit run streamlit_app.py\n```"
            )
            st.caption(
                "Existujuci index sa pouzije tak, ako je. Reindex prebehne iba po stlaceni tlacidla vyssie."
            )


if developer_mode:
    chat_tab, logs_tab = st.tabs(["Chat", "Evaluation Logs"])
else:
    chat_tab = st.container()
    logs_tab = None


with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Opytaj sa na Bang pravidla...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Hladam relevantne pravidla..."):
                chunk_count = ensure_index(pipeline)
                if chunk_count == 0:
                    answer = (
                        "Databaza pravidiel je prazdna. V developer mode skontroluj, ci deploy obsahuje BangRules/ a potom spusti reset a znovu naindexovanie pravidiel."
                    )
                else:
                    chunks = pipeline.retriever.search(question)
                    answer = pipeline.response_generator.generate_response(question, chunks)

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


if developer_mode and logs_tab is not None:
    with logs_tab:
        st.subheader("Evaluation Logs")

        log_files = list_log_files()
        if not log_files:
            st.info("V priecinku logs zatial nie je ziaden evaluation log.")
        else:
            selected_log = st.selectbox(
                "Vyber log file",
                options=log_files,
                format_func=lambda path: path.name,
            )
            log_data = load_log_file(selected_log)
            results = log_data.get("results", [])

            if not results:
                st.warning("Vybrany log neobsahuje ziadne vysledky.")
            else:
                state_key = f"log_index::{selected_log.name}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = 0

                max_index = len(results) - 1
                current_index = st.session_state[state_key]
                current_index = max(0, min(current_index, max_index))
                st.session_state[state_key] = current_index

                top_left, top_mid, top_right = st.columns([1, 2, 1])
                with top_left:
                    if st.button("Predchadzajuca", disabled=current_index == 0):
                        st.session_state[state_key] = max(0, current_index - 1)
                        st.rerun()
                with top_mid:
                    selected_number = st.number_input(
                        "Otazka",
                        min_value=1,
                        max_value=len(results),
                        value=current_index + 1,
                        step=1,
                    )
                    if selected_number - 1 != current_index:
                        st.session_state[state_key] = selected_number - 1
                        st.rerun()
                with top_right:
                    if st.button("Dalsia", disabled=current_index == max_index):
                        st.session_state[state_key] = min(max_index, current_index + 1)
                        st.rerun()

                result = results[st.session_state[state_key]]
                effective_correct = get_effective_result(result)
                verdict = "OK" if effective_correct else "FAIL"
                verdict_color = "green" if effective_correct else "red"
                auto_verdict = "OK" if result.get("is_correct") else "FAIL"
                auto_verdict_color = "green" if result.get("is_correct") else "red"
                effective_score = sum(1 for item in results if get_effective_result(item))

                st.caption(
                    f"Log: {selected_log.name} | Otazka {st.session_state[state_key] + 1}/{len(results)} | "
                    f"Skore: {effective_score}/{log_data.get('total_questions', len(results))}"
                )
                st.markdown(
                    f"**Verdict:** <span style='color:{verdict_color}'>{verdict}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Auto Verdict:** <span style='color:{auto_verdict_color}'>{auto_verdict}</span>",
                    unsafe_allow_html=True,
                )

                manual_mode_key = f"manual_mode::{selected_log.name}::{st.session_state[state_key]}"
                manual_note_key = f"manual_note::{selected_log.name}::{st.session_state[state_key]}"

                current_manual_value = result.get("manual_is_correct")
                if current_manual_value is None:
                    manual_mode_default = "Auto"
                elif current_manual_value:
                    manual_mode_default = "OK"
                else:
                    manual_mode_default = "FAIL"

                if manual_mode_key not in st.session_state:
                    st.session_state[manual_mode_key] = manual_mode_default
                if manual_note_key not in st.session_state:
                    st.session_state[manual_note_key] = result.get("manual_note", "")

                st.markdown("**Manualny Eval**")
                manual_col1, manual_col2 = st.columns([1, 2])
                with manual_col1:
                    manual_mode = st.radio(
                        "Verdict",
                        options=["Auto", "OK", "FAIL"],
                        index=["Auto", "OK", "FAIL"].index(st.session_state[manual_mode_key]),
                        key=manual_mode_key,
                        horizontal=True,
                    )
                with manual_col2:
                    manual_note = st.text_input(
                        "Poznamka",
                        key=manual_note_key,
                        placeholder="Volitelna poznamka k manualnemu vyhodnoteniu",
                    )

                save_col1, save_col2 = st.columns([1, 3])
                with save_col1:
                    if st.button("Ulozit manualny eval", use_container_width=True):
                        if manual_mode == "Auto":
                            result["manual_is_correct"] = None
                        else:
                            result["manual_is_correct"] = manual_mode == "OK"
                        result["manual_note"] = manual_note.strip()
                        log_data["correct_answers"] = sum(
                            1 for item in results if get_effective_result(item)
                        )
                        save_log_file(selected_log, log_data)
                        st.success("Manualny eval bol ulozeny.")
                        st.rerun()
                with save_col2:
                    st.caption(
                        "Manualny verdict prepise automaticky verdict len pre tento log. Rezim Auto vrati povodne AI vyhodnotenie."
                    )

                col1, col2 = st.columns(2)
                with col1:
                    render_card("Otazka", result.get("question", ""))
                    render_card("Expected Answer", result.get("expected_answer", ""))
                with col2:
                    render_card("Model Response", result.get("response", ""))
                    render_card("Reasoning", result.get("reasoning", ""))
                render_card("Manual Note", result.get("manual_note", ""))
