"""Streamlit chat UI for the Safer First-Aid Chatbots project.

Lets you pick a system (RAG / VanillaLLM / IntentClassifier) and a config,
then chat with it. Purely a research demo front-end over the existing
``safer_firstaid`` package — no new business logic lives here.

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from safer_firstaid.baselines import IntentClassifierBaseline, VanillaLLMBaseline  # noqa: E402
from safer_firstaid.config import load_config  # noqa: E402
from safer_firstaid.llm import GenerationConfig, build_backend  # noqa: E402
from safer_firstaid.pipeline import DenseRetriever, RAGChatbot, SafetyLayer  # noqa: E402

st.set_page_config(page_title="Safer First-Aid Chatbot", page_icon="🩺", layout="centered")

CONFIG_DIR = ROOT / "configs"
SYSTEM_NAMES = ["RAG", "VanillaLLM", "IntentClassifier"]


@st.cache_resource(show_spinner=False)
def load_systems(config_path: str):
    cfg = load_config(config_path)
    safety = SafetyLayer()
    gen_config = GenerationConfig(temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)

    index_dir = ROOT / cfg.paths.index_dir
    if not index_dir.exists():
        raise FileNotFoundError(
            f"No FAISS index at {index_dir}. Build it first:\n"
            f"  safer-firstaid build-index --config {config_path}"
        )

    backend = build_backend(cfg.llm.provider, cfg.llm.model)
    retriever = DenseRetriever.load(index_dir)

    rag = RAGChatbot(retriever, backend, safety, cfg.retriever.top_k, gen_config)
    vanilla = VanillaLLMBaseline(backend, safety, gen_config)

    intents_path = ROOT / cfg.paths.intents
    intent_bot = IntentClassifierBaseline.from_json(intents_path, safety=safety)

    return {"RAG": rag, "VanillaLLM": vanilla, "IntentClassifier": intent_bot}, backend.name


def render_response(resp) -> None:
    decision = resp.safety
    if decision is not None and decision.escalate:
        st.error("⚠️ Emergency indicators detected — this response includes an escalation banner.")
    if decision is not None and decision.is_dangerous:
        st.warning("Original model output was flagged as dangerous and replaced with a safe fallback.")

    st.markdown(resp.answer)

    sources = resp.provenance()
    if sources:
        st.caption("Sources: " + ", ".join(sources))

    with st.expander("Debug / audit trail"):
        st.write("**Backend:**", resp.backend_name)
        if decision is not None:
            st.write("**Escalate:**", decision.escalate)
            st.write("**Triggered terms:**", decision.triggered_terms)
            st.write("**Dangerous flags:**", decision.dangerous_flags)
        st.write("**Raw (pre-safety) answer:**")
        st.code(resp.raw_answer)
        if getattr(resp, "retrieved", None):
            st.write("**Retrieved chunks:**")
            for r in resp.retrieved:
                st.text(f"[{r.score:.3f}] {r.document.source} — {r.document.text[:200]}...")


st.title("🩺 Safer First-Aid Chatbot")
st.caption(
    "Research prototype — general first-aid guidance only, not a substitute for "
    "professional medical care. Not for use in real emergencies."
)

with st.sidebar:
    st.header("Settings")
    config_files = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
    config_name = st.selectbox("Config", config_files, index=config_files.index("offline_test.yaml")
                                if "offline_test.yaml" in config_files else 0)
    system_name = st.selectbox("System", SYSTEM_NAMES, index=0)

    if st.button("Reload systems (clear cache)"):
        st.cache_resource.clear()
        st.rerun()

try:
    systems, backend_name = load_systems(str(CONFIG_DIR / config_name))
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # backend not reachable, bad config, etc.
    st.error(f"Failed to initialise systems: {e}")
    st.stop()

with st.sidebar:
    st.caption(f"Backend: `{backend_name}`")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.markdown(turn["content"])
        else:
            render_response(turn["response"])

query = st.chat_input("Describe the first-aid situation...")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    bot = systems[system_name]
    with st.chat_message("assistant"):
        with st.spinner(f"Asking {system_name}..."):
            resp = bot.answer(query)
        render_response(resp)

    st.session_state.history.append({"role": "assistant", "response": resp})