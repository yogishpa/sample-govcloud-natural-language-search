"""Streamlit chat frontend for the NL Search Chatbot.

Provides a conversational UI that communicates with the FastAPI backend
for session management, search, and RAG-based answer generation.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

import json
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# HTTP session (persists cookies for ALB sticky sessions)
# ---------------------------------------------------------------------------

def _get_http_session() -> requests.Session:
    """Return a persistent requests.Session stored in Streamlit session state."""
    if "_http_session" not in st.session_state:
        st.session_state["_http_session"] = requests.Session()
    return st.session_state["_http_session"]


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def create_session() -> str | None:
    """Call POST /sessions on the backend and return the new session_id."""
    try:
        http = _get_http_session()
        resp = http.post(f"{BACKEND_URL}/sessions", timeout=10)
        resp.raise_for_status()
        return resp.json()["session_id"]
    except requests.exceptions.RequestException:
        st.error("Unable to connect to the backend service. Please try again later.")
        return None


def delete_session(session_id: str) -> None:
    """Call DELETE /sessions/{session_id} on the backend."""
    try:
        http = _get_http_session()
        http.delete(f"{BACKEND_URL}/sessions/{session_id}", timeout=10)
    except requests.exceptions.RequestException:
        pass  # best-effort cleanup


def stream_chat(query: str, session_id: str, search_mode: str) -> tuple[str, list[dict]]:
    """Call POST /chat/stream with SSE and yield tokens as they arrive.

    Returns the full answer text and a list of citation dicts.
    """
    payload = {
        "query": query,
        "session_id": session_id,
        "search_mode": search_mode,
    }

    full_answer = ""
    citations: list[dict] = []

    try:
        http = _get_http_session()
        with http.post(
            f"{BACKEND_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[len("data: "):])

                if "error" in data:
                    raise RuntimeError(data["error"])

                if "token" in data:
                    full_answer += data["token"]
                    yield data["token"]

                if data.get("done"):
                    citations = data.get("citations", [])

    except requests.exceptions.ConnectionError:
        st.error("Unable to reach the backend service. Please check your connection.")
    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again.")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            st.error("You are sending messages too quickly. Please wait a moment.")
        elif status == 404:
            st.error("Your session has expired. Please start a new chat.")
        else:
            st.error("Something went wrong. Please try again later.")
    except RuntimeError as exc:
        st.error(str(exc))
    except Exception:
        st.error("An unexpected error occurred. Please try again later.")

    # Store citations in session state so we can render them after streaming
    st.session_state["_pending_citations"] = citations


def render_citations(citations: list[dict]) -> None:
    """Render citations as clickable markdown links."""
    if not citations:
        return
    st.markdown("---")
    st.markdown("**Sources:**")
    for i, cite in enumerate(citations, 1):
        name = cite.get("document_name", "Unknown document")
        s3_uri = cite.get("s3_uri", "")
        score = cite.get("relevance_score", 0.0)
        st.markdown(f"{i}. [{name}]({s3_uri}) (relevance: {score:.2f})")


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NL Search Chatbot",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    search_mode = st.selectbox(
        "Search Mode",
        options=["semantic", "text", "hybrid"],
        index=0,
        help="Choose how documents are searched.",
    )

    if st.button("🗑️ New Chat", use_container_width=True):
        old_id = st.session_state.get("session_id")
        if old_id:
            delete_session(old_id)
        new_id = create_session()
        if new_id:
            st.session_state["session_id"] = new_id
            st.session_state["messages"] = []
            st.session_state["_pending_citations"] = []
            st.rerun()

# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    sid = create_session()
    if sid:
        st.session_state["session_id"] = sid
    else:
        st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "_pending_citations" not in st.session_state:
    st.session_state["_pending_citations"] = []

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

st.title("🔍 NL Search Chatbot")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            render_citations(msg["citations"])

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question…"):
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        st.session_state["_pending_citations"] = []
        response_text = st.write_stream(
            stream_chat(prompt, st.session_state["session_id"], search_mode)
        )

    citations = st.session_state.get("_pending_citations", [])

    # Render citations below the streamed answer
    if citations:
        with st.chat_message("assistant"):
            render_citations(citations)

    # Persist to message history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response_text if response_text else "",
        "citations": citations,
    })
    st.session_state["_pending_citations"] = []
