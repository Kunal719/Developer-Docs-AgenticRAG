import json
import queue
import threading
import time
import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

API_URL = f"{API_BASE_URL}/runs"
HEALTH_URL = f"{API_BASE_URL}/health"

GOOGLE_LOGIN_URL = os.getenv("GOOGLE_LOGIN_URL", f"{API_BASE_URL}/auth/google/login")


def get_api_session() -> requests.Session:
    """
    Create an HTTP session for FastAPI requests and forward
    the browser's FastAPI session cookie.
    """
    session = requests.Session()

    session_cookie = st.context.cookies.get("fastapi-session")

    if session_cookie:
        session.cookies.set(
            "fastapi-session",
            session_cookie,
            path="/",
        )

    return session

def get_runs():
    response = api_session.get(f"{API_BASE_URL}/runs", timeout=5)

    response.raise_for_status()

    return response.json()

def get_run(run_id):
    response = api_session.get(f"{API_BASE_URL}/runs/{run_id}", timeout=5)

    if response.status_code == 404:
        return None
    
    response.raise_for_status()
    
    return response.json()

def api_unavailable_message():
    st.error(
        "The RAG backend is currently unavailable. "
        "Please make sure the FastAPI server is running and try again."
    )

def check_backend() -> bool:
    try:
        response = requests.get(
            HEALTH_URL,
            timeout=1,
        )

        return (
            response.status_code == 200
            and response.json().get("status") == "ok"
        )

    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Developer Docs Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: #0b0f14;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 7rem;

        background:
            radial-gradient(
                circle at 50% 15%,
                rgba(118, 103, 255, 0.035),
                transparent 42%
            );
}

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background: #080c11;
        border-right: 1px solid #1d2630;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.8rem;
    }

    .sidebar-logo {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #6d5dfc, #8b7cff);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 19px;
    }

    .sidebar-title {
        font-size: 16px;
        font-weight: 700;
        color: #f3f5f7;
        line-height: 1.1;
    }

    .sidebar-subtitle {
        font-size: 11px;
        color: #7f8b99;
        margin-top: 3px;
    }

    .sidebar-section {
        color: #657181;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 1.5rem;
        margin-bottom: 0.65rem;
    }

    .status-card {
        background: #0f151d;
        border: 1px solid #1d2935;
        border-radius: 10px;
        padding: 12px 13px;
        margin-bottom: 8px;
    }

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 7px;
    }

    .status-row:last-child {
        margin-bottom: 0;
    }

    .status-label {
        color: #8b97a5;
        font-size: 12px;
    }

    .status-value {
        color: #d8dee6;
        font-size: 12px;
        font-weight: 600;
    }

    .status-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        margin-right: 5px;
        border-radius: 50%;
    }

    .status-dot.online {
        background: #35d07f;
        box-shadow: 0 0 8px rgba(53, 208, 127, 0.45);
    }

    .status-dot.offline {
        background: #e05a67;
        box-shadow: 0 0 8px rgba(224, 90, 103, 0.35);
    }

    .architecture-card {
        background: #0f151d;
        border: 1px solid #1d2935;
        border-radius: 10px;
        padding: 12px 13px;
        color: #8995a4;
        font-size: 11px;
        line-height: 1.8;
    }

    .architecture-card strong {
        color: #dce2e9;
    }

    /* ---------- Header ---------- */

    .hero {
        padding: 1.2rem 0 1.6rem 0;
        border-bottom: 1px solid #1a222c;
        margin-bottom: 1.5rem;
    }

    .hero-row {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .hero-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(135deg, #6d5dfc, #8b7cff);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 8px 30px rgba(109, 93, 252, 0.22);
    }

    .hero-title {
        font-size: 29px;
        font-weight: 750;
        color: #f4f6f8;
        letter-spacing: -0.025em;
    }

    .hero-subtitle {
        color: #7f8b99;
        font-size: 13px;
        margin-top: 4px;
    }

    /* ---------- Empty state ---------- */

    .empty-state {
        text-align: center;
        padding-top: 4.8rem;
        padding-bottom: 3.2rem;
    }

    .empty-icon {
        font-size: 42px;
        margin-bottom: 14px;
    }

    .empty-orb {
        width: 66px;
        height: 66px;
        margin: 0 auto 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 19px;
        background:
            radial-gradient(
                circle at 30% 25%,
                rgba(154, 142, 255, 0.35),
                transparent 45%
            ),
            linear-gradient(
                145deg,
                rgba(118, 103, 255, 0.20),
                rgba(95, 140, 255, 0.08)
            );
        border: 1px solid rgba(130, 117, 255, 0.25);
        box-shadow:
            0 0 0 8px rgba(118, 103, 255, 0.025),
            0 15px 50px rgba(90, 75, 220, 0.16);
        font-size: 27px;
        animation: orb-float 4s ease-in-out infinite;
    }

    @keyframes orb-float {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-4px);
        }
    }

    .empty-title {
        color: #f0f2f6;
        font-size: 24px;
        font-weight: 720;
        letter-spacing: -0.035em;
        margin-bottom: 8px;
    }

    .empty-text {
        max-width: 560px;
        margin: 0 auto;
        color: #707c8b;
        font-size: 12px;
        line-height: 1.7;
    }

    /* ---------- Chat ---------- */

    [data-testid="stChatMessage"] {
        background: transparent;
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background: #262d37;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #6d5dfc, #8b7cff);
    }

    [data-testid="stChatMessageContent"] {
        max-width: 850px;
    }

    /* ---------- Sources ---------- */

    .sources-header {
        color: #7f8b99;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 1.1rem;
        margin-bottom: 0.45rem;
    }

    .source-item {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 7px 9px;
        margin: 3px 0;
        border-radius: 7px;
        background: #0e141b;
        border: 1px solid #18222c;
    }

    .source-rank {
        color: #707c8b;
        font-size: 11px;
        font-weight: 700;
        min-width: 20px;
    }

    .source-path {
        color: #cbd3dc;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 11px;
        overflow-wrap: anywhere;
    }

    .source-badge {
        margin-left: auto;
        white-space: nowrap;
        padding: 2px 6px;
        border-radius: 5px;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .source-badge.documentation {
        color: #7fc7ff;
        background: #102333;
        border: 1px solid #19384f;
    }

    .source-badge.implementation {
        color: #b69cff;
        background: #1b1630;
        border: 1px solid #302456;
    }

    /* ---------- Streaming indicator ---------- */

    .streaming-indicator {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #7f8b99;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .streaming-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #8b7cff;
        box-shadow: 0 0 10px rgba(139, 124, 255, 0.8);
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #26313d;
        background: #111820;
        color: #cbd3dc;
        font-size: 12px;
    }

    .stButton > button:hover {
        border-color: #6d5dfc;
        color: #ffffff;
    }

    /* ---------- Chat input ---------- */

    [data-testid="stChatInput"] {
        border-top: none;
        background:
            linear-gradient(
                to bottom,
                rgba(11, 14, 19, 0),
                #0b0e13 25%
            );
        padding-top: 12px;
    }

    /* ---------- Expander ---------- */

    [data-testid="stExpander"] {
        border: 1px solid #1b2530;
        border-radius: 9px;
        background: #0d131a;
    }

    /* ---------- Hide Streamlit branding ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_run_id" not in st.session_state:
    st.session_state.selected_run_id = st.query_params.get("run_id")

if "loaded_run_id" not in st.session_state:
    st.session_state.loaded_run_id = None

if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None

if "current_run_question" not in st.session_state:
    st.session_state.current_run_question = None

if "backend_online" not in st.session_state:
    st.session_state.backend_online = False

if "resume_run_id" not in st.session_state:
    st.session_state.resume_run_id = None

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

# if "session" not in st.context.cookies:
#     st.info("Please sign in to use the Developer Docs Assistant.")

#     st.link_button(
#         "Sign in with Google",
#         GOOGLE_LOGIN_URL,
#         use_container_width=True,
#     )

#     st.stop()

if not st.context.cookies.get("fastapi-session"):
    st.info("Please sign in to use the Developer Docs Assistant.")

    st.link_button(
        "Sign in with Google",
        GOOGLE_LOGIN_URL,
        use_container_width=True,
    )

    st.stop()

# Create API session
api_session = get_api_session()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    # -----------------------------------------------------------------------
    # Brand
    # -----------------------------------------------------------------------

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">📚</div>
            <div>
                <div class="sidebar-title">Developer Docs</div>
                <div class="sidebar-subtitle">RAG Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # New conversation
    # -----------------------------------------------------------------------

    if st.button(
        "＋  New conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.selected_run_id = None
        st.query_params.clear()
        st.session_state.loaded_run_id = None
        st.rerun()

    # -----------------------------------------------------------------------
    # System
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">System</div>',
        unsafe_allow_html=True,
    )

    @st.fragment(run_every=5)
    def backend_status():

        backend_online = check_backend()

        previous_status = st.session_state.backend_online

        st.session_state.backend_online = backend_online

        if backend_online != previous_status:
            st.rerun(scope="app")

        if backend_online:
            status_dot = "online"
            status_text = "Connected"
        else:
            status_dot = "offline"
            status_text = "Offline"

        st.html(
            f"""
            <div class="status-card">

                <div class="status-row">

                    <span class="status-label">
                        API
                    </span>

                    <span class="status-value">
                        <span class="status-dot {status_dot}"></span>
                        {status_text}
                    </span>

                </div>

                <div class="status-row">

                    <span class="status-label">
                        Retrieval
                    </span>

                    <span class="status-value">
                        Hybrid + RRF
                    </span>

                </div>

                <div class="status-row">

                    <span class="status-label">
                        Reranking
                    </span>

                    <span class="status-value">
                        Cross-Encoder
                    </span>

                </div>

                <div class="status-row">

                    <span class="status-label">
                        Corpora
                    </span>

                    <span class="status-value">
                        2
                    </span>

                </div>

            </div>
            """
        )

    backend_status()

    # -----------------------------------------------------------------------
    # Recent conversations
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Recent conversations</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.backend_online:
        runs = get_runs()
    else:
        runs = []

    current_run_id = st.session_state.current_run_id

    if (current_run_id and current_run_id not in {run["run_id"] for run in runs}):
        runs.insert(
            0,
            {
                "run_id": current_run_id,
                "question": st.session_state.current_run_question,
                "status": "running",
            },
        )

    for run in runs:
        question_text = run["question"]
        words = question_text.split()

        label = " ".join(words[:5])

        if len(words) > 5:
            label += "..."

        if run["status"] == "running":
            label = "⏳ " + label
        elif run["status"] == "interrupted":
            label = "⚠️ " + label
        elif run["status"] == "failed":
            label = "❌ " + label

        if st.button(
            label,
            key=f"run_{run['run_id']}",
            use_container_width=True,
        ):
            st.session_state.selected_run_id = run["run_id"]
            st.query_params["run_id"] = run["run_id"]
            st.rerun()

    # -----------------------------------------------------------------------
    # Runtime Pipeline
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Runtime Pipeline</div>',
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div class="architecture-card">

            <div style="
                padding: 7px 8px;
                border: 1px solid #312b5c;
                border-radius: 6px;
                background: #17152a;
                color: #d9d5ff;
                text-align: center;
                font-weight: 600;
            ">
                ◉ User Query
            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding: 8px;
                border: 1px solid #1c2530;
                border-radius: 7px;
                background: #0f141b;
            ">

                <div style="
                    color:#8e99a7;
                    font-size:9px;
                    font-weight:700;
                    margin-bottom:5px;
                ">
                    CORPORA
                </div>

                <div style="
                    padding:5px;
                    margin:3px 0;
                    border-radius:4px;
                    background:#131921;
                    color:#aeb8c4;
                ">
                    <strong>Documentation</strong>
                    <br>

                    <span style="color:#7b8795;">
                        ├─ Dense Retrieval
                        <br>
                        └─ BM25 Retrieval
                    </span>
                </div>

                <div style="
                    padding:5px;
                    margin:3px 0;
                    border-radius:4px;
                    background:#131921;
                    color:#aeb8c4;
                ">
                    <strong>Implementation</strong>
                    <br>

                    <span style="color:#7b8795;">
                        ├─ Dense Retrieval
                        <br>
                        └─ BM25 Retrieval
                    </span>
                </div>

            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:7px 8px;
                border:1px solid #202a35;
                border-radius:6px;
                background:#111720;
                color:#b9c2cd;
                text-align:center;
                font-weight:600;
            ">
                Reciprocal Rank Fusion

                <div style="
                    color:#687583;
                    font-size:8px;
                    font-weight:500;
                    margin-top:2px;
                ">
                    RRF
                </div>
            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:7px 8px;
                border:1px solid #202a35;
                border-radius:6px;
                background:#111720;
                color:#b9c2cd;
                text-align:center;
                font-weight:600;
            ">
                Multi-Corpus Retriever
            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:7px 8px;
                border:1px solid #40366e;
                border-radius:6px;
                background:#1a1730;
                color:#c6bcff;
                text-align:center;
                font-weight:600;
            ">
                Cross-Encoder Reranker
            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:7px 8px;
                border:1px solid #202a35;
                border-radius:6px;
                background:#111720;
                color:#b9c2cd;
                text-align:center;
                font-weight:600;
            ">
                Top-K Documents
            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:7px 8px;
                border:1px solid #202a35;
                border-radius:6px;
                background:#111720;
                color:#b9c2cd;
                text-align:center;
                font-weight:600;
            ">
                Context Formatter

                <div style="
                    color:#687583;
                    font-size:8px;
                    font-weight:500;
                    margin-top:2px;
                ">
                    → RAG Prompt
                </div>
            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:7px 8px;
                border:1px solid #312b5c;
                border-radius:6px;
                background:#17152a;
                color:#d9d5ff;
                text-align:center;
                font-weight:600;
            ">
                LLM
            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:7px 8px;
                border:1px solid #214735;
                border-radius:6px;
                background:#10231b;
                color:#91ddb6;
                text-align:center;
                font-weight:600;
            ">
                ✦ Streaming Answer
            </div>

        </div>
        """
    )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div class="architecture-card">

            <div style="
                padding:7px 8px;
                border:1px solid #202a35;
                border-radius:6px;
                background:#111720;
                color:#b9c2cd;
                text-align:center;
                font-weight:600;
            ">
                Benchmark Dataset
            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:7px 8px;
                border:1px solid #312b5c;
                border-radius:6px;
                background:#17152a;
                color:#d9d5ff;
                text-align:center;
                font-weight:600;
            ">
                Benchmark Runner
            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:8px;
                border:1px solid #1c2530;
                border-radius:7px;
                background:#0f141b;
            ">

                <div style="
                    color:#8e99a7;
                    font-size:9px;
                    font-weight:700;
                    margin-bottom:5px;
                ">
                    RETRIEVAL METRICS
                </div>

                <div style="
                    color:#aeb8c4;
                    line-height:1.8;
                ">
                    ├─ Recall@K
                    <br>
                    ├─ Precision@K
                    <br>
                    ├─ Hit Rate
                    <br>
                    └─ MRR
                </div>

            </div>

            <div style="height:7px;"></div>

            <div style="
                padding:8px;
                border:1px solid #1c2530;
                border-radius:7px;
                background:#0f141b;
            ">

                <div style="
                    color:#8e99a7;
                    font-size:9px;
                    font-weight:700;
                    margin-bottom:5px;
                ">
                    PIPELINE METRICS
                </div>

                <div style="
                    color:#aeb8c4;
                    line-height:1.8;
                ">
                    ├─ Retrieval Latency
                    <br>
                    ├─ Reranker Latency
                    <br>
                    ├─ Generation Latency
                    <br>
                    ├─ Token Usage
                    <br>
                    └─ Estimated Cost
                </div>

            </div>

            <div style="text-align:center;color:#4d5866;">↓</div>

            <div style="
                padding:7px 8px;
                border:1px solid #214735;
                border-radius:6px;
                background:#10231b;
                color:#91ddb6;
                text-align:center;
                font-weight:600;
            ">
                Evaluation Results
            </div>

        </div>
        """
    )

    # -----------------------------------------------------------------------
    # Project
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Project</div>',
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div class="status-card">

            <div class="status-row">

                <span class="status-label">
                    Repository
                </span>

                <a
                    href="https://github.com/Kunal719/Developer_Docs_RAG"
                    target="_blank"
                    style="
                        color: #8b7cff;
                        font-size: 12px;
                        font-weight: 600;
                        text-decoration: none;
                    "
                >
                    GitHub ↗
                </a>

            </div>

        </div>
        """
    )

# ---------------------------------------------------------------------------
# Selected conversation
# ---------------------------------------------------------------------------

selected_run_id = st.session_state.get("selected_run_id")

if (
    selected_run_id
    and selected_run_id != st.session_state.loaded_run_id
):
    selected_run = get_run(selected_run_id)

    if selected_run is None:
        st.warning("Run not found or you don't have access to it.")
        st.session_state.selected_run_id = None
        st.session_state.loaded_run_id = None
        st.query_params.clear()
        st.stop()

    state = selected_run["current_state"]
    status = selected_run["status"]

    st.session_state.messages = [
        {
            "role": "user",
            "content": state["question"],
        }
    ]

    if status == "completed":
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": state["generated_answer"] or "",
            }
        )

    elif status == "failed":
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "This run failed.",
            }
        )

    elif status == "interrupted":
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "This run was interrupted. You can resume it.",
            }
        )

    st.session_state.loaded_run_id = selected_run_id

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-row">
            <div class="hero-icon">📚</div>
            <div>
                <div class="hero-title">Developer Docs Assistant</div>
                <div class="hero-subtitle">
                    Search LangGraph documentation and implementation source code.
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Render previous messages
# ---------------------------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            sources = message["sources"]

            with st.expander(f"Sources · {len(sources)}", expanded=False):

                for source in sources:

                    corpus = source["corpus"].lower()

                    badge_class = (
                        "implementation"
                        if corpus == "implementation"
                        else "documentation"
                    )

                    badge_label = (
                        "Implementation"
                        if corpus == "implementation"
                        else "Documentation"
                    )

                    st.markdown(
                        f"""
                        <div class="source-item">
                            <span class="source-rank">
                                [{source['rank']}]
                            </span>
                            <span class="source-path">
                                {source['source']}
                            </span>
                            <span class="source-badge {badge_class}">
                                {badge_label}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Monitor selected running conversation
# ---------------------------------------------------------------------------

selected_run_id = st.session_state.get("selected_run_id")

if selected_run_id:
    selected_run = get_run(selected_run_id)

    if selected_run["status"] in {"running", "interrupted"}:

        waiting_messages = [
            "🔎 Digging through the docs...",
            "🧠 Asking the codebase for answers...",
            "🎯 Hunting down the useful bits...",
            "🤓 Making sure I don't make things up...",
            "📚 Consulting the documentation scrolls...",
            "🕵️ Following the source...",
        ]

        @st.fragment(run_every="1s")
        def monitor_running_run():

            try:
                run = get_run(selected_run_id)

                if run["status"] == "running":

                    message = waiting_messages[
                        int(time.time()) % len(waiting_messages)
                    ]

                    st.markdown(
                        f"""
                        <div class="streaming-indicator">
                            <span class="streaming-dot"></span>
                            {message}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                elif run["status"] == "completed":

                    state = run["current_state"]

                    st.session_state.messages = [
                        {
                            "role": "user",
                            "content": state["question"],
                        },
                        {
                            "role": "assistant",
                            "content": (
                                state["generated_answer"] or ""
                            ),
                        },
                    ]

                    st.session_state.loaded_run_id = (
                        selected_run_id
                    )

                    st.rerun()

                elif run["status"] == "interrupted":

                    st.warning(
                        "This run was interrupted. "
                        "You can resume it."
                    )

                    if st.button(
                        "↻  Resume run",
                        key=f"resume_{selected_run_id}",
                        use_container_width=True,
                    ):
                        st.session_state.resume_run_id = selected_run_id
                        st.rerun()

                elif run["status"] == "failed":

                    st.error("This run failed.")

            except requests.exceptions.RequestException:
                st.warning(
                    "Unable to check the RAG run right now."
                )

        monitor_running_run()

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.html(
        """
        <div class="empty-state">

            <div class="empty-orb">
                ✦
            </div>

            <div class="empty-title">
                Explore the LangGraph codebase
            </div>

            <div class="empty-text">
                Search documentation and implementation source code
                to understand APIs, runtime behavior, architecture,
                and how components work together.
            </div>

            <div style="
                display:flex;
                justify-content:center;
                gap:8px;
                margin-top:22px;
                flex-wrap:wrap;
            ">

                <div style="
                    padding:7px 11px;
                    border:1px solid #202832;
                    border-radius:7px;
                    background:#0f141b;
                    color:#8994a2;
                    font-size:10px;
                ">
                    API behavior
                </div>

                <div style="
                    padding:7px 11px;
                    border:1px solid #202832;
                    border-radius:7px;
                    background:#0f141b;
                    color:#8994a2;
                    font-size:10px;
                ">
                    Debug code
                </div>

                <div style="
                    padding:7px 11px;
                    border:1px solid #202832;
                    border-radius:7px;
                    background:#0f141b;
                    color:#8994a2;
                    font-size:10px;
                ">
                    Runtime behavior
                </div>

                <div style="
                    padding:7px 11px;
                    border:1px solid #202832;
                    border-radius:7px;
                    background:#0f141b;
                    color:#8994a2;
                    font-size:10px;
                ">
                    Find implementation
                </div>

            </div>

            <div style="
                display:flex;
                justify-content:center;
                align-items:center;
                gap:16px;
                margin-top:18px;
                color:#596473;
                font-size:9px;
            ">

                <span>
                    <span style="color:#76c7ff;">●</span>
                    Documentation
                </span>

                <span>
                    <span style="color:#ad9cff;">●</span>
                    Implementation
                </span>

            </div>

        </div>
        """
    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

question = st.chat_input(
    "Ask about LangGraph..."
)

resume_run_id = st.session_state.get("resume_run_id")

if question or resume_run_id:

    is_resume = resume_run_id is not None

    if is_resume:
        resumed_run = get_run(resume_run_id)

        question = resumed_run["current_state"]["question"]

        # Rebuild the conversation for the resumed run.
        st.session_state.messages = [
            {
                "role": "user",
                "content": question,
            }
        ]

    else:
        st.session_state.selected_run_id = None
        st.query_params.clear()
        st.session_state.loaded_run_id = None

        # -----------------------------------------------------------------------
        # User message
        # -----------------------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

    # -----------------------------------------------------------------------
    # Assistant response
    # -----------------------------------------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        status_placeholder = st.empty()

        answer = ""
        sources = []

        # Queue used to communicate between the background HTTP
        # request and the Streamlit UI.
        event_queue = queue.Queue()

        def consume_stream(url, payload=None):
            """
            Consume the FastAPI SSE stream in the background.

            The UI thread can therefore continue updating the
            status animation while the RAG pipeline is running.
            """

            try:
                with api_session.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=(10, 300),
                ) as response:
                                        
                    if response.status_code == 429:
                        data = response.json()

                        event_queue.put(
                            (
                                "error",
                                data.get(
                                    "detail",
                                    "Question limit reached."
                                )
                            )
                        )
                        return

                    response.raise_for_status()

                    for line in response.iter_lines(
                        decode_unicode=True
                    ):

                        if not line:
                            continue

                        if line.startswith("data:"):
                            data = json.loads(line[len("data:"):].strip())

                            event_queue.put(
                                (data.get("type"), data)
                            )

                # Tell the UI that the stream has finished.
                event_queue.put(("done", None))

            except requests.exceptions.Timeout:
                event_queue.put(
                    (
                        "error",
                        "The RAG service took too long to respond."
                    )
                )

            except requests.exceptions.ConnectionError:
                event_queue.put(
                    (
                        "error",
                        "Unable to connect to the RAG API. "
                        "Make sure FastAPI is running."
                    )
                )

            except requests.RequestException as e:
                event_queue.put(
                    (
                        "error",
                        f"RAG API request failed: {e}"
                    )
                )

            except Exception as e:
                event_queue.put(
                    (
                        "error",
                        f"Unexpected error: {e}"
                    )
                )

        # Start the HTTP request in the background.
        resume_run_id = st.session_state.get("resume_run_id")

        if resume_run_id:
            stream_url = f"{API_BASE_URL}/runs/{resume_run_id}/resume"
            stream_payload = None
        else:
            stream_url = API_URL
            stream_payload = {"question": question}
            
        worker = threading.Thread(
            target=consume_stream,
            args=(stream_url, stream_payload),
            daemon=True,
        )

        worker.start()

        # -------------------------------------------------------------------
        # Waiting animation
        # -------------------------------------------------------------------

        waiting_messages = [
            "🔎 Digging through the docs...",
            "🧠 Asking the codebase for answers...",
            "🎯 Hunting down the useful bits...",
            "🤓 Making sure I don't make things up...",
            "📚 Consulting the documentation scrolls...",
            "🕵️ Following the source...",
        ]

        revision_waiting_messages = [
            "🔍 Filling in the missing pieces...",
            "📚 Searching for more evidence...",
            "🧩 Connecting the missing details...",
            "🔎 Verifying the gaps I found...",
            "🧠 Improving the answer with more evidence...",
        ]

        message_index = 0
        generation_round = 0
        first_token_received = False
        stream_finished = False
        stream_error = None
        run_accepted = False
        revision_started = False
        last_waiting_message_time = time.time()

        while not first_token_received and not stream_finished:

            current_waiting_messages = (
                revision_waiting_messages
                if generation_round > 0
                else waiting_messages
            )

            status_placeholder.markdown(
                f"""
                <div class="streaming-indicator">
                    <span class="streaming-dot"></span>
                    {current_waiting_messages[message_index]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            message_index = (
                message_index + 1
            ) % len(current_waiting_messages)

            deadline = time.time() + 0.8

            while time.time() < deadline:

                try:
                    event_type, data = event_queue.get_nowait()

                    if event_type == "run_started":
                        run_accepted = True
                        st.session_state.current_run_id = data["run_id"]
                        st.query_params["run_id"] = data["run_id"]
                        st.session_state.current_run_question = question

                    elif event_type == "run_resumed":

                        st.session_state.current_run_id = (
                            data["run_id"]
                        )
                        run_accepted = True

                    elif event_type == "generation_started":

                        generation_round = data.get(
                            "generation_round",
                            0
                        )

                        if generation_round > 0:

                            revision_started = True
                            message_index = 0

                            response_placeholder.empty()

                            status_placeholder.markdown(
                                f"""
                                <div class="streaming-indicator">
                                    <span class="streaming-dot"></span>
                                    {revision_waiting_messages[message_index]}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    elif event_type == "answer_delta":

                        first_token_received = True

                        answer += data["data"]

                        status_placeholder.empty()

                        response_placeholder.markdown(
                            answer
                        )

                        break

                    elif event_type == "run_completed":

                        stream_finished = True
                        break

                    elif event_type == "error":

                        if isinstance(data, dict):
                            stream_error = data.get(
                                "data",
                                "RAG run failed."
                            )

                        else:
                            stream_error = str(data)

                        stream_finished = True
                        break

                except queue.Empty:

                    time.sleep(0.05)


        # -------------------------------------------------------------------
        # Continue consuming the answer after first token
        # -------------------------------------------------------------------

        while not stream_finished:

            try:
                event_type, data = event_queue.get(timeout=0.1)

                if event_type == "run_started":
                    run_accepted = True
                    st.session_state.current_run_id = data["run_id"]
                    st.query_params["run_id"] = data["run_id"]

                elif event_type == "generation_started":

                    generation_round = data.get(
                        "generation_round",
                        0
                    )

                    if generation_round > 0:
                        revision_started = True
                        message_index = 0
                        last_waiting_message_time = time.time()

                        status_placeholder.markdown(
                            f"""
                            <div class="streaming-indicator">
                                <span class="streaming-dot"></span>
                                {revision_waiting_messages[message_index]}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                elif event_type == "gap_processing_started":

                    revision_started = True
                    message_index = 0
                    last_waiting_message_time = time.time()

                    status_placeholder.markdown(
                        f"""
                        <div class="streaming-indicator">
                            <span class="streaming-dot"></span>
                            {revision_waiting_messages[message_index]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                elif event_type == "answer_delta":

                    if revision_started:

                        # First token of the revised answer
                        answer = ""
                        revision_started = False

                        response_placeholder.empty()
                        status_placeholder.empty()

                    answer += data["data"]

                    response_placeholder.markdown(answer)

                elif event_type == "run_completed":
                    stream_finished = True

                elif event_type == "error":
                    if isinstance(data, dict):
                        stream_error = data.get(
                            "data",
                            "RAG run failed."
                        )
                    else:
                        stream_error = str(data)

                    stream_finished = True

            except queue.Empty:

                if revision_started:

                    current_waiting_messages = revision_waiting_messages

                    status_placeholder.markdown(
                        f"""
                        <div class="streaming-indicator">
                            <span class="streaming-dot"></span>
                            {current_waiting_messages[message_index]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if time.time() - last_waiting_message_time >= 1.0:
                        message_index = (
                            message_index + 1
                        ) % len(current_waiting_messages)

                        last_waiting_message_time = time.time()

                continue
        # -------------------------------------------------------------------
        # Error handling
        # -------------------------------------------------------------------

        if stream_error:

            status_placeholder.empty()

            response_placeholder.error(
                stream_error
            )

        # -----------------------------------------------------------------------
        # Sources
        # -----------------------------------------------------------------------

        if sources:

            with st.expander(
                f"Sources · {len(sources)}",
                expanded=False,
            ):

                for source in sources:

                    corpus = source["corpus"].lower()

                    if corpus == "implementation":
                        badge_class = "implementation"
                        badge_label = "Implementation"
                    else:
                        badge_class = "documentation"
                        badge_label = "Documentation"

                    st.html(
                        f"""
                        <div class="source-item">
                            <span class="source-rank">
                                [{source['rank']}]
                            </span>

                            <span class="source-path">
                                {source['source']}
                            </span>

                            <span class="source-badge {badge_class}">
                                {badge_label}
                            </span>
                        </div>
                        """
                    )

        # -------------------------------------------------------------------
        # Store assistant message
        # -------------------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

        if run_accepted:
            st.session_state.resume_run_id = None
            st.rerun()