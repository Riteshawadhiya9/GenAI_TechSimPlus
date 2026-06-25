import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TaskDB Agent",
    page_icon="🗃️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d0d !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background: #0d0d0d !important; }
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #1f1f1f !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px 0 20px;
    border-bottom: 1px solid #1f1f1f;
    margin-bottom: 24px;
}
.header-icon {
    width: 42px; height: 42px;
    background: #00ff88;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.header-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: #00ff88;
    letter-spacing: -0.5px;
}
.header-sub {
    font-size: 0.75rem;
    color: #555;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Chat messages ── */
.msg-wrap { margin-bottom: 18px; }
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 4px;
}
.msg-ai {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 4px;
}
.bubble-user {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    color: #c8c8ff;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    max-width: 70%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.bubble-ai {
    background: #111811;
    border: 1px solid #1a2a1a;
    border-left: 3px solid #00ff88;
    color: #d4f5d4;
    padding: 12px 16px;
    border-radius: 4px 16px 16px 16px;
    max-width: 75%;
    font-size: 0.9rem;
    line-height: 1.6;
    font-family: 'Inter', sans-serif;
}
.msg-label {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    color: #444;
    margin-bottom: 4px;
}
.msg-label.user-label { text-align: right; }

/* ── Typing ── */
.typing-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: #111811;
    border: 1px solid #1a2a1a;
    border-left: 3px solid #00ff88;
    border-radius: 4px 16px 16px 16px;
    width: fit-content;
}
.typing-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00ff88;
}
.t-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00ff88;
    animation: tpulse 1.2s infinite;
}
.t-dot:nth-child(2) { animation-delay: 0.2s; }
.t-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes tpulse {
    0%,80%,100% { opacity: 0.2; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.2); }
}

/* ── Welcome ── */
.welcome-box {
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    max-width: 600px;
    margin: 40px auto;
    background: #111111;
}
.welcome-box .wicon { font-size: 3rem; margin-bottom: 16px; }
.welcome-box h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    color: #00ff88;
    margin-bottom: 10px;
}
.welcome-box p { color: #666; font-size: 0.88rem; line-height: 1.7; margin-bottom: 24px; }
.cmd-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    text-align: left;
}
.cmd-card {
    background: #0d0d0d;
    border: 1px solid #1f1f1f;
    border-radius: 8px;
    padding: 10px 14px;
}
.cmd-card .cmd {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #00ff88;
    margin-bottom: 3px;
}
.cmd-card .desc { font-size: 0.75rem; color: #555; }

/* ── Sidebar ── */
.sb-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #444 !important;
    margin-bottom: 10px;
}
.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: #0d0d0d;
    border: 1px solid #1f1f1f;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: #888 !important;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 6px #00ff88;
    flex-shrink: 0;
}
.stat-box {
    background: #0d0d0d;
    border: 1px solid #1f1f1f;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.stat-num { font-size: 1.6rem; font-weight: 500; color: #00ff88; }
.stat-lbl { font-size: 0.68rem; color: #444; margin-top: 2px; }

/* ── Input ── */
[data-testid="stChatInput"] > div {
    background: #111111 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e0e0e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
[data-testid="stChatInput"]:focus-within > div {
    border-color: #00ff8855 !important;
    box-shadow: 0 0 0 3px #00ff8810 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "agent" not in st.session_state:
    st.session_state.agent = None


# ── Build agent ───────────────────────────────────────────────────────────────
@st.cache_resource
def build_agent():
    db = SQLDatabase.from_uri("sqlite:///my_database.db")
    try:
        db.run("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT CHECK(status IN ('pending', 'in_progress', 'completed')) NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    except Exception:
        pass

    model = ChatGroq(model="openai/gpt-oss-20b")
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()
    memory = InMemorySaver()

    system_prompt = """You are an intelligent SQL Task Management Assistant.
Help users manage tasks in a SQLite database using SQL tools.

Database table: tasks
Columns: id (INTEGER PK), title (TEXT), description (TEXT), status (TEXT: pending/in_progress/completed), created_at (TIMESTAMP)

Rules:
- Always use SQL tools to interact with the database, never fabricate data
- Ask for clarification if information is missing
- Confirm before DELETE operations
- Keep responses concise and clear
- Present results in readable format
- Never use SELECT * — select only needed columns
"""

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        system_prompt=system_prompt,
    )
    return agent


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">Connection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="status-row"><span class="status-dot"></span>SQLite · my_database.db</div>
    <div class="status-row"><span class="status-dot"></span>Groq · gpt-oss-20b</div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Session Stats</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{st.session_state.query_count}</div>
            <div class="stat-lbl">queries run</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{len(st.session_state.messages) // 2}</div>
            <div class="stat-lbl">turns</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Quick Actions</div>', unsafe_allow_html=True)

    if st.button("📋 Show all tasks", use_container_width=True):
        st.session_state["prefill"] = "Show all tasks"
    if st.button("⏳ Show pending tasks", use_container_width=True):
        st.session_state["prefill"] = "Show all pending tasks"
    if st.button("✅ Show completed tasks", use_container_width=True):
        st.session_state["prefill"] = "Show all completed tasks"
    if st.button("📊 Count by status", use_container_width=True):
        st.session_state["prefill"] = "Count tasks grouped by status"

    st.markdown('<br>' * 2, unsafe_allow_html=True)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="header-icon">🗃️</div>
  <div>
    <div class="header-title">TaskDB Agent</div>
    <div class="header-sub">Natural language → SQL · Groq + LangGraph</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Welcome screen
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        <div class="wicon">🗃️</div>
        <h2>Talk to your database</h2>
        <p>Ask in plain English — I'll write and run the SQL for you.<br>
        Create, read, update, or delete tasks naturally.</p>
        <div class="cmd-grid">
            <div class="cmd-card">
                <div class="cmd">Add a task</div>
                <div class="desc">Create a new task with title & description</div>
            </div>
            <div class="cmd-card">
                <div class="cmd">Show pending tasks</div>
                <div class="desc">List all tasks not yet started</div>
            </div>
            <div class="cmd-card">
                <div class="cmd">Mark task #3 as done</div>
                <div class="desc">Update status to completed</div>
            </div>
            <div class="cmd-card">
                <div class="cmd">Delete task #5</div>
                <div class="desc">Remove a task from the database</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render messages
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    ts = msg.get("time", "")

    if role == "user":
        st.markdown(f"""
        <div class="msg-wrap">
            <div class="msg-label user-label">YOU · {ts}</div>
            <div class="msg-user"><div class="bubble-user">{content}</div></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-wrap">
            <div class="msg-label">AGENT · {ts}</div>
            <div class="msg-ai"><div class="bubble-ai">{content}</div></div>
        </div>""", unsafe_allow_html=True)


# ── Handle prefill from sidebar buttons ──────────────────────────────────────
prefill = st.session_state.pop("prefill", None)

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("e.g. Add a task 'Fix login bug' with status in_progress")

query = prefill or user_input

if query:
    ts_now = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": query, "time": ts_now})

    typing_ph = st.empty()
    typing_ph.markdown("""
    <div class="msg-wrap">
        <div class="msg-label">AGENT</div>
        <div class="msg-ai">
            <div class="typing-wrap">
                <span class="typing-label">querying db</span>
                <div class="t-dot"></div>
                <div class="t-dot"></div>
                <div class="t-dot"></div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        if st.session_state.agent is None:
            st.session_state.agent = build_agent()

        res = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "streamlit-1"}}
        )
        response = res["messages"][-1].content
        st.session_state.query_count += 1
    except Exception as e:
        response = f"⚠️ Error: {str(e)}"

    typing_ph.empty()
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "time": time.strftime("%H:%M"),
    })
    st.rerun()