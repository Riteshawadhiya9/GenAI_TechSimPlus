import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import time

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SearchMind AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background: #0a0a0f !important; }
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e !important;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.site-header {
    display: flex; align-items: center; gap: 14px;
    padding: 28px 0 20px;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 28px;
}
.site-header .logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #6c63ff, #3ecfcf);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.site-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #6c63ff, #3ecfcf);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.site-header .tagline { font-size: 0.78rem; color: #6b6b80; font-weight: 400; margin-top: 2px; }

.msg-row { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 20px; }
.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0; margin-top: 2px;
}
.avatar.ai   { background: linear-gradient(135deg,#6c63ff22,#3ecfcf22); border:1px solid #3ecfcf44; }
.avatar.user { background: linear-gradient(135deg,#ff636322,#ff9f4322); border:1px solid #ff636344; }

.bubble {
    max-width: 75%; padding: 14px 18px; border-radius: 18px;
    line-height: 1.65; font-size: 0.92rem; position: relative;
}
.bubble.ai {
    background: #13131f; border: 1px solid #1e1e2e;
    border-top-left-radius: 4px; color: #d0d0e8;
}
.bubble.user {
    background: linear-gradient(135deg,#6c63ff18,#3ecfcf18);
    border: 1px solid #6c63ff33; border-top-right-radius: 4px;
    color: #e8e8f0; text-align: right;
}
.bubble .timestamp { font-size: 0.7rem; color: #44445a; margin-top: 8px; display: block; }
.bubble.user .timestamp { text-align: right; }

.typing-indicator {
    display: flex; gap: 5px; align-items: center;
    padding: 14px 18px; background: #13131f; border: 1px solid #1e1e2e;
    border-radius: 18px; border-top-left-radius: 4px; width: fit-content;
}
.dot { width:7px; height:7px; border-radius:50%; background:#6c63ff; animation:bounce 1.2s infinite; }
.dot:nth-child(2) { animation-delay:0.2s; background:#5a9fef; }
.dot:nth-child(3) { animation-delay:0.4s; background:#3ecfcf; }
@keyframes bounce {
    0%,80%,100% { transform:translateY(0); opacity:0.4; }
    40% { transform:translateY(-7px); opacity:1; }
}

.welcome { text-align:center; padding:60px 20px; max-width:560px; margin:0 auto; }
.welcome .big-icon { font-size:52px; margin-bottom:20px; display:block; }
.welcome h2 {
    font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700;
    background:linear-gradient(90deg,#6c63ff,#3ecfcf);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin-bottom:12px;
}
.welcome p { color:#6b6b80; font-size:0.95rem; line-height:1.7; margin-bottom:32px; }
.suggestion-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; text-align:left; }
.suggestion-card {
    background:#13131f; border:1px solid #1e1e2e; border-radius:12px;
    padding:14px 16px; font-size:0.85rem; color:#9090a8;
}
.suggestion-card .s-icon { font-size:1.2rem; margin-bottom:6px; display:block; }
.suggestion-card .s-text { color:#c0c0d8; font-size:0.83rem; }

.sb-label {
    font-size:0.7rem; font-weight:600; letter-spacing:1.5px;
    text-transform:uppercase; color:#44445a !important; margin-bottom:10px;
}
.memory-bar-bg { height:4px; background:#1e1e2e; border-radius:2px; margin-top:8px; }
.memory-bar-fill {
    height:4px; background:linear-gradient(90deg,#6c63ff,#3ecfcf);
    border-radius:2px; transition:width 0.4s;
}
.model-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:#13131f; border:1px solid #2a2a3e; border-radius:8px;
    padding:8px 12px; font-size:0.8rem; color:#9090a8 !important;
    width:100%; margin-bottom:8px;
}
.model-badge .dot-on {
    width:7px; height:7px; border-radius:50%;
    background:#3ecfcf; box-shadow:0 0 6px #3ecfcf; flex-shrink:0;
}

[data-testid="stChatInput"] > div {
    background:#13131f !important; border:1px solid #2a2a3e !important;
    border-radius:16px !important;
}
[data-testid="stChatInput"] textarea {
    background:transparent !important; color:#e8e8f0 !important;
    font-family:'Inter',sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_searches" not in st.session_state:
    st.session_state.total_searches = 0


# ── Build agent ───────────────────────────────────────────────────────────────
@st.cache_resource
def build_agent():
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0.3)
    search_api = GoogleSerperAPIWrapper()

    tools = [
        Tool(
            name="google_search",
            func=search_api.run,
            description="Search Google for current events, facts, and real-time information.",
        )
    ]

    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt="You are SearchMind, an intelligent AI assistant with Google Search. Provide accurate, well-structured, and helpful answers. Search when needed.",
    )
    return agent


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-label">Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-badge"><span class="dot-on"></span>mistral-small-2506</div>
    <div class="model-badge"><span class="dot-on"></span>Google Serper Search</div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Memory</div>', unsafe_allow_html=True)
    msg_count = len(st.session_state.messages)
    mem_pct = min(int((msg_count / 20) * 100), 100)
    st.markdown(f"""
    <div style="font-size:0.8rem;color:#6b6b80;">{msg_count} messages · {mem_pct}% window</div>
    <div class="memory-bar-bg"><div class="memory-bar-fill" style="width:{mem_pct}%"></div></div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem;color:#6b6b80;line-height:2;">
        🔍 Searches: <span style="color:#9090a8">{st.session_state.total_searches}</span><br>
        💬 Turns: <span style="color:#9090a8">{msg_count // 2}</span><br>
        🧠 Memory: <span style="color:#9090a8">LangGraph MemorySaver</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>' * 3, unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.total_searches = 0
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
  <div class="logo">🔍</div>
  <div>
    <h1>SearchMind AI</h1>
    <div class="tagline">Powered by Mistral · Google Search · LangGraph</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <span class="big-icon">✦</span>
        <h2>What do you want to know?</h2>
        <p>I search the web in real time and remember our conversation.<br>Ask me anything — news, facts, analysis, or just chat.</p>
        <div class="suggestion-grid">
            <div class="suggestion-card"><span class="s-icon">📰</span><div class="s-text">Latest AI news this week</div></div>
            <div class="suggestion-card"><span class="s-icon">📈</span><div class="s-text">Current stock market trends</div></div>
            <div class="suggestion-card"><span class="s-icon">🌍</span><div class="s-text">Recent world events</div></div>
            <div class="suggestion-card"><span class="s-icon">💡</span><div class="s-text">Explain quantum computing</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    role, content, ts = msg["role"], msg["content"], msg.get("time", "")
    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user">🧑</div>
            <div class="bubble user">{content}<span class="timestamp">{ts}</span></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar ai">✦</div>
            <div class="bubble ai">{content}<span class="timestamp">{ts}</span></div>
        </div>""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything… I'll search the web for you")

if user_input:
    ts_now = time.strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": user_input, "time": ts_now})

    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="msg-row">
        <div class="avatar ai">✦</div>
        <div class="typing-indicator">
            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        agent = build_agent()
        config = {"configurable": {"thread_id": "streamlit-session"}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        response = result["messages"][-1].content
        st.session_state.total_searches += 1
    except Exception as e:
        response = f"⚠️ Something went wrong: {str(e)}"

    typing_placeholder.empty()
    st.session_state.messages.append({"role": "assistant", "content": response, "time": time.strftime("%I:%M %p")})
    st.rerun()