import streamlit as st
from rag_engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="Cloud ☁️ • ML Foundations RAG",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light, playful, modern editorial magazine styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Force Light Theme Base on App Container */
    .stApp {
        background-color: #FBF9F5 !important;
        color: #2D3142 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Main Container max-width and padding */
    .main .block-container {
        max-width: 960px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Editorial Hero Card */
    .editorial-hero {
        background: linear-gradient(135deg, #FFF9F3 0%, #F5F0FF 55%, #EEF6FF 100%);
        border: 1px solid #EFE4DA;
        border-radius: 20px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -10px rgba(180, 150, 130, 0.2);
    }

    .editorial-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FF6B6B;
        color: #FFFFFF !important;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 0.8rem;
        box-shadow: 0 3px 10px rgba(255, 107, 107, 0.25);
    }

    .hero-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.4rem;
        font-weight: 600;
        line-height: 1.2;
        color: #1A1C24 !important;
        margin-bottom: 0.6rem;
        letter-spacing: -0.02em;
    }

    .hero-title em {
        font-style: italic;
        color: #6C5CE7 !important;
    }

    .hero-sub {
        font-size: 1rem;
        line-height: 1.55;
        color: #5C6075 !important;
        max-width: 800px;
        margin-bottom: 1.2rem;
    }

    /* Project Story Blurb Box */
    .story-blurb {
        background: rgba(255, 255, 255, 0.8);
        border: 1px dashed #DACDC2;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        font-size: 0.86rem;
        line-height: 1.55;
        color: #4A4D63 !important;
        backdrop-filter: blur(4px);
    }

    .story-blurb strong {
        color: #2D3142 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F4EFEA !important;
        border-right: 1px solid #E6DED5 !important;
    }

    .sidebar-heading {
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7D8096 !important;
        margin-bottom: 0.6rem;
    }

    .notebook-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #EAE2D8;
        border-radius: 9px;
        padding: 8px 11px;
        margin-bottom: 6px;
        font-size: 0.83rem;
        font-weight: 600;
        color: #31344A !important;
        border-left: 3.5px solid #6C5CE7;
        text-decoration: none !important;
        transition: all 0.2s ease;
    }

    .notebook-pill:hover {
        background-color: #FAF5F0 !important;
        border-color: #6C5CE7 !important;
        color: #6C5CE7 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(108, 92, 231, 0.1) !important;
    }

    /* Modern Light Buttons */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #3A3D52 !important;
        border: 1px solid #E4DBD1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 0.9rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: #FAF5F0 !important;
        border-color: #6C5CE7 !important;
        color: #6C5CE7 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(108, 92, 231, 0.12) !important;
    }

    /* Chat Messages styling */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid #EFE7DE !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.9rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02) !important;
    }

    /* Chat Input Bar */
    [data-testid="stChatInput"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
    }

    /* Expander Source Styling */
    .streamlit-expanderHeader {
        background-color: #FAF6F0 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #4B4D63 !important;
        border: 1px solid #EAE1D5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG Engine in session state so it persists across reruns
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "☁️ **Hey there, I'm Cloud!** Your ML notebook companion. Ask me anything about your preprocessing pipelines, error analyses, regression models, or classification tricks in `foundations_and_models`!",
            "sources": []
        }
    ]

# GitHub repository configuration for live links
GITHUB_REPO_NOTEBOOKS_URL = "https://github.com/claudiafarkas/crashcourses/blob/main/foundations_and_models"

# Sidebar: Controls & Info
with st.sidebar:
    st.markdown('<div class="sidebar-heading">☁️ Knowledge Base</div>', unsafe_allow_html=True)
    
    notebooks = st.session_state.rag_engine.get_notebook_list()
    if notebooks:
        for nb in notebooks:
            clean_name = nb.replace(".ipynb", "").replace("_", " ").title()
            github_url = f"{GITHUB_REPO_NOTEBOOKS_URL}/{nb}"
            st.markdown(
                f'<a href="{github_url}" target="_blank" class="notebook-pill">'
                f'<span>📖 <strong>{clean_name}</strong></span>'
                f'<span style="font-size:0.75rem; color:#8C8FA7;">↗</span>'
                f'</a>', 
                unsafe_allow_html=True
            )
    else:
        st.caption("No notebooks detected in `foundations_and_models`.")
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading">🎯 Retrieval Depth</div>', unsafe_allow_html=True)
    top_k = st.slider("Notebook chunks to fetch", min_value=1, max_value=6, value=3, help="Controls how many relevant cells are passed to the prompt.")
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading">💡 Quick Inquiries</div>', unsafe_allow_html=True)
    example_prompts = [
        ("🧹 Preprocessing", "How do we handle categorical encoding and missing values without data leakage?"),
        ("⚖️ Evaluation", "What metrics should we check for imbalanced classification?"),
        ("📈 Regularization", "When should we pick Ridge over Lasso regression?"),
        ("🔍 Error Slicing", "How do we perform error slice analysis on model predictions?")
    ]
    
    for label, prompt in example_prompts:
        if st.button(label, use_container_width=True, help=prompt):
            st.session_state.preset_prompt = prompt

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "☁️ Chat reset! What topic from your foundations notebooks shall we explore next?",
                "sources": []
            }
        ]
        st.rerun()

# Main Editorial Hero Header
st.markdown("""
<div class="editorial-hero">
    <div class="editorial-tag">☁️ Cloud • Local RAG Studio</div>
    <div class="hero-title">Ask <em>Cloud</em> anything about your Foundations.</div>
    <p class="hero-sub">
        A grounded retrieval assistant that parses, embeds, and searches your hands-on code, markdown commentary, and mathematical notes to answer modeling and pipeline questions with exact citations.
    </p>
    <div class="story-blurb">
        📍 <strong>About Cloud ☁️ & The Knowledge Base:</strong><br>
        <strong>Cloud ☁️</strong> is a dedicated local assistant created for the <em>Modern ML & Responsible AI</em> series. It treats the core curriculum in <code>foundations_and_models/</code> (covering Preprocessing, Supervised/Unsupervised Learning, and Model Evaluation) as living technical documentation. Instead of querying a generic LLM, Cloud indexes the actual notebook cells, extracts key concepts and code snippets, and grounds every answer in your verified notes.
    </div>
</div>
""", unsafe_allow_html=True)

# Display Chat Messages
for msg in st.session_state.messages:
    avatar = "☁️" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        # Display sources accordion if sources are attached
        if msg.get("sources"):
            with st.expander(f"📑 Citing {len(msg['sources'])} notebook cell(s)"):
                for idx, src in enumerate(msg["sources"], 1):
                    score_info = f" • {int(src['score'] * 100)}% match" if 'score' in src else ""
                    nb_name = src.get('notebook', 'Unknown')
                    nb_url = f"{GITHUB_REPO_NOTEBOOKS_URL}/{nb_name}" if nb_name != 'Unknown' else "#"
                    st.markdown(f"**[{idx}] [{nb_name}]({nb_url})** (Cell Type: `{src.get('cell_type', 'text')}`{score_info})")
                    st.code(src.get("content", ""), language="python" if src.get("cell_type") == "code" else "markdown")

# Handle User Input
user_input = st.chat_input("Ask Cloud a question about ML foundations, models, metrics, or preprocessing...")

# Check if an example question button was clicked
if "preset_prompt" in st.session_state:
    user_input = st.session_state.pop("preset_prompt")

if user_input:
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 2. Query RAG Engine and display response
    with st.chat_message("assistant", avatar="☁️"):
        with st.spinner("☁️ Cloud is skimming your notebooks..."):
            response = st.session_state.rag_engine.query(user_input, top_k=top_k)
            
            st.markdown(response["answer"])
            
            # Show sources
            if response.get("sources"):
                with st.expander(f"📑 Citing {len(response['sources'])} notebook cell(s)"):
                    for idx, src in enumerate(response["sources"], 1):
                        score_info = f" • {int(src['score'] * 100)}% match" if 'score' in src else ""
                        nb_name = src.get('notebook', 'Unknown')
                        nb_url = f"{GITHUB_REPO_NOTEBOOKS_URL}/{nb_name}" if nb_name != 'Unknown' else "#"
                        st.markdown(f"**[{idx}] [{nb_name}]({nb_url})** (Cell Type: `{src.get('cell_type', 'text')}`{score_info})")
                        st.code(src.get("content", ""), language="python" if src.get("cell_type") == "code" else "markdown")

    # 3. Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", [])
    })
