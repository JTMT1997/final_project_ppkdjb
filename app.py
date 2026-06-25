import streamlit as st
from rag_pipeline import build_rag_pipeline

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="IT Consulting Job Advisor Pro",
    page_icon="💼",
    layout="wide"
)

# ── HEADER STYLE + IMAGE ──────────────────────────────
st.markdown("""
<div style="text-align:center; padding:10px 0;">
    <img src="https://images.squarespace-cdn.com/content/v1/56b6ad177c65e4dd4ba03c64/1455347613224-4MS0YFU706I5TPONU9FG/Untitled-1.jpg"
         width="120"
         style="border-radius:20px; margin-bottom:10px;">
    <h1 style="margin-bottom:0;">💼 IT Consulting Job Advisor</h1>
    <p style="color:gray; margin-top:5px;">AI-powered career guidance & job recommendations for IT professionals</p>
    <p style="color:gray; font-size:0.9rem;"></p>
</div>
""", unsafe_allow_html=True)

# ── LOAD PIPELINE ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return build_rag_pipeline()

if "pipeline_loaded" not in st.session_state:
    with st.spinner("Preparing AI Engine..."):
        chain, num_chunks = load_pipeline()
        st.session_state.chain = chain
        st.session_state.num_chunks = num_chunks
        st.session_state.pipeline_loaded = True

chain = st.session_state.chain

# ── SESSION STATE ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── LAYOUT GRID ───────────────────────────────────────
left, center, right = st.columns([1, 2, 1])

# ===================== LEFT PANEL =====================
with left:
    st.markdown("### ⚡ Quick Actions")
    st.markdown("Click for instant career recommendations!")

    # Tombol 1: Cloud & DevOps
    if st.button("☁️ Cloud & DevOps"):
        st.session_state.messages.append({
            "role": "user", 
            "content": "I am a Cloud & DevOps expert with skills: AWS, Docker, Kubernetes, Terraform, Jenkins. Recommend suitable job positions!"
        })
        st.rerun()

    # Tombol 2: Data & AI
    if st.button("📊 Data & AI"):
        st.session_state.messages.append({
            "role": "user", 
            "content": "I am a Data & AI specialist with skills: Python, SQL, Spark, TensorFlow, Machine Learning. What jobs fit my profile?"
        })
        st.rerun()

    # Tombol 3: Cybersecurity
    if st.button("🔐 Cybersecurity"):
        st.session_state.messages.append({
            "role": "user", 
            "content": "I am a Cybersecurity professional with skills: CISSP, SIEM, Firewalls, ISO 27001, Penetration Testing. Any recommended roles?"
        })
        st.rerun()

    # Tombol 4: Full-Stack Developer
    if st.button("💻 Full-Stack"):
        st.session_state.messages.append({
            "role": "user", 
            "content": "I am a Full-Stack Developer with skills: React, Node.js, TypeScript, MongoDB, PostgreSQL. Find matching jobs!"
        })
        st.rerun()

    # Tombol 5: Senior Leadership
    if st.button("👔 Senior Level"):
        st.session_state.messages.append({
            "role": "user", 
            "content": "I have 10+ years experience with skills: IT Strategy, TOGAF, ITIL, PMP, Agile. What senior positions are available?"
        })
        st.rerun()

    # Tombol 6: Emerging Tech
    if st.button("🚀 Emerging Tech"):
        st.session_state.messages.append({
            "role": "user", 
            "content": "I work with Emerging Tech: AI/ML, AR/VR, Blockchain. Skills: Python, Unity, Solidity, Web3. Any recommendations?"
        })
        st.rerun()

    st.divider()

    st.markdown("### 🏷️ Top Skills in Catalog")
    st.markdown("""
    • Python, SQL, AWS\n
    • Docker, Kubernetes\n
    • React, Node.js\n
    • TensorFlow, PyTorch\n
    • Terraform, Jenkins\n
    • SAP, Oracle\n
    • Power BI, Tableau\n
    """)

    st.divider()

    st.markdown("### 📋 Domains Available")
    st.markdown("""
    ☁️ Cloud   
    📊 Data  
    🔐 Security   
    💻 Development  
    🏢 Enterprise   
    🔄 Integration   
    🚀 Emerging Tech   
    🤝 Collaboration   
    ⚡ Automation  
    📋 Governance  
    """)

# ===================== CENTER CHAT =====================
with center:
    st.markdown("### 💬 Chat with AI Advisor")

    # welcome state
    if not st.session_state.messages:
        st.info("👋 Click a button on the left panel or type your question below!")

    # chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input
    user_input = st.chat_input("Ask about job opportunities, skills, or career paths...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing job catalog..."):
                result = chain.invoke({"query": user_input})
                answer = result["result"]
                sources = result["source_documents"]

            st.markdown(answer)

            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(sources, 1):
                    st.write(f"{i}. {doc.page_content[:250]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ===================== RIGHT PANEL =====================
with right:
    st.markdown("### 📊 System Status")
    st.success("🟢 AI Active")
    st.markdown("### 🧠 AI Mode")
    st.info("🔍 RAG (Retrieval-Augmented Generation)")
    st.caption("Powered by LangChain + Groq + FAISS")

    st.markdown("### ⚙️ Pipeline")
    st.code("""
📁 Jobs Catalog (TXT)
    ↓
✂️ Text Splitter
    ↓
🔢 Embeddings (HuggingFace)
    ↓
🗂️ FAISS Vector Store
    ↓
🎯 Retriever (Top 5)
    ↓
🤖 Groq LLM (Llama 3)
    ↓
💬 Answer + Sources
""", language="text")

    st.divider()

    if st.button("🧹 Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("💼 IT Consulting Job Advisor v2.0")