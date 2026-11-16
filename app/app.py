import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.pipeline import AnimeRecommendationPipeline
from src.vector_store import VectorStoreBuilder
from config.config import (
    MODEL_NAME,
    MODEL_PROVIDER,
    RETRIEVER_TOP_K,
    TEMPERATURE,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    CHROMA_PERSIST_DIR,
    GROQ_API_KEY
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OtakuSensei | AI Anime Recommender",
    page_icon="⛩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# -----------------------------------------------------------------------------
# 2. Native Theme Adaptive Styling
# Uses Streamlit CSS variables (--text-color, --background-color, --secondary-background-color)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Hero Title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        opacity: 0.85;
        margin-bottom: 1.4rem;
    }

    /* Metric Badges (Adaptive to Streamlit theme) */
    .metric-badge {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 0.65rem 1.1rem;
        border-radius: 10px;
        font-size: 0.92rem;
        color: var(--text-color);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .metric-badge b {
        color: #FF4B4B;
    }

    /* Anime Cards (Adaptive to Streamlit theme) */
    .anime-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-radius: 12px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .anime-card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.3rem;
    }
    .anime-card-synopsis {
        line-height: 1.6;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* Badges */
    .badge-score {
        background: linear-gradient(135deg, #FFB800, #FF8C00);
        color: #000000;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .badge-genre {
        background-color: rgba(108, 92, 231, 0.15);
        color: #7C3AED;
        border: 1px solid rgba(124, 58, 237, 0.3);
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    /* Raw Context Box */
    .context-box {
        font-family: monospace;
        font-size: 0.82rem;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 1rem;
        max-height: 380px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Cached Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vector_builder():
    """Caches the dense vector embeddings and Chroma retriever loader."""
    return VectorStoreBuilder(persist_dir=CHROMA_PERSIST_DIR)

@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str, top_k: int, temperature: float):
    """Initializes and caches the agentic recommendation pipeline."""
    builder = load_vector_builder()
    return AnimeRecommendationPipeline(
        builder=builder,
        top_k=top_k,
        model_name=model_name,
        temperature=temperature
    )

@st.cache_data(show_spinner=False)
def load_catalog_data():
    """Loads anime metadata for catalog browsing and filter generation."""
    data_path = RAW_DATA_PATH
    if not data_path.exists():
        data_path = PROCESSED_DATA_PATH
    if not data_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(data_path)
    if "sypnopsis" in df.columns and "synopsis" not in df.columns:
        df["synopsis"] = df["sypnopsis"]
    if "Score" in df.columns:
        df["Score_num"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0)
    return df

# Initialize Data & State
catalog_df = load_catalog_data()

# Extract unique genres for smart filtering
all_genres = []
if not catalog_df.empty and "Genres" in catalog_df.columns:
    genre_set = set()
    for row in catalog_df["Genres"].dropna():
        for g in row.split(","):
            cleaned = g.strip()
            if cleaned and cleaned != "Unknown":
                genre_set.add(cleaned)
    all_genres = sorted(list(genre_set))

# Session State for Chat
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Konnichiwa! 🌸 I am **OtakuSensei**, your AI anime guide powered by dense vector search and Groq LLMs. Tell me what kind of stories, genres, themes, or anime you love, and I'll find your next binge-worthy masterpiece!",
            "context": None
        }
    ]

# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Engine Diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=500",
        caption="⛩️ OtakuSensei Recommender",
        use_container_width=True
    )

    st.markdown("### ⚙️ Engine Settings")

    model_options = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b"
    ]
    selected_model = st.selectbox(
        "🧠 AI Model (Groq LPU):",
        options=model_options,
        index=0,
        help="gpt-oss-20b is fast and balanced; gpt-oss-120b offers deeper reasoning; qwen offers creative nuance."
    )

    col_side1, col_side2 = st.columns(2)
    with col_side1:
        top_k_val = st.slider("🔍 Top-K Retrieval:", min_value=2, max_value=10, value=RETRIEVER_TOP_K)
    with col_side2:
        temperature_val = st.slider("🌡️ Creativity (Temp):", min_value=0.0, max_value=1.0, value=TEMPERATURE, step=0.05)

    st.divider()

    st.markdown("### 📊 Knowledge Base Status")
    db_exists = CHROMA_PERSIST_DIR.exists()
    st.markdown(f"- **Vector Store:** {'🟢 Loaded (`chroma_db`)' if db_exists else '🔴 Not Found'}")
    st.markdown(f"- **Anime Titles Indexed:** `{len(catalog_df) if not catalog_df.empty else 'N/A'}`")
    st.markdown("- **Embeddings:** `all-MiniLM-L6-v2 (384-d)`")
    st.markdown(f"- **Inference Provider:** `{MODEL_PROVIDER.upper()}`")
    st.markdown(f"- **Groq API Key:** {'🟢 Active' if GROQ_API_KEY else '⚠️ Missing'}")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared! Ready for new anime recommendations. What are you in the mood for?",
                "context": None
            }
        ]
        st.rerun()

    st.divider()
    st.caption("💡 **Tip:** To switch between **Dark** and **Light** mode, click the menu in the top-right corner (`⋮`) ➔ **Settings** ➔ **Theme**.")

# -----------------------------------------------------------------------------
# 5. Hero Header Section
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">⛩️ OtakuSensei AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Autonomous Agentic Anime Recommendation Engine powered by Dense Vector Search and Groq LPU'
    '</div>',
    unsafe_allow_html=True
)

# Metric Badges
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.markdown(f'<div class="metric-badge">📚 <b>{len(catalog_df)}</b> Curated Anime</div>', unsafe_allow_html=True)
with m_col2:
    st.markdown(f'<div class="metric-badge">⚡ <b>{top_k_val}</b> Candidates Retrieved</div>', unsafe_allow_html=True)
with m_col3:
    st.markdown(f'<div class="metric-badge">🤖 <b>{selected_model.split("/")[-1]}</b></div>', unsafe_allow_html=True)
with m_col4:
    st.markdown('<div class="metric-badge">🎯 <b>Whole-Context</b> RAG</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# 6. Tabbed User Interface
# -----------------------------------------------------------------------------
tab_chat, tab_studio, tab_vector, tab_catalog, tab_arch = st.tabs([
    "💬 Sensei Chat",
    "🎯 Recommendation Studio",
    "🔍 Vector Similarity Explorer",
    "📖 Anime Catalog",
    "⚙️ Architecture & Diagnostics"
])

# =============================================================================
# TAB 1: SENSEI CHAT
# =============================================================================
with tab_chat:
    chat_top_col1, chat_top_col2 = st.columns([3, 1.2])
    with chat_top_col1:
        st.markdown("#### 💬 Chat with OtakuSensei")
        st.caption("Ask anything naturally: describe a plot, mood, specific vibes, or ask for comparisons!")
    with chat_top_col2:
        if len(st.session_state.chat_messages) > 1:
            chat_export_md = "# OtakuSensei Chat Recommendations\n\n"
            for m in st.session_state.chat_messages:
                role_name = "Sensei" if m["role"] == "assistant" else "User"
                chat_export_md += f"### {role_name}:\n{m['content']}\n\n---\n\n"
            st.download_button(
                "📥 Export Chat (.md)",
                data=chat_export_md,
                file_name="otakusensei_recommendations.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Prompt Suggestions Chips
    st.markdown("<span style='font-size:0.85rem; opacity:0.85;'>💡 Quick prompt starters:</span>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    prompt_to_send = None
    if p_col1.button("🥋 Ninja / Shonen Combat", key="p_ninja", use_container_width=True):
        prompt_to_send = "Suggest anime with high-stakes ninja or martial combat, strong rivalries, and character growth like Naruto."
    if p_col2.button("🧠 Dark Psychological Thriller", key="p_psycho", use_container_width=True):
        prompt_to_send = "Recommend a dark, intense psychological thriller with brilliant mind games and suspense like Monster."
    if p_col3.button("🌸 Cozy Slice of Life / Romance", key="p_romance", use_container_width=True):
        prompt_to_send = "I want a heartwarming, wholesome slice-of-life or high school romance anime to relax and unwind."
    if p_col4.button("🚀 Cyberpunk & Space Opera", key="p_space", use_container_width=True):
        prompt_to_send = "Suggest a sci-fi space opera or cyberpunk anime with deep philosophical themes and jazz or electronic soundtrack."

    # Container for all chat history
    chat_container = st.container()

    # Render existing messages inside the chat container
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"], avatar="⛩️" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])
                if msg.get("context"):
                    with st.expander("🔍 Inspect ChromaDB Retrieved Context"):
                        st.markdown(f"<div class='context-box'>{msg['context']}</div>", unsafe_allow_html=True)

    # Chat Input Box strictly positioned at the bottom of the conversation
    user_input = st.chat_input("Ask OtakuSensei for recommendations (e.g., 'An emotional anime movie that will make me cry')...")
    active_prompt = prompt_to_send or user_input

    # Process prompt inside chat_container and rerun so text box stays at bottom
    if active_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": active_prompt, "context": None})
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(active_prompt)

            with st.chat_message("assistant", avatar="⛩️"):
                with st.spinner("⛩️ Searching vector knowledge base and synthesizing recommendation..."):
                    try:
                        pipeline = load_pipeline(
                            model_name=selected_model,
                            top_k=top_k_val,
                            temperature=temperature_val
                        )
                        res = pipeline.recommend(active_prompt)
                        recommendation_text = res.get("recommendation", "No response generated.")
                        retrieved_context = res.get("retrieved_context", "")

                        st.markdown(recommendation_text)
                        if retrieved_context and "No relevant anime found" not in retrieved_context:
                            with st.expander("🔍 Inspect ChromaDB Retrieved Context"):
                                st.markdown(f"<div class='context-box'>{retrieved_context}</div>", unsafe_allow_html=True)

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": recommendation_text,
                            "context": retrieved_context
                        })
                    except Exception as e:
                        err_msg = f"⚠️ **Error generating recommendation:** {str(e)}"
                        st.error(err_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": err_msg,
                            "context": None
                        })
        st.rerun()

# =============================================================================
# TAB 2: RECOMMENDATION STUDIO
# =============================================================================
with tab_studio:
    st.markdown("#### 🎯 Recommendation Studio")
    st.caption("Craft targeted queries combined with genre preferences and score requirements.")

    col_studio_in, col_studio_filters = st.columns([1.8, 1.2])

    with col_studio_in:
        studio_query = st.text_area(
            "Describe your ideal anime experience:",
            value="Looking for an anime featuring a high-stakes strategy game, supernatural intrigue, and intricate plot twists.",
            height=120,
            placeholder="Type plot elements, character archetypes, mood, or vibes..."
        )
        favorite_anime = st.text_input(
            "Optionally, name an anime you loved for calibration:",
            placeholder="e.g. Death Note, Steins;Gate, Attack on Titan"
        )

    with col_studio_filters:
        selected_genres = st.multiselect(
            "Preferred Genres (optional bias):",
            options=all_genres,
            default=[]
        )
        min_score = st.slider("Minimum Rating Score (MAL):", min_value=5.0, max_value=9.5, value=7.5, step=0.1)

    generate_btn = st.button("✨ Generate Precision Recommendations", type="primary", use_container_width=True)

    if generate_btn:
        if not studio_query.strip():
            st.warning("Please enter a description or preference above.")
        else:
            composite_query = studio_query.strip()
            if favorite_anime.strip():
                composite_query += f" (I especially loved '{favorite_anime.strip()}')."
            if selected_genres:
                composite_query += f" Desired genres: {', '.join(selected_genres)}."
            composite_query += f" Minimum MAL rating: {min_score}+."

            with st.spinner("🤖 Consulting the vector store and analyzing anime database..."):
                try:
                    pipeline = load_pipeline(
                        model_name=selected_model,
                        top_k=top_k_val,
                        temperature=temperature_val
                    )
                    res = pipeline.recommend(composite_query)

                    st.markdown("### 🎯 Your Recommendations")
                    st.markdown(res["recommendation"])

                    with st.expander("🔍 Retrieved Chroma Evidence & Raw Chunks"):
                        st.markdown(f"<div class='context-box'>{res.get('retrieved_context', '')}</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during recommendation: {str(e)}")

# =============================================================================
# TAB 3: VECTOR SIMILARITY EXPLORER
# =============================================================================
with tab_vector:
    st.markdown("#### 🔍 Direct Vector Similarity Explorer")
    st.caption("Search ChromaDB's 384-dimensional dense vector embeddings directly without LLM synthesis.")

    v_col1, v_col2 = st.columns([3, 1])
    with v_col1:
        vector_query = st.text_input(
            "Enter semantic search query:",
            value="bounty hunters in space with jazz music",
            placeholder="Search concepts, plot hooks, or styles..."
        )
    with v_col2:
        vector_k = st.number_input("Number of Matches (K):", min_value=1, max_value=15, value=5)

    if st.button("🔎 Search Vector Database", use_container_width=True):
        if vector_query.strip():
            with st.spinner("Calculating dense vector similarities..."):
                try:
                    pipeline = load_pipeline(
                        model_name=selected_model,
                        top_k=top_k_val,
                        temperature=temperature_val
                    )
                    raw_docs = pipeline.get_similar_anime(vector_query, k=vector_k)

                    if not raw_docs:
                        st.info("No matching records found in ChromaDB.")
                    else:
                        st.success(f"Retrieved {len(raw_docs)} nearest neighbor anime documents:")
                        for idx, doc in enumerate(raw_docs, 1):
                            title = doc.metadata.get("title", f"Anime #{idx}")
                            score = doc.metadata.get("score", "N/A")
                            genres = doc.metadata.get("genres", "N/A")

                            st.markdown(f"""
                            <div class="anime-card">
                                <div class="anime-card-title">#{idx}. {title}</div>
                                <div style="margin-bottom: 0.6rem;">
                                    <span class="badge-score">⭐ Score: {score}</span>
                                    <span style="opacity:0.85; font-size:0.85rem;"><b>Genres:</b> {genres}</span>
                                </div>
                                <div class="anime-card-synopsis">
                                    {doc.page_content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Vector search failed: {str(e)}")

# =============================================================================
# TAB 4: ANIME CATALOG EXPLORER
# =============================================================================
with tab_catalog:
    st.markdown("#### 📖 Anime Knowledge Base Explorer")
    st.caption(f"Browse and filter all {len(catalog_df)} anime currently indexed in your local database.")

    if catalog_df.empty:
        st.warning("No anime dataset found in `data/anime_with_synopsis.csv` or `data/anime_updated.csv`.")
    else:
        cat_col1, cat_col2, cat_col3 = st.columns([2, 1.5, 1])

        with cat_col1:
            search_title = st.text_input("Search anime by title:", placeholder="e.g. Naruto, Bebop, Death...")
        with cat_col2:
            filter_genre = st.selectbox("Filter by genre:", options=["All Genres"] + all_genres)
        with cat_col3:
            min_score_filter = st.slider("Min score:", min_value=0.0, max_value=10.0, value=7.0, step=0.5)

        # Filtering DataFrame
        filtered_df = catalog_df.copy()
        if search_title.strip():
            filtered_df = filtered_df[
                filtered_df["Name"].str.contains(search_title.strip(), case=False, na=False)
            ]
        if filter_genre != "All Genres":
            filtered_df = filtered_df[
                filtered_df["Genres"].str.contains(filter_genre, case=False, na=False)
            ]
        if "Score_num" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Score_num"] >= min_score_filter]
            filtered_df = filtered_df.sort_values(by="Score_num", ascending=False)

        st.markdown(f"**Showing {len(filtered_df)} matches:**")

        # Interactive Data Table
        display_cols = ["Name", "Score", "Genres"]
        st.dataframe(
            filtered_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=300
        )

        # Selected Anime Inspector
        st.write("")
        st.markdown("##### 🔍 Anime Detail Inspector")
        anime_names = filtered_df["Name"].tolist()
        if anime_names:
            selected_anime_name = st.selectbox("Select anime to inspect full synopsis:", options=anime_names)
            anime_row = filtered_df[filtered_df["Name"] == selected_anime_name].iloc[0]

            st.markdown(f"""
            <div class="anime-card">
                <div class="anime-card-title">{anime_row['Name']}</div>
                <div style="margin-bottom: 0.6rem;">
                    <span class="badge-score">⭐ Score: {anime_row.get('Score', 'N/A')}</span>
                    <span style="opacity:0.85; font-size:0.85rem;"><b>Genres:</b> {anime_row.get('Genres', 'N/A')}</span>
                </div>
                <p class="anime-card-synopsis">
                    {anime_row.get('synopsis', 'No synopsis available.')}
                </p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# TAB 5: ARCHITECTURE & DIAGNOSTICS
# =============================================================================
with tab_arch:
    st.markdown("#### ⚙️ System Architecture & RAG Pipeline Diagnostics")
    st.caption("Inspect the end-to-end autonomous recommendation workflow.")

    st.markdown("""
```mermaid
flowchart LR
    A["Raw Dataset<br>(MAL Titles & Synopsis)"] --> B["Document Loader &<br>Whole-Item Chunker"]
    B --> C["HuggingFace Embeddings<br>(all-MiniLM-L6-v2)"]
    C --> D[("Chroma Vector Store<br>(Dense Index)")]
    
    U["User Query / Chat"] --> E["OtakuSensei Recommender Agent<br>(LangChain Tool Calling)"]
    E -->|"Decides to Search"| F["anime_retriever_tool"]
    F -->|"Semantic Vector Search"| D
    D -->|"Top-K Intact Anime Chunks"| F
    F -->|"Retrieved Context Evidence"| E
    E -->|"LLM Synthesis<br>(Groq LPU)"| R["🎯 Personalized Recommendation"]
```
    """)

    st.markdown("### 🧪 Live Diagnostics")
    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        st.markdown("#### 📡 Vector Database Health")
        if CHROMA_PERSIST_DIR.exists():
            st.success(f"✅ Chroma database directory found at `{CHROMA_PERSIST_DIR}`")
            try:
                builder = load_vector_builder()
                store = builder.load_vector_store()
                record_count = store._collection.count()
                st.info(f"📊 Total indexed vectors in Chroma collection: **{record_count}**")
            except Exception as e:
                st.warning(f"Unable to read collection count: {e}")
        else:
            st.error(f"❌ Chroma database directory missing at `{CHROMA_PERSIST_DIR}`. Run `python main.py --build`")

    with diag_col2:
        st.markdown("#### ⚡ Groq Cloud Connectivity")
        if GROQ_API_KEY:
            st.success("✅ `GROQ_API_KEY` is present in the environment.")
            if st.button("Ping Groq API Now"):
                with st.spinner("Pinging Groq model..."):
                    try:
                        from langchain.chat_models import init_chat_model
                        test_llm = init_chat_model(selected_model, model_provider="groq")
                        ping_res = test_llm.invoke("Reply with 'OtakuSensei Online!' in 3 words.")
                        st.success(f"Groq Response: `{ping_res.content.strip()}`")
                    except Exception as e:
                        st.error(f"Ping failed: {str(e)}")
        else:
            st.error("❌ `GROQ_API_KEY` is not configured in `.env`.")

    st.divider()
    st.markdown("#### 📘 Architecture Design Principles")
    st.markdown("""
    - **Whole-Anime Document Units**: Instead of arbitrary character/token chunking that splits plot twists and summaries across separate embeddings, each anime is indexed as a cohesive, intact document preserving Title, Score, Genres, and complete Synopsis.
    - **Agentic Tool Calling**: The Groq LLM evaluates the user's intent first. If retrieval is required, it dynamically invokes the vector search tool with targeted keyword queries, retrieves evidence, and synthesizes the final recommendation grounded in real anime metadata.
    - **Zero-Latency In-Memory Caching**: Vector store embeddings and pipeline instances are cached with Streamlit's `@st.cache_resource`, ensuring subsequent queries execute without redundant model reloads.
    """)
