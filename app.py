# app.py
import streamlit as st
import os
import sys
import traceback
import db_utils  # Uses Elasticsearch now

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of sys.path modification ---

import retriever_utils as utils
import prompts
from tabs import (
    ground_truth_tab,
    rag_processor_tab,
    ragas_eval_tab,
    viz_tab,
    prompt_settings_tab,
    experiments_dashboard_tab  # Import new tab
)

RAGAS_AVAILABLE = False
try:
    from ragas import evaluate
    import langchain_openai
    import datasets
    import altair  # Needed for charts in experiments dashboard

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    # Suppress Altair import error if RAGAS itself is not available, as it's a sub-dependency for a RAGAS-related feature
    if 'altair' not in sys.modules:  # Check if altair specifically failed
        st.sidebar.warning(
            "Altair library not found. Charts in Experiments Dashboard might be affected. Install with: pip install altair")

st.set_page_config(layout="wide", page_title="Elkeval - RAG Testbed & Evaluator")


@st.cache_resource
def load_initial_config_and_clients():
    try:
        if not os.path.exists(".env"):
            st.error("`.env` file not found.");
            return None, None, None, None, None, "`.env` file not found."
        app_config = utils.load_config()
        es_client, openai_client = utils.initialize_clients(app_config)

        azure_openai_model_name_from_config = app_config.get("AZURE_OPENAI_MODEL")
        if not azure_openai_model_name_from_config: st.warning("`AZURE_OPENAI_MODEL` not found in .env.")
        default_es_index = app_config.get("ES_INDEX_NAME_ENV", utils.INDEX_NAME)

        # Store app_config in session state for other modules to access (e.g., for ES HTTP calls)
        st.session_state.app_config = app_config

        return es_client, openai_client, azure_openai_model_name_from_config, app_config, default_es_index, None
    except Exception as e:
        error_msg = f"Error during initialization: {e}\n{traceback.format_exc()}";
        st.error(error_msg)
        return None, None, None, None, None, error_msg


(es_client_instance, openai_client_instance, azure_openai_model_name,
 app_config_loaded, default_es_index,
 init_error) = load_initial_config_and_clients()  # Renamed app_config to app_config_loaded

# --- Initialize Database (Elasticsearch Index for Experiments) ---
if es_client_instance and not init_error:
    db_utils.init_db(es_client_instance)
elif not es_client_instance and not init_error:
    st.sidebar.error("Elasticsearch client not initialized. Experiment features will be unavailable.")

# --- Initialize Prompts in Session State ---
PROMPT_KEYS_FOR_APP_INIT = {
    "QA Generation": "session_qa_generation_prompt",
    "Question Groundedness Critique": "session_q_groundedness_prompt",
    "Question Relevance Critique": "session_q_relevance_prompt",
    "Question Standalone Critique": "session_q_standalone_prompt",
    "Visualization Explanation": "session_viz_explanation_prompt",
    "Translation Task": "session_translate_prompt"
}
DEFAULT_PROMPTS_FOR_APP_INIT = {
    PROMPT_KEYS_FOR_APP_INIT["QA Generation"]: prompts.QA_GENERATION_PROMPT_TEMPLATE,
    PROMPT_KEYS_FOR_APP_INIT["Question Groundedness Critique"]: prompts.QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT_TEMPLATE,
    PROMPT_KEYS_FOR_APP_INIT["Question Relevance Critique"]: prompts.QUESTION_RELEVANCE_CRITIQUE_PROMPT_TEMPLATE,
    PROMPT_KEYS_FOR_APP_INIT["Question Standalone Critique"]: prompts.QUESTION_STANDALONE_CRITIQUE_PROMPT_TEMPLATE,
    PROMPT_KEYS_FOR_APP_INIT["Visualization Explanation"]: prompts.VISUALIZATION_EXPLANATION_PROMPT_TEMPLATE,
    PROMPT_KEYS_FOR_APP_INIT["Translation Task"]: prompts.TRANSLATE_PROMPT_TEMPLATE
}
for key_name, session_key in PROMPT_KEYS_FOR_APP_INIT.items():
    if session_key not in st.session_state:
        st.session_state[session_key] = DEFAULT_PROMPTS_FOR_APP_INIT[session_key]

# --- Logo SVG ---
ELKEVAL_LOGO_SVG = """
<svg width="250" height="60" viewBox="0 0 250 60" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(10, 5) rotate(-15 22.5 22.5)"> <circle cx="22.5" cy="22.5" r="16" stroke="#4A90E2" stroke-width="4" fill="#E0EFFF" /> <rect x="36" y="33" width="20" height="6" rx="3" ry="3" fill="#4A90E2" transform="rotate(45 36 33)" /> </g>
  <text x="75" y="40" font-family="Verdana, Geneva, sans-serif" font-size="32" font-weight="bold" fill="#005A9C">Elk</text> <text x="138" y="40" font-family="Verdana, Geneva, sans-serif" font-size="32" font-weight="bold" fill="#F5A623">eval</text>
</svg>"""

# --- Sidebar ---
st.sidebar.markdown(ELKEVAL_LOGO_SVG, unsafe_allow_html=True)
st.sidebar.title("Elkeval")
st.sidebar.markdown("Navigation")

if 'current_page' not in st.session_state: st.session_state.current_page = "Homepage"
PAGES = {
    "🏠 Homepage": "Homepage", "📝 Ground Truth Generation": "Ground Truth Generation",
    "🔎 RAG Processor": "RAG Processor", "⚖️ RAGAS Evaluation": "RAGAS Evaluation",
    "📊 Evaluation Visualization": "Evaluation Visualization", "⚙️ Prompt Settings": "Prompt Settings",
    "🔬 Experiments Dashboard": "Experiments Dashboard"
}
for display_name, page_key in PAGES.items():
    if st.sidebar.button(display_name, key=f"nav_{page_key}", use_container_width=True):
        st.session_state.current_page = page_key
        if page_key != "Evaluation Visualization" and 'llm_explanation' in st.session_state:
            st.session_state.llm_explanation = None;
            st.session_state.summary_data_for_explanation = None
st.sidebar.divider();
st.sidebar.header("⚙️ Global Status")

if init_error:
    st.sidebar.error(f"Initialization Error: {init_error}")
elif app_config_loaded is None:
    st.sidebar.error("Failed to load .env config.")  # Use app_config_loaded
else:
    st.sidebar.success("Configuration loaded.")
    if azure_openai_model_name:
        st.sidebar.info(f"OpenAI Model: `{azure_openai_model_name}`")
    else:
        st.sidebar.warning("OpenAI Model for RAG not configured.")
    if default_es_index:
        st.sidebar.info(f"Default ES Index: `{default_es_index}`")
    else:
        st.sidebar.warning("Default ES Index not configured.")
    if es_client_instance:
        st.sidebar.info("Elasticsearch client connected.")
    else:
        st.sidebar.error("Elasticsearch client NOT connected.")

if not RAGAS_AVAILABLE:
    st.sidebar.warning("RAGAS features disabled. Install: `pip install ragas langchain-openai datasets altair`")
else:
    st.sidebar.info("RAGAS libraries available.")
st.sidebar.markdown("---");
st.sidebar.caption(f"App root: `{os.getcwd()}`")

# --- Main Page Content ---
if st.session_state.current_page == "Homepage":
    homepage_logo_svg_modified = ELKEVAL_LOGO_SVG.replace('width="250" height="60"', 'width="350" height="84"')
    logo_html_for_homepage = f"<div style='text-align: center; margin-bottom: 20px;'>{homepage_logo_svg_modified}</div>"
    st.markdown(logo_html_for_homepage, unsafe_allow_html=True)
    st.title("Welcome to Elkeval!")
    st.markdown("""
    **Your comprehensive suite to develop, test, and evaluate Retrieval Augmented Generation (RAG) pipelines with Elasticsearch.**
    **Key Features:**
    * **📝 Ground Truth Generation** (with language detection & prompt translation)
    * **🔎 RAG Processor** (with multi-method selection & batch processing)
    * **⚖️ RAGAS Evaluation** (with option to save runs as experiments)
    * **📊 Evaluation Visualization** (with LLM-powered explanations)
    * **⚙️ Prompt Settings**: Customize LLM prompts.
    * **🔬 Experiments Dashboard**: Track, compare, and manage your RAG experiments (using Elasticsearch!).
    """)
elif st.session_state.current_page == "Ground Truth Generation":
    if openai_client_instance and azure_openai_model_name:
        ground_truth_tab.show_ground_truth_tab(openai_client_instance, azure_openai_model_name)
    else:
        st.error("OpenAI client/model not available for Ground Truth Generation.")
elif st.session_state.current_page == "RAG Processor":
    if es_client_instance and openai_client_instance and azure_openai_model_name and default_es_index:
        rag_processor_tab.show_rag_processor_tab(es_client_instance, openai_client_instance, azure_openai_model_name,
                                                 default_es_index)
    else:
        st.error("Required clients/configs not available for RAG Processor.")
elif st.session_state.current_page == "RAGAS Evaluation":
    if app_config_loaded:
        ragas_eval_tab.show_ragas_eval_tab(app_config_loaded, RAGAS_AVAILABLE_FLAG=RAGAS_AVAILABLE,
                                           es_client=es_client_instance)  # Pass es_client
    else:
        st.error("App config not loaded for RAGAS evaluation.")
elif st.session_state.current_page == "Evaluation Visualization":
    viz_tab.show_viz_tab(openai_client_instance, azure_openai_model_name)
elif st.session_state.current_page == "Prompt Settings":
    prompt_settings_tab.show_prompt_settings_tab()
elif st.session_state.current_page == "Experiments Dashboard":
    if es_client_instance:
        experiments_dashboard_tab.show_experiments_dashboard_tab(es_client=es_client_instance)
    else:
        st.error("Elasticsearch client not available. Cannot display Experiments Dashboard.")
