# tabs/prompt_settings_tab.py
import streamlit as st
import prompts # To access default prompts

# Define keys for session state to avoid clashes and for easier management
PROMPT_KEYS = {
    "QA Generation": "session_qa_generation_prompt",
    "Question Groundedness Critique": "session_q_groundedness_prompt",
    "Question Relevance Critique": "session_q_relevance_prompt",
    "Question Standalone Critique": "session_q_standalone_prompt",
    "Visualization Explanation": "session_viz_explanation_prompt",
    "Translation Task": "session_translate_prompt"
}

DEFAULT_PROMPTS = {
    PROMPT_KEYS["QA Generation"]: prompts.QA_GENERATION_PROMPT_TEMPLATE,
    PROMPT_KEYS["Question Groundedness Critique"]: prompts.QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT_TEMPLATE,
    PROMPT_KEYS["Question Relevance Critique"]: prompts.QUESTION_RELEVANCE_CRITIQUE_PROMPT_TEMPLATE,
    PROMPT_KEYS["Question Standalone Critique"]: prompts.QUESTION_STANDALONE_CRITIQUE_PROMPT_TEMPLATE,
    PROMPT_KEYS["Visualization Explanation"]: prompts.VISUALIZATION_EXPLANATION_PROMPT_TEMPLATE,
    PROMPT_KEYS["Translation Task"]: prompts.TRANSLATE_PROMPT_TEMPLATE
}

def show_prompt_settings_tab():
    """Displays the Prompt Settings Tab."""
    st.header("⚙️ Prompt Template Settings")
    st.markdown("""
    Here you can view and modify the prompt templates used by the application for the current session.
    Changes made here will affect how the LLM generates responses for tasks like Q&A generation, critiques, and explanations.
    Click "Save Prompts" to apply your changes for this session.
    Click "Reset to Defaults" to revert to the original prompts.
    """)

    # Initialize prompts in session state if they don't exist
    for display_name, key in PROMPT_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = DEFAULT_PROMPTS[key]

    prompt_editors = {}

    for display_name, key in PROMPT_KEYS.items():
        with st.expander(f"Edit: {display_name} Prompt", expanded=False):
            prompt_editors[key] = st.text_area(
                f"Template for {display_name}:",
                value=st.session_state[key],
                height=300,
                key=f"text_area_{key}"
            )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Prompts", use_container_width=True, type="primary"):
            for key, text_area_content in prompt_editors.items():
                st.session_state[key] = text_area_content
            st.success("Prompts saved for the current session!")
            # st.experimental_rerun() # Optional: rerun to ensure UI reflects saved state immediately if needed

    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            for display_name, key in PROMPT_KEYS.items():
                st.session_state[key] = DEFAULT_PROMPTS[key]
            st.info("Prompts have been reset to their default values.")
            # We need to force a rerun to update the text areas with default values
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Current Active Prompts (Read-only)")
    for display_name, key in PROMPT_KEYS.items():
        with st.expander(f"{display_name} (Current)", expanded=False):
            st.text(st.session_state[key])
