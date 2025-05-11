# tabs/viz_tab.py
import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import prompts  # For default prompts fallback

# Define keys for session state prompts (must match those in prompt_settings_tab.py and app.py)
PROMPT_SESSION_KEYS = {
    "Visualization Explanation": "session_viz_explanation_prompt",
}


# Helper function to get current prompt from session state or default
def get_current_prompt(session_key, default_template):
    return st.session_state.get(session_key, default_template)


VIZ_METRICS_CONFIG = [
    ('faithfulness', '#1f77b4'),
    ('answer_relevancy', '#ff7f0e'),
    ('context_precision', '#2ca02c'),
    ('context_recall', '#d62728')
]
MAX_SCORE_VIZ = 1.0


def call_llm_for_explanation(client, model_deployment_name, prompt_content):
    try:
        response = client.chat.completions.create(
            model=model_deployment_name,
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": prompt_content}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling LLM for explanation: {e}");
        return None


def viz_load_data(uploaded_file_viz):
    try:
        uploaded_file_viz.seek(0);
        return json.load(uploaded_file_viz)
    except Exception as e:
        st.error(f"Error loading JSON for visualization: {e}");
        return None


def viz_extract_doc_names(data):
    return [os.path.basename(entry.get('path', f"Doc {i + 1}")).replace('_answers.json', '').replace('.json', '') for
            i, entry in enumerate(data)]


def viz_prepare_plot_data(data, metric_config_list):
    plot_data = {m_name: [] for m_name, _ in metric_config_list}
    for entry in data:
        for m_name, _ in metric_config_list:
            key = f'total_scores_{m_name}'
            scores = entry.get(key, [])
            if not scores:
                plot_data[m_name].append([])
            else:
                plot_data[m_name].append([(score / MAX_SCORE_VIZ) * 100 for score in scores if score is not None])
    return plot_data


def viz_plot_violinplots(plot_data, doc_names, metrics_config_for_plot):
    n_docs = len(doc_names)
    if n_docs == 0: st.warning("No documents to plot."); return None
    n_metrics = len(metrics_config_for_plot)
    if n_metrics == 0: st.warning("No metrics configured for plotting."); return None
    fig, axes = plt.subplots(1, n_docs, figsize=(6 * n_docs, 7), sharey=True, squeeze=False)
    axes = axes[0]
    for doc_idx, ax in enumerate(axes):
        doc_specific_plot_data, doc_specific_metric_labels, current_palette = [], [], {}
        for m_name, m_color in metrics_config_for_plot:
            scores_for_metric_in_doc = plot_data[m_name][doc_idx] if doc_idx < len(plot_data.get(m_name, [])) else []
            if scores_for_metric_in_doc:
                display_m_name = m_name.replace('_', ' ').replace('llm ', '').replace(' with reference',
                                                                                      '').capitalize()
                doc_specific_plot_data.extend(scores_for_metric_in_doc)
                doc_specific_metric_labels.extend([display_m_name] * len(scores_for_metric_in_doc))
                current_palette[display_m_name] = m_color
        if not doc_specific_plot_data:
            ax.set_title(doc_names[doc_idx], fontsize=12)
            ax.text(0.5, 0.5, 'No data for this document', ha='center', va='center', transform=ax.transAxes);
            continue
        df_doc_plot = pd.DataFrame({'Scores (%)': doc_specific_plot_data, 'Metric': doc_specific_metric_labels})
        metric_order = [m_cfg[0].replace('_', ' ').replace('llm ', '').replace(' with reference', '').capitalize() for
                        m_cfg in metrics_config_for_plot if
                        m_cfg[0].replace('_', ' ').replace('llm ', '').replace(' with reference',
                                                                               '').capitalize() in current_palette]
        sns.violinplot(x='Metric', y='Scores (%)', data=df_doc_plot, ax=ax, palette=current_palette, order=metric_order,
                       inner='box', cut=0, linewidth=1.2)
        sns.stripplot(x='Metric', y='Scores (%)', data=df_doc_plot, ax=ax, order=metric_order, color='k', size=3,
                      jitter=0.15, alpha=0.5)
        ax.set_title(doc_names[doc_idx], fontsize=12);
        ax.set_xlabel('')
        if doc_idx == 0:
            ax.set_ylabel('Score (%)', fontsize=10)
        else:
            ax.set_ylabel('')
        ax.set_ylim(0, 100);
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', labelsize=9);
        ax.tick_params(axis='y', labelsize=9)
        current_xticklabels = [label.get_text() for label in ax.get_xticklabels()]
        if current_xticklabels: ax.set_xticklabels(current_xticklabels, rotation=30, ha='right')
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=m_color,
                                  label=m_name.replace('_', ' ').replace('llm ', '').replace(' with reference',
                                                                                             '').capitalize(),
                                  markersize=10) for m_name, m_color in metrics_config_for_plot]
    if n_docs > 0 and legend_elements: fig.legend(handles=legend_elements, loc='lower center',
                                                  bbox_to_anchor=(0.5, -0.1 if n_docs > 2 else -0.15),
                                                  ncol=min(len(metrics_config_for_plot), 4), fontsize=10)
    plt.suptitle('RAG Evaluation Metrics Distribution by Document', fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95]);
    return fig


def show_viz_tab(openai_client, azure_openai_model_name):
    st.header("📊 Evaluation Visualization")
    st.markdown("Upload RAGAS evaluation JSON for visualization.")
    uploaded_ragas_json_for_viz = st.file_uploader("Upload RAGAS evaluation JSON", type=['json'],
                                                   key="ragas_viz_uploader_tab")

    if 'llm_explanation' not in st.session_state: st.session_state.llm_explanation = None
    if 'summary_data_for_explanation' not in st.session_state: st.session_state.summary_data_for_explanation = None
    if 'fig_viz' not in st.session_state: st.session_state.fig_viz = None
    if 'doc_names_for_summary' not in st.session_state: st.session_state.doc_names_for_summary = None

    if st.button("🖼️ Generate Visualization", type="primary", use_container_width=True, key="generate_viz_button_tab"):
        st.session_state.llm_explanation = None;
        st.session_state.summary_data_for_explanation = None
        st.session_state.fig_viz = None;
        st.session_state.doc_names_for_summary = None
        if uploaded_ragas_json_for_viz is None:
            st.warning("Please upload a RAGAS evaluation JSON file.")
        else:
            with st.spinner("Generating visualization..."):
                try:
                    viz_data = viz_load_data(uploaded_ragas_json_for_viz)
                    if viz_data:
                        st.session_state.doc_names_for_summary = viz_extract_doc_names(viz_data)
                        plot_data_viz = viz_prepare_plot_data(viz_data, VIZ_METRICS_CONFIG)
                        summary_data_list = []
                        for i, entry in enumerate(viz_data):  # Manually build summary_data_list for session state
                            doc_summary = {"Document": st.session_state.doc_names_for_summary[i]}
                            for m_name, _ in VIZ_METRICS_CONFIG:
                                avg_key = f'average_score_{m_name}';
                                avg = entry.get(avg_key, None)
                                display_m_name = m_name.replace('_', ' ').replace('llm ', '').replace(' with reference',
                                                                                                      '').capitalize()
                                doc_summary[display_m_name] = f"{avg * 100:.1f}%" if avg is not None else "N/A"
                            summary_data_list.append(doc_summary)
                        if summary_data_list: st.session_state.summary_data_for_explanation = summary_data_list
                        st.session_state.fig_viz = viz_plot_violinplots(plot_data_viz,
                                                                        st.session_state.doc_names_for_summary,
                                                                        VIZ_METRICS_CONFIG)
                        if st.session_state.fig_viz:
                            st.success("Visualization generated.")
                        else:
                            st.warning("Could not generate plot.")
                    else:
                        st.error("Failed to load data for visualization.")
                except Exception as e:
                    st.error(f"Visualization error: {e}");
                    st.code(traceback.format_exc())

    if st.session_state.summary_data_for_explanation and st.session_state.doc_names_for_summary:
        st.subheader("Average Scores Summary (%)")
        st.table(pd.DataFrame(st.session_state.summary_data_for_explanation))
    if st.session_state.fig_viz: st.pyplot(st.session_state.fig_viz)

    if st.session_state.summary_data_for_explanation:
        st.divider();
        st.subheader("🤖 LLM Explanation of Results")
        if openai_client and azure_openai_model_name:
            if st.button("💡 Get Explanation from LLM", key="get_llm_explanation_button"):
                summary_str = json.dumps(st.session_state.summary_data_for_explanation, indent=2)
                # Get current viz explanation prompt from session state or default
                viz_explanation_prompt_template = get_current_prompt(
                    PROMPT_SESSION_KEYS["Visualization Explanation"],
                    prompts.VISUALIZATION_EXPLANATION_PROMPT_TEMPLATE
                )
                prompt_text = viz_explanation_prompt_template.format(summary_data_string=summary_str)
                with st.spinner("Asking LLM for an explanation..."):
                    explanation = call_llm_for_explanation(openai_client, azure_openai_model_name, prompt_text)
                    st.session_state.llm_explanation = explanation if explanation else "Could not retrieve explanation."
        else:
            st.warning("OpenAI client/model not configured for LLM explanation.")
        if st.session_state.llm_explanation: st.markdown(st.session_state.llm_explanation)
