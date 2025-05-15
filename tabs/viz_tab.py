# tabs/viz_tab.py
import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# Import prompts directly (assuming project root is in sys.path via app.py)
import prompts

# --- Visualization Logic ---
# VIZ_METRICS_CONFIG should use the base metric names as keys in 'per_item_scores'
# and for constructing 'average_score_...' keys from the RAGAS output JSON.
VIZ_METRICS_CONFIG = [
    ('faithfulness', '#1f77b4'),
    ('answer_relevancy', '#ff7f0e'),
    ('llm_context_precision_with_reference', '#2ca02c'),
    ('context_recall', '#d62728')
]
MAX_SCORE_VIZ = 1.0  # RAGAS scores are 0-1


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
        st.error(f"Error calling LLM for explanation: {e}")
        return None


def viz_load_data(uploaded_file_viz):
    """Loads a list of RAGAS evaluation summaries from the uploaded JSON file."""
    try:
        uploaded_file_viz.seek(0)
        data_list = json.load(uploaded_file_viz)  # Expecting a list of dictionaries
        if not isinstance(data_list, list):
            st.error(
                "Visualization JSON does not contain a list of evaluation results as expected. Please check the file format.")
            return None
        if not all(isinstance(item, dict) for item in data_list):
            st.error("Not all items in the Visualization JSON are dictionaries. Please check the file format.")
            return None
        return data_list
    except Exception as e:
        st.error(f"Error loading JSON for visualization: {e}")
        return None


def viz_extract_doc_names(data_list):
    """
    Generates a list of representative names for each evaluation run in the data_list.
    Each item in 'data_list' is a summary object from the RAGAS evaluation output for a specific input file.
    """
    doc_names = []
    for i, entry in enumerate(data_list):
        path = entry.get('path', f"Scenario_{i + 1}")  # Fallback name
        base_name = os.path.splitext(os.path.basename(path))[0]
        base_name = base_name.replace('_answers', '').replace('_rag_output', '').replace('.json', '')
        doc_names.append(base_name if base_name else f"Scenario_{i + 1}")
    return doc_names


def viz_prepare_plot_data(data_list, metric_config_list):
    """
    Prepares data for violin plots from a list of RAGAS evaluation summaries.
    'data_list' is a list of summary objects.
    Output: {'metric1': [[scores_scenario1], [scores_scenario2]], 'metric2': [[scores_scenario1], [scores_scenario2]], ...}
    """
    plot_data = {m_name_cfg: [] for m_name_cfg, _ in metric_config_list}

    for entry_idx, entry_data_dict in enumerate(data_list):
        per_item_scores_map = entry_data_dict.get('per_item_scores', {})

        for metric_name_from_config, _ in metric_config_list:
            scores_for_metric = per_item_scores_map.get(metric_name_from_config, [])

            if scores_for_metric and isinstance(scores_for_metric, list):
                percent_scores = [(score / MAX_SCORE_VIZ) * 100 for score in scores_for_metric if score is not None]
                # If it's the first entry, initialize the list for this metric
                if entry_idx == 0:
                    plot_data[metric_name_from_config].append(percent_scores)
                # Else, if lists for this metric already exist (should for multiple scenarios)
                # This logic needs to ensure one list of scores per scenario for each metric
                # The current plot_data structure expects plot_data['faithfulness'] = [ [scen1_scores], [scen2_scores] ]
                else:
                    # Ensure the outer list for the metric exists and has enough sublists
                    while len(plot_data[metric_name_from_config]) <= entry_idx:
                        plot_data[metric_name_from_config].append(
                            [])  # Add empty list for previous scenarios if somehow missed
                    plot_data[metric_name_from_config][entry_idx] = percent_scores  # Assign scores for current scenario
            else:  # No scores or not a list
                # Ensure a placeholder empty list is added for this scenario and metric
                while len(plot_data[metric_name_from_config]) <= entry_idx:
                    plot_data[metric_name_from_config].append([])
                plot_data[metric_name_from_config][entry_idx] = []

                if not scores_for_metric:
                    st.caption(
                        f"Note: No per-item scores for metric '{metric_name_from_config}' in scenario {entry_idx + 1}.")
                else:
                    st.caption(
                        f"Warning: Per-item scores for '{metric_name_from_config}' in scenario {entry_idx + 1} not a list.")
    return plot_data


def viz_print_summary_stats_and_return_data(data_list, metric_config_list, doc_names):
    """
    Prints the summary table for multiple RAGAS evaluation runs.
    'data_list' is a list of summary objects.
    'doc_names' is a list of representative names for the runs.
    """
    st.subheader("Average Scores Summary (%)")
    summary_data_for_llm = []  # For LLM explanation

    table_data_rows = []  # For st.table

    for i, entry_data_dict in enumerate(data_list):
        run_name_for_table = doc_names[i] if i < len(doc_names) else f"Scenario {i + 1}"
        row_summary = {"Scenario": run_name_for_table}  # Changed "Document" to "Scenario"

        average_scores_map = entry_data_dict.get('average_scores', {})

        for m_name_cfg, _ in metric_config_list:
            avg_key = f'avg_{m_name_cfg}'
            avg_score_value = average_scores_map.get(avg_key)
            display_m_name = m_name_cfg.replace('_', ' ').replace('llm context ', 'ctx. ').replace(' with reference',
                                                                                                   '').capitalize()
            row_summary[display_m_name] = f"{avg_score_value * 100:.1f}%" if avg_score_value is not None else "N/A"

        table_data_rows.append(row_summary)
        summary_data_for_llm.append(
            {"Scenario": run_name_for_table, "Average Scores": average_scores_map})  # Simpler structure for LLM

    if table_data_rows:
        summary_df = pd.DataFrame(table_data_rows)
        st.table(summary_df)
        return summary_data_for_llm
    else:
        st.write("No summary data to display.")
        return None


def viz_plot_violinplots(plot_data, doc_names, metrics_config_for_plot):
    # doc_names is a list of names, one for each scenario/document in the RAGAS summary JSON
    # plot_data is like: {'faithfulness': [[scen1_scores], [scen2_scores]], ...}

    n_scenarios = len(doc_names)
    if n_scenarios == 0: st.warning("No data to plot."); return None
    n_metrics_configured = len(metrics_config_for_plot)
    if n_metrics_configured == 0: st.warning("No metrics configured for plotting."); return None

    # Create subplots, one for each scenario
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 7), sharey=True, squeeze=False)
    axes = axes[0]  # axes is now a 1D array of subplots

    for scenario_idx, ax in enumerate(axes):
        current_scenario_plot_data = []  # Data for all metrics for THIS scenario
        current_scenario_metric_labels = []  # Metric labels for THIS scenario
        current_palette = {}  # Palette for THIS scenario's subplot

        for m_name_cfg, m_color in metrics_config_for_plot:
            # plot_data[m_name_cfg] is a list of lists of scores.
            # We need the scores for the current scenario_idx
            if m_name_cfg in plot_data and scenario_idx < len(plot_data[m_name_cfg]) and \
                    isinstance(plot_data[m_name_cfg][scenario_idx], list):

                scores_for_metric_in_scenario = plot_data[m_name_cfg][scenario_idx]

                if scores_for_metric_in_scenario:  # Check if list is not empty
                    display_m_name = m_name_cfg.replace('_', ' ').replace('llm context ', 'ctx. ').replace(
                        ' with reference', '').capitalize()
                    current_scenario_plot_data.extend(scores_for_metric_in_scenario)
                    current_scenario_metric_labels.extend([display_m_name] * len(scores_for_metric_in_scenario))
                    current_palette[display_m_name] = m_color
            # else:
            # st.caption(f"Note: No plot data for metric '{m_name_cfg}' in scenario '{doc_names[scenario_idx]}'.")

        if not current_scenario_plot_data:
            ax.set_title(doc_names[scenario_idx] if scenario_idx < len(doc_names) else f"Scenario {scenario_idx + 1}",
                         fontsize=12)
            ax.text(0.5, 0.5, 'No plottable score data', horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            continue

        df_scenario_plot = pd.DataFrame(
            {'Scores (%)': current_scenario_plot_data, 'Metric': current_scenario_metric_labels})
        metric_order = [
            m_cfg[0].replace('_', ' ').replace('llm context ', 'ctx. ').replace(' with reference', '').capitalize() for
            m_cfg in metrics_config_for_plot if
            m_cfg[0].replace('_', ' ').replace('llm context ', 'ctx. ').replace(' with reference',
                                                                                '').capitalize() in current_palette]

        sns.violinplot(x='Metric', y='Scores (%)', data=df_scenario_plot, ax=ax, palette=current_palette,
                       order=metric_order, inner='box', cut=0, linewidth=1.2)
        sns.stripplot(x='Metric', y='Scores (%)', data=df_scenario_plot, ax=ax, order=metric_order, color='k', size=3,
                      jitter=0.15, alpha=0.5)

        ax.set_title(doc_names[scenario_idx] if scenario_idx < len(doc_names) else f"Scenario {scenario_idx + 1}",
                     fontsize=14)
        ax.set_xlabel('')
        if scenario_idx == 0:
            ax.set_ylabel('Score (%)', fontsize=10)
        else:
            ax.set_ylabel('')
        ax.set_ylim(0, 100);
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
        current_xticklabels = [label.get_text() for label in ax.get_xticklabels()]
        if current_xticklabels:
            ax.set_xticklabels(current_xticklabels, rotation=30, ha='right')

    # Common legend for the figure
    if current_palette:  # Check if any metrics were plotted to create a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=m_color,
                       label=m_name_cfg.replace('_', ' ').replace('llm context ', 'ctx. ').replace(' with reference',
                                                                                                   '').capitalize(),
                       markersize=10)
            for m_name_cfg, m_color in metrics_config_for_plot if
            m_name_cfg.replace('_', ' ').replace('llm context ', 'ctx. ').replace(' with reference',
                                                                                  '').capitalize() in current_palette
        ]
        if legend_elements:
            fig.legend(handles=legend_elements, loc='lower center',
                       bbox_to_anchor=(0.5, -0.05 if n_scenarios <= 2 else -0.15), ncol=min(len(legend_elements), 4),
                       fontsize=10)

    plt.suptitle('RAGAS Evaluation Metrics Distribution by Scenario', fontsize=16, y=1.0 if n_scenarios <= 2 else 1.03)
    plt.tight_layout(rect=[0, 0.08 if n_scenarios <= 2 else 0.15, 1, 0.95])
    return fig


def show_viz_tab(openai_client, azure_openai_model_name):
    st.header("📊 Evaluation Visualization")
    st.markdown("""
    Upload the JSON file generated by the "RAGAS Evaluation" tab. 
    This file should be a list, where each item represents an evaluated scenario/document.
    """)

    uploaded_ragas_json_for_viz = st.file_uploader(
        "Upload RAGAS evaluation JSON file (list of scenarios)",
        type=['json'],
        key="ragas_viz_uploader_tab"
    )

    if 'llm_explanation' not in st.session_state: st.session_state.llm_explanation = None
    if 'summary_data_for_explanation' not in st.session_state: st.session_state.summary_data_for_explanation = None
    if 'fig_viz' not in st.session_state: st.session_state.fig_viz = None
    # doc_names_for_summary is not strictly needed in session state if re-extracted each time

    if st.button("🖼️ Generate Visualization", type="primary", use_container_width=True, key="generate_viz_button_tab"):
        st.session_state.llm_explanation = None
        st.session_state.summary_data_for_explanation = None
        st.session_state.fig_viz = None
        # st.session_state.doc_names_for_summary = None # Not needed in session state

        if uploaded_ragas_json_for_viz is None:
            st.warning("Please upload a RAGAS evaluation JSON file.")
        else:
            with st.spinner("Generating visualization..."):
                try:
                    viz_data_list = viz_load_data(uploaded_ragas_json_for_viz)  # Expects a list
                    if viz_data_list:
                        doc_names = viz_extract_doc_names(viz_data_list)
                        plot_data_viz = viz_prepare_plot_data(viz_data_list, VIZ_METRICS_CONFIG)

                        summary_data_list_for_llm = viz_print_summary_stats_and_return_data(viz_data_list,
                                                                                            VIZ_METRICS_CONFIG,
                                                                                            doc_names)
                        if summary_data_list_for_llm:
                            st.session_state.summary_data_for_explanation = summary_data_list_for_llm

                        st.session_state.fig_viz = viz_plot_violinplots(plot_data_viz, doc_names, VIZ_METRICS_CONFIG)

                        if st.session_state.fig_viz:
                            st.success("Visualization generated. Scroll down for plot and LLM explanation option.")

                except Exception as e:
                    st.error(f"An error occurred during visualization: {e}");
                    st.code(traceback.format_exc())

    if st.session_state.fig_viz:
        st.pyplot(st.session_state.fig_viz)

    if st.session_state.summary_data_for_explanation:
        st.divider()
        st.subheader("🤖 LLM Explanation of Results")
        if openai_client and azure_openai_model_name:
            if st.button("💡 Get Explanation from LLM", key="get_llm_explanation_button"):
                summary_str = json.dumps(st.session_state.summary_data_for_explanation, indent=2)

                viz_explanation_prompt_template = st.session_state.get(
                    "session_viz_explanation_prompt",
                    prompts.VISUALIZATION_EXPLANATION_PROMPT_TEMPLATE
                )
                prompt_text = viz_explanation_prompt_template.format(summary_data_string=summary_str)
                with st.spinner("Asking LLM for an explanation..."):
                    explanation = call_llm_for_explanation(openai_client, azure_openai_model_name, prompt_text)
                    if explanation:
                        st.session_state.llm_explanation = explanation
                    else:
                        st.session_state.llm_explanation = "Could not retrieve an explanation from the LLM."
        else:
            st.warning("OpenAI client or model not configured. Cannot generate LLM explanation.")

        if st.session_state.llm_explanation:
            st.markdown(st.session_state.llm_explanation)
