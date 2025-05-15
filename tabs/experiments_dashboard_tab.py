# tabs/experiments_dashboard_tab.py
import streamlit as st
import pandas as pd
import db_utils  # Uses Elasticsearch backend now
import json
import os
import altair as alt


def show_experiments_dashboard_tab(es_client):  # Expects es_client
    """Displays the Experiments Dashboard Tab."""
    st.header("🔬 Experiments Dashboard")
    st.markdown("""
    View, compare, and manage your saved RAG evaluation experiments stored in Elasticsearch. 
    Select experiments from the list to see their comparative metrics.
    """)

    if es_client is None:
        st.error("Elasticsearch client is not available. Cannot load experiments.")
        return

    experiments = db_utils.load_all_experiments(es_client)

    if not experiments:
        st.info(
            "No experiments saved yet. Run an evaluation in the 'RAGAS Evaluation' tab and save it as an experiment.")
        return

    # Display experiments for management (deletion)
    experiment_display_list = [f"{exp.get('name', 'Unnamed')} (ID: {exp.get('id')}, Date: {exp.get('timestamp')})" for
                               exp in experiments]

    with st.expander("Manage Experiments", expanded=False):
        exp_to_delete_display_name = st.selectbox(
            "Select experiment to delete:",
            options=[""] + experiment_display_list,
            key="exp_delete_select"
        )
        if exp_to_delete_display_name and st.button("🗑️ Delete Selected Experiment", key="delete_exp_button"):
            try:
                # Extract ID from the display string (assuming ID is always present after "ID: ")
                exp_id_to_delete = exp_to_delete_display_name.split("ID: ")[1].split(",")[0].strip()
                if db_utils.delete_experiment(es_client, exp_id_to_delete):
                    st.success(f"Experiment '{exp_to_delete_display_name}' deleted successfully.")
                    st.rerun()
                else:
                    st.error("Failed to delete experiment. Check logs for details.")
            except (IndexError, ValueError) as e:
                st.error(f"Invalid experiment selection for deletion or error parsing ID: {e}")

    st.divider()
    st.subheader("Compare Experiments")

    # Allow selection of multiple experiments for comparison
    # Use a dictionary to map display names back to experiment IDs for reliable selection
    exp_options_for_multiselect = {f"{exp.get('name', 'Unnamed')} (ID: {exp.get('id')})": exp.get('id') for exp in
                                   experiments}

    selected_display_names = st.multiselect(
        "Select experiments to compare (up to 5 for optimal chart readability):",
        options=list(exp_options_for_multiselect.keys()),
        default=list(exp_options_for_multiselect.keys())[:min(2, len(exp_options_for_multiselect))],
        # Default to first two or all if less
        key="compare_experiments_multiselect"
    )

    if not selected_display_names:
        st.info("Select one or more experiments from the list above to compare their metrics.")
        return

    if len(selected_display_names) > 5:
        st.warning("Comparing more than 5 experiments might make the chart hard to read.")

    selected_experiment_ids = [exp_options_for_multiselect[name] for name in selected_display_names]
    selected_experiments_data = [exp for exp in experiments if exp.get('id') in selected_experiment_ids]

    if selected_experiments_data:
        comparison_data_for_table = []
        chart_data_for_altair = []

        for exp in selected_experiments_data:
            # For table display
            comparison_data_for_table.append({
                "ID": exp.get('id'),
                "Experiment Name": exp.get('name'),
                "Timestamp": exp.get('timestamp'),
                "Methods": ", ".join(exp.get('retrieval_methods_used', ['N/A'])),
                "Faithfulness": f"{exp.get('avg_faithfulness', 0) * 100:.1f}%" if exp.get(
                    'avg_faithfulness') is not None else "N/A",
                "Answer Relevancy": f"{exp.get('avg_answer_relevancy', 0) * 100:.1f}%" if exp.get(
                    'avg_answer_relevancy') is not None else "N/A",
                "Context Precision": f"{exp.get('avg_llm_context_precision_with_reference', 0) * 100:.1f}%" if exp.get(
                    'avg_llm_context_precision_with_reference') is not None else "N/A",
                "Context Recall": f"{exp.get('avg_context_recall', 0) * 100:.1f}%" if exp.get(
                    'avg_llm_context_recall') is not None else "N/A",
                # "RAGAS Inputs": ", ".join(exp.get('ragas_input_source_files', ['N/A'])[:2]) + ("..." if len(exp.get('ragas_input_source_files', [])) > 2 else ""), # Show first few
                # "RAGAS Summary File": os.path.basename(exp.get('ragas_output_summary_file', 'N/A'))
            })

            # For Altair chart (needs raw numeric scores)
            exp_name_for_chart = exp.get('name', f"ID_{exp.get('id')}")
            metrics_for_chart = {
                "Faithfulness": exp.get('avg_faithfulness', 0),
                "Answer Relevancy": exp.get('avg_answer_relevancy', 0),
                "Context Precision": exp.get('avg_llm_context_precision_with_reference', 0),
                "Context Recall": exp.get('avg_llm_context_recall', 0),
            }
            for metric_name, score in metrics_for_chart.items():
                chart_data_for_altair.append({
                    "Experiment Name": exp_name_for_chart,
                    "Metric": metric_name,
                    "Score": score if score is not None else 0  # Default to 0 if score is None for plotting
                })

        st.dataframe(pd.DataFrame(comparison_data_for_table))

        if chart_data_for_altair:
            df_chart = pd.DataFrame(chart_data_for_altair)

            if not df_chart.empty:
                st.subheader("Comparative Metrics Chart")
                df_chart["Score"] = pd.to_numeric(df_chart["Score"], errors='coerce').fillna(0) * 100

                chart = alt.Chart(df_chart).mark_bar().encode(
                    x=alt.X('Experiment Name:N', sort=None, axis=alt.Axis(labelAngle=-45, title="Experiment")),
                    y=alt.Y('Score:Q', title='Score (%)', scale=alt.Scale(domain=(0, 100))),
                    color='Experiment Name:N',
                    column=alt.Column('Metric:N', title="Metric", sort=None)
                    # Sort=None to keep order from VIZ_METRICS_CONFIG
                ).properties(
                    title='RAGAS Metrics Comparison'
                ).configure_axis(
                    labelFontSize=10,
                    titleFontSize=12
                ).configure_legend(
                    titleFontSize=12,
                    labelFontSize=10
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Not enough data to generate comparative chart.")
    else:
        st.info("No experiments selected for comparison or selected experiments not found.")

