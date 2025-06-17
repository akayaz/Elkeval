# Fixed version of experiments_dashboard_tab.py with debugging and error handling

import streamlit as st
import pandas as pd
import db_utils  # Uses Elasticsearch backend now
import json
import os
import altair as alt

from ragas_alternatives_implementation import show_alternative_visualizations

# Define the metrics to plot.
# The key in the tuple is the EXACT field name in Elasticsearch.
# The display name is for the chart/table.
DASHBOARD_METRICS_CONFIG = [
    ('avg_faithfulness', 'Faithfulness', '#1f77b4'),
    ('avg_answer_relevancy', 'Answer Relevancy', '#ff7f0e'),
    ('avg_llm_context_precision_with_reference', 'Context Precision', '#2ca02c'),
    ('avg_context_recall', 'Context Recall', '#d62728')
]


def create_parallel_coordinates_chart(df_chart):
    """Fixed Parallel Coordinates implementation"""
    try:
        # Data validation
        if df_chart.empty:
            st.error("❌ No data for parallel coordinates")
            return None

        required_cols = ['Experiment Name', 'Metric', 'Score']
        if not all(col in df_chart.columns for col in required_cols):
            st.error(f"❌ Missing columns: {required_cols}")
            return None

        # The key fix: proper data transformation
        df_pivot = df_chart.pivot(index='Experiment Name', columns='Metric', values='Score').reset_index()
        df_pivot['index'] = range(len(df_pivot))

        # Define expected metrics (adjust to match your actual metrics)
        expected_metrics = ['Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall']

        # Check available metrics
        available_metrics = [col for col in expected_metrics if col in df_pivot.columns]
        if not available_metrics:
            st.error("❌ No expected metrics found in data")
            return None

        st.info(f"✅ Creating parallel coordinates with {len(available_metrics)} metrics")

        # Create the chart with proper transform_fold
        chart = alt.Chart(df_pivot).transform_fold(
            fold=available_metrics,
            as_=['key', 'value']
        ).mark_line(
            point=True,
            strokeWidth=2,
            opacity=0.8
        ).encode(
            x=alt.X('key:N', title='RAGAS Metrics', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('value:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Experiment Name:N', legend=alt.Legend(title="Experiments")),
            detail='index:N',  # This is crucial for separate lines
            tooltip=['Experiment Name:N', 'key:N', 'value:Q']
        ).properties(
            title='Performance Profiles - Parallel Coordinates',
            width=500,
            height=300
        )

        return chart

    except Exception as e:
        st.error(f"❌ Parallel coordinates error: {str(e)}")
        return None


def create_kpi_dashboard_chart(df_chart):
    """Fixed KPI Dashboard implementation"""
    try:
        # Data validation
        if df_chart.empty:
            st.error("❌ No data for KPI dashboard")
            return None

        # Performance thresholds (customize these for your needs)
        thresholds = {
            'Faithfulness': {'poor': 0.6, 'good': 0.8, 'excellent': 1.0, 'target': 0.85},
            'Answer Relevancy': {'poor': 0.6, 'good': 0.8, 'excellent': 1.0, 'target': 0.82},
            'Context Precision': {'poor': 0.5, 'good': 0.7, 'excellent': 1.0, 'target': 0.75},
            'Context Recall': {'poor': 0.5, 'good': 0.7, 'excellent': 1.0, 'target': 0.75}
        }

        # Build bullet chart data
        bullet_data = []
        available_metrics = df_chart['Metric'].unique()

        for metric in available_metrics:
            if metric not in thresholds:
                continue

            metric_data = df_chart[df_chart['Metric'] == metric]
            config = thresholds[metric]

            # Background ranges (from largest to smallest for proper layering)
            bullet_data.extend([
                {'metric': metric, 'type': 'range_3', 'value': config['excellent'], 'color': '#eee'},
                {'metric': metric, 'type': 'range_2', 'value': config['good'], 'color': '#ddd'},
                {'metric': metric, 'type': 'range_1', 'value': config['poor'], 'color': '#bbb'}
            ])

            # Actual values (performance bars)
            for idx, (_, row) in enumerate(metric_data.iterrows()):
                bullet_data.append({
                    'metric': metric,
                    'type': f'measure_{idx}',
                    'value': row['Score'],
                    'color': 'steelblue',
                    'experiment': row['Experiment Name']
                })

            # Target line
            bullet_data.append({
                'metric': metric,
                'type': 'target',
                'value': config['target'],
                'color': 'red'
            })

        if not bullet_data:
            st.error("❌ Could not create bullet chart data")
            return None

        df_bullet = pd.DataFrame(bullet_data)
        st.info(f"✅ Creating KPI dashboard with {len(available_metrics)} metrics")

        # Create layered bullet chart
        base = alt.Chart(df_bullet)

        # Background ranges
        ranges = base.transform_filter(
            alt.expr.test(alt.expr.regexp('range'), alt.datum.type)
        ).mark_bar(height=15).encode(
            x=alt.X('value:Q', scale=alt.Scale(domain=[0, 1]), title='Score'),
            y=alt.Y('metric:N', title='RAGAS Metrics'),
            color=alt.Color('color:N', scale=None)
        )

        # Actual performance
        measures = base.transform_filter(
            alt.expr.test(alt.expr.regexp('measure'), alt.datum.type)
        ).mark_bar(height=6).encode(
            x='value:Q',
            y='metric:N',
            color=alt.value('steelblue'),
            tooltip=['metric:N', 'experiment:N', 'value:Q']
        )

        # Target markers
        targets = base.transform_filter(
            alt.datum.type == 'target'
        ).mark_tick(thickness=2, size=20, color='red').encode(
            x='value:Q',
            y='metric:N',
            tooltip=['metric:N', alt.Tooltip('value:Q', title='Target')]
        )

        chart = (ranges + measures + targets).properties(
            title='KPI Dashboard - Performance vs Targets',
            width=600,
            height=250
        ).resolve_scale(color='independent')

        return chart

    except Exception as e:
        st.error(f"❌ KPI dashboard error: {str(e)}")
        return None


def show_experiments_dashboard_tab(es_client):
    """Displays the Experiments Dashboard Tab with enhanced debugging."""
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
                    st.experimental_rerun()
                    # else: db_utils.delete_experiment will show its own error via st.error
            except (IndexError, ValueError) as e:
                st.error(f"Invalid experiment selection for deletion or error parsing ID: {e}")

    st.divider()
    st.subheader("Compare Experiments")

    exp_options_for_multiselect = {f"{exp.get('name', 'Unnamed')} (ID: {exp.get('id')})": exp.get('id') for exp in
                                   experiments}

    selected_display_names = st.multiselect(
        "Select experiments to compare (up to 5 for optimal chart readability):",
        options=list(exp_options_for_multiselect.keys()),
        default=list(exp_options_for_multiselect.keys())[:min(2, len(exp_options_for_multiselect))],
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
        # DEBUGGING: Check data structure
        with st.expander("🔍 Debug Information", expanded=False):
            st.write("**Experiment Data Structure:**")
            for i, exp in enumerate(selected_experiments_data[:1]):  # Check first experiment
                st.write(f"Experiment {i + 1}:")
                st.write(f"  - ID: {exp.get('id')}")
                st.write(f"  - Name: {exp.get('name')}")
                st.write(f"  - Available keys: {list(exp.keys())}")
                st.write("  - Field values:")
                for es_key, display_name, _ in DASHBOARD_METRICS_CONFIG:
                    value = exp.get(es_key)
                    st.write(f"    - {es_key}: {value} (type: {type(value).__name__})")

        # Validate field names exist
        all_exp_keys = set()
        for exp in selected_experiments_data:
            all_exp_keys.update(exp.keys())

        expected_keys = [key for key, _, _ in DASHBOARD_METRICS_CONFIG]
        missing_keys = set(expected_keys) - all_exp_keys
        if missing_keys:
            st.warning(f"⚠️ Missing expected keys in experiment data: {missing_keys}")
            st.write("Available keys:", sorted(all_exp_keys))

        comparison_data_for_table = []
        chart_data_for_altair = []

        for exp in selected_experiments_data:
            table_row = {
                "ID": exp.get('id'),
                "Experiment Name": exp.get('name'),
                "Timestamp": exp.get('timestamp'),
                "Methods": ", ".join(exp.get('retrieval_methods_used', ['N/A'])),
            }

            exp_name_for_chart = exp.get('name', f"ID_{exp.get('id')}")

            for es_key, display_name, _ in DASHBOARD_METRICS_CONFIG:
                score_value = exp.get(es_key)

                # Enhanced error handling for missing/invalid scores
                if score_value is None:
                    st.warning(f"Missing field '{es_key}' in experiment '{exp_name_for_chart}'")
                    score_value = 0  # Set to 0 for missing values

                # Handle non-numeric values
                try:
                    score_value = float(score_value) if score_value is not None else 0
                except (ValueError, TypeError):
                    st.warning(
                        f"Non-numeric value '{score_value}' for field '{es_key}' in experiment '{exp_name_for_chart}'")
                    score_value = 0

                # Add to table data
                table_row[display_name] = f"{score_value * 100:.1f}%" if score_value is not None else "N/A"

                # Add to chart data
                chart_data_for_altair.append({
                    "Experiment Name": exp_name_for_chart,
                    "Metric": display_name,
                    "Score": score_value
                })

            comparison_data_for_table.append(table_row)

        # Display table
        st.dataframe(pd.DataFrame(comparison_data_for_table))

        # Enhanced chart creation with debugging
        if chart_data_for_altair:
            df_chart = pd.DataFrame(chart_data_for_altair)

            # Debug chart data
            with st.expander("📊 Chart Data Debug", expanded=False):
                st.write(f"Total chart data entries: {len(chart_data_for_altair)}")
                st.write("Chart data preview:")
                st.write(df_chart.head(10))
                st.write("Metric counts:")
                st.write(df_chart['Metric'].value_counts())
                st.write("Score statistics:")
                st.write(df_chart.groupby('Metric')['Score'].describe())

            if not df_chart.empty:
                # Clean the data
                original_len = len(df_chart)
                df_chart = df_chart.dropna(subset=['Score'])
                df_chart = df_chart[df_chart['Score'] >= 0]  # Remove negative scores

                if len(df_chart) < original_len:
                    st.info(f"Removed {original_len - len(df_chart)} invalid chart entries")

                # Convert scores to percentage
                df_chart["Score"] = pd.to_numeric(df_chart["Score"], errors='coerce').fillna(0) * 100

                # Check available metrics
                available_metrics = df_chart['Metric'].unique()
                expected_metrics = [display_name for _, display_name, _ in DASHBOARD_METRICS_CONFIG]
                missing_chart_metrics = set(expected_metrics) - set(available_metrics)

                if missing_chart_metrics:
                    st.warning(f"Missing metrics in chart: {missing_chart_metrics}")

                st.subheader("Comparative Metrics Chart")

                if len(available_metrics) > 1:
                    # Multiple metrics - create faceted chart
                    metric_order = [display_name for _, display_name, _ in DASHBOARD_METRICS_CONFIG]
                    metric_colors = [color for _, _, color in DASHBOARD_METRICS_CONFIG]
                    # SOLUTION 3: Single Combined Chart (Alternative approach)
                    with st.expander("📈 Combined View (Single Chart)", expanded=False):
                        combined_chart = alt.Chart(df_chart).mark_bar(
                            size=10
                        ).encode(
                            x=alt.X('Experiment Name:N', title="Experiment"),
                            y=alt.Y('Score:Q', title='Score (%)', scale=alt.Scale(domain=(0, 100))),
                            color=alt.Color(
                                'Metric:N',
                                scale=alt.Scale(domain=expected_metrics, range=metric_colors),
                                legend=alt.Legend(title="Metrics")
                            ),
                            xOffset='Metric:N'  # Group bars by metric within each experiment
                        ).properties(
                            title='RAGAS Metrics - Grouped Bars View',
                            width=400
                        )
                        st.altair_chart(combined_chart, use_container_width=True)
                else:
                    # Single metric - create simple chart
                    st.warning(f"Only one metric available: {available_metrics[0]}")
                    chart = alt.Chart(df_chart).mark_bar().encode(
                        x=alt.X('Experiment Name:N', sort=None, axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('Score:Q', title='Score (%)', scale=alt.Scale(domain=(0, 100))),
                        color='Experiment Name:N'
                    ).properties(
                        title=f'{available_metrics[0]} Comparison'
                    )
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.error("No valid data available for chart generation.")
        else:
            st.error("No chart data could be generated.")

        # Alternative visualizations
        st.divider()
        st.subheader("📊 Alternative Visualizations")

        # create the visualization tabs
        show_alternative_visualizations(df_chart)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Performance Profiles")
            parallel_chart = create_parallel_coordinates_chart(df_chart)
            if parallel_chart:
                st.altair_chart(parallel_chart, use_container_width=True)

        with col2:
            st.subheader("🎯 KPI Dashboard")
            kpi_chart = create_kpi_dashboard_chart(df_chart)
            if kpi_chart:
                st.altair_chart(kpi_chart, use_container_width=True)
    else:
        st.info("No experiments selected for comparison or selected experiments not found.")
