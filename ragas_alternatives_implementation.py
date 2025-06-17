# ===================================================================
# RAGAS Dashboard: Alternative Visualization Implementation Guide
# ===================================================================

import pandas as pd
import altair as alt
import plotly.graph_objects as go
import streamlit as st

# Assuming you have your RAGAS data in this format:
# df_chart with columns: ['Experiment Name', 'Metric', 'Score']

# METRIC CONFIGURATION (from your existing code)
DASHBOARD_METRICS_CONFIG = [
    ('avg_faithfulness', 'Faithfulness', '#1f77b4'),
    ('avg_answer_relevancy', 'Answer Relevancy', '#ff7f0e'),
    ('avg_llm_context_precision_with_reference', 'Context Precision', '#2ca02c'),
    ('avg_context_recall', 'Context Recall', '#d62728')
]

metric_names = [display_name for _, display_name, _ in DASHBOARD_METRICS_CONFIG]
metric_colors = [color for _, _, color in DASHBOARD_METRICS_CONFIG]

# ===================================================================
# ALTERNATIVE 1: HEATMAP - RECOMMENDED FOR OVERVIEW
# ===================================================================

def create_heatmap(df_chart):
    """Create a heatmap showing all metrics across all experiments"""
    heatmap = alt.Chart(df_chart).mark_rect().encode(
        x=alt.X('Metric:N',
                sort=metric_names,
                title='RAGAS Metrics'),
        y=alt.Y('Experiment Name:N',
                title='Experiments'),
        color=alt.Color(
            'Score:Q',
            scale=alt.Scale(scheme='viridis', domain=[0.6, 1.0]),
            legend=alt.Legend(title="Performance Score")
        ),
        tooltip=['Experiment Name:N', 'Metric:N', alt.Tooltip('Score:Q', format='.3f')]
    ).properties(
        title='RAGAS Metrics Performance Heatmap',
        width=350,
        height=200
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=12
    ).configure_title(
        fontSize=14
    )
    return heatmap

# ===================================================================
# ALTERNATIVE 2: PARALLEL COORDINATES - RECOMMENDED FOR PATTERNS
# ===================================================================

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

# ===================================================================
# ALTERNATIVE 3: KPI BULLET CHART - RECOMMENDED FOR DASHBOARDS
# ===================================================================

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

# ===================================================================
# ALTERNATIVE 4: RADAR CHART (PLOTLY) - RECOMMENDED FOR PRESENTATIONS
# ===================================================================

def create_radar_chart(df_chart):
    """Create radar chart using Plotly for multi-dimensional comparison"""
    import plotly.graph_objects as go

    fig = go.Figure()

    experiments = df_chart['Experiment Name'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, exp in enumerate(experiments):
        exp_data = df_chart[df_chart['Experiment Name'] == exp]
        values = []
        for metric in metric_names:
            score = exp_data[exp_data['Metric'] == metric]['Score'].iloc[0]
            values.append(score * 100)  # Convert to percentage

        # Close the radar chart loop
        values_closed = values + [values[0]]
        metrics_closed = metric_names + [metric_names[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=metrics_closed,
            fill='toself',
            name=exp,
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[60, 100],
                tickvals=[60, 70, 80, 90, 100],
                ticktext=['60%', '70%', '80%', '90%', '100%']
            )),
        showlegend=True,
        title="RAGAS Metrics - Radar Chart Comparison",
        font=dict(size=12),
        width=600,
        height=500
    )

    return fig

# ===================================================================
# IMPLEMENTATION IN YOUR DASHBOARD
# ===================================================================

def show_alternative_visualizations(df_chart):
    """Add to your Streamlit dashboard"""
    import streamlit as st

    # Create tabs for different visualizations
    # tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Parallel Coordinates", "KPI Dashboard", "Radar Chart"])
    tab1, tab2 = st.tabs(["Heatmap", "Radar Chart"])

    with tab1:
        st.subheader("🔥 Performance Heatmap")
        st.write("Quick overview of all metrics across experiments")
        heatmap = create_heatmap(df_chart)
        st.altair_chart(heatmap, use_container_width=True)

    # with tab2:
    #     st.subheader("📈 Performance Profiles")
    #     st.write("Compare experiment patterns across all metrics")
    #     parallel = create_parallel_coordinates_chart(df_chart)
    #     st.altair_chart(parallel, use_container_width=True)
    #
    # with tab3:
    #     st.subheader("🎯 KPI Dashboard")
    #     st.write("Performance levels with targets and thresholds")
    #     bullet = create_kpi_dashboard_chart(df_chart)
    #     st.altair_chart(bullet, use_container_width=True)

    with tab2:
        st.subheader("🕸️ Radar Comparison")
        st.write("Multi-dimensional performance comparison")
        radar = create_radar_chart(df_chart)
        st.plotly_chart(radar, use_container_width=True)