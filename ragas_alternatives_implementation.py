# ===================================================================
# RAGAS Dashboard: Alternative Visualization Implementation Guide
# ===================================================================

import pandas as pd
import altair as alt
import plotly.graph_objects as go

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

def create_parallel_coordinates(df_chart):
    """Create parallel coordinates showing experiment profiles"""
    # Add index for connecting lines
    df_indexed = df_chart.copy()
    df_indexed['index'] = df_indexed.groupby('Experiment Name').ngroup()

    parallel_chart = alt.Chart(df_indexed).mark_line(
        opacity=0.7,
        strokeWidth=3
    ).encode(
        x=alt.X('Metric:N',
                sort=metric_names,
                axis=alt.Axis(labelAngle=0, title="RAGAS Metrics")),
        y=alt.Y('Score:Q',
                scale=alt.Scale(domain=[0.6, 1.0]),
                title='Performance Score'),
        color=alt.Color('Experiment Name:N',
                       legend=alt.Legend(title="Experiments")),
        detail='index:N',
        tooltip=['Experiment Name:N', 'Metric:N', alt.Tooltip('Score:Q', format='.3f')]
    ).properties(
        title='RAGAS Metrics - Performance Profiles',
        width=500,
        height=300
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=12
    ).configure_title(
        fontSize=14
    )
    return parallel_chart

# ===================================================================
# ALTERNATIVE 3: KPI BULLET CHART - RECOMMENDED FOR DASHBOARDS
# ===================================================================

def create_bullet_chart(df_chart, target_score=0.85):
    """Create KPI-style bullet chart with performance levels"""
    good_threshold = 0.80
    fair_threshold = 0.70

    # Add performance levels
    df_bullet = df_chart.copy()
    df_bullet['Performance_Level'] = df_bullet['Score'].apply(
        lambda x: 'Good' if x >= good_threshold
                 else 'Fair' if x >= fair_threshold
                 else 'Poor'
    )
    df_bullet['Score_Pct'] = df_bullet['Score'] * 100
    df_bullet['Combo'] = df_bullet['Experiment Name'] + ' - ' + df_bullet['Metric']

    bullet_chart = alt.Chart(df_bullet).mark_bar(size=15).encode(
        x=alt.X('Score_Pct:Q',
                scale=alt.Scale(domain=[60, 100]),
                title='Performance Score (%)'),
        y=alt.Y('Combo:N',
                sort=alt.SortField('Score_Pct', order='descending'),
                title='Experiment - Metric'),
        color=alt.Color('Performance_Level:N',
                       scale=alt.Scale(domain=['Poor', 'Fair', 'Good'],
                                     range=['#d62728', '#ff7f0e', '#2ca02c']),
                       legend=alt.Legend(title="Performance Level")),
        tooltip=['Experiment Name:N', 'Metric:N', alt.Tooltip('Score_Pct:Q', format='.1f')]
    ).properties(
        title='RAGAS Metrics - KPI Performance Dashboard',
        width=500,
        height=400
    ).configure_title(fontSize=14)

    return bullet_chart

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
    tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Parallel Coordinates", "KPI Dashboard", "Radar Chart"])

    with tab1:
        st.subheader("🔥 Performance Heatmap")
        st.write("Quick overview of all metrics across experiments")
        heatmap = create_heatmap(df_chart)
        st.altair_chart(heatmap, use_container_width=True)

    with tab2:
        st.subheader("📈 Performance Profiles")
        st.write("Compare experiment patterns across all metrics")
        parallel = create_parallel_coordinates(df_chart)
        st.altair_chart(parallel, use_container_width=True)

    with tab3:
        st.subheader("🎯 KPI Dashboard")
        st.write("Performance levels with targets and thresholds")
        bullet = create_bullet_chart(df_chart)
        st.altair_chart(bullet, use_container_width=True)

    with tab4:
        st.subheader("🕸️ Radar Comparison")
        st.write("Multi-dimensional performance comparison")
        radar = create_radar_chart(df_chart)
        st.plotly_chart(radar, use_container_width=True)