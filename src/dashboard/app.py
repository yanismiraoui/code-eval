import streamlit as st
import json
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import plotly.colors as pc

st.set_page_config(
    page_title="CodeEval Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS to style the multiselect
st.markdown("""
<style>
    .stMultiSelect [data-baseweb="tag"] {
        background-color: rgb(51, 102, 204) !important;    /* Royal blue */
    }
    .stMultiSelect [data-baseweb="tag"]:nth-of-type(2) {
        background-color: rgb(65, 131, 215) !important;    /* Steel blue */
    }
    .stMultiSelect [data-baseweb="tag"]:nth-of-type(3) {
        background-color: rgb(89, 171, 227) !important;    /* Sky blue */
    }
    .stMultiSelect [data-baseweb="tag"]:nth-of-type(4) {
        background-color: rgb(108, 197, 233) !important;   /* Light blue */
    }
    .stMultiSelect [data-baseweb="tag"]:nth-of-type(5) {
        background-color: rgb(140, 174, 222) !important;   /* Periwinkle */
    }
    /* Make the text white for better readability */
    .stMultiSelect [data-baseweb="tag"] span {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def load_results(data_path="streamlit_data/*.json"):
    """Load all JSON result files and combine them into a DataFrame"""
    all_results = []
    
    for file_path in glob.glob(data_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            model_name = Path(file_path).stem.split('_')[1]
            benchmark = Path(file_path).stem.split('_')[0]
            k_value = int(Path(file_path).stem.split('_')[2].replace('k', ''))
            
            result = {
                'Model': model_name,
                'Benchmark': benchmark,
                'k': k_value,
                'Pass@k': data.get('pass@k', 0)
            }
            all_results.append(result)

    df = pd.DataFrame(all_results)
    df['Pass@k'] = df['Pass@k'] * 100
    
    return df

def main():
    st.title("🚀 CodeEval Dashboard")
    st.markdown("Compare DeepSeek-Coder and CodeGemma on various benchmarks using pass@k")
    
    # Load data
    df = load_results()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=df['Model'].unique(),
        default=df['Model'].unique()
    )
    
    selected_benchmarks = st.sidebar.multiselect(
        "Select Benchmarks",
        options=df['Benchmark'].unique(),
        default=df['Benchmark'].unique()
    )
    
    selected_k = st.sidebar.multiselect(
        "Select k values",
        options=sorted(df['k'].unique()),
        default=sorted(df['k'].unique())
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['Model'].isin(selected_models)) &
        (df['Benchmark'].isin(selected_benchmarks)) &
        (df['k'].isin(selected_k))
    ]
    
    # Create visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pass@k Comparison 📊")
        fig = px.bar(
            filtered_df,
            x='Benchmark',
            y='Pass@k',
            color='Model',
            barmode='group',
            facet_col='k',
            title='Model Performance Across Benchmarks',
            labels={'Pass@k': 'Pass@k Score', 'Benchmark': 'Benchmark Name'},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Pass@k Summary Table")
        pivot_table = filtered_df.pivot_table(
            values='Pass@k',
            index=['Benchmark', 'k'],
            columns='Model',
            aggfunc='first'
        ).round(3)
        st.dataframe(pivot_table, use_container_width=True)
    
    # Radar Chart for Model Comparison
    st.subheader("Radar Charts 📡")
    for k in selected_k:
        k_filtered = filtered_df[filtered_df['k'] == k]
        fig = go.Figure()
        
        for model in selected_models:
            model_data = k_filtered[k_filtered['Model'] == model]
            fig.add_trace(go.Scatterpolar(
                r=model_data['Pass@k'],
                theta=model_data['Benchmark'],
                name=f"{model} (k={k})",
                marker=dict(
                    size=20,
                    symbol='circle'
                ),
                line=dict(width=3),
                hovertemplate='%{r:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0, 1],
                    showline=True,
                    gridcolor='rgba(255,255,255,0.3)',
                    color='white',
                    linewidth=0.5
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.3)',
                    color='white',
                    linewidth=2,
                    tickfont=dict(size=16)
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=16)
            ),
            title=f"Model Comparison (k={k})",
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 