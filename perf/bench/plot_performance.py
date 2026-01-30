"""
Performance visualization module.

Generates interactive 3D plots and heatmaps from benchmark results.
Can be used standalone or imported by benchmark scripts.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Install with: pip install plotly")


def plot_3d_scatter(df, output_path="performance_3d.html"):
    """
    Create interactive 3D scatter plot of performance landscape.
    
    Args:
        df: DataFrame with columns: K, N, threads, throughput
        output_path: Path to save HTML file
    """
    if not HAS_PLOTLY:
        print("Plotly not available, skipping 3D plot")
        return
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df['K'],
        y=df['N'],
        z=df['threads'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['throughput'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Throughput<br>(obs/s)"),
            line=dict(width=0.5, color='white')
        ),
        text=[f"K={k}, N={n}, threads={t}<br>{tp:.1f} obs/s" 
              for k, n, t, tp in zip(df['K'], df['N'], df['threads'], df['throughput'])],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title={
            'text': 'Performance Landscape: Throughput Across Configuration Space',
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title='Model Count (K)',
            yaxis_title='Particle Count (N)',
            zaxis_title='Thread Count',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=800
    )
    
    fig.write_html(output_path)
    print(f"3D plot saved to: {output_path}")


if __name__ == "__main__":
    main()
