def plot_numeric_histograms_log_paginated(df, per_page=9, bins=30):
    """
    Create a list of matplotlib Figure objects, each with up to `per_page` log-scale numeric histograms.
    Each page is a grid (3 columns, variable rows).
    Args:
        df (pd.DataFrame): DataFrame with numeric columns.
        per_page (int): Number of histograms per page.
        bins (int): Number of bins for each histogram.
    Returns:
        List[matplotlib.figure.Figure]: List of figures for PDF pagination.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rcParams
    HIST_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    font_family = "DejaVu Sans"
    font_size = 12
    rcParams.update({"font.family": font_family, "font.size": font_size})
    numeric_cols = df.select_dtypes(include='number').columns
    figs = []
    for i in range(0, len(numeric_cols), per_page):
        cols = numeric_cols[i:i+per_page]
        n = len(cols)
        nrows = int(np.ceil(n / 3))
        fig, axes = plt.subplots(nrows, 3, figsize=(15, 4 * nrows))
        axes = axes.flatten()
        for idx, (ax, col) in enumerate(zip(axes, cols)):
            data = df[col].dropna()
            data = data[data > 0]
            if len(data) > 0:
                color = HIST_COLORS[1]  # orange, matches log10 in side-by-side
                ax.hist(np.log10(data), bins=bins, color=color, edgecolor='k', alpha=0.85)
                ax.set_title(f"log10({col})", fontsize=font_size+1, fontfamily=font_family)
                ax.set_xlabel(f"log10({col})", fontsize=font_size-1, fontfamily=font_family)
                ax.set_ylabel("Count", fontsize=font_size-1, fontfamily=font_family)
            else:
                ax.set_visible(False)
        for ax in axes[n:]:
            fig.delaxes(ax)
        fig.tight_layout()
        figs.append(fig)
    return figs
# === Multi-page PDF utility for numeric histograms ===
def plot_numeric_histograms_paginated(df, per_page=9, bins=30):
    """
    Create a list of matplotlib Figure objects, each with up to `per_page` numeric histograms.
    Each page is a grid (3 columns, variable rows).
    Args:
        df (pd.DataFrame): DataFrame with numeric columns.
        per_page (int): Number of histograms per page.
        bins (int): Number of bins for each histogram.
    Returns:
        List[matplotlib.figure.Figure]: List of figures for PDF pagination.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rcParams
    HIST_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    font_family = "DejaVu Sans"
    font_size = 12
    rcParams.update({"font.family": font_family, "font.size": font_size})
    numeric_cols = df.select_dtypes(include='number').columns
    figs = []
    for i in range(0, len(numeric_cols), per_page):
        cols = numeric_cols[i:i+per_page]
        n = len(cols)
        nrows = int(np.ceil(n / 3))
        fig, axes = plt.subplots(nrows, 3, figsize=(15, 4 * nrows))
        axes = axes.flatten()
        for idx, (ax, col) in enumerate(zip(axes, cols)):
            data = df[col].dropna()
            color = HIST_COLORS[0]  # blue, matches original in side-by-side
            ax.hist(data, bins=bins, color=color, edgecolor='k', alpha=0.85)
            ax.set_title(col, fontsize=font_size+1, fontfamily=font_family)
            ax.set_xlabel(col, fontsize=font_size-1, fontfamily=font_family)
            ax.set_ylabel("Count", fontsize=font_size-1, fontfamily=font_family)
        for ax in axes[n:]:
            fig.delaxes(ax)
        fig.tight_layout()
        figs.append(fig)
    return figs

def plot_tc_histograms(df, bins=40):
    """
    Plot side-by-side histograms of thermal conductivity (original and log10 scale) using Matplotlib.
    Args:
        df (pd.DataFrame): DataFrame with 'thermal_conductivity' column.
        bins (int): Number of bins for the histograms.
    Returns:
        matplotlib.figure.Figure: The Matplotlib figure containing the histograms.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    k = df['thermal_conductivity'].dropna()
    log_k = np.log10(k[k > 0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original scale histogram
    axes[0].hist(k, bins=bins, color="#1f77b4", edgecolor='k', alpha=0.85)
    axes[0].set_title("Thermal Conductivity (Original Scale)", fontsize=14, pad=15)
    axes[0].set_xlabel("Thermal Conductivity [W/mK]", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Log scale histogram
    axes[1].hist(log_k, bins=bins, color="#ff7f0e", edgecolor='k', alpha=0.85)
    axes[1].set_title("Thermal Conductivity (Log₁₀ Scale)", fontsize=14, pad=15)
    axes[1].set_xlabel("log₁₀(Thermal Conductivity) [W/mK]", fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout(pad=2.0)
    return fig

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import textwrap

# Lazy import for heavy dependencies
try:
    from features import classify_material
except ImportError:
    classify_material = None

MODEL_COLORS = {
    "XGBoost": "#1f77b4",
    "Random Forest": "#3C9D3C",
    "Gradient Boosting": "#9467bd",
    "SVR": "#FFA94D"
}

# --- Visualization Style Constants ---
PLOTLY_TEMPLATE = "plotly_white"
FONT_FAMILY = "Segoe UI, Arial, sans-serif"
FONT_SIZE = 14
TITLE_SIZE = 20
AXIS_TITLE_SIZE = 16
TICK_SIZE = 13
LEGEND_SIZE = 13
COLOR_SEQ = px.colors.qualitative.Safe
HIST_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# --- Consistent Layout Function for Plotly ---
def _apply_common_layout(fig, title):
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font=dict(family=FONT_FAMILY, size=TITLE_SIZE, color="#222"),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color="#222"),
        margin=dict(t=60, l=60, r=40, b=50),
        autosize=True,
        legend=dict(
            font=dict(size=LEGEND_SIZE),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#DDD",
            borderwidth=1
        ),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff"
    )
    fig.update_xaxes(
        title_font=dict(size=AXIS_TITLE_SIZE, family=FONT_FAMILY, color="#222"),
        tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY, color="#222"),
        showgrid=True, gridwidth=0.5, gridcolor="#e5e5e5"
    )
    fig.update_yaxes(
        title_font=dict(size=AXIS_TITLE_SIZE, family=FONT_FAMILY, color="#222"),
        tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY, color="#222"),
        showgrid=True, gridwidth=0.5, gridcolor="#e5e5e5"
    )
    return fig


def add_subplot_border(fig, row, col, color="blue", width=3):
    """
    Adds a border around a specific subplot in a Plotly figure.
    This version dynamically calculates the border based on the subplot's domain,
    making it robust to layout changes and autosizing.
    """
    # The subplot index in plotly is 1-based and row-major.
    # An empty string is returned for the first subplot (index 1).
    subplot_count = (row - 1) * fig._get_subplot_grid_size()[1] + col
    subplot_ref = f"{subplot_count}" if subplot_count > 1 else ""

    xaxis_domain = fig.layout[f"xaxis{subplot_ref}"]["domain"]
    yaxis_domain = fig.layout[f"yaxis{subplot_ref}"]["domain"]

    # Add a shape that spans the domain of the subplot
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=xaxis_domain[0],
        y0=yaxis_domain[0],
        x1=xaxis_domain[1],
        y1=yaxis_domain[1],
        line=dict(color=color, width=width),
    )

def plot_pca_variance(pca):
    """Plots the cumulative explained variance of PCA components."""
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )
    return _apply_common_layout(fig, "PCA Explained Variance")

def plot_elbow_method(X_scaled):
    """Plot the elbow curve for K-Means clustering."""
    from sklearn.cluster import KMeans
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    fig = px.line(x=K, y=inertia)
    fig.update_layout(xaxis_title='Number of Clusters (k)', yaxis_title='Inertia')
    return _apply_common_layout(fig, 'Elbow Method for Optimal k')

def plot_clusters(X_scaled, cluster_labels, formula_list, n_clusters):
    """Create 2D and 3D PCA plots of the clusters."""
    
    def create_scatter_trace(df, dimension, hover_text):
        traces = []
        for i in range(n_clusters):
            cluster_df = df[df['Cluster'] == i]
            if dimension == '2d':
                trace = go.Scatter(
                    x=cluster_df['PCA1'], y=cluster_df['PCA2'],
                    mode='markers', name=f'Cluster {i}',
                    text=hover_text[df['Cluster'] == i],
                    hoverinfo='text+x+y'
                )
            else: # 3d
                trace = go.Scatter3d(
                    x=cluster_df['PCA1'], y=cluster_df['PCA2'], z=cluster_df['PCA3'],
                    mode='markers', name=f'Cluster {i}',
                    text=hover_text[df['Cluster'] == i],
                    hoverinfo='text'
                )
            traces.append(trace)
        return traces

    # 2D Plot
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    df_pca_2d = pd.DataFrame(data=X_pca_2d, columns=['PCA1', 'PCA2'])
    df_pca_2d['Cluster'] = cluster_labels
    fig_2d = go.Figure(data=create_scatter_trace(df_pca_2d, '2d', formula_list))
    fig_2d.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', legend_title='Cluster')
    _apply_common_layout(fig_2d, '2D PCA of Clusters')

    # 3D Plot
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    df_pca_3d = pd.DataFrame(data=X_pca_3d, columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca_3d['Cluster'] = cluster_labels
    fig_3d = go.Figure(data=create_scatter_trace(df_pca_3d, '3d', formula_list))
    fig_3d.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'), legend_title='Cluster')
    _apply_common_layout(fig_3d, '3D PCA of Clusters')
    
    return fig_2d, fig_3d

def _create_normalized_bar_plot(df, x_col, y_col, color_col, title, labels, category_orders=None):
    """Helper to create a normalized stacked bar plot."""
    df_plot = df.dropna(subset=[x_col, color_col]).copy()
    df_plot[color_col] = df_plot[color_col].astype(str)
    
    df_counts = df_plot.groupby([x_col, color_col]).size().reset_index(name='count')
    df_total = df_counts.groupby(x_col)['count'].transform('sum')
    df_counts['percent'] = df_counts['count'] / df_total * 100
    
    fig = px.bar(
        df_counts,
        x=x_col,
        y='percent',
        color=color_col,
        title=title,
        labels=labels,
        barmode='stack',
        category_orders=category_orders,
        text_auto=True
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
    fig.update_layout(yaxis_tickformat='.0f')
    return _apply_common_layout(fig, title)

def plot_cluster_crystal_structure(df):
    """Normalized crystal structure distribution within each cluster."""
    return _create_normalized_bar_plot(
        df, 'cluster_label', 'percent', 'crystal_structure',
        'Crystal Structure Composition by Cluster (Normalized)',
        {'percent': 'Percentage of Materials', 'cluster_label': 'Cluster'},
        {"crystal_structure": ["Cubic", "Tetragonal", "Orthorhombic", "Hexagonal", "Trigonal", "Monoclinic", "Triclinic", "Unknown"]}
    )

def plot_cluster_chemistry_distribution(df):
    """Normalized chemistry distribution within each cluster."""
    return _create_normalized_bar_plot(
        df, 'cluster_label', 'percent', 'chemistry',
        "Chemical Class Distribution per Cluster (Normalized)",
        {'cluster_label': 'Cluster', 'percent': 'Percentage of Materials'}
    )

def plot_structure_by_chemistry(df):
    """Distribution of crystal structures within each chemistry class."""
    return _create_normalized_bar_plot(
        df, 'chemistry', 'percent', 'crystal_structure',
        'Crystal Structure Composition per Chemistry Class (Normalized)',
        {'percent': 'Percentage of Materials', 'chemistry': 'Chemistry'},
        {'crystal_structure': ["Cubic", "Tetragonal", "Orthorhombic", "Hexagonal", "Trigonal", "Monoclinic", "Triclinic", "Unknown"]}
    )

def plot_pca_material_class(X_scaled, analysis_df):
    """3D PCA scatter colored by material class and cluster."""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'], index=analysis_df.index)
    
    plot_df = plot_df.join(analysis_df)

    color_col = 'chemistry' if 'chemistry' in plot_df.columns else 'material_class'
    symbol_col = 'cluster_label' if 'cluster_label' in plot_df.columns else 'cluster'
    hover_cols = ['crystal_structure'] if 'crystal_structure' in plot_df.columns else None

    fig = px.scatter_3d(
        plot_df.dropna(subset=[color_col]),
        x='PC1', y='PC2', z='PC3',
        color=color_col,
        symbol=symbol_col,
        hover_data=hover_cols,
    )
    return _apply_common_layout(fig, '3D PCA of Clusters Colored by Material Class')

def _plot_histograms_grid(df, log_scale=False, bins=30):
    """Helper to plot a grid of histograms."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    ncols = min(4, n_cols)
    nrows = (n_cols + ncols - 1) // ncols
    
    wrapped_titles = ['<span style="font-size:11px">' + '<br>'.join(textwrap.wrap(title, width=20)) + '</span>' for title in numeric_cols]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=wrapped_titles)
    
    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, ncols)
        data = df[col].dropna()
        if log_scale:
            data = data[data > 0].apply(np.log10)
        
        fig.add_trace(
            go.Histogram(x=data, nbinsx=bins, name=col),
            row=row + 1, col=col_idx + 1
        )
    
    title = "Feature Distributions (Log Scale)" if log_scale else "Feature Distributions"
    fig.update_layout(showlegend=False, height=300*nrows, autosize=True)
    return _apply_common_layout(fig, title)

def plot_numeric_histograms(df, bins=30):
    """Linear-scale histograms of numeric columns."""
    return _plot_histograms_grid(df, log_scale=False, bins=bins)

def plot_numeric_histograms_log(df, bins=30):
    """Log-scale histograms of numeric columns."""
    return _plot_histograms_grid(df, log_scale=True, bins=bins)

def plot_parity_logscale(model, X_test, y_test_log, model_name):
    y_pred = model.predict(X_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test_log, y=y_pred,
        mode='markers',
        marker=dict(color=MODEL_COLORS.get(model_name, 'gray'), size=8, opacity=0.7),
        name="Predicted vs. Actual"
    ))
    min_val = min(y_test_log.min(), y_pred.min())
    max_val = max(y_test_log.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(color='#111', dash='dash', width=1),
        name='Ideal'
    ))
    fig.update_layout(
        xaxis_title="Actual log(k)",
        yaxis_title="Predicted log(k)",
        xaxis_type="log", yaxis_type="log",
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
        autosize=True  # Ensure the plot scales with page size
    )
    return _apply_common_layout(fig, f"{model_name} Parity Plot (Log Scale)")

def plot_model_comparison(results_dict, best_model_name=None):
    data = []
    for model, res in results_dict.items():
        # Ensure we are accessing the correct nested dictionary for metrics
        if "log" in res and all(k in res["log"] for k in ["mae", "r2", "rmse"]):
            data.append({
                "Model": model,
                "Scale": "log",
                "MAE": res["log"]["mae"],
                "R²": res["log"]["r2"],
                "RMSE": res["log"]["rmse"]
            })

    if not data:
        return _apply_common_layout(go.Figure(), "No data to display")

    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars=['Model', 'Scale'], value_vars=['R²', 'MAE', 'RMSE'], var_name='Metric', value_name='Score')
    
    # Define colors, highlighting the best model
    color_map = {model: '#2A9D8F' if model == best_model_name else 'lightgrey' for model in df['Model'].unique()}

    fig = px.bar(
        df_melted,
        x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_map=color_map,
        text_auto=True
    )
    
    fig.update_traces(texttemplate='%{y:.3f}', textangle=0, textposition='outside')
    fig.update_layout(autosize=True)  # Make plot responsive
    return _apply_common_layout(fig, "Model Comparison (Log-Transformed Scale)")

def plot_parity_grid(models, X_test, y_test_log, best_model_name=None, title="Model Parity Plots"):
    # Determine grid size
    n_models = len(models)
    if n_models == 0:
        return _apply_common_layout(go.Figure(), "No models to display")
    
    cols = 2
    rows = (n_models + 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[name for name in models.keys()],
        horizontal_spacing=0.1, vertical_spacing=0.15
    )

    for i, (name, model) in enumerate(models.items()):
        row, col = divmod(i, cols)
        row += 1
        col += 1

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test_log, y_pred)
        
        # Add scatter plot for predictions
        fig.add_trace(go.Scatter(
            x=y_test_log,
            y=y_pred,
            mode='markers',
            marker=dict(color=MODEL_COLORS.get(name.split('_')[0], 'steelblue'), size=5, opacity=0.65),
            showlegend=False
        ), row=row, col=col)

        # Add ideal line
        min_val = min(y_test_log.min(), y_pred.min())
        max_val = max(y_test_log.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#111', dash='dash', width=1),
            showlegend=False
        ), row=row, col=col)

        # Add R² annotation
        fig.add_annotation(
            text=f"<b>R² = {r2:.3f}</b>",
            x=0.05, y=0.95,
            xref='x domain', yref='y domain',
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=12, color="#111"),
            row=row, col=col
        )
        
        # Highlight the best model
        if name == best_model_name:
            fig.update_xaxes(showline=True, linewidth=2, linecolor='#2A9D8F', row=row, col=col)
            fig.update_yaxes(showline=True, linewidth=2, linecolor='#2A9D8F', row=row, col=col)

    fig.update_layout(
        autosize=True,
        title_text=title,
        title_x=0.5
    )
    
    # Update axis labels
    for i in range(1, n_models + 1):
        row, col = divmod(i-1, cols)
        row += 1
        col += 1
        fig.update_xaxes(title_text="Actual log(k)" if row == rows else "", row=row, col=col)
        fig.update_yaxes(title_text="Predicted log(k)" if col == 1 else "", row=row, col=col)

    return _apply_common_layout(fig, title)

def plot_feature_importance(model, X_train, feature_names, model_name, y_train=None):
    try:
        importances = model.feature_importances_
    except AttributeError:
        if y_train is None:
            print("y_train must be provided for permutation importance.")
            return
        result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
        importances = np.array(result['importances_mean'])

    sorted_idx = np.argsort(importances)[-10:][::-1]
    top_features = [feature_names[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]
    fig = go.Figure(go.Bar(
        x=top_importances[::-1],
        y=top_features[::-1],
        orientation='h',
        marker_color=MODEL_COLORS.get(model_name, 'gray')
    ))
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature"
    )
    return _apply_common_layout(fig, f"{model_name} Feature Importances")

def plot_residuals(model, X_test, y_test_log, model_name):
    y_pred = model.predict(X_test)
    residuals = y_test_log - y_pred
    fig = px.histogram(residuals, nbins=30)
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_traces(marker_color=MODEL_COLORS.get(model_name, 'gray'))
    fig.update_layout(
        xaxis_title="Residual",
        yaxis_title="Frequency"
    )
    return _apply_common_layout(fig, f"{model_name} Residuals")

def plot_density_by_cluster(df):
    """Plots the density distribution for each cluster."""
    fig = px.box(
        df.dropna(subset=['mp_density']),
        x='cluster_label', 
        y='mp_density', 
        color='cluster_label',
        labels={'cluster_label': 'Cluster', 'mp_density': 'Density (g/cm³)'},
    )
    return _apply_common_layout(fig, 'Density Distribution by Cluster')

def plot_silhouette_scores(scores_list, additional_metrics=None):
    """Plots silhouette scores for a range of k values and optionally additional metrics."""
    k_range = range(2, len(scores_list) + 2)
    fig = px.line(
        x=k_range,
        y=scores_list,
        markers=True,
        labels={'x': 'Number of Clusters (k)', 'y': 'Average Silhouette Score'}
    )

    # Find the optimal k and add a vertical line
    if scores_list:
        optimal_k = k_range[np.argmax(scores_list)]
        best_score = max(scores_list)
        fig.add_vline(
            x=optimal_k, 
            line_width=2, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Optimal k = {optimal_k}<br>Score = {best_score:.3f}",
            annotation_position="top left"
        )

    # Optionally plot additional metrics
    if additional_metrics:
        for metric_name, metric_values in additional_metrics.items():
            fig.add_trace(
                go.Scatter(
                    x=k_range,
                    y=metric_values,
                    mode='lines+markers',
                    name=metric_name
                )
            )

    return _apply_common_layout(fig, 'Silhouette Scores and Additional Metrics for Optimal k')

def plot_clustering_metrics(metrics_dict, k_range):
    """Plots multiple clustering metrics for a range of k values."""
    fig = go.Figure()

    for metric_name, metric_values in metrics_dict.items():
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=metric_values,
                mode='lines+markers',
                name=metric_name
            )
        )

    fig.update_layout(
        title="Clustering Metrics for Optimal k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Metric Value",
        template=PLOTLY_TEMPLATE
    )

    return _apply_common_layout(fig, "Clustering Metrics for Optimal k")

def plot_normalized_clustering_metrics(metrics_dict, k_range):
    """Normalize and plot clustering metrics for better visualization."""
    normalized_metrics = {
        metric_name: (np.array(values) / max(values) if max(values) > 0 else np.array(values))
        for metric_name, values in metrics_dict.items()
    }

    fig = go.Figure()
    for metric_name, metric_values in normalized_metrics.items():
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=metric_values,
                mode='lines+markers',
                name=metric_name
            )
        )

    fig.update_layout(
        title="Normalized Clustering Metrics for Optimal k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Normalized Metric Value",
        template=PLOTLY_TEMPLATE
    )

    return _apply_common_layout(fig, "Normalized Clustering Metrics for Optimal k")

def plot_normalized_clustering_metrics_with_composite(metrics_dict, composite_scores, k_range, optimal_k):
    """Plot normalized clustering metrics along with composite scores."""
    fig = go.Figure()

    # Add normalized metrics
    normalized_metrics = {
        metric_name: (np.array(values) / max(values) if max(values) > 0 else np.array(values))
        for metric_name, values in metrics_dict.items()
    }
    for metric_name, metric_values in normalized_metrics.items():
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=metric_values,
                mode="lines+markers",
                name=metric_name
            )
        )

    # Add composite score
    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=composite_scores,
            mode="lines+markers",
            name="Composite Score",
            line=dict(color="blue", dash="solid")
        )
    )

    # Highlight optimal k
    fig.add_shape(
        type="line",
        x0=optimal_k,
        x1=optimal_k,
        y0=0,
        y1=1,
        line=dict(color="red", dash="dash"),
        xref="x",
        yref="paper"
    )
    fig.add_annotation(
        x=optimal_k,
        y=1,
        text=f"Optimal k = {optimal_k}",
        showarrow=False,
        yref="paper",
        xref="x",
        font=dict(color="red")
    )

    # Apply common layout
    return _apply_common_layout(fig, "Normalized Clustering Metrics with Composite Score")

def plot_model_results(model, X_test_scaled, y_test_log, y_test, plots_dir, prefix, X_train_scaled=None, X_selected_columns=None):
    """
    Generate and save all standard plots (parity, residuals, SHAP, feature importance) for a model.
    - model: fitted model
    - X_test_scaled: test features (scaled)
    - y_test_log: log-transformed test target
    - y_test: original scale test target
    - plots_dir: directory to save plots
    - prefix: filename prefix for saved plots
    - X_train_scaled: (optional) train features (for feature importance)
    - X_selected_columns: (optional) feature names for importance plot
    """
    import os
    import matplotlib.pyplot as plt
    import shap
    from src.utils import save_plot
    os.makedirs(plots_dir, exist_ok=True)
    # Parity plot
    fig_parity = plot_parity_logscale(model, X_test_scaled, y_test_log, prefix)
    save_plot(fig_parity, os.path.join(plots_dir, f"{prefix}_parity.pdf"))
    # Residuals plot
    if 'plot_residuals' in globals():
        fig_resid = plot_residuals(model, X_test_scaled, y_test_log, prefix)
        save_plot(fig_resid, os.path.join(plots_dir, f"{prefix}_residuals.pdf"))
    # SHAP summary (tree models only)
    if hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        shap.summary_plot(shap_values, X_test_scaled, show=False)
        save_plot(plt.gcf(), os.path.join(plots_dir, f"{prefix}_shap_summary.pdf"))
        plt.close()
    # Feature importance (tree models)
    if hasattr(model, 'feature_importances_') and X_train_scaled is not None and X_selected_columns is not None:
        if 'plot_feature_importance' in globals():
            fig_imp = plot_feature_importance(model, X_train_scaled, X_selected_columns, prefix)
            save_plot(fig_imp, os.path.join(plots_dir, f"{prefix}_feature_importance.pdf"))
    # Elbow method plot (inertia)
    if hasattr(model, 'inertia_'):
        inertia_values = model.inertia_
        k_range = range(1, len(inertia_values) + 1)
        optimal_k = np.argmin(np.diff(inertia_values, 2)) + 2  # Second derivative test for elbow
        fig_elbow = plot_elbow_inertia(inertia_values, k_range, optimal_k)
        save_plot(fig_elbow, os.path.join(plots_dir, f"{prefix}_elbow_method.pdf"))

def plot_elbow_inertia(inertia_values, k_range, optimal_k):
    """Plot the elbow method for inertia values."""
    fig = go.Figure()

    # Add inertia values
    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=inertia_values,
            mode="lines+markers",
            name="Inertia",
            line=dict(color="blue", dash="solid")
        )
    )

    # Highlight optimal k
    fig.add_shape(
        type="line",
        x0=optimal_k,
        x1=optimal_k,
        y0=min(inertia_values),
        y1=max(inertia_values),
        line=dict(color="red", dash="dash"),
        xref="x",
        yref="y"
    )
    fig.add_annotation(
        x=optimal_k,
        y=max(inertia_values),
        text=f"Optimal k = {optimal_k}",
        showarrow=False,
        yref="y",
        xref="x",
        font=dict(color="red")
    )

    # Apply common layout
    return _apply_common_layout(fig, "Elbow Method for Optimal k")

def plot_elbow_inertia_with_marker(inertia_values, k_range, optimal_k):
    """
    Plot the elbow method for inertia values and mark the optimal k.

    Parameters:
    - inertia_values: List of inertia values for each k.
    - k_range: Range of k values corresponding to the inertia values.
    - optimal_k: The optimal k value determined using the elbow method.

    Returns:
    - fig: Plotly figure object.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_range, y=inertia_values, mode='lines+markers', name='Inertia'))
    fig.add_trace(go.Scatter(x=[optimal_k], y=[inertia_values[k_range.index(optimal_k)]],
                             mode='markers', marker=dict(size=10, color='red'), name=f'Optimal k = {optimal_k}'))
    fig.update_layout(title='Elbow Method for Optimal k',
                      xaxis_title='Number of Clusters (k)',
                      yaxis_title='Inertia',
                      template='plotly_white')
    return fig


