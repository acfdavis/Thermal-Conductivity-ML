import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

MODEL_COLORS = {
    "XGBoost": "#1f77b4",             # steel blue
    "Random Forest": "#3C9D3C",       # forest green
    "Gradient Boosting": "#9467bd",  # purple
    "SVR": "#FFA94D"                  # soft orange
}

def plot_pca_variance(pca):
    """Plots the cumulative explained variance of PCA components."""
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )
    fig.update_layout(title_text="PCA Explained Variance")
    return fig

def plot_elbow_method(X_scaled):
    """Plot the elbow curve for K-Means clustering."""
    from sklearn.cluster import KMeans
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    fig = px.line(x=K, y=inertia, title='Elbow Method for Optimal k')
    fig.update_layout(xaxis_title='Number of Clusters (k)', yaxis_title='Inertia')
    return fig
def plot_clusters(X_scaled, cluster_labels, formula_list, n_clusters):
    """Create 2D and 3D PCA plots of the clusters."""
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    df_pca_2d = pd.DataFrame(data=X_pca_2d, columns=['PCA1', 'PCA2'])
    df_pca_2d['Cluster'] = cluster_labels

    fig_2d = go.Figure()
    for i in range(n_clusters):
        fig_2d.add_trace(go.Scatter(
            x=df_pca_2d[df_pca_2d['Cluster'] == i]['PCA1'],
            y=df_pca_2d[df_pca_2d['Cluster'] == i]['PCA2'],
            mode='markers',
            name=f'Cluster {i}',
            text=formula_list, # Add formula for hover info
            hoverinfo='text'
        ))
    fig_2d.update_layout(title='2D PCA of Clusters', xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', legend_title='Cluster')
    
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    df_pca_3d = pd.DataFrame(data=X_pca_3d, columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca_3d['Cluster'] = cluster_labels

    fig_3d = go.Figure()
    for i in range(n_clusters):
        fig_3d.add_trace(go.Scatter3d(
            x=df_pca_3d[df_pca_3d['Cluster'] == i]['PCA1'],
            y=df_pca_3d[df_pca_3d['Cluster'] == i]['PCA2'],
            z=df_pca_3d[df_pca_3d['Cluster'] == i]['PCA3'],
            mode='markers',
            name=f'Cluster {i}',
            text=formula_list, # Add formula for hover info
            hoverinfo='text'
        ))
    fig_3d.update_layout(title='3D PCA of Clusters', scene=dict(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', zaxis_title='Principal Component 3'), legend_title='Cluster')
    return fig_2d, fig_3d

def plot_material_class_distribution(df):
    """Plot the distribution of material classes within each cluster."""
    df['material_class'] = df['formula'].apply(classify_material)
    cluster_material_distribution = df.groupby(['cluster_labels', 'material_class']).size().unstack(fill_value=0)

    fig = go.Figure()
    for material_class in cluster_material_distribution.columns:
        fig.add_trace(go.Bar(
            x=cluster_material_distribution.index,
            y=cluster_material_distribution[material_class],
            name=material_class,
            hovertemplate='Cluster: %{x}<br>Class: '+material_class+'<br>Count: %{y}<extra></extra>'
        ))
    fig.update_layout(title='Distribution of Material Classes within Each Cluster', xaxis_title='Cluster', yaxis_title='Number of Materials', barmode='stack', legend_title='Material Class')
    return fig





def add_subplot_border(fig, row, col, rows=2, cols=2, color="#2A9D8F", width=2):
    # Calculate the domain for the subplot
    x0 = (col - 1) / cols
    x1 = col / cols - 0.025
    y0 = 1 - row / rows + 0.025
    y1 = 1 - (row - 1) / rows 
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=x0, x1=x1, y0=y0, y1=y1,
        line=dict(color=color, width=width),
        layer="above"
    )

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
        title=f"{model_name} Parity Plot (Log Scale)",
        xaxis_title="Actual log(k)",
        yaxis_title="Predicted log(k)",
        xaxis_type="log", yaxis_type="log",
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),

        template="plotly_white"
    )
    fig.show()

def plot_model_comparison(results_dict):
    # Extract MAE for each model and scale
    data = []
    for model, res in results_dict.items():
        for scale in ["log", "original"]:
            if scale in res and "mae" in res[scale]:
                data.append({
                    "Model": model,
                    "Scale": scale,
                    "MAE": res[scale]["mae"]
                })
    df = pd.DataFrame(data)
    fig = px.bar(
        df, x="Model", y="MAE", color="Scale", barmode="group",
        template="plotly_white", title="Model Comparison (MAE across Scales)"
    )
    fig.show()




def plot_parity_grid(models, X_test, y_test_log):
    fig = make_subplots(
        rows=2, cols=2,
        horizontal_spacing=0.05, vertical_spacing=0.05
    )

    for i, (name, model) in enumerate(models.items()):
        row, col = divmod(i, 2)
        row += 1
        col += 1

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test_log, y_pred)
        mae = mean_absolute_error(y_test_log, y_pred)

        fig.add_trace(go.Scatter(
            x=y_test_log,
            y=y_pred,
            mode='markers',
            marker=dict(color=MODEL_COLORS.get(name, 'steelblue'), size=5, opacity=0.65),
            showlegend=False
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=[0, y_test_log.max()],
            y=[0, y_test_log.max()],
            mode='lines',
            line=dict(color='#111', dash='dash', width=1),
            showlegend=False
        ), row=row, col=col)

        # Use paper coordinates for safe and consistent annotation placement
        paper_x = 0.01 if col == 1 else 0.54
        paper_y = 0.99 - 0.54 * (row - 1)

        fig.add_annotation(
            text=(
                f"<span style='font-size:16px; font-weight:bold'>{name} Parity</span><br>"
                f"<span style='font-size:12px'>R² = {r2:.2f} &nbsp; MAE = {mae:.2f}</span>"
            ),
            x=paper_x, y=paper_y,
            xref='paper', yref='paper',
            xanchor="left", yanchor="top",
            showarrow=False,
            align="left",
            font=dict(family="Arial", color="#111", size=11)
        )        
        # Determine correct xref/yref for each subplot
        subplot_idx = (row - 1) * 2 + col  # 1-based subplot index
        xref = "x domain" if subplot_idx == 1 else f"x{subplot_idx} domain"
        yref = "y domain" if subplot_idx == 1 else f"y{subplot_idx} domain"
        if row == 1 and col == 1:
            fig.add_annotation(
                text="Ideal: y = x",
                x=8.5, y=8.3,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(size=11, color='gray'),
                row=row, col=col
            )

    fig.update_layout(
        height=650,
        width=800,
        title_text="Model Parity Plots with R² and MAE (Log Scale)",
        title_x=0.5,
        template="plotly_white",
        margin=dict(t=60, l=50, r=30, b=40),
        font=dict(family="Arial", size=12)
    )

    for i in range(1, 5):
        row = ((i - 1) // 2) + 1
        col = ((i - 1) % 2) + 1
        fig.update_xaxes(title_text="Actual log(k)" if row == 2 else None, range=[0, 9], row=row, col=col)
        fig.update_yaxes(title_text="Predicted log(k)" if col == 1 else None, range=[0, 9], row=row, col=col)

    fig.show()
    return fig  



    
def plot_feature_importance(model, X_train, feature_names, model_name, y_train=None):
    try:
        importances = model.feature_importances_
    except AttributeError:
        try:
            if y_train is None:
                raise ValueError("y_train must be provided for permutation importance.")
            result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
            importances = result.importances_mean
        except Exception as e:
            print(f"Model '{model_name}' does not support feature importances or permutation failed: {e}")
            return

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
        title=f"{model_name} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature"
    )
    fig.show()


def plot_residuals(model, X_test, y_test_log, model_name):
    y_pred = model.predict(X_test)
    residuals = y_test_log - y_pred
    fig = px.histogram(residuals, nbins=30, title=f"{model_name} Residuals", template="plotly_white")
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_traces(marker_color=MODEL_COLORS.get(model_name, 'gray'))
    fig.update_layout(
        xaxis_title="Residual",
        yaxis_title="Frequency"
    )
    fig.show()


