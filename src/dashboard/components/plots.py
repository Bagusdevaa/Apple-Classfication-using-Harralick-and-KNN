"""
Plotting utility functions for the dashboard
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Define label colors and names globally
LABEL_COLORS = {0: 'yellow', 1: 'orange', 2: 'green', 3: 'blue', 4: 'red'}
LABEL_NAMES = {0: '20% Ripeness', 1: '40% Ripeness', 2: '60% Ripeness', 3: '80% Ripeness', 4: '100% Ripeness'}
LABELS_SHORT = ['20%', '40%', '60%', '80%', '100%']

def plot_accuracy_bar(summary_df):
    """Plot accuracy by d and theta as a bar chart"""
    fig = px.bar(
        summary_df, 
        x='d', 
        y='accuracy', 
        color='theta', 
        title='Accuracy by d and theta', 
        text='accuracy'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title='d', yaxis_title='Accuracy', template='plotly_white')
    return fig

def plot_accuracy_line(results, d_theta):
    """Plot accuracy vs k for a given d-theta combination"""
    accuracies = {k: result['accuracy'] for k, result in results.items()}
    line_fig = px.line(
        x=list(accuracies.keys()), 
        y=list(accuracies.values()), 
        markers=True, 
        title=f"Accuracy vs. k for d={d_theta[0]}, theta={d_theta[1]}"
    )
    line_fig.update_layout(
        xaxis_title='k (Number of Neighbors)', 
        yaxis_title='Accuracy', 
        template='plotly_white'
    )
    return line_fig

def plot_confusion_matrix(conf_matrix):
    """Plot confusion matrix as a heatmap"""
    conf_fig = px.imshow(
        conf_matrix,
        x=LABELS_SHORT,
        y=LABELS_SHORT,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        title="Confusion Matrix",
        text_auto=True
    )
    
    # Update layout for better readability
    conf_fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#00172B',
        paper_bgcolor='#00172B',
        height=400,
        xaxis=dict(
            title_font=dict(size=12),
            tickfont=dict(size=10),
            tickvals=[0, 1, 2, 3, 4],
            ticktext=LABELS_SHORT
        ),
        yaxis=dict(
            title_font=dict(size=12),
            tickfont=dict(size=10),
            tickvals=[0, 1, 2, 3, 4],
            ticktext=LABELS_SHORT
        ),
        coloraxis_showscale=True,
        margin=dict(l=50, r=50, t=80, b=50),
        title_font=dict(size=14)
    )
    return conf_fig

def plot_class_metrics(class_report):
    """Plot precision, recall, f1-score by class as a bar chart"""
    # Create a DataFrame with only the actual classes (not the averages)
    class_metrics = []
    
    for class_label, metrics in class_report.items():
        # Skip 'accuracy', 'macro avg', 'weighted avg' entries
        if class_label.isdigit():
            class_metrics.append({
                'Class': f"Class {class_label}",
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score']
            })
    
    metrics_df = pd.DataFrame(class_metrics)
    
    # If DataFrame is empty, create a placeholder message
    if metrics_df.empty:
        fig = px.bar(
            x=["No Data"],
            y=[0],
            title="No Class Metrics Available"
        )
        return fig
    
    # Create melted DataFrame for grouped bar chart
    metrics_melted = pd.melt(
        metrics_df, 
        id_vars=['Class'],
        value_vars=['Precision', 'Recall', 'F1-Score'],
        var_name='Metric',
        value_name='Score'
    )
    
    # Create the bar chart
    bar_fig = px.bar(
        metrics_melted, 
        x='Class', 
        y='Score', 
        color='Metric',
        barmode='group', 
        title="Precision, Recall, F1-Score by Class"
    )
    
    bar_fig.update_layout(
        template='plotly_white',
        height=400,
        xaxis_title='Class',
        yaxis_title='Score',
        legend_title='Metric',
        margin=dict(l=50, r=50, t=30, b=50),
        yaxis_range=[0, 1]
    )
    
    return bar_fig

def plot_neighbors_visualization(X_train, X_test, y_train, y_test, test_sample_idx, selected_k):
    """Create PCA visualization of nearest neighbors for a test sample"""
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    # Get neighbors for the selected test sample
    knn = KNeighborsClassifier(n_neighbors=selected_k)
    knn.fit(X_train, y_train)
    distances, indices = knn.kneighbors(X_test)
    neighbors_idx = indices[test_sample_idx]

    # Create a DataFrame for plotly visualization
    plot_data = []
    
    # Add training data
    y_train_array = np.array(y_train)
    for i, (x, y, label) in enumerate(zip(X_train_2d[:, 0], X_train_2d[:, 1], y_train_array)):
        is_neighbor = i in neighbors_idx
        neighbor_text = f"Neighbor ({LABEL_NAMES.get(label, 'Unknown')})" if is_neighbor else None
        
        if is_neighbor:
            marker_size = 15
            border_width = 2
            legend_group = f"Neighbor ({LABEL_NAMES.get(label, 'Unknown')})"
        else:
            marker_size = 8
            border_width = 0
            legend_group = f"Training Data ({LABEL_NAMES.get(label, 'Unknown')})"
            
        plot_data.append({
            'PCA Component 1': x, 
            'PCA Component 2': y,
            'Label': str(label),
            'Category': 'Training Data',
            'Ripeness': LABEL_NAMES.get(label, 'Unknown'),
            'Size': marker_size,
            'BorderWidth': border_width,
            'LegendGroup': legend_group,
            'IsNeighbor': is_neighbor,
            'NeighborText': neighbor_text
        })
    
    # Add test sample
    plot_data.append({
        'PCA Component 1': X_test_2d[test_sample_idx, 0],
        'PCA Component 2': X_test_2d[test_sample_idx, 1],
        'Label': str(y_test.iloc[test_sample_idx]),
        'Category': 'Test Sample',
        'Ripeness': LABEL_NAMES.get(y_test.iloc[test_sample_idx], 'Unknown'),
        'Size': 20,
        'BorderWidth': 2,
        'LegendGroup': 'Test Sample',
        'IsNeighbor': False,
        'NeighborText': None
    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create base scatter plot
    fig = px.scatter(
        plot_df, 
        x='PCA Component 1', 
        y='PCA Component 2',
        color='Ripeness',
        symbol='Category',
        size='Size',
        title=f"Visualization of {selected_k} Nearest Neighbors for Test Sample {test_sample_idx}",
        color_discrete_map={
            '20% Ripeness': 'yellow',
            '40% Ripeness': 'orange',
            '60% Ripeness': 'green',
            '80% Ripeness': 'blue',
            '100% Ripeness': 'red',
        },
        symbol_map={
            'Training Data': 'circle',
            'Test Sample': 'diamond',
        },
        hover_data={
            'Label': True,
            'Category': True,
            'Ripeness': True,
            'IsNeighbor': False,
            'Size': False,
            'BorderWidth': False,
            'LegendGroup': False,
            'NeighborText': True
        }
    )
    
    # Customize the figure
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#00172B',
        paper_bgcolor='#00172B',
        legend_title_text='Legend',
        height=600,
        title={
            'text': f"Visualization of {selected_k} Nearest Neighbors for Test Sample {test_sample_idx}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    # Customize markers for each point individually
    for i, row in plot_df.iterrows():
        fig.add_scatter(
            x=[row['PCA Component 1']],
            y=[row['PCA Component 2']],
            mode='markers',
            marker=dict(
                color=LABEL_COLORS.get(int(row['Label']), 'gray'),
                size=row['Size'],
                line=dict(width=row['BorderWidth'], color='black'),
                symbol='diamond' if row['Category'] == 'Test Sample' else 'circle'
            ),
            legendgroup=row['LegendGroup'],
            name=row['LegendGroup'],
            showlegend=False,  # Don't show in legend to avoid duplicates
            hoverinfo='text',
            hovertext=f"{row['Category']}<br>Label: {row['Label']}<br>Ripeness: {row['Ripeness']}<br>{row['NeighborText'] if row['NeighborText'] else ''}"
        )
    
    # Count occurrences of each label in neighbors
    neighbor_label_counts = {}
    for idx in neighbors_idx:
        label = y_train_array[idx]
        if label in neighbor_label_counts:
            neighbor_label_counts[label] += 1
        else:
            neighbor_label_counts[label] = 1
            
    # Add only the labels with neighbors to legend
    for label, count in sorted(neighbor_label_counts.items(), key=lambda x: x[1], reverse=True):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    color=LABEL_COLORS.get(label, 'gray'),
                    size=15,
                    line=dict(width=2, color='black'),
                    symbol='circle'
                ),
                name=f"Neighbor ({LABEL_NAMES.get(label, 'Unknown')}) - {count}/{selected_k}",
                legendgroup=f"Neighbor ({LABEL_NAMES.get(label, 'Unknown')})",
                showlegend=True
            )
        )
    
    # Add test sample to legend
    test_label = y_test.iloc[test_sample_idx]
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                color=LABEL_COLORS.get(test_label, 'gray'),
                size=20,
                line=dict(width=2, color='black'),
                symbol='diamond'
            ),
            name=f"Test Sample ({LABEL_NAMES.get(test_label, 'Unknown')})",
            legendgroup='Test Sample',
            showlegend=True
        )
    )
    
    return fig

def plot_cv_bar(cv_summary_df):
    """Plot cross validation accuracy by d and theta as a bar chart"""
    cv_bar_fig = px.bar(
        cv_summary_df, 
        x=['d', 'theta'],
        y='mean_cv_accuracy',
        error_y='std_cv_accuracy',
        color='d',
        barmode='group',
        title="Cross Validation Accuracy by (d, theta) Combination",
        labels={'mean_cv_accuracy': 'Mean CV Accuracy', 'std_cv_accuracy': 'Standard Deviation'}
    )
    cv_bar_fig.update_layout(template='plotly_white')
    return cv_bar_fig

def plot_fold_accuracies(fold_accuracies, mean_accuracy):
    """Plot accuracies by fold as a bar chart"""
    fold_df = pd.DataFrame({
        'Fold': [f"Fold {i+1}" for i in range(len(fold_accuracies))],
        'Accuracy': fold_accuracies
    })
    
    fold_fig = px.bar(
        fold_df,
        x='Fold',
        y='Accuracy',
        title="Accuracy by Fold",
        color='Accuracy',
        color_continuous_scale='Viridis'
    )
    fold_fig.add_hline(
        y=mean_accuracy, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean: {mean_accuracy:.4f}"
    )
    fold_fig.update_layout(template='plotly_white', yaxis_range=[0, 1])
    return fold_fig

def plot_class_metrics_by_fold(class_metrics_df, selected_metric):
    """Plot class metrics by fold as a bar chart"""
    class_metric_fig = px.bar(
        class_metrics_df,
        x="Class Name",
        y=selected_metric,
        color="Fold",
        barmode="group",
        title=f"{selected_metric} by Class and Fold",
        labels={selected_metric: selected_metric, "Class Name": "Ripeness Level"},
        color_continuous_scale="Viridis"
    )
    class_metric_fig.update_layout(template='plotly_white', yaxis_range=[0, 1])
    return class_metric_fig

def plot_radar_chart(class_data):
    """Plot radar chart for a class across folds"""
    radar_fig = go.Figure()
    
    for fold_idx, fold_data in class_data.groupby('Fold'):
        radar_fig.add_trace(go.Scatterpolar(
            r=[fold_data['Precision'].values[0], fold_data['Recall'].values[0], fold_data['F1-Score'].values[0]],
            theta=['Precision', 'Recall', 'F1-Score'],
            fill='toself',
            name=f'Fold {fold_idx}'
        ))
    
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title=f"Metric Comparison",
        height=400
    )
    return radar_fig

def plot_parallel_coordinates(fold_metrics_df):
    """Plot parallel coordinates for fold metrics"""
    parallel_fig = px.parallel_coordinates(
        fold_metrics_df,
        color="Accuracy",
        dimensions=['Fold', 'Accuracy', 'Macro Precision', 'Macro Recall', 
                    'Macro F1-Score', 'Weighted F1-Score'],
        labels={
            'Fold': 'Fold Number',
            'Accuracy': 'Accuracy',
            'Macro Precision': 'Macro Precision',
            'Macro Recall': 'Macro Recall',
            'Macro F1-Score': 'Macro F1-Score',
            'Weighted F1-Score': 'Weighted F1-Score'
        },
        color_continuous_scale='Viridis',
        title="Parallel Coordinates Plot of Metrics Across Folds"
    )
    return parallel_fig

def plot_metrics_boxplot(fold_metrics_df):
    """Plot boxplot of metrics across folds"""
    metrics_long_df = pd.melt(
        fold_metrics_df, 
        id_vars=['Fold'],
        value_vars=['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score', 
                    'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score'],
        var_name='Metric',
        value_name='Value'
    )
    
    box_fig = px.box(
        metrics_long_df,
        x="Metric",
        y="Value",
        points="all",
        title="Distribution of Metrics Across Folds",
    )
    box_fig.update_layout(template='plotly_white')
    return box_fig