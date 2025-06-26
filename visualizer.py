import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    """
    Handles all visualization needs for customer segmentation analysis
    """
    
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_cluster_analysis(self, features, clustered_data, optimal_k):
        """
        Create comprehensive cluster analysis plots
        
        Args:
            features (pd.DataFrame): Original features
            clustered_data (pd.DataFrame): Data with cluster labels
            optimal_k (int): Optimal number of clusters
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Recency vs Frequency scatter plot
        scatter = axes[0, 0].scatter(clustered_data['recency'], clustered_data['frequency'], 
                                   c=clustered_data['cluster'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('Recency (days)')
        axes[0, 0].set_ylabel('Frequency (orders)')
        axes[0, 0].set_title('Recency vs Frequency')
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        
        # 2. Frequency vs Monetary scatter plot
        scatter2 = axes[0, 1].scatter(clustered_data['frequency'], clustered_data['monetary_value'], 
                                    c=clustered_data['cluster'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Frequency (orders)')
        axes[0, 1].set_ylabel('Monetary Value ($)')
        axes[0, 1].set_title('Frequency vs Monetary Value')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
        
        # 3. Cluster distribution
        cluster_counts = clustered_data['cluster'].value_counts().sort_index()
        axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color=sns.color_palette("husl", len(cluster_counts)))
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].set_title('Customer Distribution by Cluster')
        
        # Add value labels on bars
        for i, v in enumerate(cluster_counts.values):
            axes[1, 0].text(i, v + max(cluster_counts) * 0.01, str(v), ha='center', va='bottom')
        
        # 4. Average metrics by cluster
        avg_metrics = clustered_data.groupby('cluster')[['recency', 'frequency', 'monetary_value']].mean()
        avg_metrics_normalized = avg_metrics / avg_metrics.max()  # Normalize for better visualization
        
        x = np.arange(len(avg_metrics))
        width = 0.25
        
        axes[1, 1].bar(x - width, avg_metrics_normalized['recency'], width, label='Recency (norm)', alpha=0.8)
        axes[1, 1].bar(x, avg_metrics_normalized['frequency'], width, label='Frequency (norm)', alpha=0.8)
        axes[1, 1].bar(x + width, avg_metrics_normalized['monetary_value'], width, label='Monetary (norm)', alpha=0.8)
        
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Normalized Average Values')
        axes[1, 1].set_title('Normalized RFM Metrics by Cluster')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'Cluster {i}' for i in avg_metrics.index])
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_segment_characteristics(self, segment_analysis):
        """
        Create detailed segment characteristics visualization
        
        Args:
            segment_analysis (pd.DataFrame): Segment analysis results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Customer Segment Characteristics', fontsize=16, fontweight='bold')
        
        clusters = segment_analysis.index
        colors = sns.color_palette("husl", len(clusters))
        
        # 1. Average Recency by Cluster
        recency_means = segment_analysis['recency_mean']
        axes[0, 0].bar(clusters, recency_means, color=colors, alpha=0.8)
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Average Recency (days)')
        axes[0, 0].set_title('Average Recency by Cluster')
        
        # Add value labels
        for i, v in enumerate(recency_means):
            axes[0, 0].text(i, v + max(recency_means) * 0.01, f'{v:.1f}', ha='center', va='bottom')
        
        # 2. Average Frequency by Cluster
        frequency_means = segment_analysis['frequency_mean']
        axes[0, 1].bar(clusters, frequency_means, color=colors, alpha=0.8)
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Average Frequency (orders)')
        axes[0, 1].set_title('Average Frequency by Cluster')
        
        for i, v in enumerate(frequency_means):
            axes[0, 1].text(i, v + max(frequency_means) * 0.01, f'{v:.1f}', ha='center', va='bottom')
        
        # 3. Average Monetary Value by Cluster
        monetary_means = segment_analysis['monetary_value_mean']
        axes[1, 0].bar(clusters, monetary_means, color=colors, alpha=0.8)
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Average Monetary Value ($)')
        axes[1, 0].set_title('Average Monetary Value by Cluster')
        
        for i, v in enumerate(monetary_means):
            axes[1, 0].text(i, v + max(monetary_means) * 0.01, f'${v:.0f}', ha='center', va='bottom')
        
        # 4. Cluster Size Distribution (Pie Chart)
        customer_counts = segment_analysis['customer_count']
        axes[1, 1].pie(customer_counts, labels=[f'Cluster {i}' for i in clusters], 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 1].set_title('Customer Distribution by Cluster')
        
        plt.tight_layout()
        plt.savefig('segment_characteristics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rfm_heatmap(self, segment_analysis):
        """
        Create RFM heatmap for better segment comparison
        
        Args:
            segment_analysis (pd.DataFrame): Segment analysis results
        """
        plt.figure(figsize=(10, 6))
        
        # Select RFM columns for heatmap
        rfm_cols = ['recency_mean', 'frequency_mean', 'monetary_value_mean']
        heatmap_data = segment_analysis[rfm_cols].copy()
        
        # Rename columns for better display
        heatmap_data.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary Value']
        
        # Create heatmap
        sns.heatmap(heatmap_data.T, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Value'}, linewidths=0.5)
        
        plt.title('RFM Analysis Heatmap by Cluster', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster')
        plt.ylabel('RFM Metrics')
        
        plt.tight_layout()
        plt.savefig('rfm_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_elbow_curve(self, k_range, inertias, silhouette_scores):
        """
        Plot elbow curve and silhouette scores for cluster optimization
        
        Args:
            k_range (range): Range of k values tested
            inertias (list): Inertia values for each k
            silhouette_scores (list): Silhouette scores for each k
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Elbow curve
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score for Different k')
        ax2.grid(True, alpha=0.3)
        
        # Mark optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                   label=f'Optimal k={optimal_k}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()