import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class CustomerSegmentation:
    """
    Handles customer segmentation using K-Means clustering
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.scaled_features = None
    
    def find_optimal_clusters(self, features, max_k=10):
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            features (pd.DataFrame): Customer features
            max_k (int): Maximum number of clusters to test
            
        Returns:
            int: Optimal number of clusters
        """
        # Scale features
        self.scaled_features = self.scaler.fit_transform(features)
        
        # Calculate inertia for different k values
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_features)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            score = silhouette_score(self.scaled_features, kmeans.labels_)
            silhouette_scores.append(score)
        
        # Find optimal k using silhouette score (highest score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Store results for plotting
        self.inertias = inertias
        self.silhouette_scores = silhouette_scores
        self.k_range = k_range
        
        return optimal_k
    
    def perform_clustering(self, features, n_clusters):
        """
        Perform K-Means clustering on customer features
        
        Args:
            features (pd.DataFrame): Customer features
            n_clusters (int): Number of clusters
            
        Returns:
            pd.DataFrame: Original features with cluster labels
        """
        # Scale features if not already done
        if self.scaled_features is None:
            self.scaled_features = self.scaler.fit_transform(features)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(self.scaled_features)
        
        # Add cluster labels to original features
        result = features.copy()
        result['cluster'] = cluster_labels
        
        return result
    
    def analyze_segments(self, clustered_data):
        """
        Analyze characteristics of each customer segment
        
        Args:
            clustered_data (pd.DataFrame): Data with cluster labels
            
        Returns:
            pd.DataFrame: Segment analysis
        """
        # Calculate segment statistics
        segment_analysis = clustered_data.groupby('cluster').agg({
            'recency': ['mean', 'median'],
            'frequency': ['mean', 'median'],
            'monetary_value': ['mean', 'median'],
            'avg_order_value': ['mean', 'median']
        }).round(2)
        
        # Flatten column names
        segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns]
        
        # Add cluster size
        segment_analysis['customer_count'] = clustered_data.groupby('cluster').size()
        segment_analysis['percentage'] = (segment_analysis['customer_count'] / len(clustered_data) * 100).round(2)
        
        return segment_analysis
    
    def print_segment_summary(self, segment_analysis):
        """
        Print a summary of customer segments
        
        Args:
            segment_analysis (pd.DataFrame): Segment analysis results
        """
        print("\n   Customer Segment Analysis:")
        print("   " + "="*50)
        
        for cluster in segment_analysis.index:
            print(f"\n   Cluster {cluster}:")
            print(f"   - Size: {segment_analysis.loc[cluster, 'customer_count']} customers ({segment_analysis.loc[cluster, 'percentage']}%)")
            print(f"   - Avg Recency: {segment_analysis.loc[cluster, 'recency_mean']:.1f} days")
            print(f"   - Avg Frequency: {segment_analysis.loc[cluster, 'frequency_mean']:.1f} orders")
            print(f"   - Avg Monetary Value: ${segment_analysis.loc[cluster, 'monetary_value_mean']:.2f}")
            print(f"   - Avg Order Value: ${segment_analysis.loc[cluster, 'avg_order_value_mean']:.2f}")
    
    def generate_business_insights(self, segment_analysis):
        """
        Generate business insights and recommendations for each segment
        
        Args:
            segment_analysis (pd.DataFrame): Segment analysis results
        """
        insights = []
        
        for cluster in segment_analysis.index:
            recency = segment_analysis.loc[cluster, 'recency_mean']
            frequency = segment_analysis.loc[cluster, 'frequency_mean']
            monetary = segment_analysis.loc[cluster, 'monetary_value_mean']
            size = segment_analysis.loc[cluster, 'customer_count']
            
            # Classify segment based on RFM values
            if recency < 50 and frequency > 10 and monetary > 1000:
                segment_type = "Champions"
                recommendation = "Reward them. They are your best customers."
            elif recency < 50 and frequency > 5:
                segment_type = "Loyal Customers"
                recommendation = "Upsell higher value products and ask for reviews."
            elif recency < 100 and monetary > 500:
                segment_type = "Potential Loyalists"
                recommendation = "Offer membership programs and recommend products."
            elif recency < 100 and frequency < 5:
                segment_type = "New Customers"
                recommendation = "Provide onboarding support and special offers."
            elif recency > 100 and frequency > 5:
                segment_type = "At Risk"
                recommendation = "Send personalized emails and offer discounts."
            elif recency > 200:
                segment_type = "Lost Customers"
                recommendation = "Win-back campaigns and surveys to understand issues."
            else:
                segment_type = "Hibernating"
                recommendation = "Offer other product categories and special discounts."
            
            insights.append({
                'cluster': cluster,
                'segment_type': segment_type,
                'recommendation': recommendation,
                'size': size
            })
        
        # Print insights
        for insight in insights:
            print(f"\n   Cluster {insight['cluster']} - {insight['segment_type']} ({insight['size']} customers):")
            print(f"   Strategy: {insight['recommendation']}")
    
    def get_cluster_centers(self):
        """
        Get the cluster centers in original feature space
        
        Returns:
            pd.DataFrame: Cluster centers
        """
        if self.kmeans is None:
            raise ValueError("Clustering not performed yet")
        
        # Transform centers back to original scale
        centers_scaled = self.kmeans.cluster_centers_
        centers_original = self.scaler.inverse_transform(centers_scaled)
        
        feature_names = ['recency', 'frequency', 'monetary_value', 'avg_order_value']
        centers_df = pd.DataFrame(centers_original, columns=feature_names)
        centers_df.index.name = 'cluster'
        
        return centers_df