import pandas as pd
import numpy as np
from data_processor import DataProcessor
from customer_segmentation import CustomerSegmentation
from visualizer import Visualizer
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the customer segmentation analysis
    """
    print("=== E-Commerce Customer Segmentation Analysis ===\n")
    
    # Initialize components
    data_processor = DataProcessor()
    segmentation = CustomerSegmentation()
    visualizer = Visualizer()
    
    try:
        # Step 1: Load and clean data
        print("1. Loading and cleaning data...")
        df = data_processor.load_data('online_retail_II.csv')  # Updated filename
        df_cleaned = data_processor.clean_data(df)
        print(f"   Original data shape: {df.shape}")
        print(f"   Cleaned data shape: {df_cleaned.shape}")
        
        # Step 2: Feature engineering
        print("\n2. Engineering customer features...")
        customer_features = data_processor.create_customer_features(df_cleaned)
        print(f"   Customer features shape: {customer_features.shape}")
        print(f"   Features: {list(customer_features.columns)}")
        
        # Step 3: Perform clustering
        print("\n3. Performing K-Means clustering...")
        optimal_k = segmentation.find_optimal_clusters(customer_features)
        print(f"   Optimal number of clusters: {optimal_k}")
        
        clustered_data = segmentation.perform_clustering(customer_features, optimal_k)
        
        # Step 4: Analyze segments
        print("\n4. Analyzing customer segments...")
        segment_analysis = segmentation.analyze_segments(clustered_data)
        segmentation.print_segment_summary(segment_analysis)
        
        # Step 5: Visualize results
        print("\n5. Generating visualizations...")
        visualizer.plot_cluster_analysis(customer_features, clustered_data, optimal_k)
        visualizer.plot_segment_characteristics(segment_analysis)
        
        # Step 6: Generate insights
        print("\n6. Business Insights and Recommendations:")
        segmentation.generate_business_insights(segment_analysis)
        
        print("\n=== Analysis Complete ===")
        print("Check the generated plots for visual insights!")
        
    except FileNotFoundError:
        print("Error: Dataset file not found. Please ensure 'online_retail_II.csv' is in the project directory.")
        print("Download the dataset from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()