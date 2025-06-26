# E-Commerce Customer Segmentation

A machine learning project that segments e-commerce customers using K-Means clustering based on RFM (Recency, Frequency, Monetary) analysis.

## Project Overview

This project analyzes customer behavior patterns in e-commerce data to identify distinct customer segments. The segmentation helps businesses understand their customer base and develop targeted marketing strategies.

## Features

- **Data Processing**: Comprehensive data cleaning and feature engineering
- **RFM Analysis**: Recency, Frequency, and Monetary value calculation
- **K-Means Clustering**: Automated optimal cluster detection
- **Visualization**: Multiple charts and plots for insights
- **Business Insights**: Actionable recommendations for each segment

## Dataset

**Required Dataset**: Online Retail II UCI Dataset from Kaggle
- Download from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- Contains ~1M transactions from UK-based online retail (2009-2011)
- File should be named `online_retail_II.csv` in the project directory

## Project Structure

```
customer-segmentation/
├── app.py                    # Main application entry point
├── data_processor.py         # Data loading and preprocessing
├── customer_segmentation.py  # K-Means clustering implementation
├── visualizer.py            # Visualization and plotting
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── online_retail_II.csv      # Dataset (download from Kaggle)
```

## Installation

1. Clone or download the project files
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Download the dataset from Kaggle and place it as `online_retail.csv`

## Usage

Run the main application:
```bash
python app.py
```

The program will:
1. Load and clean the dataset
2. Create customer features using RFM analysis
3. Find optimal number of clusters
4. Perform K-Means clustering
5. Analyze customer segments
6. Generate visualizations and business insights

## Key Components

### DataProcessor Class
- Handles data loading with multiple encoding support
- Cleans data by removing invalid entries and duplicates
- Creates RFM features for clustering
- Provides data summary statistics

### CustomerSegmentation Class
- Implements K-Means clustering with feature scaling
- Finds optimal clusters using silhouette score
- Analyzes segment characteristics
- Generates business insights and recommendations

### Visualizer Class
- Creates comprehensive cluster analysis plots
- Visualizes segment characteristics
- Generates RFM heatmaps
- Plots optimization curves

## RFM Analysis

The project uses RFM methodology to create customer features:

- **Recency**: Days since last purchase
- **Frequency**: Total number of orders
- **Monetary**: Total amount spent
- **Average Order Value**: Additional metric for better segmentation

## Customer Segments

The algorithm typically identifies segments like:

- **Champions**: Best customers with high value and frequency
- **Loyal Customers**: Regular buyers with good value
- **Potential Loyalists**: Recent customers with growth potential
- **New Customers**: Recent first-time buyers
- **At Risk**: Previously good customers showing decline
- **Lost Customers**: Haven't purchased in a long time

## Output Files

The program generates several visualization files:
- `customer_segmentation_analysis.png`: Comprehensive cluster analysis
- `segment_characteristics.png`: Detailed segment metrics
- `rfm_heatmap.png`: RFM comparison heatmap
- `cluster_optimization.png`: Elbow curve and silhouette scores

## Interview Points

**Technical Concepts to Explain:**

1. **RFM Analysis**: Why these metrics are important for customer segmentation
2. **K-Means Algorithm**: How it works and why it's suitable for this problem
3. **Feature Scaling**: Why standardization is necessary before clustering
4. **Cluster Optimization**: Elbow method vs silhouette score
5. **Business Value**: How segments translate to marketing strategies

**Potential Questions:**

- Why did you choose K-Means over other clustering algorithms?
- How do you handle missing data in customer datasets?
- What are the limitations of RFM analysis?
- How would you validate your clustering results?
- What business actions would you recommend for each segment?

## Extensions

Possible improvements for advanced implementation:
- Hierarchical clustering comparison
- Customer lifetime value prediction
- Churn prediction modeling
- Real-time segmentation updates
- A/B testing framework for marketing campaigns

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms
- matplotlib: Basic plotting
- seaborn: Statistical visualizations

## Notes

- The project handles common data quality issues (missing values, cancellations, etc.)
- Automatic encoding detection for CSV files
- Modular design for easy maintenance and extension
- Production-ready error handling
- Comprehensive documentation for interview discussions