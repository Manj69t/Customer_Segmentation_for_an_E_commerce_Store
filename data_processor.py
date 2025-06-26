import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    """
    Handles data loading, cleaning, and feature engineering for customer segmentation
    """
    
    def __init__(self):
        self.df = None
    
    def load_data(self, file_path):
        """
        Load the e-commerce dataset
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            # Try different encodings commonly used for this dataset
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"   Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not load file with any encoding")
                
            self.df = df
            return df
            
        except Exception as e:
            raise FileNotFoundError(f"Error loading data: {str(e)}")
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values, duplicates, and invalid entries
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        df_clean = df.copy()
        
        # Handle different possible column names for Online Retail II dataset
        column_mapping = {
            'Invoice': 'invoice_no',
            'InvoiceNo': 'invoice_no',
            'StockCode': 'stock_code', 
            'Description': 'description',
            'Quantity': 'quantity',
            'InvoiceDate': 'invoice_date',
            'Price': 'unit_price',
            'UnitPrice': 'unit_price',
            'Customer ID': 'customer_id',
            'CustomerID': 'customer_id',
            'Country': 'country'
        }
        
        # Rename columns to standardized names
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Convert column names to lowercase for consistency
        df_clean.columns = df_clean.columns.str.lower()
        
        print(f"   Columns in dataset: {list(df_clean.columns)}")
        
        # Remove rows with missing customer IDs
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['customer_id'])
        print(f"   Removed {initial_rows - len(df_clean)} rows with missing customer IDs")
        
        # Remove cancelled orders (negative quantities or invoice numbers starting with 'C')
        if 'invoice_no' in df_clean.columns:
            df_clean = df_clean[~df_clean['invoice_no'].astype(str).str.startswith('C')]
        
        # Remove negative quantities and prices
        if 'quantity' in df_clean.columns:
            df_clean = df_clean[df_clean['quantity'] > 0]
        if 'unit_price' in df_clean.columns:
            df_clean = df_clean[df_clean['unit_price'] > 0]
        
        # Convert invoice_date to datetime
        if 'invoice_date' in df_clean.columns:
            df_clean['invoice_date'] = pd.to_datetime(df_clean['invoice_date'])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        return df_clean
    
    def create_customer_features(self, df):
        """
        Create RFM (Recency, Frequency, Monetary) features for customer segmentation
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            
        Returns:
            pd.DataFrame: Customer features for clustering
        """
        # Calculate total amount for each transaction
        df['total_amount'] = df['quantity'] * df['unit_price']
        
        # Get the latest date in the dataset
        latest_date = df['invoice_date'].max()
        
        # Calculate RFM metrics for each customer
        customer_rfm = df.groupby('customer_id').agg({
            'invoice_date': ['max', 'count'],  # For Recency and Frequency
            'total_amount': ['sum', 'mean'],   # For Monetary
            'quantity': 'sum'                  # Additional metric
        }).round(2)
        
        # Flatten column names
        customer_rfm.columns = ['last_purchase_date', 'frequency', 'monetary_value', 'avg_order_value', 'total_quantity']
        
        # Calculate recency (days since last purchase)
        customer_rfm['recency'] = (latest_date - customer_rfm['last_purchase_date']).dt.days
        
        # Create additional features
        customer_rfm['avg_quantity_per_order'] = customer_rfm['total_quantity'] / customer_rfm['frequency']
        
        # Select features for clustering
        features_for_clustering = customer_rfm[['recency', 'frequency', 'monetary_value', 'avg_order_value']].copy()
        
        # Handle any remaining NaN values
        features_for_clustering = features_for_clustering.fillna(0)
        
        return features_for_clustering
    
    def get_data_summary(self, df):
        """
        Get summary statistics of the dataset
        
        Args:
            df (pd.DataFrame): Dataset to summarize
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_transactions': len(df),
            'unique_customers': df['customer_id'].nunique(),
            'unique_products': df['stock_code'].nunique() if 'stock_code' in df.columns else 0,
            'date_range': f"{df['invoice_date'].min()} to {df['invoice_date'].max()}",
            'total_revenue': df['total_amount'].sum() if 'total_amount' in df.columns else 0
        }
        
        return summary