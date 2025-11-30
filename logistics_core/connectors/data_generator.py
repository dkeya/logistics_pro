import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class LogisticsProDataGenerator:
    """Enhanced data generator for Logistics Pro enterprise platform"""
    
    def __init__(self):
        self.skus = self.generate_skus_data()
        self.customers = self.generate_customers_data()
        # Enhanced data storage
        self.sales_data = None
        self.inventory_data = None  
        self.logistics_data = None
    
    def generate_skus_data(self, n_skus: int = 200) -> pd.DataFrame:
        """Generate SKU master data"""
        categories = ['Beverages', 'Dairy', 'Snacks', 'Household', 'Grains']
        subcategory_map = {
            'Beverages': ['Soft Drinks', 'Juices', 'Water', 'Energy Drinks'],
            'Dairy': ['Milk', 'Yogurt', 'Cheese', 'Butter'],
            'Snacks': ['Chips', 'Cookies', 'Nuts', 'Chocolate'],
            'Household': ['Cleaning', 'Laundry', 'Personal Care'],
            'Grains': ['Rice', 'Flour', 'Pasta', 'Cereals']
        }
        
        skus = []
        for i in range(n_skus):
            category = np.random.choice(categories)
            subcategories = subcategory_map.get(category, ['General'])
            
            sku = {
                'sku_id': f"SKU{str(i+1).zfill(3)}",
                'sku_name': f"Product {i+1}",
                'category': category,
                'subcategory': np.random.choice(subcategories),
                'unit_cost': np.random.uniform(50, 500),
                'selling_price': np.random.uniform(80, 800),
                'weight_kg': np.random.uniform(0.1, 5.0),
                'volume_l': np.random.uniform(0.1, 2.0),
                'shelf_life_days': np.random.randint(7, 365),
                'brand': np.random.choice(['Premium', 'Standard', 'Budget']),
                'supplier_id': f"SUP{str(i).zfill(3)}"
            }
            skus.append(sku)
        
        return pd.DataFrame(skus)
    
    def generate_customers_data(self, n_customers: int = 150) -> pd.DataFrame:
        """Generate customer master data"""
        regions = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Western', 'Coastal', 'Central']
        region_subregions = {
            'Nairobi': ['CBD', 'Westlands', 'Karen', 'Embakasi', 'Kasarani'],
            'Mombasa': ['Island', 'Mainland', 'Nyali', 'Likoni'],
            'Kisumu': ['Central', 'Milimani', 'Nyalenda', 'Manyatta'],
            'Nakuru': ['Central', 'Lanet', 'Molo', 'Bahati'],
            'Eldoret': ['Central', 'Langas', 'Huruma', 'Ziwa'],
            'Western': ['Kakamega', 'Bungoma', 'Busia', 'Vihiga'],
            'Coastal': ['Malindi', 'Lamu', 'Kilifi', 'Kwale'],
            'Central': ['Thika', 'Kiambu', 'Muranga', 'Nyeri']
        }
        
        customers = []
        for i in range(n_customers):
            region = np.random.choice(regions)
            subregions = region_subregions.get(region, ['Central'])
            
            customer = {
                'customer_id': f"CUST{str(i+1).zfill(3)}",
                'customer_name': f"Customer {i+1}",
                'type': np.random.choice(['Retail', 'Wholesale', 'Corporate']),
                'region': region,
                'sub_region': np.random.choice(subregions),
                'credit_limit': np.random.uniform(5000, 50000),
                'payment_terms': np.random.choice([7, 14, 30, 45]),
                'tier': np.random.choice(['A', 'B', 'C'], p=[0.2, 0.3, 0.5]),
                'value_score': np.random.randint(1, 100)
            }
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_sales_data(self, n_days: int = 90) -> pd.DataFrame:
        """Generate sales data for specified number of days"""
        sales = []
        start_date = datetime.now().date() - timedelta(days=n_days)
        
        for i in range(n_days * 50):  # 50 transactions per day on average
            current_date = start_date + timedelta(days=np.random.randint(0, n_days))
            sku = self.skus.sample(1).iloc[0]
            customer = self.customers.sample(1).iloc[0]
            
            sale = {
                'sale_id': f"SALE{str(i).zfill(5)}",
                'date': current_date,
                'customer_id': customer['customer_id'],
                'sku_id': sku['sku_id'],
                'quantity': np.random.randint(1, 100),
                'unit_price': sku['selling_price'] * np.random.uniform(0.9, 1.1),
                'unit_cost': sku['unit_cost']
            }
            sales.append(sale)
        
        sales_df = pd.DataFrame(sales)
        # Calculate derived metrics
        sales_df['revenue'] = sales_df['quantity'] * sales_df['unit_price']
        sales_df['cost'] = sales_df['quantity'] * sales_df['unit_cost']
        sales_df['margin'] = sales_df['revenue'] - sales_df['cost']
        sales_df['margin_percent'] = (sales_df['margin'] / sales_df['revenue']).replace([np.inf, -np.inf], 0) * 100
        
        return sales_df
    
    def generate_inventory_data(self) -> pd.DataFrame:
        """Generate inventory data"""
        inventory = []
        for _, sku in self.skus.iterrows():
            stock = {
                'sku_id': sku['sku_id'],
                'current_stock': np.random.randint(0, 1000),
                'min_stock_level': np.random.randint(10, 100),
                'max_stock_level': np.random.randint(100, 1000),
                'reorder_point': np.random.randint(20, 200),
                'lead_time_days': np.random.randint(1, 14),
                'storage_location': np.random.choice(['A1', 'B2', 'C3', 'D4'])
            }
            inventory.append(stock)
        
        return pd.DataFrame(inventory)
    
    def generate_logistics_data(self, n_days: int = 30) -> pd.DataFrame:
        """Generate logistics data"""
        logistics = []
        start_date = datetime.now().date() - timedelta(days=n_days)
        
        for i in range(n_days * 20):  # 20 deliveries per day
            current_date = start_date + timedelta(days=np.random.randint(0, n_days))
            
            delivery = {
                'delivery_id': f"DEL{str(i).zfill(5)}",
                'date': current_date,
                'vehicle_id': f"VH{str(np.random.randint(1, 20)).zfill(2)}",
                'driver_id': f"DRV{str(np.random.randint(1, 15)).zfill(3)}",
                'route_id': f"RT{str(np.random.randint(1, 10)).zfill(3)}",
                'planned_distance_km': np.random.uniform(5, 200),
                'actual_distance_km': np.random.uniform(4, 220),
                'planned_duration_hrs': np.random.uniform(0.5, 8),
                'actual_duration_hrs': np.random.uniform(0.4, 9),
                'fuel_cost': np.random.uniform(500, 5000),
                'maintenance_cost': np.random.uniform(200, 2000),
                'weather_conditions': np.random.choice(['Clear', 'Rainy', 'Foggy', 'Stormy']),
                'otif': np.random.choice([0, 1], p=[0.1, 0.9])  # 90% on-time
            }
            logistics.append(delivery)
        
        return pd.DataFrame(logistics)
    
    def get_enhanced_sales_data(self):
        """Return enhanced sales data with all enterprise columns"""
        if self.sales_data is None:
            self.sales_data = self.generate_sales_data()
        return self.sales_data
    
    def get_enhanced_inventory_data(self):
        """Return enhanced inventory data"""
        if self.inventory_data is None:
            self.inventory_data = self.generate_inventory_data()
        return self.inventory_data
    
    def get_enhanced_logistics_data(self):
        """Return enhanced logistics data"""  
        if self.logistics_data is None:
            self.logistics_data = self.generate_logistics_data()
        return self.logistics_data
