# logistics_pro/logistics_core/connectors/data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

class LogisticsProDataGenerator:
    """Generate synthetic data for Logistics Pro demonstration"""
    
    def __init__(self):
        # Define regions first since it's used in other methods
        self.regions = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']
        self.skus = self._generate_skus()
        self.customers = self._generate_customers()
        self.suppliers = self._generate_suppliers()
        self.vehicles = self._generate_vehicles()
        
    def _generate_skus(self) -> pd.DataFrame:
        categories = {
            'Beverages': ['Soda 500ml', 'Juice 1L', 'Water 500ml', 'Energy Drink 250ml', 'Tea Bags 50pk'],
            'Sauces & Spreads': ['Tomato Sauce 500g', 'Mayonnaise 400g', 'Peanut Butter 350g', 'Ketchup 1kg'],
            'Spices': ['Salt 1kg', 'Black Pepper 200g', 'Curry Powder 100g', 'Mixed Spices 150g'],
            'Dairy': ['Milk 1L', 'Yoghurt 500g', 'Butter 250g', 'Cheese 200g'],
            'Grains': ['Rice 5kg', 'Maize Flour 2kg', 'Wheat Flour 2kg', 'Pasta 500g']
        }
        
        skus = []
        sku_id = 1000
        for category, products in categories.items():
            for product in products:
                skus.append({
                    'sku_id': f"SKU{sku_id}",
                    'sku_name': product,
                    'category': category,
                    'unit_cost': np.random.uniform(50, 500),
                    'selling_price': np.random.uniform(80, 600),
                    'weight_kg': np.random.uniform(0.5, 5),
                    'volume_l': np.random.uniform(0.5, 3),
                    'shelf_life_days': np.random.choice([30, 60, 90, 180, 365], p=[0.1, 0.2, 0.3, 0.3, 0.1])
                })
                sku_id += 1
        return pd.DataFrame(skus)
    
    def _generate_customers(self) -> pd.DataFrame:
        customer_types = ['Supermarket', 'Restaurant', 'Hotel', 'Hospital', 'School', 'Institution']
        customers = []
        
        for i in range(50):
            customers.append({
                'customer_id': f"CUST{1000 + i}",
                'customer_name': f"Customer {1000 + i}",
                'type': np.random.choice(customer_types),
                'region': np.random.choice(self.regions),
                'credit_limit': np.random.choice([50000, 100000, 200000, 500000]),
                'payment_terms': np.random.choice([7, 14, 30, 45])
            })
        return pd.DataFrame(customers)
    
    def _generate_suppliers(self) -> pd.DataFrame:
        suppliers = []
        for i in range(20):
            suppliers.append({
                'supplier_id': f"SUPP{100 + i}",
                'supplier_name': f"Supplier {100 + i}",
                'lead_time_days': np.random.choice([3, 5, 7, 10, 14]),
                'reliability_score': np.random.uniform(0.7, 0.98)
            })
        return pd.DataFrame(suppliers)
    
    def _generate_vehicles(self) -> pd.DataFrame:
        vehicles = []
        types = ['Small Truck', 'Medium Truck', 'Large Truck', 'Van']
        capacities = [2000, 5000, 10000, 1500]  # kg
        
        for i in range(10):
            vehicles.append({
                'vehicle_id': f"VH{100 + i}",
                'type': np.random.choice(types),
                'capacity_kg': np.random.choice(capacities),
                'fuel_consumption_kmpl': np.random.uniform(4, 8),
                'maintenance_cost_per_km': np.random.uniform(5, 15)
            })
        return pd.DataFrame(vehicles)
    
    
        # Enhanced data storage
        self.sales_data = None
        self.inventory_data = None  
        self.logistics_data = None
        def generate_sales_data(self, days: int = 90) -> pd.DataFrame:
        """Generate sales data for specified number of days"""
        dates = [datetime.now().date() - timedelta(days=x) for x in range(days)]
        sales_data = []
        
        for date in dates:
            daily_transactions = np.random.randint(50, 200)
            for _ in range(daily_transactions):
                sku = self.skus.sample(1).iloc[0]
                customer = self.customers.sample(1).iloc[0]
                
                # Seasonal factors
                day_factor = 1.2 if date.weekday() in [4, 5] else 1.0  # Weekend boost
                month_factor = 1.3 if date.month in [11, 12] else 1.0  # Holiday season
                
                quantity = max(1, int(np.random.poisson(10) * day_factor * month_factor))
                unit_price = sku['selling_price'] * np.random.uniform(0.95, 1.05)
                
                sales_data.append({
                    'date': date,
                    'sku_id': sku['sku_id'],
                    'customer_id': customer['customer_id'],
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'region': customer['region'],
                    'customer_type': customer['type']
                })
        
        return pd.DataFrame(sales_data)
    
    def generate_inventory_data(self) -> pd.DataFrame:
        """Generate current inventory snapshot"""
        inventory = []
        
        for _, sku in self.skus.iterrows():
            current_stock = np.random.poisson(100)
            min_stock = max(10, int(current_stock * 0.3))
            max_stock = int(current_stock * 1.5)
            
            days_until_expiry = np.random.randint(1, sku['shelf_life_days'])
            
            inventory.append({
                'sku_id': sku['sku_id'],
                'current_stock': current_stock,
                'min_stock': min_stock,
                'max_stock': max_stock,
                'days_until_expiry': days_until_expiry,
                'stock_status': 'Critical' if current_stock < min_stock else 
                               'Low' if current_stock < min_stock * 1.2 else 'Adequate',
                'expiry_risk': 'High' if days_until_expiry < 30 else 
                              'Medium' if days_until_expiry < 90 else 'Low'
            })
        
        return pd.DataFrame(inventory)
    
    def generate_logistics_data(self, days: int = 30) -> pd.DataFrame:
        """Generate delivery and route data"""
        deliveries = []
        
        for i in range(days):
            date = datetime.now().date() - timedelta(days=i)
            daily_deliveries = np.random.randint(15, 35)
            
            for j in range(daily_deliveries):
                vehicle = self.vehicles.sample(1).iloc[0]
                customer = self.customers.sample(1).iloc[0]
                
                planned_distance = np.random.uniform(5, 150)
                actual_distance = planned_distance * np.random.uniform(0.9, 1.2)
                planned_time = planned_distance / 40  # hours at 40 km/h
                actual_time = planned_time * np.random.uniform(0.8, 1.5)
                
                on_time = np.random.choice([True, False], p=[0.85, 0.15])
                in_full = np.random.choice([True, False], p=[0.90, 0.10])
                
                deliveries.append({
                    'delivery_id': f"DL{date.strftime('%Y%m%d')}{j}",
                    'date': date,
                    'vehicle_id': vehicle['vehicle_id'],
                    'customer_id': customer['customer_id'],
                    'region': customer['region'],
                    'planned_distance_km': planned_distance,
                    'actual_distance_km': actual_distance,
                    'planned_duration_hrs': planned_time,
                    'actual_duration_hrs': actual_time,
                    'fuel_consumed_l': actual_distance / vehicle['fuel_consumption_kmpl'],
                    'on_time': on_time,
                    'in_full': in_full,
                    'otif': on_time & in_full
                })
        
        return pd.DataFrame(deliveries)
    def get_enhanced_sales_data(self):
        """Return enhanced sales data with all enterprise columns"""
        if self.sales_data is None:
            self.sales_data = self.generate_sales_data()
            # Ensure unit_cost exists for margin calculations
            if 'unit_cost' not in self.sales_data.columns:
                self.sales_data['unit_cost'] = self.sales_data['unit_price'] * np.random.uniform(0.6, 0.85, len(self.sales_data))
            # Calculate derived metrics
            self.sales_data['revenue'] = self.sales_data['quantity'] * self.sales_data['unit_price']
            self.sales_data['cost'] = self.sales_data['quantity'] * self.sales_data['unit_cost']
            self.sales_data['margin'] = self.sales_data['revenue'] - self.sales_data['cost']
            self.sales_data['margin_percent'] = (self.sales_data['margin'] / self.sales_data['revenue']).replace([np.inf, -np.inf], 0) * 100
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
    