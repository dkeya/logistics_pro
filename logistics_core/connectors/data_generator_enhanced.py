# logistics_pro/logistics_core/connectors/data_generator_enhanced.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataGenerator:
    """Enhanced data generator with all required methods for the application"""
    
    def __init__(self):
        self.skus = self.generate_skus()
        self.customers = self.generate_customers()
        self.sales_data = self.generate_sales_data()
        self.inventory_data = self.generate_inventory_data()
        self.logistics_data = self.generate_logistics_data()
    
    def generate_skus(self):
        """Generate comprehensive SKU data"""
        products = [
            'Coca-Cola 500ml', 'Fanta Orange 500ml', 'Sprite 500ml', 'Stoney Tangawizi 500ml',
            'Dasani Water 500ml', 'Keringet Water 1L', 'Milk Tuzo 500ml', 'Brookside Milk 500ml',
            'White Bread', 'Brown Bread', 'Cakes Assorted', 'Blue Band 500g', 'Sunlight Soap',
            'Omo Detergent', 'Royco Cubes', 'Cooking Oil 1L', 'Rice 1kg', 'Wheat Flour 2kg',
            'Premium Water 1L', 'Specialty Coffee 200g', 'Biscuits Assorted', 'Chips Variety',
            'Toilet Paper 4-pack', 'Toothpaste', 'Shampoo', 'Body Lotion', 'Deodorant'
        ]
        
        categories = {
            'Beverages': ['Coca-Cola 500ml', 'Fanta Orange 500ml', 'Sprite 500ml', 'Stoney Tangawizi 500ml', 
                         'Dasani Water 500ml', 'Keringet Water 1L', 'Premium Water 1L'],
            'Dairy': ['Milk Tuzo 500ml', 'Brookside Milk 500ml'],
            'Bakery': ['White Bread', 'Brown Bread', 'Cakes Assorted'],
            'Household': ['Blue Band 500g', 'Sunlight Soap', 'Omo Detergent', 'Toilet Paper 4-pack'],
            'Food': ['Royco Cubes', 'Cooking Oil 1L', 'Rice 1kg', 'Wheat Flour 2kg', 'Biscuits Assorted', 'Chips Variety'],
            'Personal Care': ['Toothpaste', 'Shampoo', 'Body Lotion', 'Deodorant'],
            'Specialty': ['Specialty Coffee 200g']
        }
        
        skus = []
        for i, product in enumerate(products):
            sku_id = f"SKU{i+1:03d}"
            category = next((cat for cat, prods in categories.items() if product in prods), 'General')
            
            # Generate subcategory
            if category == 'Beverages':
                subcategory = 'Soft Drinks' if 'Water' not in product else 'Water'
            elif category == 'Food':
                subcategory = 'Cooking' if 'Oil' in product or 'Rice' in product or 'Flour' in product else 'Snacks'
            else:
                subcategory = category
            
            skus.append({
                'sku_id': sku_id,
                'sku_name': product,
                'category': category,
                'subcategory': subcategory,
                'unit_cost': np.random.uniform(30, 400),
                'selling_price': np.random.uniform(50, 600),
                'weight_kg': np.random.uniform(0.1, 2.5),
                'volume_l': np.random.uniform(0.1, 1.5),
                'shelf_life_days': np.random.randint(30, 365)
            })
        
        return pd.DataFrame(skus)
    
    def generate_customers(self):
        """Generate comprehensive customer data"""
        customer_names = [
            'Nakumatt Supermarket', 'Tuskys Hyper', 'Uchumi Supermarket', 'QuickMart Express',
            'Chandarana Foodplus', 'Naivas Supermarket', 'Eastmatt Superstores', 'Cleanshelf Supermarket',
            'Khetias Supermarket', 'Muthaiga Mini Mart', 'Karen Provision Store', 'Westgate Retail',
            'Galleria Shopping Mall', 'Sarit Centre Store', 'Yaya Centre Retail', 'TRM Supermarket',
            'Junction Retail', 'Village Market Store', 'Panari Sky Centre', 'Two Rivers Mall'
        ]
        
        regions = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Western', 'Coastal', 'Central', 'Rift Valley']
        customer_types = ['Supermarket', 'Hypermarket', 'Convenience Store', 'Mini Mart', 'Wholesaler']
        
        customers = []
        for i, name in enumerate(customer_names):
            customers.append({
                'customer_id': f"CUST{i+1:03d}",
                'customer_name': name,
                'type': np.random.choice(customer_types),
                'region': np.random.choice(regions),
                'tier': np.random.choice(['A', 'B', 'C'], p=[0.2, 0.3, 0.5]),
                'credit_limit': np.random.uniform(50000, 500000),
                'payment_terms': np.random.choice([7, 14, 30, 45])
            })
        
        return pd.DataFrame(customers)
    
    def generate_sales_data(self, days=90):
        """Generate comprehensive sales data"""
        np.random.seed(42)
        
        # Get SKUs and Customers
        skus = self.generate_skus()
        customers = self.generate_customers()
        
        sales_data = []
        start_date = datetime.now().date() - timedelta(days=days)
        
        for i in range(1000):  # Generate 1000 transactions
            sku = skus.sample(1).iloc[0]
            customer = customers.sample(1).iloc[0]
            
            date = start_date + timedelta(days=np.random.randint(0, days))
            quantity = np.random.randint(1, 50)
            unit_price = sku['selling_price'] * np.random.uniform(0.9, 1.1)  # Some price variation
            
            sales_data.append({
                'transaction_id': f"TXN{i+1:05d}",
                'date': date,
                'customer_id': customer['customer_id'],
                'sku_id': sku['sku_id'],
                'quantity': quantity,
                'unit_price': unit_price
            })
        
        return pd.DataFrame(sales_data)
    
    def generate_inventory_data(self):
        """Generate comprehensive inventory data"""
        skus = self.generate_skus()
        
        inventory_data = []
        for _, sku in skus.iterrows():
            current_stock = np.random.randint(50, 1000)
            min_stock = np.random.randint(10, 100)
            max_stock = np.random.randint(500, 2000)
            daily_sales = np.random.uniform(5, 50)
            days_cover = current_stock / daily_sales if daily_sales > 0 else 999
            
            # Stock status
            if days_cover < 7:
                stock_status = 'Critical'
            elif days_cover < 14:
                stock_status = 'Low'
            elif days_cover > 60:
                stock_status = 'Excess'
            else:
                stock_status = 'Healthy'
            
            # Expiry risk
            days_to_expiry = np.random.randint(1, 180)
            expiry_risk = 'Critical' if days_to_expiry < 30 else 'High' if days_to_expiry < 60 else 'Medium' if days_to_expiry < 90 else 'Low'
            
            inventory_data.append({
                'sku_id': sku['sku_id'],
                'sku_name': sku['sku_name'],
                'category': sku['category'],
                'current_stock': current_stock,
                'min_stock': min_stock,
                'max_stock': max_stock,
                'daily_sales_rate': daily_sales,
                'days_cover': days_cover,
                'stock_status': stock_status,
                'days_to_expiry': days_to_expiry,
                'expiry_risk': expiry_risk,
                'unit_cost': sku['unit_cost'],
                'stock_value': current_stock * sku['unit_cost'],
                'last_updated': datetime.now() - timedelta(days=np.random.randint(0, 7))
            })
        
        return pd.DataFrame(inventory_data)
    
    def generate_logistics_data(self):
        """Generate logistics data"""
        # Simplified logistics data
        return pd.DataFrame({
            'delivery_id': [f'DEL{i:04d}' for i in range(1, 101)],
            'date': [datetime.now().date() - timedelta(days=np.random.randint(0, 30)) for _ in range(100)],
            'vehicle_id': [f'VH{np.random.randint(1000, 9999)}' for _ in range(100)],
            'route': [f'Route {np.random.randint(1, 10)}' for _ in range(100)],
            'planned_distance_km': np.random.uniform(50, 300, 100),
            'actual_distance_km': np.random.uniform(45, 310, 100),
            'planned_duration_hrs': np.random.uniform(2, 8, 100),
            'actual_duration_hrs': np.random.uniform(1.5, 9, 100),
            'fuel_consumption_l': np.random.uniform(20, 150, 100),
            'delivery_status': np.random.choice(['Completed', 'In Progress', 'Delayed'], 100, p=[0.85, 0.1, 0.05])
        })

# Singleton instance
enhanced_data_generator = EnhancedDataGenerator()
