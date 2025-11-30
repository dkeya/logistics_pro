import pandas as pd
import sqlite3
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from configs.settings import APP_CONFIG
from logistics_core.connectors.data_generator import data_generator

class DatabaseManager:
    """Database management layer for multi-tenant data storage"""
    
    def __init__(self):
        self.base_dir = APP_CONFIG.DATA_DIR
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all required directories exist"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'warehouse'), exist_ok=True)
        
    def get_tenant_database_path(self, tenant_name: str) -> str:
        """Get database path for specific tenant"""
        safe_tenant = tenant_name.replace(' ', '_').lower()
        return os.path.join(self.base_dir, 'warehouse', f'{safe_tenant}.db')
    
    def initialize_tenant_database(self, tenant_name: str):
        """Initialize database schema for a tenant"""
        db_path = self.get_tenant_database_path(tenant_name)
        conn = sqlite3.connect(db_path)
        
        # Create tables
        self._create_sales_table(conn)
        self._create_inventory_table(conn)
        self._create_logistics_table(conn)
        self._create_procurement_table(conn)
        self._create_kpi_table(conn)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized for {tenant_name}")
    
    def _create_sales_table(self, conn):
        """Create sales data table"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sales_transactions (
                transaction_id TEXT PRIMARY KEY,
                transaction_date TIMESTAMP,
                customer_id TEXT,
                customer_name TEXT,
                customer_region TEXT,
                customer_tier TEXT,
                sku_id TEXT,
                product_name TEXT,
                category TEXT,
                quantity INTEGER,
                unit_price REAL,
                cost_price REAL,
                revenue REAL,
                cost_of_goods REAL,
                gross_margin REAL,
                gross_margin_percent REAL,
                sales_rep TEXT,
                tenant_id TEXT
            )
        ''')
        
        # Create indexes for better performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_transactions(transaction_date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sales_customer ON sales_transactions(customer_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sales_product ON sales_transactions(sku_id)')
    
    def _create_inventory_table(self, conn):
        """Create inventory data table"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS inventory_snapshot (
                sku_id TEXT,
                product_name TEXT,
                category TEXT,
                warehouse TEXT,
                current_stock INTEGER,
                min_stock INTEGER,
                max_stock INTEGER,
                safety_stock INTEGER,
                reorder_point INTEGER,
                stock_value REAL,
                days_of_supply REAL,
                stock_status TEXT,
                last_updated TIMESTAMP,
                tenant_id TEXT,
                PRIMARY KEY (sku_id, warehouse)
            )
        ''')
    
    def _create_logistics_table(self, conn):
        """Create logistics data table"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS logistics_deliveries (
                delivery_id TEXT PRIMARY KEY,
                delivery_date TIMESTAMP,
                route_id TEXT,
                route_name TEXT,
                vehicle_id TEXT,
                driver_id TEXT,
                customer_id TEXT,
                planned_departure TIMESTAMP,
                actual_departure TIMESTAMP,
                planned_arrival TIMESTAMP,
                actual_arrival TIMESTAMP,
                on_time BOOLEAN,
                in_full BOOLEAN,
                otif_status BOOLEAN,
                distance_km REAL,
                fuel_consumed REAL,
                delivery_cost REAL,
                issues_reported TEXT,
                tenant_id TEXT
            )
        ''')
    
    def _create_procurement_table(self, conn):
        """Create procurement data table"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS procurement_orders (
                po_id TEXT PRIMARY KEY,
                order_date TIMESTAMP,
                supplier_id TEXT,
                supplier_name TEXT,
                sku_id TEXT,
                product_name TEXT,
                quantity_ordered INTEGER,
                unit_cost REAL,
                total_cost REAL,
                lead_time_days INTEGER,
                actual_lead_time_days INTEGER,
                delivery_date TIMESTAMP,
                on_time_delivery BOOLEAN,
                quality_score REAL,
                quantity_received INTEGER,
                tenant_id TEXT
            )
        ''')
    
    def _create_kpi_table(self, conn):
        """Create KPI tracking table"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS kpi_metrics (
                kpi_id TEXT PRIMARY KEY,
                kpi_name TEXT,
                kpi_value REAL,
                kpi_target REAL,
                kpi_date TIMESTAMP,
                category TEXT,
                tenant_id TEXT
            )
        ''')
    
    def load_sample_data(self, tenant_name: str):
        """Load sample data for a tenant"""
        db_path = self.get_tenant_database_path(tenant_name)
        
        # Initialize database if it doesn't exist
        if not os.path.exists(db_path):
            self.initialize_tenant_database(tenant_name)
        
        conn = sqlite3.connect(db_path)
        
        try:
            # Generate and load sample data
            sales_data = data_generator.generate_sales_data()
            inventory_data = data_generator.generate_inventory_data()
            logistics_data = data_generator.generate_logistics_data()
            procurement_data = data_generator.generate_procurement_data()
            
            # Add tenant ID
            sales_data['tenant_id'] = tenant_name
            inventory_data['tenant_id'] = tenant_name
            logistics_data['tenant_id'] = tenant_name
            procurement_data['tenant_id'] = tenant_name
            
            # Load data into database
            sales_data.to_sql('sales_transactions', conn, if_exists='replace', index=False)
            inventory_data.to_sql('inventory_snapshot', conn, if_exists='replace', index=False)
            logistics_data.to_sql('logistics_deliveries', conn, if_exists='replace', index=False)
            procurement_data.to_sql('procurement_orders', conn, if_exists='replace', index=False)
            
            # Calculate and store KPIs
            self._calculate_initial_kpis(conn, tenant_name)
            
            conn.commit()
            print(f"✅ Sample data loaded for {tenant_name}")
            
        except Exception as e:
            print(f"❌ Error loading sample data: {e}")
        finally:
            conn.close()
    
    def _calculate_initial_kpis(self, conn, tenant_name: str):
        """Calculate initial KPI values"""
        from datetime import datetime
        
        # Calculate OTIF
        otif_query = "SELECT AVG(otif_status) * 100 FROM logistics_deliveries WHERE tenant_id = ?"
        otif_value = conn.execute(otif_query, (tenant_name,)).fetchone()[0]
        
        # Calculate inventory turnover
        turnover_query = """
            SELECT SUM(revenue) / AVG(stock_value) 
            FROM sales_transactions s
            JOIN inventory_snapshot i ON s.sku_id = i.sku_id 
            WHERE s.tenant_id = ? AND i.tenant_id = ?
        """
        turnover_value = conn.execute(turnover_query, (tenant_name, tenant_name)).fetchone()[0]
        
        # Calculate gross margin
        margin_query = "SELECT AVG(gross_margin_percent) FROM sales_transactions WHERE tenant_id = ?"
        margin_value = conn.execute(margin_query, (tenant_name,)).fetchone()[0]
        
        # Insert KPI values
        kpis = [
            ('otif', 'OTIF Performance', otif_value, 95.0, datetime.now(), 'logistics', tenant_name),
            ('inventory_turnover', 'Inventory Turnover', turnover_value, 8.0, datetime.now(), 'inventory', tenant_name),
            ('gross_margin', 'Gross Margin %', margin_value, 30.0, datetime.now(), 'sales', tenant_name)
        ]
        
        conn.executemany('''
            INSERT OR REPLACE INTO kpi_metrics 
            (kpi_id, kpi_name, kpi_value, kpi_target, kpi_date, category, tenant_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', kpis)
    
    def get_sales_data(self, tenant_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get sales data for a tenant with optional date filtering"""
        db_path = self.get_tenant_database_path(tenant_name)
        
        if not os.path.exists(db_path):
            self.load_sample_data(tenant_name)
        
        conn = sqlite3.connect(db_path)
        
        query = "SELECT * FROM sales_transactions WHERE tenant_id = ?"
        params = [tenant_name]
        
        if start_date and end_date:
            query += " AND transaction_date BETWEEN ? AND ?"
            params.extend([start_date, end_date])
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_inventory_data(self, tenant_name: str) -> pd.DataFrame:
        """Get inventory data for a tenant"""
        db_path = self.get_tenant_database_path(tenant_name)
        
        if not os.path.exists(db_path):
            self.load_sample_data(tenant_name)
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT * FROM inventory_snapshot WHERE tenant_id = ?", 
            conn, 
            params=[tenant_name]
        )
        conn.close()
        
        return df
    
    def get_logistics_data(self, tenant_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get logistics data for a tenant"""
        db_path = self.get_tenant_database_path(tenant_name)
        
        if not os.path.exists(db_path):
            self.load_sample_data(tenant_name)
        
        conn = sqlite3.connect(db_path)
        
        query = "SELECT * FROM logistics_deliveries WHERE tenant_id = ?"
        params = [tenant_name]
        
        if start_date and end_date:
            query += " AND delivery_date BETWEEN ? AND ?"
            params.extend([start_date, end_date])
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_kpi_data(self, tenant_name: str, category: str = None) -> pd.DataFrame:
        """Get KPI data for a tenant"""
        db_path = self.get_tenant_database_path(tenant_name)
        
        if not os.path.exists(db_path):
            self.load_sample_data(tenant_name)
        
        conn = sqlite3.connect(db_path)
        
        query = "SELECT * FROM kpi_metrics WHERE tenant_id = ?"
        params = [tenant_name]
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df

# Global database manager instance
db_manager = DatabaseManager()