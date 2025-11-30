# logistics_pro/validate_data.py
import pandas as pd
import numpy as np
import sys
import os

def validate_data_structure():
    """Validate that all data structures are working correctly"""
    print("🔍 VALIDATING DATA STRUCTURE...")
    
    # Add current directory to path
    sys.path.append('.')
    
    try:
        # Test enhanced data generator
        from logistics_core.connectors.data_generator_enhanced import EnhancedDataGenerator
        generator = EnhancedDataGenerator()
        
        print("✅ EnhancedDataGenerator imported successfully")
        
        # Check all required attributes
        required_attrs = ['skus', 'customers', 'sales_data', 'inventory_data', 'logistics_data']
        for attr in required_attrs:
            if hasattr(generator, attr):
                data = getattr(generator, attr)
                print(f"✅ {attr}: {len(data)} records, columns: {list(data.columns)}")
            else:
                print(f"❌ {attr}: Missing")
        
        # Test analytics engine
        from app import AnalyticsEngine
        analytics = AnalyticsEngine(generator)
        
        print("✅ AnalyticsEngine created successfully")
        
        # Check analytics data
        for data_type in ['sales_data', 'inventory_data', 'logistics_data']:
            if hasattr(analytics, data_type):
                data = getattr(analytics, data_type)
                print(f"✅ analytics.{data_type}: {len(data)} records")
            else:
                print(f"❌ analytics.{data_type}: Missing")
        
        print("🎉 ALL DATA VALIDATION PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False

if __name__ == "__main__":
    validate_data_structure()
