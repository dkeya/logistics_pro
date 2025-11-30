# logistics_pro/configs/settings.py
import os
from datetime import datetime

# Application Settings
APP_NAME = "Logistics Pro"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "FMCG Distribution Intelligence Platform"

# Data Settings
DEFAULT_DATE_RANGE_DAYS = 90
DEFAULT_FORECAST_PERIODS = 28

# Business Rules
INVENTORY_RULES = {
    'critical_stock_threshold': 0.3,  # 30% of normal stock level
    'low_stock_threshold': 0.5,       # 50% of normal stock level  
    'excess_stock_threshold': 2.0,    # 200% of normal stock level
    'high_expiry_risk_days': 30,
    'medium_expiry_risk_days': 90,
    'target_service_level': 0.95,     # 95% service level
}

LOGISTICS_RULES = {
    'target_otif': 0.95,              # 95% OTIF target
    'max_route_hours': 10,
    'max_stops_per_route': 20,
    'fuel_cost_per_liter': 150,       # KES per liter
    'driver_cost_per_hour': 500,      # KES per hour
}

SALES_RULES = {
    'low_margin_threshold': 0.15,     # 15% margin threshold
    'high_margin_threshold': 0.30,    # 30% margin threshold
    'customer_segment_thresholds': {
        'platinum': 1000000,          # KES monthly revenue
        'gold': 500000,
        'silver': 100000,
        'bronze': 0
    }
}

# AI Model Settings
FORECASTING_SETTINGS = {
    'model_type': 'random_forest',
    'training_window_days': 365,
    'confidence_level': 0.95,
    'retraining_frequency_days': 30
}

# Visualization Settings
CHART_SETTINGS = {
    'color_scale': 'Viridis',
    'template': 'plotly_white',
    'height': 400
}

# Export the settings as a dictionary for easy access
def get_settings():
    return {
        'app': {
            'name': APP_NAME,
            'version': APP_VERSION,
            'description': APP_DESCRIPTION
        },
        'inventory_rules': INVENTORY_RULES,
        'logistics_rules': LOGISTICS_RULES,
        'sales_rules': SALES_RULES,
        'forecasting': FORECASTING_SETTINGS,
        'visualization': CHART_SETTINGS
    }