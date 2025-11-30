from typing import Dict, Any
from enum import Enum

class KPICategory(Enum):
    SALES = "sales"
    INVENTORY = "inventory"
    LOGISTICS = "logistics"
    PROCUREMENT = "procurement"

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# KPI Targets and Thresholds
KPI_TARGETS = {
    "otif_performance": {
        "target": 95.0,
        "warning": 90.0,
        "critical": 85.0
    },
    "inventory_turnover": {
        "target": 8.0,
        "warning": 6.0,
        "critical": 4.0
    },
    "gross_margin": {
        "target": 30.0,
        "warning": 25.0,
        "critical": 20.0
    },
    "fleet_utilization": {
        "target": 80.0,
        "warning": 70.0,
        "critical": 60.0
    },
    "stock_availability": {
        "target": 98.0,
        "warning": 95.0,
        "critical": 90.0
    }
}

# Business Rules
BUSINESS_RULES = {
    "inventory": {
        "safety_stock_days": 7,
        "reorder_point_coverage": 14,
        "max_stock_days": 45,
        "expiry_warning_days": 30
    },
    "logistics": {
        "delivery_time_window": 2,  # hours
        "route_optimization_frequency": "daily",
        "fleet_maintenance_interval": 90  # days
    },
    "procurement": {
        "supplier_evaluation_frequency": "quarterly",
        "lead_time_buffer": 3,  # days
        "minimum_order_quantity": 1000  # units
    }
}

# Color Schemes
COLOR_SCHEMES = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#ffa500",
    "danger": "#d62728",
    "info": "#9467bd",
    "light": "#f8f9fa",
    "dark": "#343a40"
}

# Chart Configuration
CHART_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d']
}

# Data Generation Parameters
DATA_GENERATION = {
    "sales": {
        "min_transactions": 1000,
        "max_transactions": 10000,
        "date_range": 365,
        "sku_count": 200,
        "customer_count": 50
    },
    "inventory": {
        "sku_count": 500,
        "warehouse_count": 10,
        "movement_frequency": "daily"
    },
    "logistics": {
        "route_count": 25,
        "vehicle_count": 15,
        "delivery_points": 100
    }
}