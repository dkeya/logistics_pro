# logistics_pro/logistics_core/analytics/optimization.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class RouteOptimizer:
    """Route optimization engine"""
    
    def __init__(self):
        pass
    
    def optimize_routes(self, deliveries: pd.DataFrame, vehicles: pd.DataFrame, 
                       constraints: Optional[Dict] = None) -> pd.DataFrame:
        """Optimize delivery routes"""
        
        if constraints is None:
            constraints = {
                'max_route_time': 8,  # hours
                'max_stops_per_route': 15,
                'time_window_start': 8,  # 8 AM
                'time_window_end': 17   # 5 PM
            }
        
        optimized_routes = []
        
        # Group by region first
        for region in deliveries['region'].unique():
            region_deliveries = deliveries[deliveries['region'] == region]
            
            # Simple assignment to vehicles
            vehicles_in_region = vehicles.sample(min(3, len(vehicles)))
            
            for i, (_, vehicle) in enumerate(vehicles_in_region.iterrows()):
                vehicle_deliveries = region_deliveries.sample(
                    min(constraints['max_stops_per_route'], len(region_deliveries))
                )
                
                route = {
                    'vehicle_id': vehicle['vehicle_id'],
                    'region': region,
                    'stops': len(vehicle_deliveries),
                    'estimated_distance': vehicle_deliveries['planned_distance_km'].sum(),
                    'estimated_duration': vehicle_deliveries['planned_duration_hrs'].sum(),
                    'customers': vehicle_deliveries['customer_id'].tolist()
                }
                
                optimized_routes.append(route)
        
        return pd.DataFrame(optimized_routes)
    
    def calculate_savings(self, current_routes: pd.DataFrame, 
                         optimized_routes: pd.DataFrame) -> Dict:
        """Calculate savings from route optimization"""
        
        current_total_distance = current_routes['actual_distance_km'].sum()
        optimized_total_distance = optimized_routes['estimated_distance'].sum()
        
        current_total_time = current_routes['actual_duration_hrs'].sum()
        optimized_total_time = optimized_routes['estimated_duration'].sum()
        
        distance_savings = (current_total_distance - optimized_total_distance) / current_total_distance * 100
        time_savings = (current_total_time - optimized_total_time) / current_total_time * 100
        
        return {
            'distance_savings_percent': distance_savings,
            'time_savings_percent': time_savings,
            'fuel_savings_percent': distance_savings * 0.8,  # Rough estimate
            'vehicles_required': len(optimized_routes['vehicle_id'].unique())
        }

class InventoryOptimizer:
    """Inventory optimization engine"""
    
    def __init__(self):
        pass
    
    def generate_recommendations(self, inventory_data: pd.DataFrame, 
                               sales_data: pd.DataFrame, 
                               sku_data: pd.DataFrame) -> pd.DataFrame:
        """Generate inventory optimization recommendations"""
        recommendations = []
        
        for _, item in inventory_data.iterrows():
            sku_sales = sales_data[sales_data['sku_id'] == item['sku_id']]
            avg_daily_demand = sku_sales['quantity'].mean() if not sku_sales.empty else 10
            
            current_cover = item['current_stock'] / avg_daily_demand if avg_daily_demand > 0 else 999
            
            if current_cover < 7:
                action = "URGENT: Reorder immediately"
                priority = "High"
            elif current_cover < 14:
                action = "Plan reorder"
                priority = "Medium"
            elif current_cover > 60:
                action = "Excess stock - consider promotions"
                priority = "Medium"
            else:
                action = "Monitor"
                priority = "Low"
                
            sku_name = sku_data[sku_data['sku_id'] == item['sku_id']]['sku_name'].iloc[0]
                
            recommendations.append({
                'sku_id': item['sku_id'],
                'sku_name': sku_name,
                'current_stock': item['current_stock'],
                'daily_demand': round(avg_daily_demand, 1),
                'days_cover': round(current_cover, 1),
                'action': action,
                'priority': priority,
                'recommended_order': max(0, int(avg_daily_demand * 21 - item['current_stock'])) if current_cover < 14 else 0
            })
        
        return pd.DataFrame(recommendations)
    
    def calculate_economic_order_quantity(self, annual_demand: float, 
                                        ordering_cost: float, 
                                        holding_cost: float) -> float:
        """Calculate Economic Order Quantity (EOQ)"""
        return np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)