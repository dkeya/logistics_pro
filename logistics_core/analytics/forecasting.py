# logistics_pro/logistics_core/analytics/forecasting.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    """AI-powered demand forecasting engine"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_importance = {}
        
    def prepare_features(self, sales_data: pd.DataFrame, sku_data: pd.DataFrame, 
                        calendar_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare features for demand forecasting"""
        
        # Aggregate sales by SKU and date
        daily_sales = sales_data.groupby(['date', 'sku_id']).agg({
            'quantity': 'sum'
        }).reset_index()
        
        # Create time-based features
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
        daily_sales['day_of_month'] = daily_sales['date'].dt.day
        daily_sales['month'] = daily_sales['date'].dt.month
        daily_sales['quarter'] = daily_sales['date'].dt.quarter
        daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)
        
        # Merge with SKU data
        daily_sales = daily_sales.merge(sku_data, on='sku_id')
        
        # Create lag features
        daily_sales = daily_sales.sort_values(['sku_id', 'date'])
        for lag in [1, 7, 14, 30]:
            daily_sales[f'lag_{lag}'] = daily_sales.groupby('sku_id')['quantity'].shift(lag)
        
        # Rolling statistics
        daily_sales['rolling_mean_7'] = daily_sales.groupby('sku_id')['quantity'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        daily_sales['rolling_std_7'] = daily_sales.groupby('sku_id')['quantity'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
        
        return daily_sales.dropna()
    
    def train(self, sales_data: pd.DataFrame, sku_data: pd.DataFrame) -> Dict:
        """Train the forecasting model"""
        try:
            # Prepare features
            features_df = self.prepare_features(sales_data, sku_data)
            
            # Select features and target
            feature_columns = [
                'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
                'unit_cost', 'selling_price', 'weight_kg', 'volume_l',
                'lag_1', 'lag_7', 'lag_14', 'lag_30',
                'rolling_mean_7', 'rolling_std_7'
            ]
            
            X = features_df[feature_columns]
            y = features_df['quantity']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.is_trained = True
            self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
            
            return {
                'mae': mae,
                'rmse': rmse,
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def forecast(self, sku_id: str, sku_data: pd.DataFrame, history_data: pd.DataFrame, 
                periods: int = 28) -> Optional[pd.DataFrame]:
        """Generate forecast for a specific SKU"""
        if not self.is_trained:
            # Use simple forecasting if model not trained
            return self._simple_forecast(sku_id, history_data, periods)
            
        try:
            # Get historical data for the SKU
            sku_history = history_data[history_data['sku_id'] == sku_id].copy()
            if sku_history.empty:
                return self._simple_forecast(sku_id, history_data, periods)
                
            # Get SKU features
            sku_info = sku_data[sku_data['sku_id'] == sku_id].iloc[0]
            
            # Generate future dates
            last_date = pd.to_datetime(sku_history['date'].max())
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            forecasts = []
            
            for i, date in enumerate(future_dates):
                # Prepare features for this date
                features = {
                    'day_of_week': date.dayofweek,
                    'day_of_month': date.day,
                    'month': date.month,
                    'quarter': date.quarter,
                    'is_weekend': 1 if date.dayofweek in [5, 6] else 0,
                    'unit_cost': sku_info['unit_cost'],
                    'selling_price': sku_info['selling_price'],
                    'weight_kg': sku_info['weight_kg'],
                    'volume_l': sku_info['volume_l'],
                }
                
                # Add lag features (using recent history)
                recent_data = sku_history.tail(30)
                if len(recent_data) >= 1:
                    features['lag_1'] = recent_data.iloc[-1]['quantity']
                if len(recent_data) >= 7:
                    features['lag_7'] = recent_data.iloc[-7]['quantity']
                if len(recent_data) >= 14:
                    features['lag_14'] = recent_data.iloc[-14]['quantity']
                if len(recent_data) >= 30:
                    features['lag_30'] = recent_data.iloc[-30]['quantity']
                
                # Add rolling statistics
                if len(recent_data) >= 7:
                    features['rolling_mean_7'] = recent_data['quantity'].mean()
                    features['rolling_std_7'] = recent_data['quantity'].std()
                
                # Ensure all features are present
                feature_columns = [
                    'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
                    'unit_cost', 'selling_price', 'weight_kg', 'volume_l',
                    'lag_1', 'lag_7', 'lag_14', 'lag_30',
                    'rolling_mean_7', 'rolling_std_7'
                ]
                
                for col in feature_columns:
                    if col not in features:
                        features[col] = 0  # Default value for missing features
                
                # Create feature vector
                feature_vector = pd.DataFrame([features])[feature_columns]
                
                # Make prediction
                prediction = max(0, self.model.predict(feature_vector)[0])
                
                forecasts.append({
                    'date': date.date(),
                    'forecast': prediction,
                    'week': i // 7 + 1
                })
            
            return pd.DataFrame(forecasts)
            
        except Exception as e:
            print(f"Error generating forecast: {e}")
            return self._simple_forecast(sku_id, history_data, periods)
    
    def _simple_forecast(self, sku_id: str, history_data: pd.DataFrame, 
                        periods: int) -> pd.DataFrame:
        """Simple forecasting method as fallback"""
        sku_history = history_data[history_data['sku_id'] == sku_id]
        if sku_history.empty:
            base_demand = 50
        else:
            base_demand = sku_history['quantity'].mean()
            
        forecasts = []
        for week in range(periods):
            # Add some seasonality and noise
            seasonal_factor = 1 + 0.1 * np.sin(week * np.pi / 2)
            week_demand = base_demand * seasonal_factor * np.random.uniform(0.9, 1.1)
            forecasts.append({
                'week': week + 1,
                'forecast_demand': max(0, int(week_demand)),
                'confidence_interval_lower': max(0, int(week_demand * 0.8)),
                'confidence_interval_upper': int(week_demand * 1.2)
            })
        return pd.DataFrame(forecasts)