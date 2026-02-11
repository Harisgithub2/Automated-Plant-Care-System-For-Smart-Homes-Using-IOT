import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
import os

def prepare_training_data(conn):
    """Prepare data for model training"""
    
    # Query to get historical data with features
    query = '''
    SELECT 
        timestamp,
        temperature,
        humidity,
        soil,
        strftime('%H', timestamp) as hour,
        strftime('%w', timestamp) as day_of_week,
        strftime('%m', timestamp) as month,
        CASE WHEN pump = 'ON' THEN 1 ELSE 0 END as pump_status,
        pump_duration
    FROM sensor_data 
    WHERE timestamp > datetime('now', '-90 days')
    AND temperature IS NOT NULL
    AND humidity IS NOT NULL
    AND soil IS NOT NULL
    ORDER BY timestamp
    '''
    
    df = pd.read_sql_query(query, conn)
    
    if len(df) < 50:
        print("Not enough data for training")
        return None, None, None
    
    # Create time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'].astype(int) / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'].astype(int) / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'].astype(int) / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'].astype(int) / 7)
    
    # Create lag features (previous values)
    for lag in [1, 2, 3, 24]:  # 1 hour, 2 hours, 3 hours, 1 day lags
        df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
        df[f'soil_lag_{lag}'] = df['soil'].shift(lag)
    
    # Create rolling averages
    for window in [3, 6, 12, 24]:  # 3h, 6h, 12h, 24h windows
        df[f'temp_ma_{window}'] = df['temperature'].rolling(window=window).mean()
        df[f'humidity_ma_{window}'] = df['humidity'].rolling(window=window).mean()
        df[f'soil_ma_{window}'] = df['soil'].rolling(window=window).mean()
    
    # Create interaction features
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    df['temp_soil_interaction'] = df['temperature'] * df['soil']
    df['humidity_soil_interaction'] = df['humidity'] * df['soil']
    
    # Target: predict if pump should be ON (1) or OFF (0)
    df['target'] = df['pump_status']
    
    # Drop rows with NaN values (from lag features)
    df = df.dropna()
    
    # Feature selection
    feature_columns = [
        'temperature', 'humidity', 'soil',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'temp_lag_1', 'humidity_lag_1', 'soil_lag_1',
        'temp_ma_3', 'humidity_ma_3', 'soil_ma_3',
        'temp_humidity_interaction', 'humidity_soil_interaction'
    ]
    
    X = df[feature_columns]
    y = df['target']
    
    return X, y, df

def train_ensemble_model(X, y):
    """Train ensemble of models for better predictions"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'linear_regression': LinearRegression()
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred = np.clip(y_pred, 0, 1)  # Clip to [0, 1] range
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        trained_models[name] = {
            'model': model,
            'mse': mse,
            'r2': r2
        }
    
    # Select best model based on R2 score
    best_model_name = max(trained_models, key=lambda x: trained_models[x]['r2'])
    best_model = trained_models[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"R2 Score: {trained_models[best_model_name]['r2']:.4f}")
    
    return best_model, scaler, trained_models

def save_model_artifacts(model, scaler, feature_names, model_info):
    """Save model and related artifacts"""
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/irrigation_model.pkl')
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature names
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save model info
    model_info['trained_at'] = datetime.now().isoformat()
    joblib.dump(model_info, 'models/model_info.pkl')
    
    print("Model artifacts saved successfully")

def generate_training_report(trained_models, X_test, y_test):
    """Generate training report with insights"""
    
    report = {
        'training_date': datetime.now().isoformat(),
        'models_trained': len(trained_models),
        'best_model': None,
        'performance_metrics': {},
        'feature_importance': None
    }
    
    # Find best model
    best_model_name = max(trained_models, key=lambda x: trained_models[x]['r2'])
    best_model = trained_models[best_model_name]['model']
    
    report['best_model'] = {
        'name': best_model_name,
        'r2_score': float(trained_models[best_model_name]['r2']),
        'mse': float(trained_models[best_model_name]['mse'])
    }
    
    # Get feature importance if available
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = dict(zip(
            X_test.columns if hasattr(X_test, 'columns') else range(len(best_model.feature_importances_)),
            best_model.feature_importances_
        ))
        report['feature_importance'] = feature_importance
    
    # Save report
    report_path = f"models/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Training report saved to {report_path}")
    return report

def main():
    """Main training function"""
    
    print("Starting ML model training...")
    
    try:
        # Connect to database
        conn = sqlite3.connect('irrigation.db')
        
        # Prepare data
        X, y, df = prepare_training_data(conn)
        
        if X is None:
            print("Insufficient data for training")
            conn.close()
            return
        
        print(f"Training data shape: {X.shape}")
        print(f"Number of samples: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Pump ON events: {df['pump_status'].sum()} ({df['pump_status'].mean()*100:.1f}%)")
        
        # Train model
        model, scaler, trained_models = train_ensemble_model(X, y)
        
        # Generate report
        report = generate_training_report(trained_models, X, y)
        
        # Save artifacts
        save_model_artifacts(
            model, 
            scaler, 
            list(X.columns), 
            {
                'training_date': datetime.now().isoformat(),
                'best_model': report['best_model']['name'],
                'r2_score': report['best_model']['r2_score'],
                'data_points': len(df),
                'feature_count': X.shape[1]
            }
        )
        
        # Test predictions
        print("\nSample predictions:")
        sample_data = X.iloc[:5].copy()
        sample_scaled = scaler.transform(sample_data)
        predictions = model.predict(sample_scaled)
        
        for i, (idx, row) in enumerate(sample_data.iterrows()):
            print(f"Sample {i+1}: Temp={row['temperature']:.1f}°C, "
                  f"Humidity={row['humidity']:.1f}%, Soil={row['soil']:.0f} "
                  f"-> Pump probability: {predictions[i]:.1%}")
        
        conn.close()
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()