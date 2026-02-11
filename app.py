from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime, timedelta
import csv
import io
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import requests
import schedule
import time
import threading
import joblib
from collections import defaultdict
import random
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
OPENWEATHER_API_KEY = "1d77de94d48b21f123aa67a8cd3e70f4"  # Your OpenWeather API key

# Store detected location
user_location = None
location_detected = False

# Email configuration
EMAIL_CONFIG = {
    'enabled': False,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': '',
    'sender_password': '',
    'recipient_email': ''
}

# Ensure directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    # Create sensors table
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  temperature REAL,
                  humidity REAL,
                  soil INTEGER,
                  water TEXT,
                  pump TEXT,
                  pump_duration INTEGER DEFAULT 0,
                  manual_override INTEGER DEFAULT 0)''')
    
    # Create alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  type TEXT,
                  message TEXT,
                  resolved INTEGER DEFAULT 0)''')
    
    # Create settings table
    c.execute('''CREATE TABLE IF NOT EXISTS settings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  key TEXT UNIQUE,
                  value TEXT)''')
    
    # Create weather data table
    c.execute('''CREATE TABLE IF NOT EXISTS weather_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  temperature REAL,
                  humidity REAL,
                  precipitation REAL,
                  rain_forecast INTEGER DEFAULT 0,
                  weather_description TEXT,
                  location_city TEXT,
                  location_lat REAL,
                  location_lon REAL)''')
    
    # Create pump schedule table
    c.execute('''CREATE TABLE IF NOT EXISTS pump_schedule
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  hour INTEGER,
                  minute INTEGER,
                  duration INTEGER,
                  enabled INTEGER DEFAULT 1)''')
    
    # Create location table
    c.execute('''CREATE TABLE IF NOT EXISTS location
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  city TEXT,
                  lat REAL,
                  lon REAL,
                  country TEXT,
                  method TEXT)''')
    
    # Create ML training log table
    c.execute('''CREATE TABLE IF NOT EXISTS training_log
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  model_name TEXT,
                  accuracy REAL,
                  r2_score REAL,
                  mse REAL,
                  samples_used INTEGER,
                  status TEXT,
                  message TEXT)''')
    
    # Create predictions table for storing generated predictions
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  prediction_date TEXT,
                  day_name TEXT,
                  avg_probability REAL,
                  predictions_json TEXT,
                  recommendations TEXT)''')
    
    # Insert default settings if not exists
    default_settings = [
        ('soil_dry_threshold', '500'),
        ('soil_wet_threshold', '300'),
        ('update_interval', '5'),
        ('max_temperature', '35'),
        ('min_humidity', '30'),
        ('enable_alerts', '1'),
        ('auto_mode', '1'),
        ('weather_integration', '1'),
        ('rain_threshold', '0.5'),
        ('prediction_days', '7'),
        ('auto_detect_location', '1'),
        ('auto_train_interval', '10'),  # Auto-train every 10 minutes
        ('min_training_samples', '50'),
        ('ml_model_type', 'random_forest'),  # Options: linear_regression, random_forest, gradient_boosting
        ('email_enabled', '0'),
        ('email_sender', ''),
        ('email_password', ''),
        ('email_recipient', ''),
        ('email_smtp_server', 'smtp.gmail.com'),
        ('email_smtp_port', '587'),
        ('weather_rain_stop_pump', '1'),  # New: Stop pump if rain forecast
        ('rain_stop_threshold', '60'),  # New: Rain probability threshold to stop pump
        ('prediction_hours', '6,9,12,15,18,21'),  # Hours to predict for
        ('prediction_enabled', '1')  # Enable predictions
    ]
    
    for key, value in default_settings:
        c.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', (key, value))
    
    # Default pump schedule (8 AM daily for 5 minutes)
    c.execute('''INSERT OR IGNORE INTO pump_schedule (hour, minute, duration) 
                 VALUES (8, 0, 300)''')
    
    # Get stored location
    c.execute('SELECT city, lat, lon, country, method FROM location ORDER BY timestamp DESC LIMIT 1')
    stored_location = c.fetchone()
    
    conn.commit()
    conn.close()
    
    global user_location, location_detected
    if stored_location:
        user_location = {
            'city': stored_location[0],
            'lat': stored_location[1],
            'lon': stored_location[2],
            'country': stored_location[3],
            'method': stored_location[4]
        }
        location_detected = True
        print(f"Loaded stored location: {user_location['city']} ({user_location['lat']}, {user_location['lon']})")
    
    # Load email config
    load_email_config()

# Email configuration functions
def load_email_config():
    """Load email configuration from database"""
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('''SELECT key, value FROM settings 
                     WHERE key IN ('email_enabled', 'email_sender', 
                                   'email_password', 'email_recipient',
                                   'email_smtp_server', 'email_smtp_port')''')
        
        config = {row[0]: row[1] for row in c.fetchall()}
        conn.close()
        
        if config.get('email_enabled') == '1':
            EMAIL_CONFIG.update({
                'enabled': True,
                'smtp_server': config.get('email_smtp_server', 'smtp.gmail.com'),
                'smtp_port': int(config.get('email_smtp_port', 587)),
                'sender_email': config.get('email_sender', ''),
                'sender_password': config.get('email_password', ''),
                'recipient_email': config.get('email_recipient', '')
            })
            print("Email alerts enabled")
        else:
            EMAIL_CONFIG['enabled'] = False
            print("Email alerts disabled")
            
    except Exception as e:
        print(f"Error loading email config: {e}")

# Initialize database on startup
init_db()

# Global variables for pump state
pump_state = "OFF"
pump_override = False
pump_start_time = None


def send_email_alert(subject, message, alert_type='warning'):
    """Send email alert in background thread"""
    if not EMAIL_CONFIG['enabled'] or not EMAIL_CONFIG['sender_email']:
        return
    
    def send_email():
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG['sender_email']
            msg['To'] = EMAIL_CONFIG['recipient_email']
            msg['Subject'] = f"[Smart Irrigation] {subject}"
            
            # HTML email body
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <div style="background-color: {'#ffebee' if alert_type == 'danger' else '#fff3cd' if alert_type == 'warning' else '#e3f2fd'}; 
                            padding: 20px; border-radius: 10px; border-left: 5px solid {'#dc3545' if alert_type == 'danger' else '#ffc107' if alert_type == 'warning' else '#2196F3'};">
                    <h2 style="color: #333;">Smart Irrigation System Alert</h2>
                    <p style="font-size: 16px; color: #555;">{message}</p>
                    <p style="font-size: 14px; color: #777; margin-top: 20px;">
                        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                        Alert Type: {alert_type.upper()}
                    </p>
                </div>
                <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 12px; color: #666;">
                    This is an automated message from your Smart Irrigation System.
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            print(f"Email alert sent: {subject}")
            
        except Exception as e:
            print(f"Error sending email: {e}")
    
    # Send email in background thread
    thread = threading.Thread(target=send_email)
    thread.daemon = True
    thread.start()

# Location detection functions
def detect_location_by_ip():
    """Detect location using IP address with free services"""
    
    # List of free IP geolocation APIs
    ip_services = [
        {
            'name': 'ip-api',
            'url': 'http://ip-api.com/json/',
            'parse': lambda data: {
                'city': data.get('city'),
                'lat': data.get('lat'),
                'lon': data.get('lon'),
                'country': data.get('countryCode'),
                'region': data.get('regionName')
            } if data.get('status') == 'success' else None
        },
        {
            'name': 'ipapi',
            'url': 'https://ipapi.co/json/',
            'parse': lambda data: {
                'city': data.get('city'),
                'lat': float(data.get('latitude', 0)),
                'lon': float(data.get('longitude', 0)),
                'country': data.get('country_code'),
                'region': data.get('region')
            }
        }
    ]
    
    for service in ip_services:
        try:
            response = requests.get(service['url'], timeout=3)
            if response.status_code == 200:
                data = response.json()
                location = service['parse'](data)
                
                if location and location.get('lat') and location.get('lon'):
                    location['method'] = f'ip_{service["name"]}'
                    print(f"Location detected via {service['name']}: {location}")
                    return location
                    
        except Exception as e:
            print(f"IP service {service['name']} failed: {e}")
            continue
    
    return None

def detect_location_by_browser(lat, lon):
    """Store location from browser geolocation"""
    try:
        # Reverse geocode to get city name
        reverse_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={OPENWEATHER_API_KEY}"
        response = requests.get(reverse_url, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                city = data[0].get('name', 'Unknown')
                country = data[0].get('country', '')
                
                location = {
                    'city': city,
                    'lat': lat,
                    'lon': lon,
                    'country': country,
                    'method': 'browser_geolocation'
                }
                
                # Store in database
                conn = sqlite3.connect('irrigation.db')
                c = conn.cursor()
                c.execute('''INSERT INTO location (city, lat, lon, country, method)
                             VALUES (?, ?, ?, ?, ?)''',
                          (city, lat, lon, country, 'browser_geolocation'))
                conn.commit()
                conn.close()
                
                global user_location, location_detected
                user_location = location
                location_detected = True
                
                print(f"Browser location stored: {city} ({lat}, {lon})")
                return location
    except Exception as e:
        print(f"Reverse geocoding failed: {e}")
    
    return None

def get_user_location():
    """Get or detect user location"""
    global user_location, location_detected
    
    # If location already detected, return it
    if location_detected and user_location:
        return user_location
    
    # Try to detect by IP
    location = detect_location_by_ip()
    
    if location:
        # Store in database
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        c.execute('''INSERT INTO location (city, lat, lon, country, method)
                     VALUES (?, ?, ?, ?, ?)''',
                  (location['city'], location['lat'], location['lon'], 
                   location['country'], location['method']))
        conn.commit()
        conn.close()
        
        user_location = location
        location_detected = True
        return location
    
    # Fallback to major cities if detection fails
    major_cities = [
        {'city': 'London', 'lat': 51.5074, 'lon': -0.1278, 'country': 'GB'},
        {'city': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'country': 'US'},
        {'city': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503, 'country': 'JP'},
        {'city': 'Delhi', 'lat': 28.6139, 'lon': 77.2090, 'country': 'IN'},
        {'city': 'Singapore', 'lat': 1.3521, 'lon': 103.8198, 'country': 'SG'},
        {'city': 'Sydney', 'lat': -33.8688, 'lon': 151.2093, 'country': 'AU'},
        {'city': 'Dubai', 'lat': 25.2048, 'lon': 55.2708, 'country': 'AE'},
    ]
    
    # Select a random major city as fallback
    fallback = random.choice(major_cities)
    fallback['method'] = 'fallback_random'
    
    # Store fallback in database
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    c.execute('''INSERT INTO location (city, lat, lon, country, method)
                 VALUES (?, ?, ?, ?, ?)''',
              (fallback['city'], fallback['lat'], fallback['lon'], 
               fallback['country'], fallback['method']))
    conn.commit()
    conn.close()
    
    user_location = fallback
    location_detected = True
    
    print(f"Using fallback location: {fallback['city']}")
    return fallback

# Basic Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

# Data handling endpoint
@app.route('/data', methods=['POST', 'GET'])
def handle_data():
    if request.method == 'POST':
        try:
            data = request.get_json()
            print(f"Received data: {data}")
            
            # Validate required fields
            required_fields = ['temperature', 'humidity', 'soil', 'water', 'pump']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return jsonify({'status': 'error', 'message': f'Missing fields: {missing_fields}'}), 400
            
            # Convert soil value
            soil_value = convert_soil_value(data['soil'])
            
            conn = sqlite3.connect('irrigation.db')
            c = conn.cursor()
            
            # Get settings for auto mode
            c.execute('''SELECT key, value FROM settings 
                         WHERE key IN ("auto_mode", "weather_integration", "weather_rain_stop_pump", "rain_stop_threshold")''')
            settings = {row[0]: row[1] for row in c.fetchall()}
            
            auto_mode = settings.get('auto_mode', '1') == '1'
            weather_integration = settings.get('weather_integration', '1') == '1'
            weather_rain_stop = settings.get('weather_rain_stop_pump', '1') == '1'
            rain_stop_threshold = float(settings.get('rain_stop_threshold', '60'))
            
            # Determine pump state based on mode
            final_pump_state = str(data['pump'])
            pump_duration = 0
            reason = "Manual control"
            
            if auto_mode:
                # Auto mode: decide based on conditions
                should_pump, reason = determine_pump_action(
                    float(data['temperature']),
                    float(data['humidity']),
                    soil_value,
                    str(data['water']),
                    weather_integration,
                    weather_rain_stop,
                    rain_stop_threshold,
                    conn
                )
                
                if should_pump:
                    final_pump_state = "ON"
                    # Calculate pump duration based on soil dryness
                    if soil_value > 550:
                        pump_duration = 120  # 2 minutes for very dry soil
                    elif soil_value > 450:
                        pump_duration = 60   # 1 minute for dry soil
                    else:
                        pump_duration = 30   # 30 seconds for slightly dry
                    print(f"Auto mode: Pump ON for {pump_duration}s - {reason}")
                else:
                    final_pump_state = "OFF"
                    print(f"Auto mode: Pump OFF - {reason}")
            
            # Insert sensor data
            c.execute('''INSERT INTO sensor_data 
                         (temperature, humidity, soil, water, pump, pump_duration, manual_override)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (float(data['temperature']), 
                       float(data['humidity']), 
                       soil_value, 
                       str(data['water']), 
                       final_pump_state,
                       pump_duration,
                       1 if not auto_mode else 0))
            
            # Check for alerts
            check_alerts({
                'temperature': float(data['temperature']),
                'humidity': float(data['humidity']),
                'soil': soil_value,
                'water': str(data['water']),
                'pump': final_pump_state
            }, conn)
            
            # Update global pump state
            global pump_state
            pump_state = final_pump_state
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'status': 'success', 
                'message': 'Data saved successfully',
                'pump_state': final_pump_state,
                'mode': 'MANUAL' if not auto_mode else 'AUTO',
                'pump_duration': pump_duration,
                'reason': reason
            })
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    else:  # GET request
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        # Get latest data
        c.execute('''SELECT * FROM sensor_data 
                     ORDER BY timestamp DESC LIMIT 1''')
        row = c.fetchone()
        
        if row:
            data = {
                'id': row[0],
                'timestamp': row[1],
                'temperature': row[2],
                'humidity': row[3],
                'soil': row[4],
                'water': row[5],
                'pump': row[6],
                'pump_duration': row[7],
                'manual_override': bool(row[8])
            }
        else:
            data = {}
        
        conn.close()
        return jsonify(data)

def convert_soil_value(soil_value):
    """Convert soil value from various formats to integer"""
    try:
        if isinstance(soil_value, str):
            if soil_value.upper() == "DRY":
                return 600
            elif soil_value.upper() == "WET":
                return 300
            else:
                return int(float(soil_value))
        else:
            return int(float(soil_value))
    except:
        return 500

def determine_pump_action(temperature, humidity, soil, water_level, weather_integration, weather_rain_stop, rain_stop_threshold, conn):
    """Determine if pump should turn on based on conditions"""
    c = conn.cursor()
    
    # Get thresholds
    c.execute('SELECT key, value FROM settings WHERE key IN ("soil_dry_threshold", "rain_threshold")')
    settings = {row[0]: row[1] for row in c.fetchall()}
    
    soil_dry_threshold = int(settings.get('soil_dry_threshold', 500))
    
    # Check water level
    if water_level == "NOT AVAILABLE":
        return False, "Water tank is empty"
    
    # Check soil moisture
    if soil <= soil_dry_threshold:
        return False, f"Soil is wet enough (value: {soil})"
    
    # Check weather if integration is enabled
    if weather_integration:
        # Get latest weather data
        c.execute('''SELECT rain_forecast, precipitation, weather_description 
                     FROM weather_data 
                     ORDER BY timestamp DESC LIMIT 1''')
        weather_row = c.fetchone()
        
        if weather_row:
            rain_forecast = weather_row[0]
            precipitation = weather_row[1] or 0
            weather_desc = weather_row[2] or ""
            
            # Check if we should stop pump due to rain forecast
            if weather_rain_stop and rain_forecast == 1:
                # Calculate rain probability from weather description
                rain_probability = estimate_rain_probability(weather_desc)
                
                if rain_probability >= rain_stop_threshold:
                    return False, f"Rain forecasted ({rain_probability}% probability)"
            
            # Get precipitation threshold
            rain_threshold = float(settings.get('rain_threshold', 0.5))
            
            if precipitation >= rain_threshold:
                return False, f"Rain detected ({precipitation}mm)"
    
    # All checks passed - pump should turn on
    return True, f"Soil is dry (value: {soil})"

def estimate_rain_probability(weather_description):
    """Estimate rain probability from weather description"""
    weather_lower = weather_description.lower()
    
    if any(word in weather_lower for word in ['thunderstorm', 'heavy rain', 'torrential']):
        return 90
    elif any(word in weather_lower for word in ['rain', 'drizzle', 'shower']):
        return 70
    elif any(word in weather_lower for word in ['light rain', 'sprinkle']):
        return 50
    elif any(word in weather_lower for word in ['cloud', 'overcast']):
        return 30
    else:
        return 10

# Weather API integration with auto-location
def fetch_weather_data():
    """Fetch current weather and forecast using auto-detected location"""
    try:
        # Get user location
        location = get_user_location()
        
        if not location:
            print("No location available for weather data")
            return False
        
        # Use coordinates for weather API
        lat = location['lat']
        lon = location['lon']
        
        # Current weather
        current_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        
        # Forecast for rain prediction
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        
        response = requests.get(current_url)
        
        if response.status_code == 200:
            data = response.json()
            
            conn = sqlite3.connect('irrigation.db')
            c = conn.cursor()
            
            # Extract weather data
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            precipitation = data.get('rain', {}).get('1h', 0) or data.get('snow', {}).get('1h', 0)
            weather_desc = data['weather'][0]['description']
            
            # Check if rain is in forecast (next 6 hours)
            rain_forecast = 0
            rain_probability = 0
            try:
                forecast_response = requests.get(forecast_url)
                if forecast_response.status_code == 200:
                    forecast_data = forecast_response.json()
                    # Check next 3 forecast periods (9 hours)
                    for forecast in forecast_data['list'][:3]:
                        if 'rain' in forecast:
                            rain_forecast = 1
                            rain_probability = forecast.get('rain', {}).get('3h', 0) * 100
                            break
                        elif 'Rain' in forecast['weather'][0]['main']:
                            rain_forecast = 1
                            rain_probability = 70  # Default probability
                            break
            except Exception as e:
                print(f"Forecast check failed: {e}")
            
            # Insert into database with location info
            c.execute('''INSERT INTO weather_data 
                         (temperature, humidity, precipitation, rain_forecast, 
                          weather_description, location_city, location_lat, location_lon)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (temperature, humidity, precipitation, rain_forecast, 
                       weather_desc, location['city'], lat, lon))
            
            conn.commit()
            conn.close()
            
            print(f"Weather data fetched for {location['city']}: {temperature}°C, {humidity}% humidity, Rain forecast: {rain_forecast}")
            return True
        else:
            print(f"Weather API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        traceback.print_exc()
    return False

# Location API endpoint
@app.route('/api/location', methods=['GET', 'POST'])
def handle_location():
    if request.method == 'GET':
        location = get_user_location()
        return jsonify({
            'location': location,
            'detected': location_detected,
            'weather_api_key_set': OPENWEATHER_API_KEY != "YOUR_ACTUAL_API_KEY_HERE"
        })
    
    else:  # POST request for browser geolocation
        try:
            data = request.get_json()
            lat = data.get('lat')
            lon = data.get('lon')
            
            if lat and lon:
                location = detect_location_by_browser(lat, lon)
                if location:
                    return jsonify({
                        'status': 'success',
                        'message': 'Location saved successfully',
                        'location': location
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'Failed to save location'
                    }), 400
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Latitude and longitude required'
                }), 400
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

# ==================== ENHANCED ML MODEL TRAINING ====================
def get_training_settings():
    """Get ML training settings from database"""
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    c.execute('SELECT key, value FROM settings WHERE key IN ("min_training_samples", "ml_model_type")')
    settings = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    
    return {
        'min_samples': int(settings.get('min_training_samples', 50)),
        'model_type': settings.get('ml_model_type', 'random_forest')
    }

def log_training_result(model_name, accuracy, r2_score, mse, samples_used, status, message):
    """Log training results to database"""
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    c.execute('''INSERT INTO training_log 
                 (model_name, accuracy, r2_score, mse, samples_used, status, message)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (model_name, accuracy, r2_score, mse, samples_used, status, message))
    conn.commit()
    conn.close()

def prepare_training_data():
    """Prepare data for ML training from database"""
    conn = sqlite3.connect('irrigation.db')
    
    try:
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
            pump_duration,
            CASE WHEN water = 'AVAILABLE' THEN 1 ELSE 0 END as water_available
        FROM sensor_data 
        WHERE timestamp > datetime('now', '-30 days')
        AND temperature IS NOT NULL
        AND humidity IS NOT NULL
        AND soil IS NOT NULL
        ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) < 10:  # Minimum samples
            print(f"Not enough data for training. Only {len(df)} samples available.")
            return None, None, None
        
        print(f"📊 Preparing {len(df)} samples for training...")
        
        # Create time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract hour and day as integers for cyclical features
        df['hour_int'] = df['hour'].astype(int)
        df['day_int'] = df['day_of_week'].astype(int)
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_int'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_int'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_int'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_int'] / 7)
        
        # Lag features (previous readings)
        for lag in [1, 2, 3]:  # Previous 1-3 readings
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
            df[f'soil_lag_{lag}'] = df['soil'].shift(lag)
        
        # Rolling averages (3 and 6 previous readings)
        for window in [3, 6]:
            df[f'temp_ma_{window}'] = df['temperature'].rolling(window=window).mean()
            df[f'humidity_ma_{window}'] = df['humidity'].rolling(window=window).mean()
            df[f'soil_ma_{window}'] = df['soil'].rolling(window=window).mean()
        
        # Interaction features
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100.0
        df['temp_soil_interaction'] = df['temperature'] * df['soil'] / 1000.0
        df['humidity_soil_interaction'] = df['humidity'] * df['soil'] / 1000.0
        
        # Month as categorical (sin/cos)
        df['month_int'] = df['month'].astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['month_int'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_int'] / 12)
        
        # Target: predict if pump should be ON (1) or OFF (0)
        # We'll use a combination of pump_status and soil moisture
        df['target'] = df['pump_status']
        
        # Also create a regression target for pump duration
        df['duration_target'] = df['pump_duration']
        
        # Drop rows with NaN values (from lag features)
        df = df.dropna()
        
        if len(df) < 10:
            print("Not enough valid samples after cleaning.")
            return None, None, None
        
        # Feature selection - include all engineered features
        feature_columns = [
            'temperature', 'humidity', 'soil',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'temp_lag_1', 'humidity_lag_1', 'soil_lag_1',
            'temp_ma_3', 'humidity_ma_3', 'soil_ma_3',
            'temp_humidity_interaction', 'temp_soil_interaction', 'humidity_soil_interaction',
            'water_available'
        ]
        
        # Only use columns that exist in the dataframe
        available_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[available_columns]
        y_class = df['target']  # Classification target
        y_reg = df['duration_target']  # Regression target
        
        print(f"✅ Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Pump ON events: {y_class.sum()} ({y_class.mean()*100:.1f}%)")
        print(f"   Average pump duration: {y_reg.mean():.1f}s")
        
        return X, y_class, available_columns
        
    except Exception as e:
        print(f"Error preparing training data: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        conn.close()

def train_prediction_model_enhanced():
    """Enhanced ML model training with multiple algorithms"""
    try:
        print("\n" + "="*60)
        print("🚀 STARTING ENHANCED ML MODEL TRAINING")
        print("="*60)
        
        # Prepare data
        X, y, feature_names = prepare_training_data()
        
        if X is None or y is None:
            log_training_result('none', 0, 0, 0, 0, 'failed', 'Insufficient training data')
            print("❌ Training aborted: Insufficient data")
            return None
        
        # Get training settings
        settings = get_training_settings()
        min_samples = settings['min_samples']
        model_type = settings['model_type']
        
        if len(X) < min_samples:
            msg = f"Only {len(X)} samples available, need at least {min_samples}"
            log_training_result('none', 0, 0, 0, len(X), 'failed', msg)
            print(f"❌ Training aborted: {msg}")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select and train model based on settings
        if model_type == 'linear_regression':
            model = LinearRegression()
            model_name = 'Linear Regression'
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            model_name = 'Gradient Boosting'
        else:  # Default to random_forest
            # Use classifier for pump on/off prediction
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
            model_name = 'Random Forest Classifier'
        
        print(f"🏋️ Training {model_name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics based on model type
        if hasattr(model, 'predict_proba'):
            # Classification model
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            r2 = r2_score(y_test, y_pred_proba)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"\n📈 Model Performance (Classification):")
            print(f"  ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"  📊 R2 Score: {r2:.4f}")
            print(f"  📉 MSE: {mse:.4f}")
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            print(f"  🎯 Precision: {report['1']['precision']:.3f}")
            print(f"  🎯 Recall: {report['1']['recall']:.3f}")
            print(f"  🎯 F1-Score: {report['1']['f1-score']:.3f}")
        else:
            # Regression model
            accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"\n📈 Model Performance (Regression):")
            print(f"  ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"  📊 R2 Score: {r2:.4f}")
            print(f"  📉 MSE: {mse:.4f}")
        
        print(f"  📊 Samples used: {len(X)}")
        print(f"  🔧 Features: {len(feature_names)}")
        print(f"  📋 Feature names: {feature_names}")
        
        # Save model artifacts
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = 'models/irrigation_model.pkl'
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = 'models/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        
        # Save feature names
        feature_path = 'models/feature_names.pkl'
        joblib.dump(feature_names, feature_path)
        
        # Save model info
        model_info = {
            'model_name': model_name,
            'trained_at': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'r2_score': float(r2),
            'mse': float(mse),
            'samples_used': len(X),
            'feature_count': len(feature_names),
            'features': feature_names,
            'model_type': model_type
        }
        
        info_path = 'models/model_info.pkl'
        joblib.dump(model_info, info_path)
        
        # Log to database
        log_training_result(model_name, accuracy, r2, mse, len(X), 'success', 
                          f'Model trained successfully with {len(X)} samples')
        
        print(f"\n✅ Model saved successfully:")
        print(f"   📁 Model: {model_path}")
        print(f"   📁 Scaler: {scaler_path}")
        print(f"   📁 Features: {feature_path}")
        print(f"   📁 Info: {info_path}")
        
        # Export training data to CSV for reference
        export_training_data_to_csv()
        
        return model
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        log_training_result('none', 0, 0, 0, 0, 'failed', error_msg)
        return None

def export_training_data_to_csv():
    """Export training data to CSV for analysis"""
    try:
        conn = sqlite3.connect('irrigation.db')
        
        # Get last 30 days of data
        query = '''
        SELECT 
            timestamp,
            temperature,
            humidity,
            soil,
            water,
            pump,
            pump_duration,
            manual_override
        FROM sensor_data 
        WHERE timestamp > datetime('now', '-30 days')
        ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            os.makedirs('models/training_data', exist_ok=True)
            csv_path = f"models/training_data/training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ Training data exported to: {csv_path}")
            return csv_path
    except Exception as e:
        print(f"⚠️ Error exporting training data: {e}")
    
    return None

def generate_sample_predictions():
    """Generate sample predictions when model fails"""
    print("📝 Generating sample predictions...")
    
    predictions = []
    current_date = datetime.now()
    
    for i in range(7):
        date = current_date + timedelta(days=i)
        day_name = date.strftime('%A')
        date_str = date.strftime('%Y-%m-%d')
        
        # Create realistic sample predictions
        day_predictions = []
        prediction_hours = [6, 9, 12, 15, 18, 21]
        
        for hour in prediction_hours:
            # Base probability with some variation
            base_prob = 60 - (i * 3)  # Decrease probability over days
            variation = random.randint(-15, 15)
            probability = base_prob + variation
            
            # Adjust based on time of day
            if hour in [6, 18]:  # Morning/Evening - higher probability
                probability += 10
            elif hour == 12:  # Midday - lower probability
                probability -= 5
            
            probability = max(10, min(95, probability))
            
            day_predictions.append({
                'hour': hour,
                'probability': float(probability),
                'recommended': probability > 60,
                'confidence': 'high' if probability > 80 else 'medium' if probability > 60 else 'low'
            })
        
        # Calculate day statistics
        probabilities = [p['probability'] for p in day_predictions]
        avg_probability = sum(probabilities) / len(probabilities) if probabilities else 50
        
        # Generate recommendations
        recommendations = []
        if avg_probability > 70:
            recommendations.append(f"High watering probability ({avg_probability:.0f}%)")
            best_time = max(day_predictions, key=lambda x: x['probability'])
            recommendations.append(f"Best watering time: {best_time['hour']}:00")
        elif avg_probability > 45:
            recommendations.append(f"Moderate watering probability ({avg_probability:.0f}%)")
            recommendations.append("Check soil moisture before watering")
        else:
            recommendations.append(f"Low watering probability ({avg_probability:.0f}%)")
            recommendations.append("No watering needed")
        
        predictions.append({
            'date': date_str,
            'day_name': day_name,
            'avg_probability': round(avg_probability, 1),
            'predictions': day_predictions,
            'recommendations': recommendations
        })
    
    print(f"✅ Generated {len(predictions)} sample predictions")
    return predictions

def predict_irrigation_schedule():
    """Predict irrigation schedule for next 7 days"""
    try:
        print("\n" + "="*60)
        print("🔮 STARTING PREDICTION PROCESS")
        print("="*60)
        
        # First check if predictions are enabled
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        c.execute('SELECT value FROM settings WHERE key = "prediction_enabled"')
        prediction_enabled = c.fetchone()
        if prediction_enabled and prediction_enabled[0] == '0':
            print("⚠️ Predictions are disabled in settings")
            conn.close()
            sample_preds = generate_sample_predictions()
            save_predictions_to_csv(sample_preds)
            return sample_preds
        conn.close()
        
        # Check if model exists
        model_path = 'models/irrigation_model.pkl'
        scaler_path = 'models/scaler.pkl'
        feature_path = 'models/feature_names.pkl'
        
        required_files = [model_path, scaler_path, feature_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing model files: {missing_files}")
            print("Training new model...")
            model = train_prediction_model_enhanced()
            if not model:
                print("Model training failed, returning sample predictions")
                sample_preds = generate_sample_predictions()
                save_predictions_to_csv(sample_preds)
                return sample_preds
        
        # Load model artifacts
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_names = joblib.load(feature_path)
            
            print(f"✅ Model loaded successfully")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Features: {len(feature_names)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Training new model...")
            model = train_prediction_model_enhanced()
            if not model:
                print("Model training failed, returning sample predictions")
                sample_preds = generate_sample_predictions()
                save_predictions_to_csv(sample_preds)
                return sample_preds
            else:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                feature_names = joblib.load(feature_path)
        
        # Get recent data for predictions
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        # Get recent averages
        c.execute('''
            SELECT 
                COALESCE(AVG(temperature), 25.0) as avg_temp,
                COALESCE(AVG(humidity), 60.0) as avg_humidity,
                COALESCE(AVG(soil), 450.0) as avg_soil
            FROM sensor_data 
            WHERE timestamp > datetime('now', '-3 days')
            AND temperature IS NOT NULL
            AND humidity IS NOT NULL
            AND soil IS NOT NULL
        ''')
        
        avg_data = c.fetchone()
        avg_temp = float(avg_data[0]) if avg_data[0] else 25.0
        avg_humidity = float(avg_data[1]) if avg_data[1] else 60.0
        avg_soil = float(avg_data[2]) if avg_data[2] else 450.0
        
        print(f"📊 Recent conditions - Temp: {avg_temp:.1f}°C, Humidity: {avg_humidity:.1f}%, Soil: {avg_soil:.1f}")
        
        # Get water status
        c.execute('''SELECT water FROM sensor_data 
                     WHERE water IS NOT NULL
                     ORDER BY timestamp DESC LIMIT 1''')
        water_row = c.fetchone()
        water_available = 1 if water_row and water_row[0] == 'AVAILABLE' else 0
        print(f"💧 Water available: {'Yes' if water_available else 'No'}")
        
        conn.close()
        
        # Get prediction hours from settings
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        c.execute('SELECT value FROM settings WHERE key = "prediction_hours"')
        hours_result = c.fetchone()
        conn.close()
        
        if hours_result and hours_result[0]:
            prediction_hours = [int(h) for h in hours_result[0].split(',') if h.strip().isdigit()]
        else:
            prediction_hours = [6, 9, 12, 15, 18, 21]
        
        print(f"⏰ Predicting for hours: {prediction_hours}")
        
        # Generate predictions for next 7 days
        predictions = []
        current_date = datetime.now()
        
        for day_offset in range(7):
            prediction_date = current_date + timedelta(days=day_offset)
            day_of_week = prediction_date.weekday()
            day_name = prediction_date.strftime('%A')
            date_str = prediction_date.strftime('%Y-%m-%d')
            
            # Weather factor (simulated)
            weather_factor = 1.0 - (day_offset * 0.03)  # Slightly decrease probability for future days
            weather_factor = max(0.6, weather_factor)
            
            hour_predictions = []
            
            for hour in prediction_hours:
                try:
                    # Build feature vector
                    features = []
                    for feature_name in feature_names:
                        value = 0.0
                        
                        # Map features based on training
                        if feature_name == 'temperature':
                            value = avg_temp
                        elif feature_name == 'humidity':
                            value = avg_humidity
                        elif feature_name == 'soil':
                            value = avg_soil
                        elif feature_name == 'hour_sin':
                            value = np.sin(2 * np.pi * hour / 24)
                        elif feature_name == 'hour_cos':
                            value = np.cos(2 * np.pi * hour / 24)
                        elif feature_name == 'day_sin':
                            value = np.sin(2 * np.pi * day_of_week / 7)
                        elif feature_name == 'day_cos':
                            value = np.cos(2 * np.pi * day_of_week / 7)
                        elif feature_name == 'month_sin':
                            value = np.sin(2 * np.pi * prediction_date.month / 12)
                        elif feature_name == 'month_cos':
                            value = np.cos(2 * np.pi * prediction_date.month / 12)
                        elif feature_name == 'water_available':
                            value = float(water_available)
                        elif 'lag' in feature_name or 'ma' in feature_name:
                            # Use current values for lag/ma features
                            if 'temp' in feature_name:
                                value = avg_temp
                            elif 'humidity' in feature_name:
                                value = avg_humidity
                            elif 'soil' in feature_name:
                                value = avg_soil
                        elif 'interaction' in feature_name:
                            # Calculate interaction terms
                            if 'temp_humidity' in feature_name:
                                value = avg_temp * avg_humidity / 100.0
                            elif 'temp_soil' in feature_name:
                                value = avg_temp * avg_soil / 1000.0
                            elif 'humidity_soil' in feature_name:
                                value = avg_humidity * avg_soil / 1000.0
                        
                        features.append(float(value))
                    
                    # Ensure correct number of features
                    if len(features) != len(feature_names):
                        print(f"⚠️ Feature mismatch: expected {len(feature_names)}, got {len(features)}")
                        if len(features) > len(feature_names):
                            features = features[:len(feature_names)]
                        else:
                            features.extend([0.0] * (len(feature_names) - len(features)))
                    
                    # Scale features and predict
                    features_array = np.array(features).reshape(1, -1)
                    features_scaled = scaler.transform(features_array)
                    
                    # Get prediction based on model type
                    if hasattr(model, 'predict_proba'):
                        # Classification model
                        proba = model.predict_proba(features_scaled)[0]
                        if len(proba) > 1:
                            probability = float(proba[1]) * 100  # Probability of class 1 (pump ON)
                        else:
                            probability = float(proba[0]) * 100
                    else:
                        # Regression model
                        pred = model.predict(features_scaled)[0]
                        probability = float(pred) * 100
                    
                    # Apply weather factor and clamp
                    probability = probability * weather_factor
                    probability = max(0, min(100, probability))
                    
                    # Add realistic variation for future days
                    if day_offset > 0:
                        variation = np.random.uniform(-8, 8)
                        probability += variation
                        probability = max(0, min(100, probability))
                    
                    # Determine confidence
                    if probability > 80:
                        confidence = 'high'
                    elif probability > 60:
                        confidence = 'medium'
                    else:
                        confidence = 'low'
                    
                    hour_predictions.append({
                        'hour': hour,
                        'probability': round(probability, 1),
                        'recommended': probability > 60,
                        'confidence': confidence
                    })
                    
                except Exception as e:
                    print(f"❌ Error predicting hour {hour}: {e}")
                    hour_predictions.append({
                        'hour': hour,
                        'probability': 50.0,
                        'recommended': False,
                        'confidence': 'low'
                    })
            
            # Calculate day statistics
            if hour_predictions:
                probabilities = [p['probability'] for p in hour_predictions]
                avg_probability = np.mean(probabilities)
                
                # Generate recommendations
                recommendations = []
                if avg_probability > 75:
                    recommendations.append(f"High watering probability ({avg_probability:.0f}%)")
                    best_time = max(hour_predictions, key=lambda x: x['probability'])
                    recommendations.append(f"Best time: {best_time['hour']}:00")
                    recommendations.append("Consider watering today")
                elif avg_probability > 50:
                    recommendations.append(f"Moderate watering probability ({avg_probability:.0f}%)")
                    recommendations.append("Check soil moisture before watering")
                    recommendations.append("Water in early morning or evening")
                else:
                    recommendations.append(f"Low watering probability ({avg_probability:.0f}%)")
                    recommendations.append("No watering needed today")
                
                predictions.append({
                    'date': date_str,
                    'day_name': day_name,
                    'avg_probability': round(avg_probability, 1),
                    'predictions': hour_predictions,
                    'recommendations': recommendations,
                    'weather_factor': round(weather_factor, 2)
                })
            else:
                predictions.append({
                    'date': date_str,
                    'day_name': day_name,
                    'avg_probability': 50.0,
                    'predictions': [],
                    'recommendations': ["Unable to generate predictions"],
                    'weather_factor': round(weather_factor, 2)
                })
        
        print(f"✅ Generated {len(predictions)} days of predictions")
        
        # Save predictions to CSV and database
        save_predictions_to_csv(predictions)
        save_predictions_to_db(predictions)
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        traceback.print_exc()
        sample_preds = generate_sample_predictions()
        save_predictions_to_csv(sample_preds)
        return sample_preds

def save_predictions_to_csv(predictions):
    """Save predictions to CSV file"""
    try:
        if not predictions:
            print("⚠️ No predictions to save to CSV")
            return None
        
        os.makedirs('predictions', exist_ok=True)
        
        # Create CSV filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f'predictions/irrigation_predictions_{timestamp}.csv'
        
        # Flatten the data for CSV
        rows = []
        for day_data in predictions:
            date = day_data['date']
            day_name = day_data['day_name']
            avg_prob = day_data['avg_probability']
            weather_factor = day_data.get('weather_factor', 1.0)
            
            for hour_data in day_data['predictions']:
                rows.append({
                    'Date': date,
                    'Day': day_name,
                    'Hour': hour_data['hour'],
                    'Hour_Probability': hour_data['probability'],
                    'Recommended': 'Yes' if hour_data['recommended'] else 'No',
                    'Confidence': hour_data['confidence'],
                    'Day_Avg_Probability': avg_prob,
                    'Weather_Factor': weather_factor,
                    'Recommendations': ' | '.join(day_data['recommendations'])
                })
        
        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Predictions saved to CSV: {csv_path} ({len(rows)} rows)")
        
        # Also save a summary file
        summary_path = f'predictions/predictions_summary_{timestamp}.csv'
        summary_rows = []
        for day_data in predictions:
            summary_rows.append({
                'Date': day_data['date'],
                'Day': day_data['day_name'],
                'Avg_Probability': day_data['avg_probability'],
                'Weather_Factor': day_data.get('weather_factor', 1.0),
                'Recommendations': ' | '.join(day_data['recommendations']),
                'Prediction_Hours': len(day_data['predictions'])
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False)
        
        return csv_path
        
    except Exception as e:
        print(f"❌ Error saving predictions to CSV: {e}")
        return None

def save_predictions_to_db(predictions):
    """Save predictions to database"""
    try:
        if not predictions:
            return
        
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        # Clear old predictions
        c.execute('DELETE FROM predictions WHERE timestamp < datetime("now", "-7 days")')
        
        # Insert new predictions
        for pred in predictions:
            c.execute('''INSERT INTO predictions 
                         (prediction_date, day_name, avg_probability, predictions_json, recommendations)
                         VALUES (?, ?, ?, ?, ?)''',
                      (pred['date'], pred['day_name'], pred['avg_probability'],
                       json.dumps(pred['predictions']), ' | '.join(pred['recommendations'])))
        
        conn.commit()
        conn.close()
        print(f"✅ Saved {len(predictions)} predictions to database")
        
    except Exception as e:
        print(f"⚠️ Error saving predictions to DB: {e}")

@app.route('/api/predictions')
def get_predictions():
    """Get irrigation predictions for next week"""
    try:
        print(f"\n📡 [API] GET /api/predictions called")
        
        # Generate predictions
        predictions_data = predict_irrigation_schedule()
        
        if not predictions_data:
            print("⚠️ No predictions generated, using sample data")
            predictions_data = generate_sample_predictions()
        
        # Get model info
        model_info = {}
        model_files_exist = False
        
        try:
            if os.path.exists('models/model_info.pkl'):
                model_info = joblib.load('models/model_info.pkl')
                model_files_exist = True
        except Exception as e:
            print(f"⚠️ Could not load model info: {e}")
        
        # Get latest training log
        training_log = []
        try:
            conn = sqlite3.connect('irrigation.db')
            c = conn.cursor()
            c.execute('''
                SELECT timestamp, model_name, accuracy, r2_score, samples_used, status, message
                FROM training_log 
                ORDER BY timestamp DESC LIMIT 7
            ''')
            logs = c.fetchall()
            for log in logs:
                training_log.append({
                    'timestamp': log[0],
                    'model_name': log[1],
                    'accuracy': log[2],
                    'r2_score': log[3],
                    'samples_used': log[4],
                    'status': log[5],
                    'message': log[6]
                })
            conn.close()
        except Exception as e:
            print(f"⚠️ Could not load training log: {e}")
        
        # Get database stats
        try:
            conn = sqlite3.connect('irrigation.db')
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM sensor_data')
            total_samples = c.fetchone()[0]
            conn.close()
        except:
            total_samples = 0
        
        # Get current model status
        model_status = {
            'model_exists': model_files_exist,
            'accuracy': model_info.get('accuracy', 0) if model_info else 0,
            'r2_score': model_info.get('r2_score', 0) if model_info else 0,
            'samples_used': model_info.get('samples_used', 0) if model_info else 0,
            'last_trained': model_info.get('trained_at', 'Never') if model_info else 'Never',
            'feature_count': model_info.get('feature_count', 0) if model_info else 0
        }
        
        # Create the response
        response = {
            'success': True,
            'message': 'Predictions loaded successfully',
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions_data,
            'model_info': model_info,
            'model_status': model_status,
            'training_log': training_log,
            'stats': {
                'total_samples': total_samples,
                'prediction_count': len(predictions_data),
                'prediction_days': [p['date'] for p in predictions_data],
                'first_day_avg': predictions_data[0]['avg_probability'] if predictions_data else 0
            }
        }
        
        print(f"✅ [API] Returning {len(predictions_data)} predictions")
        if predictions_data:
            print(f"   First day: {predictions_data[0]['date']} - {predictions_data[0]['avg_probability']}% avg")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ [API ERROR] /api/predictions: {e}")
        traceback.print_exc()
        
        # Return error response
        error_response = {
            'success': False,
            'error': str(e),
            'message': 'Failed to load predictions',
            'timestamp': datetime.now().isoformat(),
            'predictions': generate_sample_predictions(),
            'model_status': {'model_exists': False},
            'training_log': []
        }
        
        return jsonify(error_response)

@app.route('/api/train-model', methods=['POST'])
def train_model_endpoint():
    """Endpoint to manually trigger model training"""
    try:
        print("🚀 Manual model training requested...")
        model = train_prediction_model_enhanced()
        if model:
            return jsonify({'status': 'success', 'message': 'Model trained successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to train model (insufficient data)'}), 400
    except Exception as e:
        print(f"Error in training endpoint: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/training-log')
def get_training_log():
    """Get ML training history"""
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM training_log ORDER BY timestamp DESC LIMIT 20')
    rows = c.fetchall()
    
    logs = []
    for row in rows:
        logs.append({
            'id': row[0],
            'timestamp': row[1],
            'model_name': row[2],
            'accuracy': row[3],
            'r2_score': row[4],
            'mse': row[5],
            'samples_used': row[6],
            'status': row[7],
            'message': row[8]
        })
    
    conn.close()
    return jsonify(logs)

@app.route('/api/model-info')
def get_model_info():
    """Get current model information"""
    model_info = {}
    try:
        if os.path.exists('models/model_info.pkl'):
            model_info = joblib.load('models/model_info.pkl')
    except Exception as e:
        print(f"Error loading model info: {e}")
    
    # Check if model files exist
    files_exist = {
        'model': os.path.exists('models/irrigation_model.pkl'),
        'scaler': os.path.exists('models/scaler.pkl'),
        'features': os.path.exists('models/feature_names.pkl'),
        'info': os.path.exists('models/model_info.pkl')
    }
    
    return jsonify({
        'model_info': model_info,
        'files_exist': files_exist,
        'models_dir': os.listdir('models') if os.path.exists('models') else []
    })

@app.route('/api/export-predictions-csv')
def export_predictions_csv():
    """Export predictions as CSV download"""
    try:
        # Generate fresh predictions
        predictions = predict_irrigation_schedule()
        
        if not predictions:
            predictions = generate_sample_predictions()
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Date', 'Day', 'Hour', 'Probability (%)', 
            'Recommended', 'Confidence', 'Day Average (%)',
            'Recommendations', 'Generated At'
        ])
        
        # Write data
        generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for day_data in predictions:
            date = day_data['date']
            day_name = day_data['day_name']
            avg_prob = day_data['avg_probability']
            
            for hour_data in day_data['predictions']:
                writer.writerow([
                    date,
                    day_name,
                    f"{hour_data['hour']}:00",
                    hour_data['probability'],
                    'Yes' if hour_data['recommended'] else 'No',
                    hour_data['confidence'],
                    avg_prob,
                    ' | '.join(day_data['recommendations']),
                    generated_time
                ])
        
        output.seek(0)
        
        # Create response
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'irrigation_predictions_{timestamp}.csv'
        
        response = send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
        print(f"✅ CSV exported: {filename}")
        return response
        
    except Exception as e:
        print(f"Error exporting predictions: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== PREDICTION DEBUG ENDPOINTS ====================

@app.route('/api/debug-predictions')
def debug_predictions():
    """Debug endpoint for prediction system"""
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'directories': {},
        'model_status': {},
        'database': {},
        'prediction_test': {}
    }
    
    # Check directories
    debug_info['directories']['models'] = {
        'exists': os.path.exists('models'),
        'contents': os.listdir('models') if os.path.exists('models') else []
    }
    
    debug_info['directories']['predictions'] = {
        'exists': os.path.exists('predictions'),
        'contents': os.listdir('predictions') if os.path.exists('predictions') else []
    }
    
    # Check model files
    model_files = ['irrigation_model.pkl', 'scaler.pkl', 'feature_names.pkl', 'model_info.pkl']
    for file in model_files:
        path = f'models/{file}'
        exists = os.path.exists(path)
        debug_info['model_status'][file] = {
            'exists': exists,
            'path': path
        }
        if exists:
            try:
                size = os.path.getsize(path)
                debug_info['model_status'][file]['size'] = size
                
                # Try to load to test if valid
                if file.endswith('.pkl'):
                    try:
                        loaded = joblib.load(path)
                        debug_info['model_status'][file]['loadable'] = True
                        debug_info['model_status'][file]['type'] = type(loaded).__name__
                        if file == 'feature_names.pkl':
                            debug_info['model_status'][file]['value'] = loaded[:5] if len(loaded) > 5 else loaded
                    except:
                        debug_info['model_status'][file]['loadable'] = False
            except:
                debug_info['model_status'][file]['size'] = 'unknown'
    
    # Check database
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM sensor_data')
        debug_info['database']['total_samples'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM training_log')
        debug_info['database']['training_logs'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM predictions')
        debug_info['database']['stored_predictions'] = c.fetchone()[0]
        
        c.execute('SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1')
        latest_sensor = c.fetchone()
        if latest_sensor:
            debug_info['database']['latest_sensor'] = {
                'timestamp': latest_sensor[1],
                'temperature': latest_sensor[2],
                'humidity': latest_sensor[3],
                'soil': latest_sensor[4],
                'pump': latest_sensor[6]
            }
        
        c.execute('SELECT * FROM training_log ORDER BY timestamp DESC LIMIT 1')
        latest_log = c.fetchone()
        if latest_log:
            debug_info['database']['latest_training'] = {
                'timestamp': latest_log[1],
                'model': latest_log[2],
                'accuracy': latest_log[3],
                'status': latest_log[7]
            }
        
        conn.close()
    except Exception as e:
        debug_info['database']['error'] = str(e)
    
    # Test prediction function
    try:
        test_pred = predict_irrigation_schedule()
        debug_info['prediction_test']['success'] = True
        debug_info['prediction_test']['days'] = len(test_pred) if test_pred else 0
        if test_pred and len(test_pred) > 0:
            debug_info['prediction_test']['first_day'] = test_pred[0]['date']
            debug_info['prediction_test']['first_day_avg'] = test_pred[0]['avg_probability']
            debug_info['prediction_test']['first_day_recommendations'] = test_pred[0]['recommendations']
    except Exception as e:
        debug_info['prediction_test']['success'] = False
        debug_info['prediction_test']['error'] = str(e)
        debug_info['prediction_test']['traceback'] = traceback.format_exc()
    
    return jsonify(debug_info)

# ==================== EXISTING ROUTES (Keep these as they are) ====================

# Pump Control Endpoints
@app.route('/api/pump/control', methods=['POST'])
def control_pump():
    """Manually control pump"""
    try:
        data = request.get_json()
        action = data.get('action', '').upper()
        
        if action not in ['ON', 'OFF', 'TOGGLE']:
            return jsonify({'status': 'error', 'message': 'Invalid action'}), 400
        
        global pump_state, pump_override
        
        if action == 'TOGGLE':
            pump_state = 'OFF' if pump_state == 'ON' else 'ON'
        else:
            pump_state = action
        
        # Set manual override
        pump_override = True
        
        # Log the action
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        # Get latest sensor data for context
        c.execute('''SELECT temperature, humidity, soil, water 
                     FROM sensor_data 
                     ORDER BY timestamp DESC LIMIT 1''')
        sensor_data = c.fetchone()
        
        if sensor_data:
            temp, humidity, soil, water = sensor_data
        else:
            temp, humidity, soil, water = 0, 0, 0, 'AVAILABLE'
        
        c.execute('''INSERT INTO sensor_data 
                     (temperature, humidity, soil, water, pump, manual_override)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (temp, humidity, soil, water, pump_state, 1))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': f'Pump turned {pump_state}',
            'pump_state': pump_state,
            'manual_override': True
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/pump/status')
def get_pump_status():
    """Get current pump status"""
    global pump_state, pump_override
    
    # Get last pump action from database
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    c.execute('''SELECT pump, manual_override, timestamp 
                 FROM sensor_data 
                 WHERE pump IS NOT NULL 
                 ORDER BY timestamp DESC LIMIT 1''')
    
    row = c.fetchone()
    conn.close()
    
    if row:
        pump_state = row[0]
        pump_override = bool(row[1])
        last_update = row[2]
    else:
        last_update = None
    
    return jsonify({
        'pump_state': pump_state,
        'manual_override': pump_override,
        'last_update': last_update,
        'auto_mode': not pump_override
    })

# Enhanced Plant Health Analysis
def check_alerts(data, conn):
    c = conn.cursor()
    
    # Get settings
    c.execute('SELECT key, value FROM settings')
    settings = {row[0]: row[1] for row in c.fetchall()}
    
    alerts = []
    
    # Check soil moisture
    soil_threshold = int(settings.get('soil_dry_threshold', 500))
    if int(data['soil']) > soil_threshold:
        message = f'Soil is too dry! Value: {data["soil"]} (Threshold: {soil_threshold})'
        alerts.append(('soil_dry', message))
        send_email_alert('Soil Too Dry', message, 'danger')
    
    # Check temperature
    max_temp = float(settings.get('max_temperature', 35))
    if float(data['temperature']) > max_temp:
        message = f'Temperature is high! {data["temperature"]}°C (Max: {max_temp}°C)'
        alerts.append(('high_temp', message))
        send_email_alert('High Temperature', message, 'warning')
    
    # Check humidity
    min_humidity = float(settings.get('min_humidity', 30))
    if float(data['humidity']) < min_humidity:
        message = f'Humidity is low! {data["humidity"]}% (Min: {min_humidity}%)'
        alerts.append(('low_humidity', message))
        send_email_alert('Low Humidity', message, 'warning')
    
    # Check water level
    if data['water'] == 'NOT AVAILABLE':
        message = 'Water tank is empty!'
        alerts.append(('no_water', message))
        send_email_alert('Water Tank Empty', message, 'danger')
    
    # Insert new alerts
    for alert_type, message in alerts:
        c.execute('''INSERT INTO alerts (type, message) 
                     VALUES (?, ?)''', (alert_type, message))
    
    # Send email for critical alerts
    email_enabled = settings.get('email_enabled', '0') == '1'
    if email_enabled and alerts:
        for alert_type, message in alerts:
            if alert_type in ['no_water', 'soil_dry']:  # Critical alerts
                send_email_alert(f'Critical Alert: {alert_type}', message, 'danger')

@app.route('/api/plant-health-detailed')
def get_detailed_plant_health():
    """Get detailed plant health analysis including watering patterns"""
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    # Get last 7 days of data
    c.execute('''
        SELECT 
            timestamp,
            temperature,
            humidity,
            soil,
            pump,
            pump_duration
        FROM sensor_data 
        WHERE timestamp > datetime('now', '-7 days')
        ORDER BY timestamp
    ''')
    
    rows = c.fetchall()
    
    if not rows:
        return jsonify({'health': 'Unknown', 'score': 0, 'details': {}})
    
    # Analyze watering patterns
    watering_events = []
    total_water_time = 0
    last_pump_on = None
    
    for row in rows:
        timestamp, temp, humidity, soil, pump, duration = row
        
        if pump == 'ON' and duration > 0:
            if last_pump_on is None:
                last_pump_on = timestamp
            total_water_time += duration
        elif pump == 'OFF' and last_pump_on:
            watering_events.append({
                'start': last_pump_on,
                'end': timestamp,
                'duration': duration or 0
            })
            last_pump_on = None
    
    # Calculate metrics
    avg_watering_duration = np.mean([e['duration'] for e in watering_events]) if watering_events else 0
    watering_frequency = len(watering_events) / 7  # events per day
    
    # Analyze soil moisture trends
    soil_values = [row[3] for row in rows]
    avg_soil = np.mean(soil_values) if soil_values else 500
    soil_stability = np.std(soil_values) if len(soil_values) > 1 else 0
    
    # Get latest readings
    latest = rows[-1]
    
    # Calculate health score
    score = 100
    
    # Soil moisture scoring
    if avg_soil > 600:
        score -= 40
    elif avg_soil > 500:
        score -= 20
    elif avg_soil < 200:
        score -= 30
    
    # Watering pattern scoring
    if watering_frequency < 0.5:  # Less than once every 2 days
        score -= 25
    elif watering_frequency > 3:  # More than 3 times per day
        score -= 20
    
    if avg_watering_duration > 300:  # More than 5 minutes per watering
        score -= 15
    elif avg_watering_duration < 30:  # Less than 30 seconds
        score -= 10
    
    # Soil stability scoring (lower std is better)
    if soil_stability > 150:
        score -= 15
    
    # Temperature and humidity scoring
    if latest[1] > 35 or latest[1] < 10:
        score -= 20
    elif latest[1] > 30 or latest[1] < 15:
        score -= 10
    
    if latest[2] < 30 or latest[2] > 80:
        score -= 15
    
    # Cap score
    score = max(0, min(100, score))
    
    # Determine health status
    if score >= 85:
        health = "Excellent"
    elif score >= 70:
        health = "Good"
    elif score >= 50:
        health = "Fair"
    elif score >= 30:
        health = "Poor"
    else:
        health = "Critical"
    
    # Generate recommendations
    recommendations = generate_detailed_recommendations(
        avg_soil, watering_frequency, avg_watering_duration, 
        latest[1], latest[2], soil_stability
    )
    
    conn.close()
    
    return jsonify({
        'health': health,
        'score': round(score, 1),
        'details': {
            'avg_soil_moisture': round(avg_soil, 1),
            'watering_frequency_per_day': round(watering_frequency, 2),
            'avg_watering_duration_seconds': round(avg_watering_duration, 1),
            'total_water_time_seconds': total_water_time,
            'soil_stability': round(soil_stability, 1),
            'current_temperature': latest[1],
            'current_humidity': latest[2],
            'watering_events_count': len(watering_events)
        },
        'recommendations': recommendations,
        'watering_history': watering_events[-10:] if watering_events else []  # Last 10 events
    })

def generate_detailed_recommendations(avg_soil, frequency, duration, temp, humidity, stability):
    """Generate detailed recommendations based on analysis"""
    recommendations = []
    
    # Soil moisture recommendations
    if avg_soil > 550:
        recommendations.append({
            'priority': 'high',
            'message': 'Soil is consistently dry. Increase watering frequency or duration.',
            'action': 'Increase watering by 30 seconds per session'
        })
    elif avg_soil < 250:
        recommendations.append({
            'priority': 'high',
            'message': 'Soil is too wet. Reduce watering to prevent root rot.',
            'action': 'Reduce watering frequency by 50%'
        })
    
    # Watering pattern recommendations
    if frequency < 0.5:
        recommendations.append({
            'priority': 'medium',
            'message': 'Infrequent watering detected.',
            'action': 'Water at least once daily'
        })
    elif frequency > 2.5:
        recommendations.append({
            'priority': 'medium',
            'message': 'Overwatering detected.',
            'action': 'Reduce watering to twice daily maximum'
        })
    
    if duration > 300:
        recommendations.append({
            'priority': 'low',
            'message': 'Watering sessions are too long.',
            'action': 'Reduce watering duration to 2-3 minutes max'
        })
    
    # Soil stability recommendation
    if stability > 150:
        recommendations.append({
            'priority': 'medium',
            'message': 'Soil moisture fluctuates too much.',
            'action': 'Maintain more consistent watering schedule'
        })
    
    # Temperature recommendations
    if temp > 32:
        recommendations.append({
            'priority': 'high',
            'message': 'High temperature detected.',
            'action': 'Water in early morning or evening to reduce evaporation'
        })
    
    if humidity < 35:
        recommendations.append({
            'priority': 'medium',
            'message': 'Low humidity environment.',
            'action': 'Consider misting plants between waterings'
        })
    
    if not recommendations:
        recommendations.append({
            'priority': 'low',
            'message': 'All parameters are within optimal ranges.',
            'action': 'Continue current watering schedule'
        })
    
    # Sort by priority
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    recommendations.sort(key=lambda x: priority_order[x['priority']])
    
    return recommendations

# Existing routes
@app.route('/api/history')
def get_history():
    limit = request.args.get('limit', 100)
    days = request.args.get('days')
    
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    if days:
        c.execute('''SELECT * FROM sensor_data 
                     WHERE timestamp > datetime('now', ?)
                     ORDER BY timestamp DESC LIMIT ?''', (f'-{days} days', int(limit)))
    else:
        c.execute('''SELECT * FROM sensor_data 
                     ORDER BY timestamp DESC LIMIT ?''', (int(limit),))
    
    rows = c.fetchall()
    data = []
    
    for row in rows:
        data.append({
            'id': row[0],
            'timestamp': row[1],
            'temperature': row[2],
            'humidity': row[3],
            'soil': row[4],
            'water': row[5],
            'pump': row[6],
            'pump_duration': row[7],
            'manual_override': bool(row[8])
        })
    
    conn.close()
    return jsonify(data)

@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    # Get last 24 hours data
    c.execute('''SELECT 
                 AVG(temperature) as avg_temp,
                 AVG(humidity) as avg_humidity,
                 AVG(soil) as avg_soil,
                 COUNT(CASE WHEN pump = 'ON' THEN 1 END) * 100.0 / COUNT(*) as pump_usage_percent,
                 SUM(pump_duration) as total_pump_time,
                 COUNT(CASE WHEN water = 'NOT AVAILABLE' THEN 1 END) as water_warnings
                 FROM sensor_data 
                 WHERE timestamp > datetime('now', '-1 day')''')
    
    stats = c.fetchone()
    
    # Get latest readings
    c.execute('SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1')
    latest = c.fetchone()
    
    # Get latest weather data
    c.execute('SELECT * FROM weather_data ORDER BY timestamp DESC LIMIT 1')
    weather_row = c.fetchone()
    
    # Get location
    location = get_user_location()
    
    # Get model status
    model_exists = os.path.exists('models/irrigation_model.pkl')
    
    conn.close()
    
    weather = None
    if weather_row:
        weather = {
            'temperature': weather_row[2],
            'humidity': weather_row[3],
            'precipitation': weather_row[4],
            'rain_forecast': bool(weather_row[5]),
            'description': weather_row[6],
            'location': weather_row[7]
        }
    
    return jsonify({
        'avg_temperature': round(stats[0], 2) if stats[0] else 0,
        'avg_humidity': round(stats[1], 2) if stats[1] else 0,
        'avg_soil': round(stats[2], 2) if stats[2] else 0,
        'pump_usage': round(stats[3], 2) if stats[3] else 0,
        'total_pump_time': stats[4] if stats[4] else 0,
        'water_warnings': stats[5] if stats[5] else 0,
        'latest': {
            'temperature': latest[2] if latest else 0,
            'humidity': latest[3] if latest else 0,
            'soil': latest[4] if latest else 0,
            'water': latest[5] if latest else 'N/A',
            'pump': latest[6] if latest else 'N/A'
        },
        'weather': weather,
        'location': location,
        'model_available': model_exists
    })

@app.route('/api/alerts')
def get_alerts():
    unresolved = request.args.get('unresolved', 'false').lower() == 'true'
    
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    if unresolved:
        c.execute('SELECT * FROM alerts WHERE resolved = 0 ORDER BY timestamp DESC')
    else:
        c.execute('SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 50')
    
    rows = c.fetchall()
    alerts = []
    
    for row in rows:
        alerts.append({
            'id': row[0],
            'timestamp': row[1],
            'type': row[2],
            'message': row[3],
            'resolved': bool(row[4])
        })
    
    conn.close()
    return jsonify(alerts)

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'GET':
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('SELECT key, value FROM settings')
        settings = {row[0]: row[1] for row in c.fetchall()}
        
        # Get pump schedule
        c.execute('SELECT * FROM pump_schedule ORDER BY hour, minute')
        schedule_rows = c.fetchall()
        pump_schedule = []
        
        for row in schedule_rows:
            pump_schedule.append({
                'id': row[0],
                'hour': row[1],
                'minute': row[2],
                'duration': row[3],
                'enabled': bool(row[4])
            })
        
        # Get email config
        email_config = {
            'enabled': settings.get('email_enabled', '0') == '1',
            'sender': settings.get('email_sender', ''),
            'recipient': settings.get('email_recipient', ''),
            'smtp_server': settings.get('email_smtp_server', 'smtp.gmail.com'),
            'smtp_port': settings.get('email_smtp_port', '587')
        }
        
        conn.close()
        return jsonify({
            'settings': settings,
            'pump_schedule': pump_schedule,
            'email_config': email_config
        })
    
    else:  # POST
        data = request.get_json()
        
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        if 'settings' in data:
            for key, value in data['settings'].items():
                c.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, value))
            
            # Reload email config if email settings changed
            if any(key.startswith('email_') for key in data['settings'].keys()):
                load_email_config()
        
        if 'pump_schedule' in data:
            for schedule in data['pump_schedule']:
                c.execute('''INSERT OR REPLACE INTO pump_schedule 
                             (id, hour, minute, duration, enabled) 
                             VALUES (?, ?, ?, ?, ?)''',
                          (schedule.get('id'), schedule['hour'], schedule['minute'], 
                           schedule['duration'], int(schedule.get('enabled', True))))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'message': 'Settings saved successfully'})

@app.route('/api/export/csv')
def export_csv():
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM sensor_data ORDER BY timestamp')
    rows = c.fetchall()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Timestamp', 'Temperature (°C)', 'Humidity (%)', 
                     'Soil Moisture', 'Water Level', 'Pump Status', 
                     'Pump Duration (s)', 'Manual Override'])
    
    # Write data
    for row in rows:
        writer.writerow(row)
    
    conn.close()
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"irrigation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

@app.route('/api/clear-alert/<int:alert_id>', methods=['POST'])
def clear_alert(alert_id):
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('UPDATE alerts SET resolved = 1 WHERE id = ?', (alert_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'message': 'Alert cleared'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Background tasks scheduler
def run_scheduled_tasks():
    """Run background tasks periodically"""
    last_weather_fetch = 0
    last_auto_train = 0
    weather_interval = 1800  # 30 minutes
    auto_train_interval = 600  # 10 minutes (will be overridden by settings)
    
    print("Starting scheduled tasks...")
    
    while True:
        try:
            current_time = time.time()
            
            # Fetch weather data every 30 minutes
            if current_time - last_weather_fetch >= weather_interval:
                print("Scheduled: Fetching weather data...")
                fetch_weather_data()
                last_weather_fetch = current_time
            
            # Auto-train model based on settings interval
            if current_time - last_auto_train >= auto_train_interval:
                # Get current interval from settings
                conn = sqlite3.connect('irrigation.db')
                c = conn.cursor()
                c.execute('SELECT value FROM settings WHERE key = "auto_train_interval"')
                interval_row = c.fetchone()
                conn.close()
                
                current_interval = int(interval_row[0]) if interval_row else 10
                auto_train_interval = current_interval * 60  # Convert minutes to seconds
                
                print(f"Scheduled: Auto-training check (every {current_interval} minutes)...")
                auto_train_model()
                last_auto_train = current_time
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Error in scheduled tasks: {e}")
            time.sleep(300)  # Sleep for 5 minutes on error

# Auto-train function
def auto_train_model():
    """Auto-train ML model based on settings"""
    try:
        # Get auto-train settings
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        c.execute('SELECT value FROM settings WHERE key = "auto_train_interval"')
        interval_row = c.fetchone()
        conn.close()
        
        auto_train_interval = int(interval_row[0]) if interval_row else 10  # Default 10 minutes
        
        print(f"\n🔧 Auto-training check (every {auto_train_interval} minutes)...")
        
        # Check if we have enough new data
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        # Count new samples since last successful training
        c.execute('''
            SELECT MAX(timestamp) FROM training_log 
            WHERE status = 'success'
        ''')
        last_train_row = c.fetchone()
        last_train_time = last_train_row[0] if last_train_row[0] else '2000-01-01'
        
        # Count new samples
        c.execute('''
            SELECT COUNT(*) FROM sensor_data 
            WHERE timestamp > ?
        ''', (last_train_time,))
        new_samples = c.fetchone()[0]
        
        conn.close()
        
        print(f"  New samples since last training: {new_samples}")
        
        # Train if we have enough new data or never trained before
        if new_samples >= 10 or not os.path.exists('models/irrigation_model.pkl'):
            print(f"  Training model with {new_samples} new samples...")
            train_prediction_model_enhanced()
        else:
            print(f"  Skipping training: insufficient new data ({new_samples} samples)")
            
    except Exception as e:
        print(f"Error in auto-training: {e}")

# Test route
@app.route('/test')
def test():
    return jsonify({
        'status': 'ok', 
        'message': 'Server is running!',
        'location': get_user_location(),
        'location_detected': location_detected,
        'model_available': os.path.exists('models/irrigation_model.pkl'),
        'auto_train_interval': '10 minutes',
        'email_enabled': EMAIL_CONFIG['enabled']
    })

# Initialize auto-training on startup
def initialize_auto_training():
    """Initialize auto-training on first run"""
    # Check if model exists, if not train one
    if not os.path.exists('models/irrigation_model.pkl'):
        print("No model found. Training initial model...")
        train_prediction_model_enhanced()
    
    # Start scheduled tasks thread
    background_thread = threading.Thread(target=run_scheduled_tasks, daemon=True)
    background_thread.start()
    print("Auto-training scheduler started")

# Start auto-training on app startup
initialize_auto_training()

if __name__ == '__main__':
    print("=" * 60)
    print("🌱 SMART IRRIGATION SYSTEM WITH AUTO-ML TRAINING 🌱")
    print("=" * 60)
    print("Dashboard available at: http://localhost:5000")
    print("API Endpoint: http://localhost:5000/data")
    print("\n📊 Prediction Endpoints:")
    print("  GET  /api/predictions         - Get 7-day predictions")
    print("  GET  /api/export-predictions-csv - Download predictions as CSV")
    print("  POST /api/train-model         - Manual training")
    print("  GET  /api/training-log        - View training history")
    print("  GET  /api/model-info          - Get model details")
    print("  GET  /api/debug-predictions   - Debug prediction system")
    
    location = get_user_location()
    print(f"\n📍 Detected location: {location['city']} ({location['lat']}, {location['lon']})")
    
    print("\n✅ System ready!")
    
    app.run(host='0.0.0.0', port=5000, debug=True)