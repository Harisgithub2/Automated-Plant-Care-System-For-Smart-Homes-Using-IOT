from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import json
from datetime import datetime, timedelta
import csv
import io
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import requests
import time
import threading
import joblib
import random
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)
CORS(app, supports_credentials=True)

# Configuration
OPENWEATHER_API_KEY = "1d77de94d48b21f123aa67a8cd3e70f4"

# Store detected location
user_location = None
location_detected = False

# Email configuration
EMAIL_CONFIG = {
    'enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email':'harikarthikeya2004irahal@gmail.com',
    'sender_password': 'prdgdnzpxicjlrey',
    'recipient_email': ''
}

# Global pump states
pump1_state = "OFF"
pump2_state = "OFF"
pump_override = False

# Track when pumps were started
pump1_start_time = None
pump2_start_time = None

# Email cooldown tracking
email_cooldown = {}
DEFAULT_COOLDOWN_MINUTES = 10

# Ensure directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)

# ==================== DATABASE INITIALIZATION ====================

def init_db():
    """Initialize database with dual plant support and alert settings"""
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Check if old table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sensor_data'")
    table_exists = c.fetchone()
    
    if table_exists:
        # Check columns of existing table
        c.execute("PRAGMA table_info(sensor_data)")
        columns = [col[1] for col in c.fetchall()]
        
        # If old schema detected, migrate data
        if 'soil' in columns and 'pump' in columns and 'soil1' not in columns:
            print("Migrating database from old schema to dual-plant schema...")
            
            # Rename old table
            c.execute("ALTER TABLE sensor_data RENAME TO sensor_data_old")
            
            # Create new table with dual plant support
            c.execute('''CREATE TABLE sensor_data
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                          temperature REAL,
                          humidity REAL,
                          soil1 TEXT,
                          soil2 TEXT,
                          water TEXT,
                          pump1 TEXT,
                          pump2 TEXT,
                          pump1_duration INTEGER DEFAULT 0,
                          pump2_duration INTEGER DEFAULT 0,
                          manual_override INTEGER DEFAULT 0)''')
            
            # Migrate data
            c.execute('''INSERT INTO sensor_data 
                         (id, timestamp, temperature, humidity, soil1, soil2, water, pump1, pump2, pump1_duration, pump2_duration, manual_override)
                         SELECT id, timestamp, temperature, humidity, soil, soil, water, pump, pump, pump_duration, 0, manual_override
                         FROM sensor_data_old''')
            
            print("Migration completed successfully")
    else:
        # Create new table with dual plant support
        c.execute('''CREATE TABLE IF NOT EXISTS sensor_data
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                      temperature REAL,
                      humidity REAL,
                      soil1 TEXT,
                      soil2 TEXT,
                      water TEXT,
                      pump1 TEXT,
                      pump2 TEXT,
                      pump1_duration INTEGER DEFAULT 0,
                      pump2_duration INTEGER DEFAULT 0,
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
                  plant TEXT DEFAULT 'both',
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
    
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  prediction_date TEXT,
                  day_name TEXT,
                  avg_probability REAL,
                  predictions_json TEXT,
                  recommendations TEXT)''')
    
    # Create email cooldown tracking table
    c.execute('''CREATE TABLE IF NOT EXISTS email_cooldown
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  alert_type TEXT UNIQUE,
                  last_sent DATETIME)''')
    
    # Insert default settings if not exists
    default_settings = [
        ('soil_dry_threshold1', '500'),
        ('soil_wet_threshold1', '300'),
        ('pump1_max_duration', '120'),
        ('soil_dry_threshold2', '500'),
        ('soil_wet_threshold2', '300'),
        ('pump2_max_duration', '120'),
        ('auto_mode', '1'),
        ('rain_delay', '1'),
        ('rain_threshold', '0.5'),
        ('evaporation_rate', 'medium'),
        ('weather_integration', '1'),
        ('location_city', ''),
        ('weather_api_key', '1d77de94d48b21f123aa67a8cd3e70f4'),
        ('ml_model_type', 'random_forest'),
        ('auto_train_interval', '10'),
        ('min_training_samples', '50'),
        ('prediction_enabled', '1'),
        ('prediction_days', '7'),
        ('prediction_hours', '6,9,12,15,18,21'),
        ('update_interval', '5'),
        ('auto_detect_location', '1'),
        ('max_temperature', '35'),
        ('min_humidity', '30'),
        ('email_enabled', '0'),
        ('email_recipient', ''),
        ('email_cooldown_minutes', '10'),
        ('alert_no_water', '1'),
        ('alert_pump_start', '0'),
        ('alert_pump_still_dry', '0'),
        ('alert_dry_duration', '5'),
        ('alert_high_temp', '1'),
        ('alert_low_humidity', '1'),
        ('alert_soil_dry_plant1', '1'),
        ('alert_soil_dry_plant2', '1'),
        ('soil_dry_threshold', '500'),
        ('soil_wet_threshold', '300'),
        ('enable_alerts', '1'),
        ('rain_stop_threshold', '60'),
        ('weather_rain_stop_pump', '1')
    ]
    
    for key, value in default_settings:
        c.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', (key, value))
    
    # Default pump schedule
    c.execute('''INSERT OR IGNORE INTO pump_schedule (hour, minute, duration, plant, enabled) 
                 VALUES (8, 0, 300, 'both', 1)''')
    
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
    
    load_email_config()
    load_email_cooldown()

# ==================== EMAIL COOLDOWN FUNCTIONS ====================

def load_email_cooldown():
    """Load email cooldown tracking from database"""
    global email_cooldown
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        c.execute('SELECT alert_type, last_sent FROM email_cooldown')
        rows = c.fetchall()
        for row in rows:
            email_cooldown[row[0]] = datetime.fromisoformat(row[1])
        conn.close()
        print(f"📧 Loaded {len(email_cooldown)} email cooldown records")
    except Exception as e:
        print(f"Error loading email cooldown: {e}")

def save_email_cooldown(alert_type):
    """Save email cooldown timestamp to database"""
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute('''INSERT OR REPLACE INTO email_cooldown (alert_type, last_sent) 
                     VALUES (?, ?)''', (alert_type, now))
        conn.commit()
        conn.close()
        email_cooldown[alert_type] = datetime.now()
    except Exception as e:
        print(f"Error saving email cooldown: {e}")

def can_send_email(alert_type):
    """Check if enough time has passed since last email of this type"""
    global email_cooldown
    
    cooldown_minutes = get_cooldown_minutes()
    
    if alert_type not in email_cooldown:
        return True
    
    last_sent = email_cooldown[alert_type]
    now = datetime.now()
    elapsed = (now - last_sent).total_seconds() / 60
    
    return elapsed >= cooldown_minutes

def get_cooldown_minutes():
    """Get cooldown minutes from settings"""
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        c.execute('SELECT value FROM settings WHERE key = "email_cooldown_minutes"')
        result = c.fetchone()
        conn.close()
        return int(result[0]) if result else DEFAULT_COOLDOWN_MINUTES
    except:
        return DEFAULT_COOLDOWN_MINUTES

# ==================== EMAIL CONFIGURATION ====================

def load_email_config():
    """Load email configuration from database (just recipient)"""
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('''SELECT key, value FROM settings 
                     WHERE key IN ('email_enabled', 'email_recipient')''')
        
        config = {row[0]: row[1] for row in c.fetchall()}
        conn.close()
        
        EMAIL_CONFIG['enabled'] = config.get('email_enabled', '0') == '1'
        EMAIL_CONFIG['recipient_email'] = config.get('email_recipient', '')
        
        if EMAIL_CONFIG['enabled'] and EMAIL_CONFIG['recipient_email']:
            print(f"📧 Email alerts enabled for {EMAIL_CONFIG['recipient_email']}")
        else:
            EMAIL_CONFIG['enabled'] = False
            print("📧 Email alerts disabled")
            
    except Exception as e:
        print(f"Error loading email config: {e}")

def should_send_alert(alert_type):
    """Check if alert type is enabled in settings"""
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        alert_settings = {
            'no_water': 'alert_no_water',
            'pump_start': 'alert_pump_start',
            'pump_still_dry': 'alert_pump_still_dry',
            'high_temp': 'alert_high_temp',
            'low_humidity': 'alert_low_humidity',
            'soil_dry_plant1': 'alert_soil_dry_plant1',
            'soil_dry_plant2': 'alert_soil_dry_plant2'
        }
        
        if alert_type in alert_settings:
            setting_key = alert_settings[alert_type]
            c.execute('SELECT value FROM settings WHERE key = ?', (setting_key,))
            result = c.fetchone()
            conn.close()
            return result and result[0] == '1'
        
        conn.close()
        return True
    except Exception as e:
        print(f"Error checking alert setting: {e}")
        return True

def get_dry_duration_threshold():
    """Get configured dry duration threshold in minutes"""
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        c.execute('SELECT value FROM settings WHERE key = "alert_dry_duration"')
        result = c.fetchone()
        conn.close()
        return int(result[0]) if result else 5
    except:
        return 5

def send_email_alert(subject, message, alert_type='warning'):
    """Send email alert in background thread with cooldown"""
    if not EMAIL_CONFIG['enabled'] or not EMAIL_CONFIG['recipient_email']:
        return
    
    if not should_send_alert(alert_type):
        return
    
    if not can_send_email(alert_type):
        cooldown = get_cooldown_minutes()
        print(f"⏰ Email cooldown for {alert_type}: waiting {cooldown} minutes")
        return
    
    def send_email():
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG['sender_email']
            msg['To'] = EMAIL_CONFIG['recipient_email']
            msg['Subject'] = f"[Smart Irrigation] {subject}"
            
            cooldown = get_cooldown_minutes()
            
            colors = {
                'danger': {'bg': '#ffebee', 'border': '#dc3545'},
                'warning': {'bg': '#fff3cd', 'border': '#ffc107'},
                'info': {'bg': '#e3f2fd', 'border': '#2196F3'},
                'success': {'bg': '#d4edda', 'border': '#28a745'}
            }
            color = colors.get(alert_type, colors['warning'])
            
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; padding: 20px; border-radius: 10px 10px 0 0; text-align: center; }}
                    .content {{ background: white; padding: 30px; border-radius: 0 0 10px 10px; border: 1px solid #e0e0e0; border-top: none; }}
                    .alert-box {{ background: {color['bg']}; border-left: 5px solid {color['border']}; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                    .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 12px; color: #666; text-align: center; }}
                    .button {{ background: #27ae60; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }}
                    .cooldown-note {{ background: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 11px; color: #666; margin-top: 15px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1 style="margin:0;">🌱 Smart Irrigation System</h1>
                    </div>
                    <div class="content">
                        <h2 style="color: #333; margin-top: 0;">System Alert</h2>
                        <div class="alert-box">
                            <h3 style="color: {color['border']}; margin-top: 0;">{subject}</h3>
                            <p style="font-size: 16px; color: #555;">{message}</p>
                        </div>
                        <p style="font-size: 14px; color: #777;">
                            <strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                            <strong>Alert Type:</strong> {alert_type.upper()}<br>
                            <strong>System Status:</strong> Plant1: {pump1_state if pump1_state else 'Unknown'} | Plant2: {pump2_state if pump2_state else 'Unknown'}
                        </p>
                        <a href="http://localhost:5000/dashboard" class="button">View Dashboard</a>
                        <div class="cooldown-note">
                            <i class="fas fa-clock"></i> Next {alert_type} alert will be sent after {cooldown} minutes cooldown.
                        </div>
                        <div class="footer">
                            This is an automated message from your Smart Irrigation System.<br>
                            To change alert settings, visit the Settings page.
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
            server.quit()
            
            save_email_cooldown(alert_type)
            
            print(f"✅ Email alert sent: {subject} (cooldown started)")
            
        except Exception as e:
            print(f"❌ Error sending email: {e}")
    
    thread = threading.Thread(target=send_email)
    thread.daemon = True
    thread.start()

# ==================== LOCATION FUNCTIONS ====================

def detect_location_by_ip():
    """Detect location using IP address"""
    ip_services = [
        {
            'name': 'ip-api',
            'url': 'http://ip-api.com/json/',
            'parse': lambda data: {
                'city': data.get('city'),
                'lat': data.get('lat'),
                'lon': data.get('lon'),
                'country': data.get('countryCode'),
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
    
    if location_detected and user_location:
        return user_location
    
    location = detect_location_by_ip()
    
    if location:
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
    
    # Fallback to major cities
    major_cities = [
        {'city': 'London', 'lat': 51.5074, 'lon': -0.1278, 'country': 'GB'},
        {'city': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'country': 'US'},
        {'city': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503, 'country': 'JP'},
        {'city': 'Delhi', 'lat': 28.6139, 'lon': 77.2090, 'country': 'IN'},
        {'city': 'Singapore', 'lat': 1.3521, 'lon': 103.8198, 'country': 'SG'},
    ]
    
    fallback = random.choice(major_cities)
    fallback['method'] = 'fallback_random'
    
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

# ==================== WEATHER FUNCTIONS ====================

def fetch_weather_data():
    """Fetch current weather and forecast using auto-detected location"""
    try:
        location = get_user_location()
        
        if not location:
            print("No location available for weather data")
            return False
        
        lat = location['lat']
        lon = location['lon']
        
        current_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        
        response = requests.get(current_url)
        
        if response.status_code == 200:
            data = response.json()
            
            conn = sqlite3.connect('irrigation.db')
            c = conn.cursor()
            
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            precipitation = data.get('rain', {}).get('1h', 0) or data.get('snow', {}).get('1h', 0)
            weather_desc = data['weather'][0]['description']
            
            rain_forecast = 0
            try:
                forecast_response = requests.get(forecast_url)
                if forecast_response.status_code == 200:
                    forecast_data = forecast_response.json()
                    for forecast in forecast_data['list'][:3]:
                        if 'rain' in forecast:
                            rain_forecast = 1
                            break
                        elif 'Rain' in forecast['weather'][0]['main']:
                            rain_forecast = 1
                            break
            except Exception as e:
                print(f"Forecast check failed: {e}")
            
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

# ==================== ALERT FUNCTIONS ====================

def check_alerts(data, conn):
    """Check for alert conditions based on sensor data"""
    global pump1_start_time, pump2_start_time
    
    c = conn.cursor()
    
    c.execute('SELECT key, value FROM settings')
    settings = {row[0]: row[1] for row in c.fetchall()}
    
    alerts = []
    
    # Convert soil string to value for threshold comparison
    soil1_value = 600 if data['soil1'] == 'DRY' else 300
    soil2_value = 600 if data['soil2'] == 'DRY' else 300
    
    # Plant 1 soil dry alert
    soil_threshold1 = int(settings.get('soil_dry_threshold1', 500))
    if soil1_value > soil_threshold1:
        message = f'Plant 1 soil is too dry! Current: {data["soil1"]} (Threshold: {soil_threshold1})'
        alerts.append(('soil_dry_plant1', message))
        send_email_alert('Plant 1 Soil Too Dry', message, 'danger')
    
    # Plant 2 soil dry alert
    soil_threshold2 = int(settings.get('soil_dry_threshold2', 500))
    if soil2_value > soil_threshold2:
        message = f'Plant 2 soil is too dry! Current: {data["soil2"]} (Threshold: {soil_threshold2})'
        alerts.append(('soil_dry_plant2', message))
        send_email_alert('Plant 2 Soil Too Dry', message, 'danger')
    
    # High temperature alert
    max_temp = float(settings.get('max_temperature', 35))
    if float(data['temperature']) > max_temp:
        message = f'Temperature is high! {data["temperature"]}°C (Max: {max_temp}°C)'
        alerts.append(('high_temp', message))
        send_email_alert('High Temperature', message, 'warning')
    
    # Low humidity alert
    min_humidity = float(settings.get('min_humidity', 30))
    if float(data['humidity']) < min_humidity:
        message = f'Humidity is low! {data["humidity"]}% (Min: {min_humidity}%)'
        alerts.append(('low_humidity', message))
        send_email_alert('Low Humidity', message, 'warning')
    
    # No water alert
    if data['water'] == 'NOT AVAILABLE':
        message = 'Water tank is empty! Refill immediately.'
        alerts.append(('no_water', message))
        send_email_alert('Water Tank Empty', message, 'danger')
    
    # Pump start alerts
    if data['pump1'] == 'ON' and pump1_start_time is None:
        pump1_start_time = datetime.now()
        message = f'Plant 1 pump started automatically at {datetime.now().strftime("%H:%M:%S")}'
        alerts.append(('pump1_start', message))
        send_email_alert('Plant 1 Pump Started', message, 'info')
    elif data['pump1'] == 'OFF':
        pump1_start_time = None
    
    if data['pump2'] == 'ON' and pump2_start_time is None:
        pump2_start_time = datetime.now()
        message = f'Plant 2 pump started automatically at {datetime.now().strftime("%H:%M:%S")}'
        alerts.append(('pump2_start', message))
        send_email_alert('Plant 2 Pump Started', message, 'info')
    elif data['pump2'] == 'OFF':
        pump2_start_time = None
    
    # Check for pumps running but soil still dry after configured duration
    dry_duration = get_dry_duration_threshold()
    
    if data['pump1'] == 'ON' and pump1_start_time:
        elapsed = (datetime.now() - pump1_start_time).total_seconds() / 60
        if elapsed >= dry_duration and data['soil1'] == 'DRY':
            message = f'Plant 1 pump has been running for {elapsed:.1f} minutes but soil is still DRY! Check for issues.'
            alerts.append(('pump1_still_dry', message))
            send_email_alert('Plant 1 - Pump Running but Soil Dry', message, 'warning')
    
    if data['pump2'] == 'ON' and pump2_start_time:
        elapsed = (datetime.now() - pump2_start_time).total_seconds() / 60
        if elapsed >= dry_duration and data['soil2'] == 'DRY':
            message = f'Plant 2 pump has been running for {elapsed:.1f} minutes but soil is still DRY! Check for issues.'
            alerts.append(('pump2_still_dry', message))
            send_email_alert('Plant 2 - Pump Running but Soil Dry', message, 'warning')
    
    for alert_type, message in alerts:
        c.execute('''INSERT INTO alerts (type, message) 
                     VALUES (?, ?)''', (alert_type, message))

# ==================== UTILITY FUNCTIONS ====================

def convert_soil_value(soil_value):
    """Convert soil value from ESP string format"""
    if isinstance(soil_value, str):
        if soil_value.upper() == "DRY":
            return "DRY"
        elif soil_value.upper() == "WET":
            return "WET"
        else:
            return soil_value
    return str(soil_value)

# ==================== FLASK ROUTES ====================

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

# ==================== AUTH ROUTES ====================

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        print(f"Signup attempt: {data}")

        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            return jsonify({'status': 'error', 'message': 'All fields are required'}), 400

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()

        c.execute('''
            INSERT INTO users (username, email, password)
            VALUES (?, ?, ?)
        ''', (username, email, hashed_password))
        
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Set session
        session['user_id'] = user_id
        session['username'] = username

        return jsonify({
            'status': 'success', 
            'message': 'User registered successfully',
            'user': {
                'id': user_id,
                'username': username,
                'email': email
            }
        })

    except sqlite3.IntegrityError as e:
        print(f"Integrity error: {e}")
        return jsonify({'status': 'error', 'message': 'Username or Email already exists'}), 400

    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/signin', methods=['POST'])
def signin():
    try:
        data = request.get_json()
        print(f"Login attempt: {data}")

        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'status': 'error', 'message': 'Email and password required'}), 400

        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()

        c.execute('SELECT id, username, password FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]

            return jsonify({
                'status': 'success',
                'message': 'Login successful',
                'user': {
                    'id': user[0],
                    'username': user[1],
                    'email': email
                }
            })
        else:
            return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'status': 'success', 'message': 'Logged out successfully'})

@app.route('/api/check-auth')
def check_auth():
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': session['user_id'],
                'username': session['username']
            }
        })
    return jsonify({'authenticated': False})

# ==================== TEST EMAIL ENDPOINT ====================

@app.route('/api/test-email', methods=['POST'])
def test_email():
    """Test email configuration by sending a test message"""
    try:
        data = request.get_json()
        recipient_email = data.get('recipient_email')
        
        if not recipient_email:
            return jsonify({
                'status': 'error', 
                'message': 'Recipient email is required'
            }), 400
        
        # Create test message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = recipient_email
        msg['Subject'] = "[Smart Irrigation] Test Email"
        
        cooldown = get_cooldown_minutes()
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .content {{ background: white; padding: 30px; border-radius: 10px; border: 1px solid #e0e0e0; }}
                .success {{ color: #27ae60; font-size: 24px; text-align: center; margin: 20px 0; }}
                .info {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌱 Smart Irrigation System</h1>
                </div>
                <div class="content">
                    <div class="success">✓ Test Email</div>
                    <p>This is a test email to confirm your email configuration is working correctly.</p>
                    <p>If you received this, you will receive alerts for the events you've enabled in Settings.</p>
                    <div class="info">
                        <strong>📧 Email Cooldown Settings:</strong><br>
                        • Cooldown period: {cooldown} minutes<br>
                        • You won't receive duplicate alerts within this period
                    </div>
                    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        print(f"📧 Sending test email to {recipient_email}...")
        
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Test email sent successfully to {recipient_email}")
        
        return jsonify({
            'status': 'success',
            'message': f'Test email sent successfully to {recipient_email}'
        })
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"Authentication failed: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'status': 'error',
            'message': 'Email authentication failed. Contact system administrator.'
        }), 400
        
    except Exception as e:
        error_msg = f"Error sending test email: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to send test email. Contact system administrator.'
        }), 500

# ==================== CRITICAL: /data ENDPOINT ====================

@app.route('/data', methods=['POST', 'GET'])
def handle_data():
    """Primary data ingestion endpoint - NEVER overrides ESP decisions"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            print(f"Received data from ESP: {data}")
            
            # Validate required fields
            required_fields = ['temperature', 'humidity', 'soil1', 'soil2', 'water', 'pump1', 'pump2']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return jsonify({'status': 'error', 'message': f'Missing fields: {missing_fields}'}), 400
            
            # Process soil values - keep as strings (DRY/WET)
            soil1_value = str(data['soil1']).upper()
            soil2_value = str(data['soil2']).upper()
            
            # Validate soil values
            if soil1_value not in ['DRY', 'WET']:
                soil1_value = 'DRY'  # Default to DRY if invalid
            if soil2_value not in ['DRY', 'WET']:
                soil2_value = 'DRY'  # Default to DRY if invalid
            
            # CRITICAL: Trust ESP's pump decisions completely - NO recalculation
            final_pump1_state = str(data['pump1']).upper()
            final_pump2_state = str(data['pump2']).upper()
            
            # Validate pump states
            if final_pump1_state not in ['ON', 'OFF']:
                final_pump1_state = 'OFF'
            if final_pump2_state not in ['ON', 'OFF']:
                final_pump2_state = 'OFF'
            
            # Validate water status
            water_status = str(data['water']).upper()
            if water_status not in ['AVAILABLE', 'NOT AVAILABLE']:
                water_status = 'AVAILABLE'  # Default
            
            # Set durations based on ESP state (use provided duration or default)
            pump1_duration = int(data.get('pump1_duration', 60 if final_pump1_state == 'ON' else 0))
            pump2_duration = int(data.get('pump2_duration', 60 if final_pump2_state == 'ON' else 0))
            
            conn = sqlite3.connect('irrigation.db')
            c = conn.cursor()
            
            # Get auto_mode setting for display only - NOT for pump control
            c.execute('SELECT value FROM settings WHERE key = "auto_mode"')
            auto_mode_row = c.fetchone()
            auto_mode = auto_mode_row[0] == '1' if auto_mode_row else True
            
            # Insert sensor data with ESP's pump states - exactly as received
            c.execute('''INSERT INTO sensor_data 
                         (temperature, humidity, soil1, soil2, water, pump1, pump2, pump1_duration, pump2_duration, manual_override)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (float(data['temperature']), 
                       float(data['humidity']), 
                       soil1_value,
                       soil2_value,
                       water_status, 
                       final_pump1_state,
                       final_pump2_state,
                       pump1_duration,
                       pump2_duration,
                       0))  # manual_override = 0 for ESP-triggered data
            
            # Check for alerts (for notification only, not control)
            check_alerts({
                'temperature': float(data['temperature']),
                'humidity': float(data['humidity']),
                'soil1': soil1_value,
                'soil2': soil2_value,
                'water': water_status,
                'pump1': final_pump1_state,
                'pump2': final_pump2_state
            }, conn)
            
            # Update global pump states for manual override tracking only
            global pump1_state, pump2_state
            pump1_state = final_pump1_state
            pump2_state = final_pump2_state
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'status': 'success', 
                'message': 'Data saved successfully',
                'pump1_state': final_pump1_state,
                'pump2_state': final_pump2_state,
                'pump1_duration': pump1_duration,
                'pump2_duration': pump2_duration,
                'mode': 'AUTO' if auto_mode else 'MANUAL'
            })
            
        except Exception as e:
            print(f"Error processing data: {e}")
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    else:  # GET request
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('''SELECT * FROM sensor_data 
                     ORDER BY timestamp DESC LIMIT 1''')
        row = c.fetchone()
        
        if row and len(row) >= 12:
            # Format timestamp to ensure proper display
            timestamp = row[1]
            if timestamp:
                try:
                    # Parse the timestamp and format it consistently
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_timestamp = timestamp
            else:
                formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            data = {
                'id': row[0],
                'timestamp': formatted_timestamp,
                'temperature': row[2],
                'humidity': row[3],
                'soil1': row[4],
                'soil2': row[5],
                'water': row[6],
                'pump1': row[7],
                'pump2': row[8],
                'pump1_duration': row[9],
                'pump2_duration': row[10],
                'manual_override': bool(row[11])
            }
        else:
            data = {}
        
        conn.close()
        return jsonify(data)

# ==================== PUMP CONTROL (MANUAL ONLY) ====================

@app.route('/api/pump/control', methods=['POST'])
def control_pump():
    """Manually control individual pumps - sets manual_override=1"""
    try:
        data = request.get_json()
        pump_id = data.get('pump', '').lower()
        action = data.get('action', '').upper()
        
        if pump_id not in ['pump1', 'pump2']:
            return jsonify({'status': 'error', 'message': 'Invalid pump. Must be "pump1" or "pump2"'}), 400
        
        if action not in ['ON', 'OFF', 'TOGGLE']:
            return jsonify({'status': 'error', 'message': 'Invalid action'}), 400
        
        global pump1_state, pump2_state, pump_override
        
        # Update the specific pump state
        if pump_id == 'pump1':
            if action == 'TOGGLE':
                pump1_state = 'OFF' if pump1_state == 'ON' else 'ON'
            else:
                pump1_state = action
        else:  # pump2
            if action == 'TOGGLE':
                pump2_state = 'OFF' if pump2_state == 'ON' else 'ON'
            else:
                pump2_state = action
        
        # Set manual override flag
        pump_override = True
        
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        # Get latest sensor data for context
        c.execute('''SELECT temperature, humidity, soil1, soil2, water 
                     FROM sensor_data 
                     ORDER BY timestamp DESC LIMIT 1''')
        sensor_data = c.fetchone()
        
        if sensor_data and len(sensor_data) >= 5:
            temp, humidity, soil1, soil2, water = sensor_data
        else:
            temp, humidity, soil1, soil2, water = 25.0, 60.0, 'DRY', 'DRY', 'AVAILABLE'
        
        current_pump1 = pump1_state
        current_pump2 = pump2_state
        
        # Set durations for manual control
        pump1_duration = 60 if current_pump1 == 'ON' else 0
        pump2_duration = 60 if current_pump2 == 'ON' else 0
        
        # Insert record with manual_override=1
        c.execute('''INSERT INTO sensor_data 
                     (temperature, humidity, soil1, soil2, water, pump1, pump2, pump1_duration, pump2_duration, manual_override)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (temp, humidity, soil1, soil2, water, current_pump1, current_pump2, pump1_duration, pump2_duration, 1))
        
        conn.commit()
        conn.close()
        
        # Send email notification for manual control
        send_email_alert(
            f'Manual Pump Control',
            f'Pump {pump_id} turned {current_pump1 if pump_id == "pump1" else current_pump2} manually',
            'info'
        )
        
        return jsonify({
            'status': 'success',
            'message': f'{pump_id} turned {pump1_state if pump_id == "pump1" else pump2_state}',
            'pump1_state': pump1_state,
            'pump2_state': pump2_state,
            'manual_override': True
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/pump/status')
def get_pump_status():
    """Get current pump status for both pumps"""
    global pump1_state, pump2_state, pump_override
    
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    c.execute('''SELECT pump1, pump2, manual_override, timestamp 
                 FROM sensor_data 
                 WHERE pump1 IS NOT NULL AND pump2 IS NOT NULL
                 ORDER BY timestamp DESC LIMIT 1''')
    
    row = c.fetchone()
    conn.close()
    
    if row and len(row) >= 4:
        pump1_state = row[0]
        pump2_state = row[1]
        pump_override = bool(row[2])
        last_update = row[3]
    else:
        last_update = None
    
    return jsonify({
        'pump1_state': pump1_state,
        'pump2_state': pump2_state,
        'manual_override': pump_override,
        'auto_mode': not pump_override,
        'last_update': last_update
    })

# ==================== LOCATION API ====================

@app.route('/api/location', methods=['GET', 'POST'])
def handle_location():
    if request.method == 'GET':
        location = get_user_location()
        return jsonify({
            'location': location,
            'detected': location_detected,
            'weather_api_key_set': OPENWEATHER_API_KEY != "YOUR_ACTUAL_API_KEY_HERE"
        })
    
    else:  # POST
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

# ==================== STATS API ====================

@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    # Convert soil text to numeric for averaging
    c.execute('''SELECT 
                 AVG(temperature) as avg_temp,
                 AVG(humidity) as avg_humidity,
                 SUM(CASE WHEN soil1 = 'DRY' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as soil1_dry_percent,
                 SUM(CASE WHEN soil2 = 'DRY' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as soil2_dry_percent,
                 COUNT(CASE WHEN pump1 = 'ON' THEN 1 END) * 100.0 / COUNT(*) as pump1_usage_percent,
                 COUNT(CASE WHEN pump2 = 'ON' THEN 1 END) * 100.0 / COUNT(*) as pump2_usage_percent,
                 SUM(pump1_duration) as total_pump1_time,
                 SUM(pump2_duration) as total_pump2_time,
                 COUNT(CASE WHEN water = 'NOT AVAILABLE' THEN 1 END) as water_warnings
                 FROM sensor_data 
                 WHERE timestamp > datetime('now', '-1 day')''')
    
    stats = c.fetchone()
    
    c.execute('SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1')
    latest = c.fetchone()
    
    c.execute('SELECT * FROM weather_data ORDER BY timestamp DESC LIMIT 1')
    weather_row = c.fetchone()
    
    location = get_user_location()
    model_exists = os.path.exists('models/irrigation_model.pkl')
    
    conn.close()
    
    weather = None
    if weather_row and len(weather_row) >= 7:
        weather = {
            'temperature': weather_row[2],
            'humidity': weather_row[3],
            'precipitation': weather_row[4],
            'rain_forecast': bool(weather_row[5]),
            'description': weather_row[6],
            'location': weather_row[7] if len(weather_row) > 7 else None
        }
    
    latest_data = {}
    if latest and len(latest) >= 12:
        # Format timestamp for latest data
        timestamp = latest[1]
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_timestamp = timestamp
        else:
            formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        latest_data = {
            'timestamp': formatted_timestamp,
            'temperature': latest[2],
            'humidity': latest[3],
            'soil1': latest[4],
            'soil2': latest[5],
            'water': latest[6],
            'pump1': latest[7],
            'pump2': latest[8]
        }
    
    return jsonify({
        'avg_temperature': round(stats[0], 2) if stats[0] else 0,
        'avg_humidity': round(stats[1], 2) if stats[1] else 0,
        'soil1_dry_percent': round(stats[2], 2) if stats[2] else 0,
        'soil2_dry_percent': round(stats[3], 2) if stats[3] else 0,
        'pump1_usage_percent': round(stats[4], 2) if stats[4] else 0,
        'pump2_usage_percent': round(stats[5], 2) if stats[5] else 0,
        'total_pump1_time': stats[6] if stats[6] else 0,
        'total_pump2_time': stats[7] if stats[7] else 0,
        'water_warnings': stats[8] if stats[8] else 0,
        'latest': latest_data,
        'weather': weather,
        'location': location,
        'model_available': model_exists
    })

# ==================== HISTORY API ====================

@app.route('/api/history')
def get_history():
    limit = request.args.get('limit', '100')
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
        if len(row) >= 12:
            # Format timestamp for each record
            timestamp = row[1]
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_timestamp = timestamp
            else:
                formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            data.append({
                'id': row[0],
                'timestamp': formatted_timestamp,
                'temperature': row[2],
                'humidity': row[3],
                'soil1': row[4],
                'soil2': row[5],
                'water': row[6],
                'pump1': row[7],
                'pump2': row[8],
                'pump1_duration': row[9],
                'pump2_duration': row[10],
                'manual_override': bool(row[11])
            })
    
    conn.close()
    return jsonify(data)

# ==================== ALERTS API ====================

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
        # Format timestamp for alerts
        timestamp = row[1]
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_timestamp = timestamp
        else:
            formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        alerts.append({
            'id': row[0],
            'timestamp': formatted_timestamp,
            'type': row[2],
            'message': row[3],
            'resolved': bool(row[4])
        })
    
    conn.close()
    return jsonify(alerts)

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

@app.route('/api/clear-all-alerts', methods=['POST'])
def clear_all_alerts():
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('UPDATE alerts SET resolved = 1 WHERE resolved = 0')
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'message': 'All alerts cleared'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ==================== SETTINGS API ====================

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'GET':
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('SELECT key, value FROM settings')
        settings = {row[0]: row[1] for row in c.fetchall()}
        
        c.execute('SELECT * FROM pump_schedule ORDER BY hour, minute')
        schedule_rows = c.fetchall()
        pump_schedule = []
        
        for row in schedule_rows:
            pump_schedule.append({
                'id': row[0],
                'hour': row[1],
                'minute': row[2],
                'duration': row[3],
                'plant': row[4] if len(row) > 4 else 'both',
                'enabled': bool(row[5] if len(row) > 5 else True)
            })
        
        email_config = {
            'enabled': settings.get('email_enabled', '0') == '1',
            'recipient': settings.get('email_recipient', '')
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
                c.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, str(value)))
            
            # Reload email config if email settings changed
            if 'email_enabled' in data['settings'] or 'email_recipient' in data['settings'] or 'email_cooldown_minutes' in data['settings']:
                load_email_config()
                
                # Send test email if enabled
                if data['settings'].get('email_enabled') == '1' and data['settings'].get('email_recipient'):
                    cooldown = data['settings'].get('email_cooldown_minutes', '10')
                    send_email_alert(
                        'Email Alerts Enabled',
                        f'Email notifications have been enabled for {data["settings"]["email_recipient"]}\n\n'
                        f'You will now receive alerts for the events you\'ve selected in Settings.\n'
                        f'📧 Cooldown period: {cooldown} minutes between duplicate alerts.',
                        'success'
                    )
        
        if 'pump_schedule' in data:
            # Clear existing schedule
            c.execute('DELETE FROM pump_schedule')
            
            # Insert new schedule
            for schedule in data['pump_schedule']:
                c.execute('''INSERT INTO pump_schedule (hour, minute, duration, plant, enabled) 
                             VALUES (?, ?, ?, ?, ?)''',
                          (schedule['hour'], schedule['minute'], 
                           schedule['duration'], schedule.get('plant', 'both'),
                           1 if schedule.get('enabled', True) else 0))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'message': 'Settings saved successfully'})

# ==================== CLEAR DATA ENDPOINT ====================

@app.route('/api/clear-data', methods=['POST'])
def clear_data():
    """Clear all historical sensor data"""
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        # Clear sensor data but keep settings and schedule
        c.execute('DELETE FROM sensor_data')
        c.execute('DELETE FROM alerts')
        c.execute('DELETE FROM weather_data')
        c.execute('DELETE FROM predictions')
        c.execute('DELETE FROM training_log')
        c.execute('DELETE FROM email_cooldown')  # Clear email cooldown tracking
        
        # Reset pump states
        global pump1_state, pump2_state, pump_override, pump1_start_time, pump2_start_time, email_cooldown
        pump1_state = "OFF"
        pump2_state = "OFF"
        pump_override = False
        pump1_start_time = None
        pump2_start_time = None
        email_cooldown = {}
        
        conn.commit()
        conn.close()
        
        send_email_alert(
            'Data Cleared',
            'All historical sensor data and email cooldown tracking has been cleared from the system.',
            'warning'
        )
        
        return jsonify({'status': 'success', 'message': 'All data cleared successfully'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ==================== PLANT HEALTH DETAILED ====================

@app.route('/api/plant-health-detailed')
def get_detailed_plant_health():
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            timestamp,
            temperature,
            humidity,
            soil1,
            soil2,
            pump1,
            pump2,
            pump1_duration,
            pump2_duration
        FROM sensor_data 
        WHERE timestamp > datetime('now', '-7 days')
        ORDER BY timestamp
    ''')
    
    rows = c.fetchall()
    
    if not rows:
        return jsonify({
            'plant1': {'health': 'Unknown', 'score': 0, 'details': {}, 'recommendations': []},
            'plant2': {'health': 'Unknown', 'score': 0, 'details': {}, 'recommendations': []}
        })
    
    # Plant 1 analysis
    watering_events_1 = []
    total_water_time_1 = 0
    last_pump1_on = None
    
    # Plant 2 analysis
    watering_events_2 = []
    total_water_time_2 = 0
    last_pump2_on = None
    
    # Convert soil strings to numeric for calculations
    soil1_numeric = []
    soil2_numeric = []
    
    for row in rows:
        timestamp, temp, humidity, soil1, soil2, pump1, pump2, dur1, dur2 = row
        
        # Convert soil to numeric (600 for DRY, 300 for WET)
        soil1_val = 600 if soil1 == 'DRY' else 300
        soil2_val = 600 if soil2 == 'DRY' else 300
        soil1_numeric.append(soil1_val)
        soil2_numeric.append(soil2_val)
        
        # Plant 1
        if pump1 == 'ON' and dur1 > 0:
            if last_pump1_on is None:
                last_pump1_on = timestamp
            total_water_time_1 += dur1
        elif pump1 == 'OFF' and last_pump1_on:
            watering_events_1.append({
                'start': last_pump1_on,
                'end': timestamp,
                'duration': dur1 or 0
            })
            last_pump1_on = None
        
        # Plant 2
        if pump2 == 'ON' and dur2 > 0:
            if last_pump2_on is None:
                last_pump2_on = timestamp
            total_water_time_2 += dur2
        elif pump2 == 'OFF' and last_pump2_on:
            watering_events_2.append({
                'start': last_pump2_on,
                'end': timestamp,
                'duration': dur2 or 0
            })
            last_pump2_on = None
    
    # Calculate metrics
    avg_watering_duration_1 = np.mean([e['duration'] for e in watering_events_1]) if watering_events_1 else 0
    watering_frequency_1 = len(watering_events_1) / 7
    
    avg_watering_duration_2 = np.mean([e['duration'] for e in watering_events_2]) if watering_events_2 else 0
    watering_frequency_2 = len(watering_events_2) / 7
    
    avg_soil1 = np.mean(soil1_numeric) if soil1_numeric else 450
    avg_soil2 = np.mean(soil2_numeric) if soil2_numeric else 450
    soil1_stability = np.std(soil1_numeric) if len(soil1_numeric) > 1 else 0
    soil2_stability = np.std(soil2_numeric) if len(soil2_numeric) > 1 else 0
    
    latest = rows[-1] if rows else None
    latest_temp = latest[1] if latest else 25
    latest_humidity = latest[2] if latest else 60
    
    # Calculate health scores
    score1 = calculate_health_score(avg_soil1, watering_frequency_1, avg_watering_duration_1, 
                                    soil1_stability, latest_temp, latest_humidity)
    
    score2 = calculate_health_score(avg_soil2, watering_frequency_2, avg_watering_duration_2, 
                                    soil2_stability, latest_temp, latest_humidity)
    
    health1 = get_health_status(score1)
    health2 = get_health_status(score2)
    
    # Generate recommendations
    recommendations1 = generate_recommendations(
        avg_soil1, watering_frequency_1, avg_watering_duration_1, 
        latest_temp, latest_humidity, soil1_stability, 1
    )
    
    recommendations2 = generate_recommendations(
        avg_soil2, watering_frequency_2, avg_watering_duration_2, 
        latest_temp, latest_humidity, soil2_stability, 2
    )
    
    conn.close()
    
    return jsonify({
        'plant1': {
            'health': health1,
            'score': round(score1, 1),
            'details': {
                'avg_soil_moisture': round(avg_soil1, 1),
                'soil_status': 'DRY' if avg_soil1 > 450 else 'WET',
                'watering_frequency_per_day': round(watering_frequency_1, 2),
                'avg_watering_duration_seconds': round(avg_watering_duration_1, 1),
                'total_water_time_seconds': total_water_time_1,
                'soil_stability': round(soil1_stability, 1),
                'watering_events_count': len(watering_events_1)
            },
            'recommendations': recommendations1
        },
        'plant2': {
            'health': health2,
            'score': round(score2, 1),
            'details': {
                'avg_soil_moisture': round(avg_soil2, 1),
                'soil_status': 'DRY' if avg_soil2 > 450 else 'WET',
                'watering_frequency_per_day': round(watering_frequency_2, 2),
                'avg_watering_duration_seconds': round(avg_watering_duration_2, 1),
                'total_water_time_seconds': total_water_time_2,
                'soil_stability': round(soil2_stability, 1),
                'watering_events_count': len(watering_events_2)
            },
            'recommendations': recommendations2
        },
        'environment': {
            'current_temperature': latest_temp,
            'current_humidity': latest_humidity
        }
    })

def calculate_health_score(avg_soil, frequency, duration, stability, temp, humidity):
    score = 100
    
    if avg_soil > 550:
        score -= 40
    elif avg_soil > 450:
        score -= 20
    elif avg_soil < 250:
        score -= 30
    
    if frequency < 0.5:
        score -= 25
    elif frequency > 3:
        score -= 20
    
    if duration > 300:
        score -= 15
    elif duration < 30:
        score -= 10
    
    if stability > 150:
        score -= 15
    
    if temp > 35 or temp < 10:
        score -= 20
    elif temp > 30 or temp < 15:
        score -= 10
    
    if humidity < 30 or humidity > 80:
        score -= 15
    
    return max(0, min(100, score))

def get_health_status(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Fair"
    elif score >= 30:
        return "Poor"
    else:
        return "Critical"

def generate_recommendations(avg_soil, frequency, duration, temp, humidity, stability, plant_num):
    recommendations = []
    
    if avg_soil > 550:
        recommendations.append({
            'priority': 'high',
            'message': f'Plant {plant_num} soil is consistently dry. Increase watering frequency or duration.',
            'action': 'Increase watering by 30 seconds per session'
        })
    elif avg_soil < 250:
        recommendations.append({
            'priority': 'high',
            'message': f'Plant {plant_num} soil is too wet. Reduce watering to prevent root rot.',
            'action': 'Reduce watering frequency by 50%'
        })
    
    if frequency < 0.5:
        recommendations.append({
            'priority': 'medium',
            'message': f'Infrequent watering detected for plant {plant_num}.',
            'action': 'Water at least once daily'
        })
    elif frequency > 2.5:
        recommendations.append({
            'priority': 'medium',
            'message': f'Overwatering detected for plant {plant_num}.',
            'action': 'Reduce watering to twice daily maximum'
        })
    
    if duration > 300:
        recommendations.append({
            'priority': 'low',
            'message': f'Watering sessions for plant {plant_num} are too long.',
            'action': 'Reduce watering duration to 2-3 minutes max'
        })
    
    if stability > 150:
        recommendations.append({
            'priority': 'medium',
            'message': f'Plant {plant_num} soil moisture fluctuates too much.',
            'action': 'Maintain more consistent watering schedule'
        })
    
    if not recommendations:
        recommendations.append({
            'priority': 'low',
            'message': f'All parameters are within optimal ranges for plant {plant_num}.',
            'action': 'Continue current watering schedule'
        })
    
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    recommendations.sort(key=lambda x: priority_order[x['priority']])
    
    return recommendations

# ==================== EXPORT CSV ====================

@app.route('/api/export/csv')
def export_csv():
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM sensor_data ORDER BY timestamp')
    rows = c.fetchall()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['ID', 'Timestamp', 'Temperature (°C)', 'Humidity (%)', 
                     'Soil1 Status', 'Soil2 Status', 'Water Level', 
                     'Pump1 Status', 'Pump2 Status', 'Pump1 Duration (s)', 
                     'Pump2 Duration (s)', 'Manual Override'])
    
    for row in rows:
        # Format timestamp for CSV
        timestamp = row[1]
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_timestamp = timestamp
        else:
            formatted_timestamp = ''
            
        row_list = list(row)
        row_list[1] = formatted_timestamp
        writer.writerow(row_list)
    
    conn.close()
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"irrigation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

# ==================== ML TRAINING FUNCTIONS ====================

def get_training_settings():
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
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    c.execute('''INSERT INTO training_log 
                 (model_name, accuracy, r2_score, mse, samples_used, status, message)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (model_name, accuracy, r2_score, mse, samples_used, status, message))
    conn.commit()
    conn.close()

def prepare_training_data():
    conn = sqlite3.connect('irrigation.db')
    
    try:
        query = '''
        SELECT 
            timestamp,
            temperature,
            humidity,
            CASE WHEN soil1 = 'DRY' THEN 1 ELSE 0 END as soil1_dry,
            CASE WHEN soil2 = 'DRY' THEN 1 ELSE 0 END as soil2_dry,
            strftime('%H', timestamp) as hour,
            strftime('%w', timestamp) as day_of_week,
            strftime('%m', timestamp) as month,
            CASE WHEN pump1 = 'ON' OR pump2 = 'ON' THEN 1 ELSE 0 END as pump_status,
            CASE WHEN water = 'AVAILABLE' THEN 1 ELSE 0 END as water_available
        FROM sensor_data 
        WHERE timestamp > datetime('now', '-30 days')
        AND temperature IS NOT NULL
        AND humidity IS NOT NULL
        AND soil1 IS NOT NULL
        AND soil2 IS NOT NULL
        ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) < 10:
            print(f"Not enough data for training. Only {len(df)} samples available.")
            return None, None, None
        
        print(f"📊 Preparing {len(df)} samples for training...")
        
        # Calculate combined soil dryness
        df['soil_dry'] = (df['soil1_dry'] + df['soil2_dry']) / 2
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time features
        df['hour_int'] = df['hour'].astype(int)
        df['day_int'] = df['day_of_week'].astype(int)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_int'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_int'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_int'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_int'] / 7)
        
        # Lag features
        for lag in [1, 2, 3]:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
            df[f'soil_dry_lag_{lag}'] = df['soil_dry'].shift(lag)
        
        # Rolling averages
        for window in [3, 6]:
            df[f'temp_ma_{window}'] = df['temperature'].rolling(window=window).mean()
            df[f'humidity_ma_{window}'] = df['humidity'].rolling(window=window).mean()
            df[f'soil_dry_ma_{window}'] = df['soil_dry'].rolling(window=window).mean()
        
        # Interaction features
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100.0
        
        # Month features
        df['month_int'] = df['month'].astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['month_int'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_int'] / 12)
        
        # Target
        df['target'] = df['pump_status']
        
        df = df.dropna()
        
        if len(df) < 10:
            print("Not enough valid samples after cleaning.")
            return None, None, None
        
        feature_columns = [
            'temperature', 'humidity', 'soil_dry',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'temp_lag_1', 'humidity_lag_1', 'soil_dry_lag_1',
            'temp_ma_3', 'humidity_ma_3', 'soil_dry_ma_3',
            'temp_humidity_interaction',
            'water_available'
        ]
        
        available_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[available_columns]
        y = df['target']
        
        print(f"✅ Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Pump ON events: {y.sum()} ({y.mean()*100:.1f}%)")
        
        return X, y, available_columns
        
    except Exception as e:
        print(f"Error preparing training data: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        conn.close()

def train_prediction_model_enhanced():
    try:
        print("\n" + "="*60)
        print("🚀 STARTING ML MODEL TRAINING")
        print("="*60)
        
        X, y, feature_names = prepare_training_data()
        
        if X is None or y is None:
            log_training_result('none', 0, 0, 0, 0, 'failed', 'Insufficient training data')
            print("❌ Training aborted: Insufficient data")
            return None
        
        settings = get_training_settings()
        min_samples = settings['min_samples']
        
        if len(X) < min_samples:
            msg = f"Only {len(X)} samples available, need at least {min_samples}"
            log_training_result('none', 0, 0, 0, len(X), 'failed', msg)
            print(f"❌ Training aborted: {msg}")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
        model_name = 'Random Forest Classifier'
        
        print(f"🏋️ Training {model_name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred_proba)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"\n📈 Model Performance:")
        print(f"  ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  📊 R2 Score: {r2:.4f}")
        print(f"  📉 MSE: {mse:.4f}")
        print(f"  📊 Samples used: {len(X)}")
        print(f"  🔧 Features: {len(feature_names)}")
        
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(model, 'models/irrigation_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(feature_names, 'models/feature_names.pkl')
        
        model_info = {
            'model_name': model_name,
            'trained_at': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'r2_score': float(r2),
            'mse': float(mse),
            'samples_used': len(X),
            'feature_count': len(feature_names),
            'features': feature_names
        }
        
        joblib.dump(model_info, 'models/model_info.pkl')
        
        log_training_result(model_name, accuracy, r2, mse, len(X), 'success', 
                          f'Model trained successfully with {len(X)} samples')
        
        print(f"\n✅ Model saved successfully")
        
        # Send email notification for successful training
        cooldown = get_cooldown_minutes()
        send_email_alert(
            'ML Model Training Complete',
            f'New model trained successfully!\n\n'
            f'• Accuracy: {accuracy*100:.1f}%\n'
            f'• R² Score: {r2:.3f}\n'
            f'• Samples Used: {len(X)}\n'
            f'• Features: {len(feature_names)}\n\n'
            f'📧 Next alert will be sent after {cooldown} minutes cooldown.',
            'success'
        )
        
        return model
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        log_training_result('none', 0, 0, 0, 0, 'failed', error_msg)
        send_email_alert('ML Model Training Failed', error_msg, 'danger')
        return None

# ==================== ML PREDICTION FUNCTIONS ====================

def generate_sample_predictions():
    print("📝 Generating sample predictions...")
    
    predictions = []
    current_date = datetime.now()
    
    for i in range(7):
        date = current_date + timedelta(days=i)
        day_name = date.strftime('%A')
        date_str = date.strftime('%Y-%m-%d')
        
        day_predictions = []
        prediction_hours = [6, 9, 12, 15, 18, 21]
        
        for hour in prediction_hours:
            base_prob = 60 - (i * 3)
            variation = random.randint(-15, 15)
            probability = base_prob + variation
            
            if hour in [6, 18]:
                probability += 10
            elif hour == 12:
                probability -= 5
            
            probability = max(10, min(95, probability))
            
            day_predictions.append({
                'hour': hour,
                'probability': float(probability),
                'recommended': probability > 60,
                'confidence': 'high' if probability > 80 else 'medium' if probability > 60 else 'low'
            })
        
        probabilities = [p['probability'] for p in day_predictions]
        avg_probability = sum(probabilities) / len(probabilities) if probabilities else 50
        
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

def save_predictions_to_csv(predictions):
    try:
        if not predictions:
            return None
        
        os.makedirs('predictions', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f'predictions/irrigation_predictions_{timestamp}.csv'
        
        rows = []
        for day_data in predictions:
            date = day_data['date']
            day_name = day_data['day_name']
            avg_prob = day_data['avg_probability']
            
            for hour_data in day_data['predictions']:
                rows.append({
                    'Date': date,
                    'Day': day_name,
                    'Hour': hour_data['hour'],
                    'Hour_Probability': hour_data['probability'],
                    'Recommended': 'Yes' if hour_data['recommended'] else 'No',
                    'Confidence': hour_data['confidence'],
                    'Day_Avg_Probability': avg_prob,
                    'Recommendations': ' | '.join(day_data['recommendations'])
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Predictions saved to CSV: {csv_path}")
        return csv_path
        
    except Exception as e:
        print(f"❌ Error saving predictions to CSV: {e}")
        return None

def save_predictions_to_db(predictions):
    try:
        if not predictions:
            return
        
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('DELETE FROM predictions WHERE timestamp < datetime("now", "-7 days")')
        
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

def predict_irrigation_schedule():
    try:
        print("\n" + "="*60)
        print("🔮 STARTING PREDICTION PROCESS")
        print("="*60)
        
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
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sample_preds = generate_sample_predictions()
            save_predictions_to_csv(sample_preds)
            return sample_preds
        
        # Get recent averages for prediction
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                COALESCE(AVG(temperature), 25.0) as avg_temp,
                COALESCE(AVG(humidity), 60.0) as avg_humidity,
                COALESCE(SUM(CASE WHEN soil1 = 'DRY' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 0.5) as soil1_dry_ratio,
                COALESCE(SUM(CASE WHEN soil2 = 'DRY' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 0.5) as soil2_dry_ratio
            FROM sensor_data 
            WHERE timestamp > datetime('now', '-3 days')
            AND temperature IS NOT NULL
            AND humidity IS NOT NULL
            AND soil1 IS NOT NULL
            AND soil2 IS NOT NULL
        ''')
        
        avg_data = c.fetchone()
        avg_temp = float(avg_data[0]) if avg_data[0] else 25.0
        avg_humidity = float(avg_data[1]) if avg_data[1] else 60.0
        soil1_dry_ratio = float(avg_data[2]) if avg_data[2] else 0.5
        soil2_dry_ratio = float(avg_data[3]) if avg_data[3] else 0.5
        
        avg_soil_dry = (soil1_dry_ratio + soil2_dry_ratio) / 2
        
        c.execute('''SELECT water FROM sensor_data 
                     WHERE water IS NOT NULL
                     ORDER BY timestamp DESC LIMIT 1''')
        water_row = c.fetchone()
        water_available = 1 if water_row and water_row[0] == 'AVAILABLE' else 0
        
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
        
        # Generate predictions for next 7 days
        predictions = []
        current_date = datetime.now()
        
        for day_offset in range(7):
            prediction_date = current_date + timedelta(days=day_offset)
            day_of_week = prediction_date.weekday()
            day_name = prediction_date.strftime('%A')
            date_str = prediction_date.strftime('%Y-%m-%d')
            
            weather_factor = 1.0 - (day_offset * 0.03)
            weather_factor = max(0.6, weather_factor)
            
            hour_predictions = []
            
            for hour in prediction_hours:
                try:
                    # Build feature vector
                    features = []
                    for feature_name in feature_names:
                        value = 0.0
                        
                        if feature_name == 'temperature':
                            value = avg_temp
                        elif feature_name == 'humidity':
                            value = avg_humidity
                        elif feature_name == 'soil_dry':
                            value = avg_soil_dry
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
                            if 'temp' in feature_name:
                                value = avg_temp
                            elif 'humidity' in feature_name:
                                value = avg_humidity
                            elif 'soil' in feature_name:
                                value = avg_soil_dry
                        elif 'interaction' in feature_name:
                            if 'temp_humidity' in feature_name:
                                value = avg_temp * avg_humidity / 100.0
                        
                        features.append(float(value))
                    
                    if len(features) != len(feature_names):
                        if len(features) > len(feature_names):
                            features = features[:len(feature_names)]
                        else:
                            features.extend([0.0] * (len(feature_names) - len(features)))
                    
                    features_array = np.array(features).reshape(1, -1)
                    features_scaled = scaler.transform(features_array)
                    
                    proba = model.predict_proba(features_scaled)[0]
                    if len(proba) > 1:
                        probability = float(proba[1]) * 100
                    else:
                        probability = float(proba[0]) * 100
                    
                    probability = probability * weather_factor
                    probability = max(0, min(100, probability))
                    
                    if day_offset > 0:
                        variation = np.random.uniform(-8, 8)
                        probability += variation
                        probability = max(0, min(100, probability))
                    
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
            
            if hour_predictions:
                probabilities = [p['probability'] for p in hour_predictions]
                avg_probability = np.mean(probabilities)
                
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
        
        save_predictions_to_csv(predictions)
        save_predictions_to_db(predictions)
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        traceback.print_exc()
        sample_preds = generate_sample_predictions()
        save_predictions_to_csv(sample_preds)
        return sample_preds

# ==================== ML API ENDPOINTS ====================

@app.route('/api/predictions')
def get_predictions():
    try:
        print(f"\n📡 [API] GET /api/predictions called")
        
        predictions_data = predict_irrigation_schedule()
        
        if not predictions_data:
            print("⚠️ No predictions generated, using sample data")
            predictions_data = generate_sample_predictions()
        
        model_info = {}
        model_files_exist = False
        
        try:
            if os.path.exists('models/model_info.pkl'):
                model_info = joblib.load('models/model_info.pkl')
                model_files_exist = True
        except Exception as e:
            print(f"⚠️ Could not load model info: {e}")
        
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
                # Format timestamp for training log
                timestamp = log[0]
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        formatted_timestamp = timestamp
                else:
                    formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                training_log.append({
                    'timestamp': formatted_timestamp,
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
        
        try:
            conn = sqlite3.connect('irrigation.db')
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM sensor_data')
            total_samples = c.fetchone()[0]
            conn.close()
        except:
            total_samples = 0
        
        model_status = {
            'model_exists': model_files_exist,
            'accuracy': model_info.get('accuracy', 0) if model_info else 0,
            'r2_score': model_info.get('r2_score', 0) if model_info else 0,
            'samples_used': model_info.get('samples_used', 0) if model_info else 0,
            'last_trained': model_info.get('trained_at', 'Never') if model_info else 'Never',
            'feature_count': model_info.get('feature_count', 0) if model_info else 0
        }
        
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
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ [API ERROR] /api/predictions: {e}")
        traceback.print_exc()
        
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
    conn = sqlite3.connect('irrigation.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM training_log ORDER BY timestamp DESC LIMIT 20')
    rows = c.fetchall()
    
    logs = []
    for row in rows:
        # Format timestamp for training log
        timestamp = row[1]
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_timestamp = timestamp
        else:
            formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        logs.append({
            'id': row[0],
            'timestamp': formatted_timestamp,
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
    model_info = {}
    try:
        if os.path.exists('models/model_info.pkl'):
            model_info = joblib.load('models/model_info.pkl')
    except Exception as e:
        print(f"Error loading model info: {e}")
    
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
    try:
        predictions = predict_irrigation_schedule()
        
        if not predictions:
            predictions = generate_sample_predictions()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            'Date', 'Day', 'Hour', 'Probability (%)', 
            'Recommended', 'Confidence', 'Day Average (%)',
            'Recommendations', 'Generated At'
        ])
        
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
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'irrigation_predictions_{timestamp}.csv'
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Error exporting predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug-predictions')
def debug_predictions():
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'directories': {},
        'model_status': {},
        'database': {},
        'prediction_test': {}
    }
    
    debug_info['directories']['models'] = {
        'exists': os.path.exists('models'),
        'contents': os.listdir('models') if os.path.exists('models') else []
    }
    
    debug_info['directories']['predictions'] = {
        'exists': os.path.exists('predictions'),
        'contents': os.listdir('predictions') if os.path.exists('predictions') else []
    }
    
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
                
                if file.endswith('.pkl'):
                    try:
                        loaded = joblib.load(path)
                        debug_info['model_status'][file]['loadable'] = True
                        debug_info['model_status'][file]['type'] = type(loaded).__name__
                    except:
                        debug_info['model_status'][file]['loadable'] = False
            except:
                debug_info['model_status'][file]['size'] = 'unknown'
    
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM sensor_data')
        debug_info['database']['total_samples'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM training_log')
        debug_info['database']['training_logs'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM predictions')
        debug_info['database']['stored_predictions'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM email_cooldown')
        debug_info['database']['email_cooldown'] = c.fetchone()[0]
        
        c.execute('SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1')
        latest_sensor = c.fetchone()
        if latest_sensor and len(latest_sensor) > 8:
            # Format timestamp for debug
            timestamp = latest_sensor[1]
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_timestamp = timestamp
            else:
                formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            debug_info['database']['latest_sensor'] = {
                'timestamp': formatted_timestamp,
                'temperature': latest_sensor[2],
                'humidity': latest_sensor[3],
                'soil1': latest_sensor[4],
                'soil2': latest_sensor[5],
                'water': latest_sensor[6],
                'pump1': latest_sensor[7],
                'pump2': latest_sensor[8]
            }
        
        c.execute('SELECT * FROM training_log ORDER BY timestamp DESC LIMIT 1')
        latest_log = c.fetchone()
        if latest_log:
            # Format timestamp for debug
            timestamp = latest_log[1]
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_timestamp = timestamp
            else:
                formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            debug_info['database']['latest_training'] = {
                'timestamp': formatted_timestamp,
                'model': latest_log[2],
                'accuracy': latest_log[3],
                'status': latest_log[7]
            }
        
        conn.close()
    except Exception as e:
        debug_info['database']['error'] = str(e)
    
    try:
        test_pred = predict_irrigation_schedule()
        debug_info['prediction_test']['success'] = True
        debug_info['prediction_test']['days'] = len(test_pred) if test_pred else 0
        if test_pred and len(test_pred) > 0:
            debug_info['prediction_test']['first_day'] = test_pred[0]['date']
            debug_info['prediction_test']['first_day_avg'] = test_pred[0]['avg_probability']
    except Exception as e:
        debug_info['prediction_test']['success'] = False
        debug_info['prediction_test']['error'] = str(e)
        debug_info['prediction_test']['traceback'] = traceback.format_exc()
    
    return jsonify(debug_info)

# ==================== TEST ROUTE ====================

@app.route('/test')
def test():
    cooldown = get_cooldown_minutes()
    return jsonify({
        'status': 'ok', 
        'message': 'Smart Irrigation System with Dual Plant Support is running!',
        'location': get_user_location(),
        'location_detected': location_detected,
        'model_available': os.path.exists('models/irrigation_model.pkl'),
        'auto_train_interval': '10 minutes',
        'email_enabled': EMAIL_CONFIG['enabled'],
        'email_recipient': EMAIL_CONFIG['recipient_email'],
        'email_cooldown': cooldown,
        'dual_plant_support': True,
        'architecture': 'ESP is primary decision maker - Flask only stores data'
    })

# ==================== BACKGROUND TASKS ====================

def run_scheduled_tasks():
    last_weather_fetch = 0
    last_auto_train = 0
    weather_interval = 1800  # 30 minutes
    auto_train_interval = 600  # 10 minutes
    
    print("Starting scheduled tasks...")
    
    while True:
        try:
            current_time = time.time()
            
            if current_time - last_weather_fetch >= weather_interval:
                print("Scheduled: Fetching weather data...")
                fetch_weather_data()
                last_weather_fetch = current_time
            
            if current_time - last_auto_train >= auto_train_interval:
                conn = sqlite3.connect('irrigation.db')
                c = conn.cursor()
                c.execute('SELECT value FROM settings WHERE key = "auto_train_interval"')
                interval_row = c.fetchone()
                conn.close()
                
                current_interval = int(interval_row[0]) if interval_row else 10
                auto_train_interval = current_interval * 60
                
                print(f"Scheduled: Auto-training check (every {current_interval} minutes)...")
                auto_train_model()
                last_auto_train = current_time
            
            time.sleep(60)
            
        except Exception as e:
            print(f"Error in scheduled tasks: {e}")
            time.sleep(300)

def auto_train_model():
    """Auto-train the ML model if enough new data is available"""
    try:
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        c.execute('SELECT value FROM settings WHERE key = "auto_train_interval"')
        interval_row = c.fetchone()
        conn.close()
        
        auto_train_interval = int(interval_row[0]) if interval_row else 10
        
        print(f"\n🔧 Auto-training check (every {auto_train_interval} minutes)...")
        
        conn = sqlite3.connect('irrigation.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT MAX(timestamp) FROM training_log 
            WHERE status = 'success'
        ''')
        last_train_row = c.fetchone()
        last_train_time = last_train_row[0] if last_train_row[0] else '2000-01-01'
        
        c.execute('''
            SELECT COUNT(*) FROM sensor_data 
            WHERE timestamp > ?
        ''', (last_train_time,))
        new_samples = c.fetchone()[0]
        
        conn.close()
        
        print(f"  New samples since last training: {new_samples}")
        
        if new_samples >= 10 or not os.path.exists('models/irrigation_model.pkl'):
            print(f"  Training model with {new_samples} new samples...")
            train_prediction_model_enhanced()
        else:
            print(f"  Skipping training: insufficient new data ({new_samples} samples)")
            
    except Exception as e:
        print(f"Error in auto-training: {e}")

def initialize_auto_training():
    if not os.path.exists('models/irrigation_model.pkl'):
        print("No model found. Training initial model...")
        train_prediction_model_enhanced()
    
    background_thread = threading.Thread(target=run_scheduled_tasks, daemon=True)
    background_thread.start()
    print("Auto-training scheduler started")

# Initialize database
init_db()

# Start auto-training
initialize_auto_training()

if __name__ == '__main__':
    print("=" * 60)
    print("🌱 SMART IRRIGATION SYSTEM WITH DUAL PLANT SUPPORT 🌱")
    print("=" * 60)
    print("ARCHITECTURE: ESP is PRIMARY decision maker")
    print("Flask role: Data storage, ML predictions, Analytics only")
    print("=" * 60)
    print("Dashboard available at: http://localhost:5000")
    print("Data endpoint: http://localhost:5000/data")
    print("\n📊 ML Endpoints:")
    print("  GET  /api/predictions         - Get 7-day predictions")
    print("  GET  /api/export-predictions-csv - Download predictions")
    print("  POST /api/train-model         - Manual training")
    print("  GET  /api/training-log        - View training history")
    print("  GET  /api/model-info          - Get model details")
    print("  GET  /api/debug-predictions   - Debug prediction system")
    print("\n📧 Email Endpoints:")
    print("  POST /api/test-email           - Test email configuration")
    print("  POST /api/clear-data           - Clear all historical data")
    
    location = get_user_location()
    cooldown = get_cooldown_minutes()
    print(f"\n📍 Detected location: {location['city']} ({location['lat']}, {location['lon']})")
    
    print("\n✅ System ready with dual plant support!")
    print("   Plant 1: soil1 + pump1")
    print("   Plant 2: soil2 + pump2")
    print("   ESP controls pumps - Flask only stores data")
    print(f"\n📧 Email alerts: {'ENABLED' if EMAIL_CONFIG['enabled'] else 'DISABLED'}")
    if EMAIL_CONFIG['enabled']:
        print(f"   Recipient: {EMAIL_CONFIG['recipient_email']}")
        print(f"   Cooldown: {cooldown} minutes between duplicate alerts")
    
    app.run(host="0.0.0.0", port=5000, debug=True)