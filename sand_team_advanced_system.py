import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
import sqlite3
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime
import bcrypt
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from transformers import pipeline, DistilBertTokenizer, TFDistilBertForSequenceClassification

# Set page config as the FIRST Streamlit command
st.set_page_config(page_title="نظام سَند - لوحة تحكم متقدمة", layout="wide", initial_sidebar_state="expanded")

# إعداد التسجيل (Logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تعطيل تحذيرات PyTorch
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# إعداد حلقة الأحداث لـ asyncio
import nest_asyncio
nest_asyncio.apply()

# إعداد Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

# إعداد نموذج LLM (DistilBERT)
@st.cache_resource(show_spinner=False)
def load_llm_model():
    try:
        logger.info("جاري تحميل نموذج DistilBERT...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        nlp = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2")
        return tokenizer, model, nlp
    except Exception as e:
        logger.error(f"خطأ في تحميل النموذج: {str(e)}")
        return None, None, None

tokenizer, bert_model, nlp = load_llm_model()

# إعداد قاعدة البيانات
def init_db():
    conn = sqlite3.connect('sand_team_advanced_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT,
        timestamp TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS team_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_date TEXT,
        team_name TEXT,
        player_name TEXT,
        sprints INTEGER,
        passes INTEGER,
        passes_successful INTEGER,
        shots INTEGER,
        shots_on_target INTEGER,
        contacts INTEGER,
        tackles_successful INTEGER,
        distance_covered REAL,
        fatigue REAL,
        fatigue_trend REAL,
        joint_angles TEXT,
        speed REAL,
        injury_risk REAL,
        possession REAL,
        pressing REAL,
        offensive_patterns REAL,
        defensive_patterns REAL,
        temperature REAL,
        humidity REAL,
        pitch_type TEXT,
        inbody_fat_percentage REAL,
        inbody_muscle_mass REAL,
        inbody_weight REAL,
        inbody_bmi REAL,
        training_plan TEXT,
        timestamp TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS match_timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_date TEXT,
        player_name TEXT,
        time_segment TEXT,
        sprints INTEGER,
        contacts INTEGER,
        fatigue REAL,
        timestamp TEXT
    )''')
    conn.commit()
    return conn

# تشفير كلمة المرور
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# التحقق من كلمة المرور
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# جمع بيانات من StatsBomb للفريق
@st.cache_data(ttl=3600)
def fetch_statsbomb_data(team_name):
    try:
        logger.info(f"جاري جمع بيانات الفريق {team_name} من StatsBomb...")
        url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/3788741.json"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.error("فشل في جلب بيانات StatsBomb")
            return None
        data = response.json()
        events = pd.DataFrame(data)
        
        # قراءة بيانات اللاعبين المحلية
        with open('player_data_realistic.json', 'r', encoding='utf-8') as f:
            local_data = json.load(f)
        
        team_events = events[events['team'] == team_name]
        players = team_events['player'].unique()
    except Exception as e:
        logger.error(f"حدث خطأ أثناء جمع البيانات: {str(e)}")
        return None
    team_stats = {}
    
    # دمج البيانات المحلية مع بيانات StatsBomb
    local_players = {player['name']: player for player in local_data}
    
    for player in players:
        player_events = team_events[team_events['player'] == player]
        local_player_data = local_players.get(player, {})
        
        # إحصائيات المباراة
        sprints = player_events[player_events['type'] == 'Sprint'].shape[0]
        passes = player_events[player_events['type'] == 'Pass'].shape[0]
        passes_successful = player_events[(player_events['type'] == 'Pass') & (player_events['pass_outcome'].isna())].shape[0]
        shots = player_events[player_events['type'] == 'Shot'].shape[0]
        shots_on_target = player_events[(player_events['type'] == 'Shot') & (player_events['shot_outcome'] == 'On Target')].shape[0]
        contacts = player_events[player_events['type'] == 'Duel'].shape[0]
        tackles = player_events[player_events['type'] == 'Tackle'].shape[0]
        tackles_successful = player_events[(player_events['type'] == 'Tackle') & (player_events['tackle_outcome'] == 'Won')].shape[0]
        distance_covered = player_events['distance'].sum() if 'distance' in player_events.columns else 0
        
        # دمج مع البيانات المحلية
        if local_player_data:
            inbody_data = local_player_data.get('inbody', {})
            matches_data = local_player_data.get('matches', [])
            
            # حساب متوسط معدل ضربات القلب والإجهاد
            avg_heart_rate = np.mean([match['heart_rate'] for match in matches_data]) if matches_data else 0
            avg_fatigue = np.mean([match['fatigue'] for match in matches_data]) if matches_data else 0
        
        time_segments = {'0-15': (0, 15), '16-30': (16, 30), '31-45': (31, 45), '46-60': (46, 60), '61-75': (61, 75), '76-90': (76, 90)}
        timeline_data = {}
        for segment, (start, end) in time_segments.items():
            segment_events = player_events[(player_events['minute'] >= start) & (player_events['minute'] <= end)]
            segment_sprints = segment_events[segment_events['type'] == 'Sprint'].shape[0]
            segment_contacts = segment_events[segment_events['type'] == 'Duel'].shape[0]
            
            # حساب الإجهاد مع مراعاة معدل ضربات القلب
            segment_fatigue = (segment_sprints * 0.3 + segment_contacts * 0.3 + (avg_heart_rate/180) * 0.4) if local_player_data else (segment_sprints * 0.4 + segment_contacts * 0.6) / 100
            
            # حساب مؤشر الأداء
            performance_index = ((segment_sprints/10) * 0.4 + (1 - segment_fatigue) * 0.6) * 100
            
            timeline_data[segment] = {
                "sprints": segment_sprints,
                "contacts": segment_contacts,
                "fatigue": segment_fatigue,
                "heart_rate": avg_heart_rate if local_player_data else 0,
                "performance_index": performance_index
            }
        
        # حساب مؤشرات الأداء المتقدمة
        advanced_metrics = {
            "physical_index": (sprints/20 * 0.4 + distance_covered/10000 * 0.6) * 100,
            "technical_index": (passes_successful/passes if passes > 0 else 0) * 0.5 + (shots_on_target/shots if shots > 0 else 0) * 0.5 * 100,
            "tactical_index": (tackles_successful/contacts if contacts > 0 else 0) * 100,
            "overall_performance": 0  # سيتم حسابه لاحقاً
        }

        # دمج البيانات المحلية
        if local_player_data:
            inbody_metrics = {
                "fitness_score": (100 - inbody_data.get('fat_percentage', 20)) * 0.4 + inbody_data.get('muscle_mass', 30) * 0.6,
                "health_index": (1 - abs(inbody_data.get('bmi', 25) - 22) / 10) * 100 if inbody_data.get('bmi') else 70
            }
            advanced_metrics["overall_performance"] = (
                advanced_metrics["physical_index"] * 0.3 +
                advanced_metrics["technical_index"] * 0.3 +
                advanced_metrics["tactical_index"] * 0.2 +
                inbody_metrics["fitness_score"] * 0.2
            )
        else:
            advanced_metrics["overall_performance"] = (
                advanced_metrics["physical_index"] * 0.4 +
                advanced_metrics["technical_index"] * 0.3 +
                advanced_metrics["tactical_index"] * 0.3
            )

        team_stats[player] = {
            "sprints": sprints,
            "passes": passes,
            "passes_successful": passes_successful,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "contacts": contacts,
            "tackles_successful": tackles_successful,
            "distance_covered": distance_covered,
            "timeline": timeline_data,
            "advanced_metrics": advanced_metrics,
            "inbody_data": inbody_data if local_player_data else {},
            "avg_heart_rate": avg_heart_rate if local_player_data else 0,
            "avg_fatigue": avg_fatigue if local_player_data else 0
        }
    
    possession = team_events[team_events['type'] == 'Pass'].shape[0] / len(events) * 100
    pressing = team_events[team_events['type'] == 'Pressure'].shape[0] / len(events) * 100
    
    return {
        "team_stats": team_stats,
        "possession": possession,
        "pressing": pressing
    }

# جمع بيانات من WyScout (افتراضي)
def fetch_wyscout_data():
    logger.info("جاري جمع بيانات من WyScout...")
    return {"offensive_patterns": 0.65, "defensive_patterns": 0.35}

# جمع بيانات بيئية (افتراضي)
def fetch_environmental_data():
    logger.info("جاري جمع بيانات بيئية...")
    return {
        "temperature": 27.0,
        "humidity": 65.0,
        "pitch_type": "عشب طبيعي"
    }

# جمع بيانات InBody (افتراضي)
def fetch_inbody_data(player_name):
    logger.info(f"جاري جمع بيانات InBody لـ {player_name}...")
    return {
        "fat_percentage": 12.5,
        "muscle_mass": 38.0,
        "weight": 75.0,
        "bmi": 22.5
    }

# تحليل الفيديو لتتبع اللاعبين
def analyze_video(video_path, team_stats):
    logger.info(f"جاري تحليل الفيديو: {video_path}")
    cap = cv2.VideoCapture(video_path)
    players_data = {}
    frame_count = 0
    
    player_names = list(team_stats.keys())
    player_index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            if player_index >= len(player_names):
                player_index = 0
            player_id = player_names[player_index]
            if player_id not in players_data:
                players_data[player_id] = {
                    "joint_angles": {"knee": [], "shoulder": [], "elbow": [], "ankle": [], "hip": []},
                    "speeds": [],
                    "times": [],
                    "prev_position": None,
                    "contacts": []
                }
            
            left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y])
            left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
            left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y])
            v1 = left_hip - left_knee
            v2 = left_ankle - left_knee
            angle_knee = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            players_data[player_id]["joint_angles"]["knee"].append(angle_knee)
            
            left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
            left_elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y])
            left_hip_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
            v1_shoulder = left_elbow - left_shoulder
            v2_shoulder = left_hip_shoulder - left_shoulder
            angle_shoulder = np.degrees(np.arccos(np.dot(v1_shoulder, v2_shoulder) / (np.linalg.norm(v1_shoulder) * np.linalg.norm(v2_shoulder))))
            players_data[player_id]["joint_angles"]["shoulder"].append(angle_shoulder)
            
            left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y])
            v1_elbow = left_shoulder - left_elbow
            v2_elbow = left_wrist - left_elbow
            angle_elbow = np.degrees(np.arccos(np.dot(v1_elbow, v2_elbow) / (np.linalg.norm(v1_elbow) * np.linalg.norm(v2_elbow))))
            players_data[player_id]["joint_angles"]["elbow"].append(angle_elbow)
            
            left_heel = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y])
            v1_ankle = left_knee - left_ankle
            v2_ankle = left_heel - left_ankle
            angle_ankle = np.degrees(np.arccos(np.dot(v1_ankle, v2_ankle) / (np.linalg.norm(v1_ankle) * np.linalg.norm(v2_ankle))))
            players_data[player_id]["joint_angles"]["ankle"].append(angle_ankle)
            
            right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])
            v1_hip = left_shoulder - left_hip
            v2_hip = right_hip - left_hip
            angle_hip = np.degrees(np.arccos(np.dot(v1_hip, v2_hip) / (np.linalg.norm(v1_hip) * np.linalg.norm(v2_hip))))
            players_data[player_id]["joint_angles"]["hip"].append(angle_hip)
            
            current_position = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
            if players_data[player_id]["prev_position"] is not None:
                distance = np.linalg.norm(current_position - players_data[player_id]["prev_position"])
                speed = distance * 30
                players_data[player_id]["speeds"].append(speed)
            players_data[player_id]["prev_position"] = current_position
            players_data[player_id]["times"].append(frame_count / 30)
            
            if angle_knee > 160 and speed > 0.5:
                players_data[player_id]["contacts"].append(frame_count / 30)
            
            player_index += 1
        
        if frame_count % 100 == 0:
            logger.info(f"تم تحليل {frame_count} إطار")
    
    cap.release()
    
    processed_data = {}
    for player_id, data in players_data.items():
        processed_data[player_id] = {
            "joint_angles": {
                "knee": np.mean(data["joint_angles"]["knee"]) if data["joint_angles"]["knee"] else 0,
                "shoulder": np.mean(data["joint_angles"]["shoulder"]) if data["joint_angles"]["shoulder"] else 0,
                "elbow": np.mean(data["joint_angles"]["elbow"]) if data["joint_angles"]["elbow"] else 0,
                "ankle": np.mean(data["joint_angles"]["ankle"]) if data["joint_angles"]["ankle"] else 0,
                "hip": np.mean(data["joint_angles"]["hip"]) if data["joint_angles"]["hip"] else 0
            },
            "speed": np.mean(data["speeds"]) if data["speeds"] else 0,
            "speeds": data["speeds"],
            "times": data["times"],
            "contacts": len(data["contacts"])
        }
    return processed_data

# تحليل لحظي للفيديو
def analyze_video_realtime(video_path, team_stats, player_name):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    data = {"times": [], "speeds": [], "fatigue": [], "contacts": []}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y])
            left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
            left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y])
            v1 = left_hip - left_knee
            v2 = left_ankle - left_knee
            angle_knee = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            
            current_position = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
            speed = 0
            if "prev_position" in data:
                distance = np.linalg.norm(current_position - data["prev_position"])
                speed = distance * 30
            data["prev_position"] = current_position
            data["speeds"].append(speed)
            data["times"].append(frame_count / 30)
            
            if angle_knee > 160 and speed > 0.5:
                data["contacts"].append(frame_count / 30)
            fatigue = (speed * 0.3 + len(data["contacts"]) * 0.7) / 100
            data["fatigue"].append(fatigue)
        
        if frame_count >= 300:
            break
    
    cap.release()
    return data

# حساب الإجهاد
def calculate_fatigue(sprints, contacts, speed, distance_covered, temperature, humidity, pitch_type, inbody_data, heart_rate=0):
    try:
        # حساب الإجهاد الأساسي مع مراعاة معدل ضربات القلب
        base_fatigue = (
            sprints * 0.15 +
            contacts * 0.2 +
            speed * 0.15 +
            distance_covered * 0.1 +
            (heart_rate/180) * 0.2
        ) / 100

        # عوامل البيئة
        env_factor = (
            (temperature/40) * 0.1 +
            (humidity/100) * 0.1
        ) / 100

        # عامل نوع الملعب
        pitch_factor = 1.2 if pitch_type == "عشب صناعي" else 1.0

        # عوامل اللياقة البدنية
        if inbody_data:
            inbody_factor = (
                (inbody_data.get("fat_percentage", 20) * 0.05) +
                ((100 - inbody_data.get("muscle_mass", 40)) * 0.05) +
                (abs(inbody_data.get("bmi", 25) - 22) * 0.02)
            ) / 100
        else:
            inbody_factor = 0.1

        # حساب الإجهاد النهائي
        fatigue = base_fatigue + env_factor + (inbody_factor * pitch_factor)
        
        # تطبيق عوامل التصحيح
        if heart_rate > 160:
            fatigue *= 1.2
        if temperature > 35:
            fatigue *= 1.15
        if humidity > 80:
            fatigue *= 1.1

        return min(fatigue * 10, 10.0)
    except Exception as e:
        logger.error(f"خطأ في حساب الإجهاد: {str(e)}")
        return 5.0  # قيمة افتراضية معتدلة

# حساب اتجاه الإجهاد واحتمالية الإصابة
def calculate_fatigue_trend_and_injury_risk(timeline_data, player_stats, env_data):
    try:
        segments = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
        
        # تحليل الإجهاد
        fatigue_values = [timeline_data[seg]["fatigue"] for seg in segments]
        performance_values = [timeline_data[seg]["performance_index"] for seg in segments]
        
        # حساب الاتجاهات
        fatigue_trend = (fatigue_values[-1] - fatigue_values[0]) / max(fatigue_values[0], 1) * 100
        performance_trend = (performance_values[-1] - performance_values[0]) / max(performance_values[0], 1) * 100
        
        # عوامل الخطورة
        risk_factors = {
            "high_fatigue": len([f for f in fatigue_values if f > 7]) / len(fatigue_values),
            "performance_drop": abs(min(0, performance_trend)) / 100,
            "high_intensity": player_stats["sprints"] / 30,
            "environmental_stress": (env_data["temperature"] / 40 + env_data["humidity"] / 100) / 2
        }
        
        # حساب احتمالية الإصابة
        injury_risk = (
            risk_factors["high_fatigue"] * 0.35 +
            risk_factors["performance_drop"] * 0.25 +
            risk_factors["high_intensity"] * 0.25 +
            risk_factors["environmental_stress"] * 0.15
        )
        
        return {
            "fatigue_trend": fatigue_trend,
            "performance_trend": performance_trend,
            "injury_risk": min(injury_risk, 1.0),
            "risk_factors": risk_factors
        }
    except Exception as e:
        logger.error(f"خطأ في حساب اتجاه الإجهاد واحتمالية الإصابة: {str(e)}")
        return {
            "fatigue_trend": 0,
            "performance_trend": 0,
            "injury_risk": 0.5,
            "risk_factors": {}
        }

# تدريب نموذج تحليل المباراة
def train_match_model(data):
    try:
        logger.info("جاري تدريب نموذج تحليل المباراة...")
        
        # تحضير البيانات
        features = [
            "sprints", "contacts", "fatigue", 
            "joint_angles_knee", "joint_angles_shoulder", "joint_angles_elbow", "joint_angles_ankle", "joint_angles_hip",
            "speed", "distance_covered", "temperature", "humidity",
            "inbody_fat_percentage", "inbody_muscle_mass", "inbody_bmi",
            "heart_rate", "performance_index", "technical_index", "tactical_index"
        ]
        
        X = data[features]
        y = data["injury_risk"]
        
        # معالجة القيم المفقودة
        X = X.fillna(X.mean())
        
        # تطبيع البيانات
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # بناء النموذج
        inputs = Input(shape=(X_train.shape[1],))
        
        # طبقة المعالجة الأولية
        x = Dense(512, activation='relu')(inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(0.3)(x)
        
        # طبقة الانتباه متعدد الرؤوس
        x = tf.expand_dims(x, axis=1)
        attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        # طبقات التحليل العميق
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(0.3)(x)
        
        # طبقة التجميع والإخراج
        x = tf.reduce_mean(x, axis=1)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # تجميع وتدريب النموذج
        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # تدريب النموذج مع التحقق المبكر
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # تقييم النموذج
        test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"نتائج النموذج - الدقة: {test_accuracy:.4f}, AUC: {test_auc:.4f}")
        
        return model, scaler, features
        
    except Exception as e:
        logger.error(f"خطأ في تدريب النموذج: {str(e)}")
        return None, None, None

# تحليل بيانات اللاعب باستخدام LLM
def analyze_player_with_llm(player_name, stats, fatigue, injury_risk, advanced_metrics=None, inbody_data=None):
    try:
        if not nlp:
            return "نظام التحليل غير متاح حالياً"
            
        # تجميع البيانات الأساسية
        base_text = f"اللاعب: {player_name}\n"
        base_text += f"الإحصائيات الأساسية:\n"
        base_text += f"- الركضات السريعة: {stats['sprints']}\n"
        base_text += f"- التمريرات: {stats.get('passes', 0)} (ناجحة: {stats.get('passes_successful', 0)})\n"
        base_text += f"- التدخلات: {stats['contacts']}\n"
        base_text += f"- المسافة المقطوعة: {stats.get('distance_covered', 0):.1f} متر\n"
        base_text += f"- الإجهاد: {fatigue:.2f}/10\n"
        base_text += f"- احتمالية الإصابة: {injury_risk:.1%}\n"
        
        # إضافة المؤشرات المتقدمة
        if advanced_metrics:
            base_text += f"\nالمؤشرات المتقدمة:\n"
            base_text += f"- مؤشر الأداء البدني: {advanced_metrics.get('physical_index', 0):.1f}%\n"
            base_text += f"- مؤشر الأداء الفني: {advanced_metrics.get('technical_index', 0):.1f}%\n"
            base_text += f"- مؤشر الأداء التكتيكي: {advanced_metrics.get('tactical_index', 0):.1f}%\n"
            base_text += f"- الأداء الإجمالي: {advanced_metrics.get('overall_performance', 0):.1f}%\n"
        
        # إضافة بيانات InBody
        if inbody_data:
            base_text += f"\nبيانات اللياقة البدنية:\n"
            base_text += f"- نسبة الدهون: {inbody_data.get('fat_percentage', 0):.1f}%\n"
            base_text += f"- كتلة العضلات: {inbody_data.get('muscle_mass', 0):.1f} كجم\n"
            base_text += f"- مؤشر كتلة الجسم: {inbody_data.get('bmi', 0):.1f}\n"
        
        # توليد التحليل
        prompt = f"قدم تحليلاً شاملاً لأداء اللاعب بناءً على البيانات التالية:\n{base_text}\n"
        prompt += "قدم توصيات محددة لتحسين الأداء وتقليل مخاطر الإصابة.\n"
        
        result = nlp(prompt, max_length=300, num_return_sequences=1)
        analysis = result[0]["generated_text"]
        
        # تنظيف وتنسيق التحليل
        analysis = analysis.replace(prompt, "").strip()
        analysis = "\n".join(line.strip() for line in analysis.split("\n") if line.strip())
        
        return analysis
        
    except Exception as e:
        logger.error(f"خطأ في تحليل بيانات اللاعب: {str(e)}")
        return "عذراً، حدث خطأ في تحليل البيانات"

# إنشاء خطة تدريبية
def generate_training_plan(injury_risk, fatigue, speed, distance_covered, joint_angles, temperature, humidity, pitch_type, inbody_data):
    plan = []
    if injury_risk > 0.8:
        plan.append("🚨 تحذير عاجل: مخاطر الإصابة مرتفعة جدًا! توقف عن التمارين عالية الكثافة فورًا.")
        plan.append("الراحة: خذ استراحة كاملة لمدة 5 أيام مع تمارين استشفاء خفيفة (المشي 20 دقيقة يوميًا).")
    elif injury_risk > 0.6:
        plan.append("⚠️ تقليل الحمل التدريبي: قلل التمارين عالية الكثافة بنسبة 50%.")
        plan.append("زيادة الراحة: أضف 3 أيام راحة إضافية هذا الأسبوع.")
    elif injury_risk > 0.4:
        plan.append("⚠️ مراقبة: قلل التمارين عالية الكثافة بنسبة 30%.")
        plan.append("تمارين استشفاء: أضف جلسات يوغا يومية لمدة 30 دقيقة.")
    else:
        plan.append("✅ استمر في الروتين: يمكن زيادة الشدة بنسبة 20% مع الحفاظ على التوازن.")
    
    if fatigue > 8:
        plan.append("تقليل الإجهاد: توقف عن التمارين الشاقة لمدة 3 أيام، ركز على التمدد والاستشفاء.")
    elif fatigue > 6:
        plan.append("مراقبة الإجهاد: أضف جلسات استشفاء (مثل الساونا أو المساج) بعد كل تمرين.")
    elif fatigue > 4:
        plan.append("تحسين التعافي: أضف تمارين تنفس عميق لمدة 10 دقائق يوميًا.")
    else:
        plan.append("تمارين القوة: أضف تمارين مقاومة 3 مرات أسبوعيًا لتحسين القدرة البدنية.")
    
    if speed < 0.4:
        plan.append("تحسين السرعة: أضف تمارين ركض سريع (Sprints) 4 مرات أسبوعيًا، 10 ركضات لكل جلسة.")
    elif speed < 0.6:
        plan.append("زيادة السرعة: أضف تمارين ركض سريع (Sprints) مرتين أسبوعيًا، 8 ركضات لكل جلسة.")
    
    if distance_covered < 8:
        plan.append("زيادة التحمل: أضف تمارين جري طويلة (5 كم) مرتين أسبوعيًا.")
    elif distance_covered > 12:
        plan.append("تقليل الحمل: قلل المسافات المقطوعة بنسبة 20% لتجنب الإرهاق.")
    
    if joint_angles["knee"] > 165:
        plan.append("تحسين مرونة الركبة: تمارين تمدد يومية للركبة لمدة 20 دقيقة.")
    if joint_angles["shoulder"] < 75:
        plan.append("تقوية الكتف: تمارين مقاومة للكتف بأوزان خفيفة 3 مرات أسبوعيًا.")
    if joint_angles["ankle"] > 40:
        plan.append("تحسين مرونة الكاحل: تمارين دوران الكاحل لمدة 10 دقائق يوميًا.")
    if joint_angles["hip"] < 90:
        plan.append("تقوية الورك: تمارين السكوات 3 مرات أسبوعيًا، 3 مجموعات بـ 15 تكرار.")
    
    if temperature > 32 or humidity > 85:
        plan.append("تأثير بيئي: تجنب التمارين في الهواء الطلق، ركز على التمارين الداخلية.")
    elif temperature > 28 or humidity > 75:
        plan.append("تأثير بيئي: قلل التمارين الخارجية بنسبة 30%، اشرب 2 لتر ماء إضافي يوميًا.")
    
    if pitch_type == "عشب صناعي":
        plan.append("تأثير الأرضية: ارتد أحذية بقبضة قوية لتجنب الانزلاق، ركز على تقوية الكاحل.")
    
    if inbody_data["fat_percentage"] > 15:
        plan.append("تقليل الدهون: أضف تمارين كارديو 4 مرات أسبوعيًا، 30 دقيقة لكل جلسة.")
    if inbody_data["muscle_mass"] < 35:
        plan.append("زيادة الكتلة العضلية: أضف تمارين رفع أثقال 4 مرات أسبوعيًا، ركز على العضلات الكبيرة.")
    if inbody_data["bmi"] > 25:
        plan.append("تقليل الوزن: اتبع نظام غذائي منخفض السعرات مع زيادة الكارديو إلى 5 جلسات أسبوعيًا.")
    elif inbody_data["bmi"] < 18.5:
        plan.append("زيادة الوزن: زِد السعرات اليومية بـ 500 سعرة، ركز على البروتين.")
    
    return "\n".join(plan)

# تخزين البيانات
def store_data(conn, match_date, team_name, player_name, performance, fatigue, fatigue_trend, joint_angles, speed, distance_covered, injury_risk, possession, pressing, offensive_patterns, defensive_patterns, temperature, humidity, pitch_type, inbody_data, training_plan):
    c = conn.cursor()
    joint_angles_str = json.dumps(joint_angles)
    c.execute('''INSERT INTO team_data (match_date, team_name, player_name, sprints, passes, passes_successful, shots, shots_on_target, contacts, tackles_successful, distance_covered, fatigue, fatigue_trend, joint_angles, speed, injury_risk, possession, pressing, offensive_patterns, defensive_patterns, temperature, humidity, pitch_type, inbody_fat_percentage, inbody_muscle_mass, inbody_weight, inbody_bmi, training_plan, timestamp) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (match_date, team_name, player_name, performance["sprints"], performance["passes"], performance["passes_successful"], performance["shots"], performance["shots_on_target"], performance["contacts"], performance["tackles_successful"], distance_covered, fatigue, fatigue_trend, joint_angles_str, speed, injury_risk, possession, pressing, offensive_patterns, defensive_patterns, temperature, humidity, pitch_type, inbody_data["fat_percentage"], inbody_data["muscle_mass"], inbody_data["weight"], inbody_data["bmi"], training_plan, datetime.now().isoformat()))
    conn.commit()

# تخزين بيانات الجدول الزمني
def store_timeline_data(conn, match_date, player_name, timeline_data):
    c = conn.cursor()
    for segment, data in timeline_data.items():
        c.execute('''INSERT INTO match_timeline (match_date, player_name, time_segment, sprints, contacts, fatigue, timestamp) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (match_date, player_name, segment, data["sprints"], data["contacts"], data["fatigue"], datetime.now().isoformat()))
    conn.commit()

# إنشاء تقرير PDF
def generate_pdf_report(team_name, match_date, team_data, players_analysis, env_data, wyscout_data):
    filename = f"match_report_{team_name}_{match_date}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph(f"تقرير مباراة - {team_name} ({match_date})", styles['Title']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("تحليل الفريق", styles['Heading1']))
    story.append(Paragraph(f"نسبة الاستحواذ: {team_data['possession']:.2f}%", styles['BodyText']))
    story.append(Paragraph(f"نسبة الضغط العالي: {team_data['pressing']:.2f}%", styles['BodyText']))
    story.append(Paragraph(f"الأنماط الهجومية: {wyscout_data['offensive_patterns']:.2%}", styles['BodyText']))
    story.append(Paragraph(f"الأنماط الدفاعية: {wyscout_data['defensive_patterns']:.2%}", styles['BodyText']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("البيانات البيئية", styles['Heading1']))
    story.append(Paragraph(f"درجة الحرارة: {env_data['temperature']}°C", styles['BodyText']))
    story.append(Paragraph(f"الرطوبة: {env_data['humidity']}%", styles['BodyText']))
    story.append(Paragraph(f"نوع الأرضية: {env_data['pitch_type']}", styles['BodyText']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("تحليل اللاعبين", styles['Heading1']))
    for player_name, analysis in players_analysis.items():
        story.append(Paragraph(f"اللاعب: {player_name}", styles['Heading2']))
        story.append(Paragraph(f"الركضات السريعة: {analysis['stats']['sprints']}", styles['BodyText']))
        story.append(Paragraph(f"التمريرات: {analysis['stats']['passes']}", styles['BodyText']))
        story.append(Paragraph(f"التمريرات الناجحة: {analysis['stats']['passes_successful']}", styles['BodyText']))
        story.append(Paragraph(f"التسديدات: {analysis['stats']['shots']}", styles['BodyText']))
        story.append(Paragraph(f"التسديدات على المرمى: {analysis['stats']['shots_on_target']}", styles['BodyText']))
        story.append(Paragraph(f"التدخلات: {analysis['stats']['contacts']}", styles['BodyText']))
        story.append(Paragraph(f"التدخلات الناجحة: {analysis['stats']['tackles_successful']}", styles['BodyText']))
        story.append(Paragraph(f"المسافة المقطوعة: {analysis['stats']['distance_covered']:.2f} كم", styles['BodyText']))
        story.append(Paragraph(f"الإجهاد: {analysis['fatigue']:.2f}", styles['BodyText']))
        story.append(Paragraph(f"زاوية الركبة: {analysis['video_data']['joint_angles']['knee']:.2f} درجة", styles['BodyText']))
        story.append(Paragraph(f"زاوية الكتف: {analysis['video_data']['joint_angles']['shoulder']:.2f} درجة", styles['BodyText']))
        story.append(Paragraph(f"السرعة المتوسطة: {analysis['video_data']['speed']:.2f} وحدة/ثانية", styles['BodyText']))
        story.append(Paragraph(f"احتمالية الإصابة: {analysis['injury_risk']:.2%}", styles['BodyText']))
        story.append(Paragraph(f"نسبة الدهون (InBody): {analysis['inbody_data']['fat_percentage']:.2f}%", styles['BodyText']))
        story.append(Paragraph(f"الكتلة العضلية (InBody): {analysis['inbody_data']['muscle_mass']:.2f} كجم", styles['BodyText']))
        story.append(Paragraph(f"مؤشر كتلة الجسم (InBody): {analysis['inbody_data']['bmi']:.2f}", styles['BodyText']))
        story.append(Paragraph("خطة تدريبية:", styles['Heading3']))
        story.append(Paragraph(analysis['training_plan'].replace('\n', '<br/>'), styles['BodyText']))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    return filename

# CSS مخصص لتصميم احترافي يشبه Apple/Google
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Almarai:wght@400;700&display=swap');

    /* إعدادات عامة */
    html, body, .stApp {
        background: #121212 !important;
        color: #FFFFFF !important;
        font-family: 'SF Pro Display', 'Almarai', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }

    /* تنسيق العناوين */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'SF Pro Display', 'Almarai', sans-serif !important;
        font-weight: 600 !important;
    }

    /* تنسيق النصوص */
    p, div, span, label {
        color: #E0E0E0 !important;
        font-family: 'SF Pro Display', 'Almarai', sans-serif !important;
        font-size: 16px !important;
    }

    /* تصميم البطاقات */
    .card {
        background: #1F1F1F !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin: 10px 0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        animation: fadeIn 0.5s ease-in-out !important;
    }
    .card:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5) !important;
    }

    /* تصميم الأزرار */
    .stButton>button {
        background: linear-gradient(90deg, #00DDEB, #007BFF) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-family: 'SF Pro Display', 'Almarai', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2) !important;
        animation: slideIn 0.5s ease-in-out !important;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00B4D8, #0056D2) !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
    }

    /* تنسيق حقول الإدخال */
    .stTextInput>div>input, .stDateInput>div>input {
        background: #2A2A2A !important;
        color: #FFFFFF !important;
        border: 1px solid #404040 !important;
        border-radius: 12px !important;
        padding: 10px !important;
        font-family: 'SF Pro Display', 'Almarai', sans-serif !important;
        animation: fadeIn 0.5s ease-in-out !important;
    }

    /* تنسيق القوائم المنسدلة */
    .stSelectbox>div>div {
        background: #2A2A2A !important;
        color: #FFFFFF !important;
        border: 1px solid #404040 !important;
        border-radius: 12px !important;
        font-family: 'SF Pro Display', 'Almarai', sans-serif !important;
        animation: slideIn 0.5s ease-in-out !important;
    }

    /* تصميم قائمة اللاعبين */
    .player-card {
        background: #1F1F1F !important;
        border-radius: 12px !important;
        padding: 15px !important;
        margin: 5px 0 !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.3s ease !important;
        animation: zoomIn 0.5s ease-in-out !important;
    }
    .player-card:hover {
        background: #2A2A2A !important;
        transform: scale(1.02) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    }
    .player-card img {
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        margin-left: 10px !important;
    }

    /* تصميم الشريط الجانبي */
    .css-1d391kg, .stSidebar {
        background: #1F1F1F !important;
        border-left: 1px solid #404040 !important;
        animation: slideInLeft 0.5s ease-in-out !important;
    }
    .stSidebar .stButton>button {
        background: transparent !important;
        color: #E0E0E0 !important;
        border: 1px solid #404040 !important;
        border-radius: 12px !important;
        padding: 10px !important;
        transition: all 0.3s ease !important;
    }
    .stSidebar .stButton>button:hover {
        background: #00DDEB !important;
        color: #121212 !important;
    }

    /* تنسيق المخططات */
    .plotly-graph-div {
        background: #1F1F1F !important;
        border-radius: 12px !important;
        padding: 10px !important;
        animation: fadeIn 0.5s ease-in-out !important;
    }

    /* الرسوم المتحركة */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes slideInLeft {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes zoomIn {
        from { transform: scale(0.95); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }

    /* تنسيق الرسائل */
    .stAlert {
        background: #2A2A2A !important;
        border-radius: 12px !important;
        border: 1px solid #404040 !important;
        color: #FFFFFF !important;
        font-family: 'SF Pro Display', 'Almarai', sans-serif !important;
        animation: fadeIn 0.5s ease-in-out !important;
    }

    /* تنسيق الفواصل */
    hr {
        border: 0 !important;
        height: 1px !important;
        background: #404040 !important;
        margin: 20px 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# واجهة Streamlit
def main():
    # تحميل CSS
    load_css()
    
    # إعداد قاعدة البيانات
    conn = init_db()
    
    # إدارة حالة تسجيل الدخول
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
    
    # صفحة تسجيل الدخول/التسجيل
    if not st.session_state['logged_in']:
        st.markdown("<h1 style='text-align: center;'>نظام سَند - لوحة تحكم متقدمة 🏆</h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("تسجيل الدخول")
            login_username = st.text_input("اسم المستخدم", key="login_username")
            login_password = st.text_input("كلمة المرور", type="password", key="login_password")
            if st.button("تسجيل الدخول"):
                c = conn.cursor()
                c.execute("SELECT password FROM users WHERE username = ?", (login_username,))
                result = c.fetchone()
                if result and check_password(login_password, result[0]):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = login_username
                    st.success("تم تسجيل الدخول بنجاح!")
                    st.rerun()
                else:
                    st.error("اسم المستخدم أو كلمة المرور غير صحيحة")
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("إنشاء حساب جديد")
            signup_username = st.text_input("اسم المستخدم", key="signup_username")
            signup_password = st.text_input("كلمة المرور", type="password", key="signup_password")
            signup_role = st.selectbox("الدور", ["مدرب", "محلل", "إداري"])
            if st.button("إنشاء الحساب"):
                try:
                    c = conn.cursor()
                    hashed_password = hash_password(signup_password)
                    c.execute("INSERT INTO users (username, password, role, timestamp) VALUES (?, ?, ?, ?)",
                              (signup_username, hashed_password, signup_role, datetime.now().isoformat()))
                    conn.commit()
                    st.success("تم إنشاء الحساب بنجاح! يرجى تسجيل الدخول.")
                except sqlite3.IntegrityError:
                    st.error("اسم المستخدم موجود بالفعل، اختر اسمًا آخر.")
            st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("<h2>لوحة التحكم</h2>", unsafe_allow_html=True)
        page = st.radio("التنقل", ["اختيار الفريق", "التحليل اللحظي", "تحليل الفريق", "تحليل اللاعبين", "التوصيات", "تصدير التقارير"])
        if st.button("تسجيل الخروج"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.rerun()
    
    # الواجهة الرئيسية
    st.markdown(f"<h1 style='text-align: center;'>نظام سَند - لوحة تحكم متقدمة (مرحبًا، {st.session_state['username']})</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # قسم اختيار الفريق واللاعب
    if page == "اختيار الفريق":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("اختيار الفريق واللاعب")
        teams = ["الفريق الأول", "الفريق الثاني", "الفريق الثالث"]
        team_logos = {
            "الفريق الأول": "https://via.placeholder.com/50?text=فريق1",
            "الفريق الثاني": "https://via.placeholder.com/50?text=فريق2",
            "الفريق الثالث": "https://via.placeholder.com/50?text=فريق3"
        }
        team_name = st.selectbox("اختر الفريق", teams, format_func=lambda x: f"{x}")
        st.markdown(f"""
        <div class='card'>
            <img src='{team_logos[team_name]}' style='width: 50px; height: 50px; border-radius: 50%; margin-bottom: 10px;'>
            <p>الفريق: {team_name}</p>
            <p>عدد اللاعبين: 11</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("اختيار اللاعب")
        players = ["لاعب 1", "لاعب 2", "لاعب 3", "لاعب 4", "لاعب 5"]
        player_images = {player: f"https://via.placeholder.com/40?text={player}" for player in players}
        selected_player = st.session_state.get('selected_player', players[0])
        for player in players:
            if st.button(f"{player}", key=f"player_{player}"):
                st.session_state['selected_player'] = player
                st.rerun()
            st.markdown(f"""
            <div class='player-card'>
                <img src='{player_images[player]}'>
                <p>{player}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("إدخال بيانات المباراة")
        match_date = st.date_input("تاريخ المباراة", value=datetime.now())
        video_path = st.file_uploader("رفع فيديو المباراة", type=["mp4", "avi"])
        if st.button("بدء التحليل"):
            st.session_state['match_date'] = str(match_date)
            st.session_state['team_name'] = team_name
            st.session_state['video_path'] = video_path
            st.session_state['analysis_done'] = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # قسم التحليل اللحظي
    elif page == "التحليل اللحظي":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("التحليل اللحظي")
        if 'video_path' not in st.session_state or not st.session_state['video_path']:
            st.warning("يرجى رفع فيديو المباراة من قسم اختيار الفريق أولاً.")
        else:
            with open("temp_video.mp4", "wb") as f:
                f.write(st.session_state['video_path'].read())
            selected_player = st.session_state.get('selected_player', "لاعب 1")
            st_autorefresh(interval=10000, key="realtime_analysis")
            team_stats = {"لاعب 1": {}, "لاعب 2": {}, "لاعب 3": {}, "لاعب 4": {}, "لاعب 5": {}}  # افتراضي
            realtime_data = analyze_video_realtime("temp_video.mp4", team_stats, selected_player)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h5>السرعة اللحظية</h5>", unsafe_allow_html=True)
                fig_speed = go.Figure()
                fig_speed.add_trace(go.Scatter(x=realtime_data["times"], y=realtime_data["speeds"], mode='lines+markers', name='السرعة', line=dict(color='#00DDEB')))
                if len(realtime_data["speeds"]) > 0:
                    max_speed_idx = np.argmax(realtime_data["speeds"])
                    max_speed_time = realtime_data["times"][max_speed_idx]
                    max_speed = realtime_data["speeds"][max_speed_idx]
                    fig_speed.add_annotation(x=max_speed_time, y=max_speed, text="أعلى سرعة", showarrow=True, arrowhead=1, font=dict(color="#FFFFFF"))
                fig_speed.update_layout(
                    plot_bgcolor='#1F1F1F', paper_bgcolor='#1F1F1F', font=dict(color='#FFFFFF'),
                    xaxis_title="الوقت (ثواني)", yaxis_title="السرعة", title=f"السرعة اللحظية - {selected_player}"
                )
                st.plotly_chart(fig_speed, use_container_width=True)
            
            with col2:
                st.markdown("<h5>الإجهاد اللحظي</h5>", unsafe_allow_html=True)
                fig_fatigue = go.Figure()
                fig_fatigue.add_trace(go.Scatter(x=realtime_data["times"], y=realtime_data["fatigue"], mode='lines+markers', name='الإجهاد', line=dict(color='#FF4D4F')))
                max_fatigue_idx = np.argmax(realtime_data["fatigue"])
                max_fatigue_time = realtime_data["times"][max_fatigue_idx]
                max_fatigue = realtime_data["fatigue"][max_fatigue_idx]
                fig_fatigue.add_annotation(x=max_fatigue_time, y=max_fatigue, text="أعلى إجهاد", showarrow=True, arrowhead=1, font=dict(color="#FFFFFF"))
                fig_fatigue.update_layout(
                    plot_bgcolor='#1F1F1F', paper_bgcolor='#1F1F1F', font=dict(color='#FFFFFF'),
                    xaxis_title="الوقت (ثواني)", yaxis_title="الإجهاد", title=f"الإجهاد اللحظي - {selected_player}"
                )
                st.plotly_chart(fig_fatigue, use_container_width=True)
            
            os.remove("temp_video.mp4")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # قسم تحليل الفريق
    elif page == "تحليل الفريق":
        if 'team_name' not in st.session_state or 'match_date' not in st.session_state:
            st.warning("يرجى اختيار الفريق وتاريخ المباراة من قسم اختيار الفريق أولاً.")
        else:
            team_name = st.session_state['team_name']
            match_date = st.session_state['match_date']
            team_data = fetch_statsbomb_data(team_name)
            if not team_data:
                st.error("فشل في جمع بيانات الفريق")
                return
            
            wyscout_data = fetch_wyscout_data()
            env_data = fetch_environmental_data()
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("تحليل الفريق")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"نسبة الاستحواذ: {team_data['possession']:.2f}%")
                st.write(f"نسبة الضغط العالي: {team_data['pressing']:.2f}%")
                st.write(f"الأنماط الهجومية: {wyscout_data['offensive_patterns']:.2%}")
                st.write(f"الأنماط الدفاعية: {wyscout_data['defensive_patterns']:.2%}")
                fig1 = px.pie(
                    values=[team_data['possession'], 100 - team_data['possession']],
                    names=["الاستحواذ", "فقدان الاستحواذ"],
                    title="نسبة الاستحواذ",
                    color_discrete_sequence=["#00DDEB", "#404040"]
                )
                fig1.update_layout(
                    plot_bgcolor='#1F1F1F', paper_bgcolor='#1F1F1F', font=dict(color='#FFFFFF')
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.write(f"درجة الحرارة: {env_data['temperature']}°C")
                st.write(f"الرطوبة: {env_data['humidity']}%")
                st.write(f"نوع الأرضية: {env_data['pitch_type']}")
                fig2 = px.bar(
                    x=["درجة الحرارة", "الرطوبة"],
                    y=[env_data['temperature'], env_data['humidity']],
                    title="البيانات البيئية",
                    labels={"x": "الفئة", "y": "القيمة"},
                    color_discrete_sequence=["#00DDEB"]
                )
                fig2.update_layout(
                    plot_bgcolor='#1F1F1F', paper_bgcolor='#1F1F1F', font=dict(color='#FFFFFF')
                )
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # قسم تحليل اللاعبين
    elif page == "تحليل اللاعبين":
        if 'team_name' not in st.session_state or 'match_date' not in st.session_state:
            st.warning("يرجى اختيار الفريق وتاريخ المباراة من قسم اختيار الفريق أولاً.")
        else:
            team_name = st.session_state['team_name']
            match_date = st.session_state['match_date']
            if 'video_path' in st.session_state and st.session_state['video_path']:
                with open("temp_video.mp4", "wb") as f:
                    f.write(st.session_state['video_path'].read())
                video_path = "temp_video.mp4"
            else:
                video_path = None
            
            team_data = fetch_statsbomb_data(team_name)
            if not team_data:
                st.error("فشل في جمع بيانات الفريق")
                return
            
            wyscout_data = fetch_wyscout_data()
            env_data = fetch_environmental_data()
            
            if video_path:
                video_data = analyze_video(video_path, team_data["team_stats"])
                os.remove(video_path)
            else:
                video_data = {}
                for player in team_data["team_stats"].keys():
                    video_data[player] = {
                        "joint_angles": {"knee": 0, "shoulder": 0, "elbow": 0, "ankle": 0, "hip": 0},
                        "speed": 0,
                        "speeds": [0],
                        "times": [0],
                        "contacts": 0
                    }
            
            players_analysis = {}
            for player_name, stats in team_data["team_stats"].items():
                inbody_data = fetch_inbody_data(player_name)
                fatigue = calculate_fatigue(
                    stats["sprints"], video_data[player_name]["contacts"],
                    video_data[player_name]["speed"], stats["distance_covered"],
                    env_data["temperature"], env_data["humidity"], env_data["pitch_type"],
                    inbody_data
                )
                fatigue_trend = calculate_fatigue_trend(stats["timeline"])
                
                player_data = pd.DataFrame({
                    "sprints": [stats["sprints"]],
                    "contacts": [video_data[player_name]["contacts"]],
                    "fatigue": [fatigue],
                    "joint_angles_knee": [video_data[player_name]["joint_angles"]["knee"]],
                    "joint_angles_shoulder": [video_data[player_name]["joint_angles"]["shoulder"]],
                    "joint_angles_elbow": [video_data[player_name]["joint_angles"]["elbow"]],
                    "joint_angles_ankle": [video_data[player_name]["joint_angles"]["ankle"]],
                    "joint_angles_hip": [video_data[player_name]["joint_angles"]["hip"]],
                    "speed": [video_data[player_name]["speed"]],
                    "distance_covered": [stats["distance_covered"]],
                    "temperature": [env_data["temperature"]],
                    "humidity": [env_data["humidity"]],
                    "inbody_fat_percentage": [inbody_data["fat_percentage"]],
                    "inbody_muscle_mass": [inbody_data["muscle_mass"]],
                    "inbody_bmi": [inbody_data["bmi"]],
                    "injury_risk": [np.random.uniform(0, 1)]  # قيمة افتراضية لاحتمالية الإصابة
                })
                
                model, scaler = train_match_model(player_data)
                X = scaler.transform(player_data[["sprints", "contacts", "fatigue", "joint_angles_knee", "joint_angles_shoulder", "joint_angles_elbow", "joint_angles_ankle", "joint_angles_hip", "speed", "distance_covered", "temperature", "humidity", "inbody_fat_percentage", "inbody_muscle_mass", "inbody_bmi"]])
                injury_risk = model.predict(X)[0][0]
                
                training_plan = generate_training_plan(
                    injury_risk, fatigue, video_data[player_name]["speed"], stats["distance_covered"],
                    video_data[player_name]["joint_angles"], env_data["temperature"], env_data["humidity"],
                    env_data["pitch_type"], inbody_data
                )
                
                llm_analysis = analyze_player_with_llm(player_name, stats, fatigue, injury_risk)
                
                players_analysis[player_name] = {
                    "stats": stats,
                    "fatigue": fatigue,
                    "fatigue_trend": fatigue_trend,
                    "video_data": video_data[player_name],
                    "injury_risk": injury_risk,
                    "training_plan": training_plan,
                    "llm_analysis": llm_analysis,
                    "inbody_data": inbody_data
                }
                
                store_data(
                    conn, match_date, team_name, player_name, stats, fatigue, fatigue_trend,
                    video_data[player_name]["joint_angles"], video_data[player_name]["speed"],
                    stats["distance_covered"], injury_risk, team_data["possession"], team_data["pressing"],
                    wyscout_data["offensive_patterns"], wyscout_data["defensive_patterns"],
                    env_data["temperature"], env_data["humidity"], env_data["pitch_type"],
                    inbody_data, training_plan
                )
                store_timeline_data(conn, match_date, player_name, stats["timeline"])
            
            st.session_state['players_analysis'] = players_analysis
            st.session_state['team_data'] = team_data
            st.session_state['env_data'] = env_data
            st.session_state['wyscout_data'] = wyscout_data
            st.session_state['analysis_done'] = True
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("تحليل اللاعبين")
            for player_name, analysis in players_analysis.items():
                st.markdown(f"""
                <div class='card'>
                    <h4>{player_name}</h4>
                    <p>الركضات السريعة: {analysis['stats']['sprints']}</p>
                    <p>التمريرات: {analysis['stats']['passes']}</p>
                    <p>التمريرات الناجحة: {analysis['stats']['passes_successful']}</p>
                    <p>التسديدات: {analysis['stats']['shots']}</p>
                    <p>التسديدات على المرمى: {analysis['stats']['shots_on_target']}</p>
                    <p>التدخلات: {analysis['stats']['contacts']}</p>
                    <p>التدخلات الناجحة: {analysis['stats']['tackles_successful']}</p>
                    <p>المسافة المقطوعة: {analysis['stats']['distance_covered']:.2f} كم</p>
                    <p>الإجهاد: {analysis['fatigue']:.2f}</p>
                    <p>اتجاه الإجهاد: {analysis['fatigue_trend']:.2f}%</p>
                    <p>زاوية الركبة: {analysis['video_data']['joint_angles']['knee']:.2f} درجة</p>
                    <p>زاوية الكتف: {analysis['video_data']['joint_angles']['shoulder']:.2f} درجة</p>
                    <p>زاوية الكوع: {analysis['video_data']['joint_angles']['elbow']:.2f} درجة</p>
                    <p>زاوية الكاحل: {analysis['video_data']['joint_angles']['ankle']:.2f} درجة</p>
                    <p>زاوية الورك: {analysis['video_data']['joint_angles']['hip']:.2f} درجة</p>
                    <p>السرعة المتوسطة: {analysis['video_data']['speed']:.2f} وحدة/ثانية</p>
                    <p>احتمالية الإصابة: {analysis['injury_risk']:.2%}</p>
                    <p>نسبة الدهون (InBody): {analysis['inbody_data']['fat_percentage']:.2f}%</p>
                    <p>الكتلة العضلية (InBody): {analysis['inbody_data']['muscle_mass']:.2f} كجم</p>
                    <p>مؤشر كتلة الجسم (InBody): {analysis['inbody_data']['bmi']:.2f}</p>
                    <p>تحليل الأداء (LLM): {analysis['llm_analysis']}</p>
                    <h5>خطة تدريبية</h5>
                    <p>{analysis['training_plan'].replace('\n', '<br>')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # عرض الرسوم البيانية للسرعة والإجهاد
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<h5>السرعة - {player_name}</h5>", unsafe_allow_html=True)
                    fig_speed = go.Figure()
                    fig_speed.add_trace(go.Scatter(
                        x=analysis['video_data']['times'],
                        y=analysis['video_data']['speeds'],
                        mode='lines+markers',
                        name='السرعة',
                        line=dict(color='#00DDEB')
                    ))
                    fig_speed.update_layout(
                        plot_bgcolor='#1F1F1F', paper_bgcolor='#1F1F1F', font=dict(color='#FFFFFF'),
                        xaxis_title="الوقت (ثواني)", yaxis_title="السرعة", title=f"السرعة - {player_name}"
                    )
                    st.plotly_chart(fig_speed, use_container_width=True)
                
                with col2:
                    st.markdown(f"<h5>الإجهاد - {player_name}</h5>", unsafe_allow_html=True)
                    fig_fatigue = go.Figure()
                    fig_fatigue.add_trace(go.Scatter(
                        x=analysis['video_data']['times'],
                        y=[analysis['fatigue']] * len(analysis['video_data']['times']),  # الإجهاد ثابت هنا لعرض تقديري
                        mode='lines+markers',
                        name='الإجهاد',
                        line=dict(color='#FF4D4F')
                    ))
                    fig_fatigue.update_layout(
                        plot_bgcolor='#1F1F1F', paper_bgcolor='#1F1F1F', font=dict(color='#FFFFFF'),
                        xaxis_title="الوقت (ثواني)", yaxis_title="الإجهاد", title=f"الإجهاد - {player_name}"
                    )
                    st.plotly_chart(fig_fatigue, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    # قسم التوصيات
    elif page == "التوصيات":
        if 'analysis_done' not in st.session_state or not st.session_state['analysis_done']:
            st.warning("يرجى إجراء التحليل أولاً من قسم تحليل اللاعبين.")
        else:
            team_data = st.session_state['team_data']
            players_analysis = st.session_state['players_analysis']
            env_data = st.session_state['env_data']
            possession = team_data['possession']
            pressing = team_data['pressing']
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("التوصيات")

            # توصيات الفريق
            team_recommendations = []
            if possession < 50:
                team_recommendations.append("تحسين الاستحواذ: زيادة التمارين التكتيكية للحفاظ على الكرة.")
            else:
                team_recommendations.append("الاستحواذ جيد: حافظ على الإيقاع مع التركيز على تحسين التمريرات الطولية.")

            if pressing < 30:
                team_recommendations.append("زيادة الضغط العالي: أضف تمارين ضغط جماعي في التدريب.")
            elif pressing > 60:
                team_recommendations.append("تقليل الضغط العالي: ركز على التوازن بين الدفاع والهجوم لتجنب الإرهاق.")
            else:
                team_recommendations.append("الضغط العالي متوازن: استمر مع إضافة تدريبات لتحسين التنظيم الدفاعي.")

            if env_data["temperature"] > 30 or env_data["humidity"] > 80:
                team_recommendations.append("تأثير بيئي: قلل التدريبات الخارجية، ركز على الترطيب واستخدام مرافق داخلية.")
            else:
                team_recommendations.append("الظروف البيئية مناسبة: يمكن إجراء التدريبات في الهواء الطلق مع مراقبة الترطيب.")

            st.markdown("<h5>توصيات الفريق</h5>", unsafe_allow_html=True)
            for rec in team_recommendations:
                st.markdown(f"<p>{rec}</p>", unsafe_allow_html=True)

            # توصيات للاعبين
            st.markdown("<h4>توصيات اللاعبين</h4>", unsafe_allow_html=True)
            for player_name, analysis in players_analysis.items():
                st.markdown(f"""
                <div class='card'>
                    <h5>{player_name}</h5>
                    <p>احتمالية الإصابة: {analysis['injury_risk']:.2%}</p>
                    <p>الإجهاد: {analysis['fatigue']:.2f}</p>
                    <p>اتجاه الإجهاد: {analysis['fatigue_trend']:.2f}%</p>
                    <h6>خطة تدريبية مقترحة</h6>
                    <p>{analysis['training_plan'].replace('\n', '<br>')}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # قسم تصدير التقارير
    elif page == "تصدير التقارير":
        if 'analysis_done' not in st.session_state or not st.session_state['analysis_done']:
            st.warning("يرجى إجراء التحليل أولاً من قسم تحليل اللاعبين.")
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("تصدير التقارير")
            team_name = st.session_state['team_name']
            match_date = st.session_state['match_date']
            team_data = st.session_state['team_data']
            players_analysis = st.session_state['players_analysis']
            env_data = st.session_state['env_data']
            wyscout_data = st.session_state['wyscout_data']

            if st.button("إنشاء تقرير PDF"):
                pdf_file = generate_pdf_report(team_name, match_date, team_data, players_analysis, env_data, wyscout_data)
                with open(pdf_file, "rb") as f:
                    pdf_data = f.read()
                b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_file}">تنزيل تقرير PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success(f"تم إنشاء التقرير: {pdf_file}")

            if st.button("تصدير البيانات إلى CSV"):
                c = conn.cursor()
                c.execute("SELECT * FROM team_data WHERE match_date = ? AND team_name = ?", (match_date, team_name))
                data = c.fetchall()
                columns = [desc[0] for desc in c.description]
                df = pd.DataFrame(data, columns=columns)
                csv = df.to_csv(index=False)
                b64_csv = base64.b64encode(csv.encode()).decode('utf-8')
                href = f'<a href="data:file/csv;base64,{b64_csv}" download="team_data_{team_name}_{match_date}.csv">تنزيل ملف CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("تم تصدير البيانات إلى CSV بنجاح!")

            st.markdown("<h4>معاينة البيانات</h4>", unsafe_allow_html=True)
            c = conn.cursor()
            c.execute("SELECT * FROM team_data WHERE match_date = ? AND team_name = ?", (match_date, team_name))
            data = c.fetchall()
            columns = [desc[0] for desc in c.description]
            df = pd.DataFrame(data, columns=columns)
            st.dataframe(df)

            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()