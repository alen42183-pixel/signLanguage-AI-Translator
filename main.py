import cv2
import numpy as np
import os
import mediapipe as mp
import pyttsx3  # Библиотека для голоса
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- НАСТРОЙКИ (Твои 3 жеста) ---
# Убедись, что список ТОЧНО совпадает с train.py
actions = np.array(['help', 'SOS', 'nothing'])

# --- 1. НАСТРОЙКА ГОЛОСА ---
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Скорость речи (можно менять: 100 - медленно, 200 - быстро)

# --- 2. ЗАГРУЗКА МОДЕЛИ (МОЗГА) ---
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Загружаем веса
model.load_weights('action.h5')

# --- 3. ИНИЦИАЛИЗАЦИЯ MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        res = []
        for res_point in hand.landmark:
            res.append(np.array([res_point.x, res_point.y, res_point.z]))
        return np.array(res).flatten()
    else:
        return np.zeros(21*3)

# Переменные
sequence = [] 
threshold = 0.8         # Порог уверенности (80%)
last_spoken = ""        # Память (что мы сказали последним)
display_text = "..."    # Текст на экране

cap = cv2.VideoCapture(0)

# Запуск камеры
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Обработка изображения
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Рисуем скелет руки
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Собираем данные
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Храним последние 30 кадров

        # --- ГЛАВНАЯ ЛОГИКА ---
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            best_guess = actions[np.argmax(res)]
            confidence = res[np.argmax(res)]
            
            # Пишем в терминал текущее состояние (для проверки)
            print(f"Жест: {best_guess} ({confidence:.2f})")

            # СЦЕНАРИЙ 1: Нейросеть уверена (> 80%)
            if confidence > threshold:
                
                # Если это жест "nothing" (тишина)
                if best_guess == 'nothing':
                    display_text = "..."
                    # Если до этого мы что-то говорили — сбрасываем память
                    if last_spoken != "":
                        print(">>> СБРОС ПАМЯТИ (Вижу nothing) <<<")
                        last_spoken = ""
                
                # Если это реальный жест (Help или SOS)
                else:
                    display_text = best_guess
                    
                    # Проверяем, говорили ли мы это уже
                    if best_guess != last_spoken:
                        print(f"--> ГОВОРЮ: {best_guess}") 
                        engine.say(best_guess)
                        engine.runAndWait()
                        last_spoken = best_guess # Запоминаем, чтобы не повторять
            
            # СЦЕНАРИЙ 2: Нейросеть сомневается (< 80%)
            # (Например, ты опускаешь руки, меняешь жест или ушел из кадра)
            else:
                display_text = "..."
                # Это тоже повод сбросить память!
                if last_spoken != "":
                    print(">>> СБРОС ПАМЯТИ (Низкая уверенность) <<<")
                    last_spoken = ""

        # Рисуем красивую плашку и текст
        cv2.rectangle(image, (0,0), (640, 60), (245, 117, 16), -1)
        cv2.putText(image, display_text, (200, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow('Sign Language Translator', image)

        # Выход по кнопке 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()