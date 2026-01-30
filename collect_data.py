import cv2
import numpy as np
import os
import mediapipe as mp
import time # Добавили библиотеку для пауз

# --- НАСТРОЙКИ ---
DATA_PATH = os.path.join('MP_Data') 

# Список из 12 главных жестов
actions = np.array(['Help me', 'SOS', 'nothing'])

no_sequences = 30   # Количество видео для каждого жеста
sequence_length = 30 # Длина каждого видео (в кадрах)

# --- ИНИЦИАЛИЗАЦИЯ ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Извлекает координаты точек руки. Если рук нет — возвращает нули."""
    if results.multi_hand_landmarks:
        # Берем только первую найденную руку (для упрощения)
        hand = results.multi_hand_landmarks[0]
        res = []
        for res_point in hand.landmark:
            res.append(np.array([res_point.x, res_point.y, res_point.z]))
        return np.array(res).flatten()
    else:
        return np.zeros(21*3)

# Создаем папки
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# --- ЗАПУСК ---
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5) as hands:
    
    # ЦИКЛ ПО ЖЕСТАМ
    for action in actions:
        
        # --- БОЛЬШАЯ ПАУЗА ПЕРЕД НОВЫМ ЖЕСТОМ ---
        # Чтобы ты успел понять, что сейчас будем снимать
        print(f"ГОТОВИМСЯ К ЖЕСТУ: {action.upper()}")
        for i in range(5, 0, -1): # Обратный отсчет 5 секунд
            ret, frame = cap.read()
            image = cv2.flip(frame, 1) # Зеркалим для удобства
            
            cv2.putText(image, f'NEXT ACTION: {action.upper()}', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'GET READY: {i}', (200, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(1000) # Ждем 1 секунду
        
        # ЦИКЛ ПО ВИДЕО (30 раз для одного жеста)
        for sequence in range(no_sequences):
            # ЦИКЛ ПО КАДРАМ
            for frame_num in range(sequence_length):

                ret, frame = cap.read()
                if not ret: break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Рисуем скелет
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # ИНФОРМАЦИЯ НА ЭКРАНЕ
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting {} Video #{}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000) # Пауза 1 сек перед началом записи видео
                else: 
                    cv2.putText(image, 'Collecting {} Video #{}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Сохраняем
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()