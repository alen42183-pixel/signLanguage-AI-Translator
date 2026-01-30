import customtkinter as ctk
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- НАСТРОЙКИ ---
actions = np.array(['help', 'SOS', 'nothing'])
MODEL_PATH = 'action.h5'

class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Настройка окна
        self.title("AI Sign Language Translator")
        self.geometry("1000x700") 
        ctk.set_appearance_mode("Dark")
        
        self.init_backend()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- 1. БОКОВАЯ ПАНЕЛЬ (Слева) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SignAI", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Переключатель голоса
        self.voice_switch = ctk.CTkSwitch(self.sidebar_frame, text="Озвучка")
        self.voice_switch.select()
        self.voice_switch.grid(row=1, column=0, padx=20, pady=10)

        # Кнопка выхода
        self.quit_button = ctk.CTkButton(self.sidebar_frame, text="Выход", command=self.close_app, fg_color="#C0392B", hover_color="#E74C3C")
        self.quit_button.grid(row=5, column=0, padx=20, pady=20)

        # --- 2. ГЛАВНЫЙ ЭКРАН (Видео на всё окно) ---
        self.video_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        self.cap = cv2.VideoCapture(0)
        self.update_feed()

    def init_backend(self):
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(actions.shape[0], activation='softmax'))
        self.model.load_weights(MODEL_PATH)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        self.sequence = []
        self.last_spoken = ""
        self.threshold = 0.8 

    def speak_threaded(self, text):
        if self.voice_switch.get() == 1:
            def run():
                self.engine.say(text)
                self.engine.runAndWait()
            threading.Thread(target=run).start()

    def extract_keypoints(self, results):
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            res = []
            for res_point in hand.landmark:
                res.append(np.array([res_point.x, res_point.y, res_point.z]))
            return np.array(res).flatten()
        else:
            return np.zeros(21*3)

    def update_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Превращаем в RGB для Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True

            # Рисуем скелет
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Логика нейросети
            keypoints = self.extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]

            # Текст по умолчанию (пустота)
            display_text = ""
            
            if len(self.sequence) == 30:
                res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                best_guess = actions[np.argmax(res)]
                confidence = res[np.argmax(res)]

                if confidence > self.threshold:
                    if best_guess == 'nothing':
                        self.last_spoken = ""
                        display_text = "..."
                    else:
                        display_text = best_guess.upper()
                        
                        # Озвучка
                        if best_guess != self.last_spoken:
                            self.speak_threaded(best_guess)
                            self.last_spoken = best_guess
                else:
                    self.last_spoken = ""
                    display_text = "..."

            # --- РИСУЕМ ТЕКСТ ПРЯМО НА КАРТИНКЕ (Overlay) ---
            # 1. Рисуем красивую плашку сверху
            # (Координаты: x1=0, y1=0, x2=640, y2=60)
            # Цвет (R, G, B) = (245, 117, 16) - Оранжевый
            cv2.rectangle(image, (0,0), (640, 60), (245, 117, 16), -1)
            
            # 2. Пишем текст поверх плашки
            cv2.putText(image, display_text, (200, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

            # Конвертируем картинку для приложения
            img = Image.fromarray(image)
            
            # Растягиваем на всё окно (например 800x600)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
            
            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img

        self.after(10, self.update_feed)

    def close_app(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()