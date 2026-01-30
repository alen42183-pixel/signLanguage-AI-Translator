import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# --- НАСТРОЙКИ ---
DATA_PATH = os.path.join('MP_Data')

# !!! СЮДА ВСТАВЬ ТОТ ЖЕ СПИСОК, ЧТО И В COLLECT_DATA.PY !!!
actions = np.array(['Help me', 'SOS', 'nothing'])

no_sequences = 30
sequence_length = 30

# --- 1. ЗАГРУЗКА ДАННЫХ ---
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Загружаю данные...")
for action in actions:
    for sequence in range(no_sequences):
        window = []
        try:
            # Загружаем все 30 кадров одного видео
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
        except Exception as e:
            print(f"Ошибка в файле {action}/{sequence}: {e}")

X = np.array(sequences)
y = to_categorical(labels).astype(int) # Превращаем метки в категории (0 -> [1,0,0], 1 -> [0,1,0])

# Разделяем на данные для обучения (95%) и теста (5%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- 2. СОЗДАНИЕ МОДЕЛИ (МОЗГ) ---
model = Sequential()

# Слой 1: LSTM (Запоминает последовательность)
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
# Слой 2: LSTM
model.add(LSTM(128, return_sequences=True, activation='relu'))
# Слой 3: LSTM (Последний запоминающий слой)
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Слой 4: Dense (Классификация)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Слой 5: Финал (Выдает вероятности для каждого из 12 жестов)
model.add(Dense(actions.shape[0], activation='softmax'))

# Компиляция
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 3. ОБУЧЕНИЕ ---
print("Начинаю обучение нейросети... Это займет время.")
# epochs=200 — значит, нейросеть пройдет по данным 200 раз
model.fit(X_train, y_train, epochs=200, callbacks=[TensorBoard(log_dir='logs')])

# --- 4. СОХРАНЕНИЕ ---
model.summary()
model.save('action.h5')
print("Модель сохранена как action.h5! Готово!")