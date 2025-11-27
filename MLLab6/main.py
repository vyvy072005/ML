import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, labels


path_to_cifar10 = 'cifar-10-batches-py/'  # замените на ваш путь

train_data_list = []
train_labels_list = []

for i in range(1, 6):
    data, labels = load_cifar10_batch(f'{path_to_cifar10}data_batch_{i}')
    train_data_list.append(data)
    train_labels_list.extend(labels)

train_data = np.concatenate(train_data_list)
train_labels = np.array(train_labels_list)

# --------- Предобработка данных ---------
train_data = train_data.astype('float32') / 255.0
train_labels = to_categorical(train_labels, num_classes=10)

# --------- Создание модели ---------
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

model = create_cnn_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --------- Обучение модели ---------
model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_split=0.1)

# --------- Сохранение модели ---------
model.save('cnn_cifar10_model.h5')
print("Модель сохранена в файл 'cnn_cifar10_model.h5'.")

# --------- Загрузка модели и оценка ---------
# Для проверки: загрузим модель из файла
loaded_model = load_model('cnn_cifar10_model.h5')

# Функция для оценки точности на новых данных
def evaluate_new_data(data, labels):
    data = data.astype('float32') / 255.0
    labels_categorical = to_categorical(labels, num_classes=10)
    loss, accuracy = loaded_model.evaluate(data, labels_categorical, verbose=0)
    print(f'Точность на новых данных: {accuracy:.4f}')

# --------- Пример использования ---------
# Для проверки на новых данных, повторите загрузку данных, как выше
# Например, загрузим еще один батч или другой набор данных
new_data, new_labels = load_cifar10_batch('data_batch_5')
evaluate_new_data(new_data, new_labels)