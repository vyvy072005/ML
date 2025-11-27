import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Загрузка сохраненной модели
model = load_model('cnn_cifar10_model.h5')

# Функция для загрузки данных из файла CIFAR-10
def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, labels

# Функция для предобработки данных
def preprocess_data(data):
    data = data.astype('float32') / 255.0
    return data

# Функция для оценки модели на новых данных
def evaluate_data(data, labels):
    data = preprocess_data(data)
    labels_categorical = to_categorical(labels, num_classes=10)
    loss, accuracy = model.evaluate(data, labels_categorical, verbose=0)
    print(f'Точность модели: {accuracy:.4f}')

# Основная часть программы
def main():
    print("Выберите способ загрузки данных для проверки модели:")
    print("1. Загрузить из файла CIFAR-10 (batch файл)")
    print("2. Загрузить изображение из файла")

    choice = input("Введите номер варианта (1/2): ")

    if choice == '1':
        filename = input("Введите путь к файлу CIFAR-10 (например, 'cifar-10-batches-py/data_batch_5'): ")
        try:
            data, labels = load_cifar10_batch(filename)
            # Вариант для проверки
            evaluate_data(data, labels)
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")


    elif choice == '2':

        image_path = input("Введите путь к изображению (например, 'test_image.png'): ")

        try:

            # Загружаем изображение и изменяем размер

            img = load_img(image_path, target_size=(32, 32))

            img_array = img_to_array(img)

            # Расширяем размерность для предсказания

            img_array = np.expand_dims(img_array, axis=0)

            # Предобработка

            img_array = preprocess_data(img_array)

            # Предсказание

            prediction = model.predict(img_array)

            predicted_class_index = np.argmax(prediction)

            # Названия классов CIFAR-10

            class_names = ['самолёт', 'автомобиль', 'птица', 'кот', 'олень',

                           'собака', 'лягушка', 'лошадь', 'корабль', '']

            predicted_class_name = class_names[predicted_class_index]

            print(f"Предсказанный класс: {predicted_class_name} (номер: {predicted_class_index})")

        except Exception as e:

            print(f"Ошибка при загрузке изображения: {e}")


    else:
        print("Некорректный выбор. Попробуйте снова.")

if __name__ == '__main__':
    main()