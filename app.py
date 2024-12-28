#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:


import streamlit as st
import glob
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO


# 1. Задать путь к модели и данным
model_path = 'https://github.com/AVGorbulya/Pastic_CV/blob/main/best.pt'
source_folder = 'https://github.com/AVGorbulya/Pastic_CV/tree/main/test/images'
output_folder = 'https://github.com/AVGorbulya/Pastic_CV/tree/main/predict'

# 2. Запуск модели для предсказаний
@st.cache_resource
def run_model():
    model = YOLO(model_path)
    results = model.predict(source=source_folder, conf=0.25, save=True, verbose=False)
    return results

# 3. Отображение результатов
def show_random_images():
    # Получаем все изображения в папке с результатами
    latest_folder = max(glob.glob(f'{output_folder}predict*/'), key=os.path.getmtime)
    image_paths = glob.glob(f'{latest_folder}/*.jpg')

    # Проверяем, что есть хотя бы 10 изображений
    if len(image_paths) >= 10:
        # Выбираем случайные 10 изображений
        random_images = random.sample(image_paths, 10)
    else:
        # Если изображений меньше 10, выбираем все
        random_images = image_paths

    # Создаем сетку для отображения изображений (например, 2 строки и 5 столбцов)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 строки, 5 столбцов
    axes = axes.ravel()  # Преобразуем в одномерный массив для удобства

    # Отображаем случайные изображения
    for i, img_path in enumerate(random_images):
        img = Image.open(img_path)  # Открываем изображение с помощью PIL
        axes[i].imshow(img)
        axes[i].axis('off')  # Скрываем оси

    # Показываем все изображения в одном окне
    plt.tight_layout()
    st.pyplot(fig)  # Используем Streamlit для отображения

# 4. Интерфейс Streamlit
def main():
    st.title("Marine Plastic Detection with YOLO")

    # Заголовок для предсказания
    st.header("Predictions on Test Images")

    # Кнопка для запуска модели и отображения изображений
    if st.button('Run Predictions'):
        run_model()  # Запуск модели
        st.success("Predictions completed successfully!")
        show_random_images()  # Отображение случайных изображений с результатами

    # Информация о модели
    st.sidebar.header("About the model")
    st.sidebar.text("This model detects marine plastic using YOLOv11 trained on Roboflow dataset.")

if __name__ == '__main__':
    main()
