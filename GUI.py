import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.core")
# Загрузка модели (укажите актуальное имя)
model = joblib.load('trained_ensemble_20250524_145441.joblib')  # 👈 замените!

feature_names = [
    'Gender', 'Age', 'Driving_License', 'Region_Code',
    'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
    'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'
]

# Коды категориальных признаков
category_maps = {
    'Gender': {'Мужской': 1, 'Женский': 0},
    'Vehicle_Age': {'< 1 год': 0, '1-2 года': 1, '> 2 лет': 2},
    'Vehicle_Damage': {'Да': 1, 'Нет': 0}
}


def predict_response():
    try:
        input_data = {
            'Gender': category_maps['Gender'][gender_combo.get()],
            'Age': float(entry_fields[0].get()),
            'Driving_License': float(entry_fields[1].get()),
            'Region_Code': float(entry_fields[2].get()),
            'Previously_Insured': float(entry_fields[3].get()),
            'Vehicle_Age': category_maps['Vehicle_Age'][vehicle_age_combo.get()],
            'Vehicle_Damage': category_maps['Vehicle_Damage'][vehicle_damage_combo.get()],
            'Annual_Premium': float(entry_fields[4].get()),
            'Policy_Sales_Channel': float(entry_fields[5].get()),
            'Vintage': float(entry_fields[6].get())
        }

        input_df = pd.DataFrame([input_data])[feature_names]
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][prediction] * 100

        label_result.config(
            text=f"Предсказание: {'Откликнется' if prediction == 1 else 'Не откликнется'}\n"
                 f"Уверенность: {confidence:.2f}%"
        )

    except Exception as e:
        messagebox.showerror("Ошибка", f"Неверный ввод: {str(e)}")


# Инициализация окна
root = tk.Tk()
root.title("Прогноз отклика на страхование (Ансамбль)")

# Числовые признаки
labels = [
    "Возраст", "Водительское удостоверение (1/0)", "Код региона", "Ранее застрахован (1/0)",
    "Годовая премия", "Канал продаж полиса", "Стаж клиента (Vintage)"
]

entry_fields = []

# Поля для ввода числовых данных
for i, label_text in enumerate(labels):
    tk.Label(root, text=label_text).grid(row=i, column=0, padx=5, pady=5, sticky='e')
    entry = tk.Entry(root, width=30)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entry_fields.append(entry)

row_index = len(labels)

# Поля для категориальных переменных
tk.Label(root, text="Пол").grid(row=row_index, column=0, padx=5, pady=5, sticky='e')
gender_combo = ttk.Combobox(root, values=["Мужской", "Женский"], state="readonly", width=28)
gender_combo.current(0)
gender_combo.grid(row=row_index, column=1, padx=5, pady=5)

tk.Label(root, text="Возраст автомобиля").grid(row=row_index+1, column=0, padx=5, pady=5, sticky='e')
vehicle_age_combo = ttk.Combobox(root, values=["< 1 год", "1-2 года", "> 2 лет"], state="readonly", width=28)
vehicle_age_combo.current(0)
vehicle_age_combo.grid(row=row_index+1, column=1, padx=5, pady=5)

tk.Label(root, text="Повреждение автомобиля").grid(row=row_index+2, column=0, padx=5, pady=5, sticky='e')
vehicle_damage_combo = ttk.Combobox(root, values=["Да", "Нет"], state="readonly", width=28)
vehicle_damage_combo.current(0)
vehicle_damage_combo.grid(row=row_index+2, column=1, padx=5, pady=5)

# Кнопка предсказания
tk.Button(root, text="Сделать предсказание", command=predict_response).grid(row=row_index+3, columnspan=2, pady=10)

# Вывод результата
label_result = tk.Label(root, text="Модель: Ансамбль\nПредсказание: Нет данных\nУверенность: Нет данных", font=("Arial", 12), fg="blue")
label_result.grid(row=row_index+4, columnspan=2, pady=10)

# Запуск интерфейса
root.mainloop()
