import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
from datetime import datetime
# Настройка ширины и расположения графика
sns.set_theme(style='whitegrid')
palette='viridis'

# Предупреждения удаление предупреждений
import warnings
warnings.filterwarnings("ignore")

# Установите для параметра display.max_columns значение "Нет"
pd.set_option('display.max_columns', None)

## Данные 1
# Последовательность данных
train_df = pd.read_csv("train.csv")

# Данные 2
test_df = pd.read_csv("test.csv")

# Данные 3
df = pd.read_csv("htrain.csv")

train_df.head()

# Просмотр 5 последних данных
train_df.tail()

# Иноформация о данных
train_df.info()

# Типы данных
train_df.dtypes

# Просмотр строк и столбцов
train_df.shape

# Исследовательский анализ данных (EDA)
print("\nОписательная статистика обучающего набора:")
train_df.describe().T

# Анализ категориальных и числовых переменных
categorical_features = train_df.select_dtypes(include=['object']).columns
numerical_features = train_df.select_dtypes(include=[np.number]).columns

print("\nКатегориальные переменные:", categorical_features)
print("Числовые переменные:", numerical_features)

# Анализ категориальных переменных
for col in categorical_features:
    print(f"\nРаспределение категориальной переменной {col}:")
    print(train_df[col].value_counts())

    # Анализ целевой переменной 'Target'
    print("\nРаспределение целевой переменной 'Target':")
    print(train_df['Response'].value_counts())
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_df, x='Response')
    plt.title("Распределение целевой переменной 'Target'")
    plt.grid(False)
    plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Vehicle_Age', y='Annual_Premium', hue='Response', data=train_df)
plt.title('График ежегодных премий в зависимости от возраста автомобиля и отзывчивости')
plt.grid(False)
plt.show()

# Группировка данных по полу и рассчет общей годовой премию для каждой группы.
total_premium_by_gender = train_df.groupby('Gender')['Annual_Premium'].sum().reset_index()

# Просмотр общего количества призов в разбивке по полу
plt.figure(figsize=(10, 6))
sns.barplot(x='Gender', y='Annual_Premium', data=total_premium_by_gender)
plt.title('Общая страховая премия по полу')
plt.xlabel('Пол')
plt.ylabel('Общая годовая страховая премия')
plt.grid(False)
plt.show()

# Создание возрастных диапазонов
train_df['Age_Bucket'] = pd.cut(train_df['Age'], bins=[18, 25, 35, 50, np.inf], labels=['18-25', '26-35', ' 36-50', '51+'])

# Группировка данных по возрастным группам и полу и рассчет средней годовой премии для каждой группы.
average_premium_by_age_gender = train_df.groupby(['Age_Bucket', 'Gender'])['Annual_Premium'].mean().reset_index()

# Просмотр средней годовой премии в разбивке по возрастным группам и полу
plt.figure(figsize=(20, 10))
sns.barplot(x='Age_Bucket', y='Annual_Premium', hue='Gender', data=average_premium_by_age_gender)
plt.title('Средняя годовая страховая премия по возрастным группам и полу')
plt.xlabel('Возрастной диапазон')
plt.ylabel('Средняя годовая страховая премия')
plt.legend(title='Пол')
plt.grid(False)
plt.show()

# Группировка данных по полу и предыдущему страховому статусу и рассчет средней годовой премии для каждой группы.
average_premium_by_gender_insured = train_df.groupby(['Gender', 'Previously_Insured'])['Annual_Premium'].mean().reset_index()

# Преобразование переменной "Previously_Insured" в более удобочитаемую категорию
average_premium_by_gender_insured['Previously_Insured'] = average_premium_by_gender_insured['Previously_Insured'].map({0: 'No', 1: 'Yes'})

# Просмотр средней годовой страховой премии в разбивке по полу и предыдущему страховому статусу
plt.figure(figsize=(10, 6))
sns.barplot(x='Previously_Insured', y='Annual_Premium', hue='Gender', data=average_premium_by_gender_insured)
plt.title('Средняя годовая премия в разбивке по предыдущему страховому статусу и полу')
plt.xlabel('Предыдущее страхование')
plt.ylabel('Средняя годовая премия')
plt.legend(title='Пол')
plt.grid(False)
plt.show()

# Категориальные переменные для итерации
categorical_variables = ['Gender', 'Vehicle_Damage', 'Vehicle_Age', 'Response']

# Размер фигуры
plt.figure(figsize=(15, 10))

# Цикл по категориальным переменным
for i, var in enumerate(categorical_variables, 1):
 plt.subplot(2, 2, i) # Subplots 2x2
 sns.boxplot(data=train_df, x=var, y='Annual_Premium', palette='viridis')
 plt.title(f'Ежегодная премия за {var}')
 plt.xlabel(var)
 plt.ylabel('Ежегодная премия')
 plt.xticks(rotation=45)

plt.tight_layout()
plt.grid(False)
plt.show()

# Группировка по возрасту и полу и автомобилю, добавив ежегодные страховые взносы
grouped_data = train_df.groupby(['Vehicle_Age', 'Gender'])['Annual_Premium'].sum().reset_index()

# Сгруппированная гистограмма
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_data, x='Vehicle_Age', y='Annual_Premium', hue='Gender', palette='viridis')
plt.title('Общая страховая премия по возрасту автомобиля и полу')
plt.xlabel('Возраст автомобиля')
plt.ylabel('Общая годовая страховая премия')
plt.legend(title='Пол')
plt.grid(False)
plt.show()

# Список жанров
genders = train_df['Gender'].unique()

# Размер фигуры
plt.figure(figsize=(15, 8))

# Перебираем жанры
for i, gender in enumerate(genders, 1):
    plt.subplot(1, 2, i)
    gender_data = train_df[train_df['Gender'] == gender]
    gender_grouped = gender_data.groupby('Vehicle_Age')['Annual_Premium'].sum().reset_index()
    sns.barplot(data=gender_grouped, x='Vehicle_Age', y='Annual_Premium', palette='viridis')
    plt.title(f'Общая годовая страховая премия по возрасту автомобиля ({gender})')
    plt.xlabel('Возраст автомобиля')
    plt.ylabel('Общая годовая страховая премия')

plt.tight_layout()
plt.grid(False)
plt.show()

# Группировка по возрасту транспортного средства и страховому статусу, добавив ежегодные страховые взносы
grouped_data = train_df.groupby(['Vehicle_Age', 'Previously_Insured'])['Annual_Premium'].sum().reset_index()

# Преобразование столбца "Previously_Insured" в строку для лучшей визуализации
grouped_data['Previously_Insured'] = grouped_data['Previously_Insured'].astype(str)

# Сгруппированная гистограмма
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_data, x='Vehicle_Age', y='Annual_Premium', hue='Previously_Insured', palette='viridis')
plt.title('Общая годовая страховая премия по возрасту автомобиля и страховому статусу')
plt.xlabel('Возраст автомобиля')
plt.ylabel('Общая годовая страховая премия')
plt.legend(title='Ранее застрахован')
plt.grid(False)
plt.show()

# Список страховых статусов
statuses = train_df['Previously_Insured'].unique()

# Размер фигуры
plt.figure(figsize=(15, 8))

# Информация о страховых статусах
for i, status in enumerate(statuses, 1):
    plt.subplot(1, 2, i)
    status_data = df[df['Previously_Insured'] == status]
    status_grouped = status_data.groupby('Vehicle_Age')['Annual_Premium'].sum().reset_index()
    sns.barplot(data=status_grouped, x='Vehicle_Age', y='Annual_Premium', palette='viridis')
    plt.title(f'Общая годовая страховая премия по возрасту автомобиля (Страховой статус: {status})')
    plt.xlabel('Возраст автомобиля')
    plt.ylabel('Общая годовая страховая премия')

plt.tight_layout()
plt.grid(False)
plt.show()

# Группировка по возрасту, полу и возрасту транспортного средства, добавив ежегодные страховые взносы
grouped_data = train_df.groupby(['Age', 'Gender', 'Vehicle_Age'])['Annual_Premium'].sum().reset_index()

# Преобразование столбца "Пол" в строку для лучшей визуализации
grouped_data['Gender'] = grouped_data['Gender'].map({'Male': 'Man', 'Female': 'Woman'})

# Настройка размера фигуры
plt.figure(figsize=(30.5, 10))

# Цикл для создания графиков, разделенных по возрасту автомобиля
vehicle_ages = grouped_data['Vehicle_Age'].unique()

for i, vehicle_age in enumerate(vehicle_ages, 1):
    plt.subplot(2, 2, i)
    subset = grouped_data[grouped_data['Vehicle_Age'] == vehicle_age]
    sns.barplot(data=subset, x='Age', y='Annual_Premium', hue='Gender', palette='viridis')
    plt.title(f'Общая годовая страховая премия по возрасту и полу (Возраст автомобиля: {vehicle_age})')
    plt.xlabel('Возраст')
    plt.ylabel('Общая годовая страховая премия')
    plt.legend(title='Пол')
    plt.xticks(rotation=50)

plt.tight_layout()
plt.grid(False)
plt.show()

# Выбор указанных столбцов
columns_of_interest = ["id", "Age", "Driving_License", "Region_Code", "Previously_Insured", "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"]
df_selected = df[columns_of_interest]

# Вычисление корреляционной матрицы
correlation_matrix = df_selected.corr()

# Настройка размера фигуры
plt.figure(figsize=(14, 10))

# Тепловая карта корреляции
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
plt.title('Тепловая карта матрицы корреляций')
plt.show()

# Выбор числовых столбцов
numeric_columns = ["Age", "Annual_Premium", "Vintage"]

# Настройка размера фигуры
plt.figure(figsize=(18, 6))

# Цикл для создания прямоугольных диаграмм для каждого числового столбца
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=train_df, y=column, palette='viridis')
    plt.title(f'Боксплот: {column}')
    plt.ylabel(column)

plt.tight_layout()
plt.grid(False)
plt.show()

# Выбор числовых столбцов
numeric_columns = ["Age", "Annual_Premium", "Vintage"]

# Настройка размера фигуры
plt.figure(figsize=(18, 6))

# Цикл для создания графиков скриптов для каждого числового столбца, разделенных ответом
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=train_df, x='Response', y=column, palette='viridis')
    plt.title(f'Боксплот: {column} по целевой переменной')
    plt.xlabel('Отклик')
    plt.ylabel(column)

plt.tight_layout()
plt.grid(False)
plt.show()

# Описательная статистика, разделенная по ответам
for column in numeric_columns:
    print(f'\nОписательная статистика для {column} по целевой переменной:')
    print(train_df.groupby('Response')[column].describe())

# Выбор числовых столбцов
numeric_columns = ["Age", "Annual_Premium", "Vintage"]

def remove_outliers(train_df, columns):
    for column in columns:
        Q1 = train_df[column].quantile(0.25)
        Q3 = train_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        train_df = train_df[(train_df[column] >= lower_bound) & (train_df[column] <= upper_bound)]
    return train_df


# Удаление выбросов из фрейма данных
df_cleaned = remove_outliers(train_df, numeric_columns)

# Проверка измерения фрейма данных после удаления выбросов
print(f"Исходная размерность: {train_df.shape}")
print(f"Размерность после удаления выбросов: {df_cleaned.shape}")

# Просмотр первых записей очищенного фрейма данных
df_cleaned.head()

# Выбор числовых столбцов
numeric_columns = ["Age", "Annual_Premium", "Vintage"]

# Настройка размера фигуры
plt.figure(figsize=(18, 6))

# Цикл для создания графиков скриптов для каждого числового столбца, разделенных ответом
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=df_cleaned, x='Response', y=column, hue="Response", palette='viridis')
    plt.title(f'Боксплот: {column} по целевой переменной (после удаления выбросов)')
    plt.xlabel('Отклик')
    plt.ylabel(column)

plt.tight_layout()
plt.grid(False)
plt.show()

# Описательная статистика, разделенная по ответам
for column in numeric_columns:
    print(f'\nОписательная статистика для {column} по целевой переменной (после удаления выбросов):')
    print(df_cleaned.groupby('Response')[column].describe())


def optimize_memory_usage(df):
    df = df.copy()

    print("Использование памяти до оптимизации:")
    print(df.memory_usage(deep=True))
    print()

    # Преобразование столбцов в тип "категория", если они присутствуют
    categorical_columns = [
        'Gender', 'Driving_License', 'Region_Code', 'Previously_Insured',
        'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel',
        'Response', 'Age_Bucket'
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Оптимизация числовых столбцов
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], downcast='integer')

    if 'Annual_Premium' in df.columns:
        df['Annual_Premium'] = pd.to_numeric(df['Annual_Premium'], downcast='integer')

    if 'Vintage' in df.columns:
        df['Vintage'] = pd.to_numeric(df['Vintage'], downcast='integer')

    print("Использование памяти после оптимизации:")
    print(df.memory_usage(deep=True))
    print()

    print("Информация о DataFrame после оптимизации:")
    df.info(memory_usage='deep')
    print()

    return df_cleaned


# Оптимизация фреймворков данных
df_cleaned_optimized_train = optimize_memory_usage(df_cleaned)
test_df_optimized_test = optimize_memory_usage(test_df)
df_optimized = optimize_memory_usage(df)

# Коппирование набора данных
train_df = df_cleaned_optimized_train.copy()
test_df = df_optimized.copy()

# 1. Обработка пропущенных значений
print("Количество пропущенных значений в каждом столбце:")
print(df_optimized.isnull().sum())

# Визуализация пропущенных значений
plt.figure(figsize=(10, 6))
sns.heatmap(df_optimized.isnull(), cbar=False, cmap="viridis")
plt.title("Визуализация пропущенных значений в обучающем наборе")
plt.show()

# Импорт библиотеки
from sklearn.preprocessing import LabelEncoder

# Кодирование категориальных переменных
label_encoder = LabelEncoder()
df_optimized['Gender'] = label_encoder.fit_transform(df_optimized['Gender'])
df_optimized['Vehicle_Age'] = label_encoder.fit_transform(df_optimized['Vehicle_Age'])
df_optimized['Vehicle_Damage'] = label_encoder.fit_transform(df_optimized['Vehicle_Damage'])

# Просмотр
label_encoder

# Просмотр первых записей фрейма данных после кодирования
df_optimized

# Заполнение пропущенных значений
df_optimized.fillna(method='ffill', inplace=True)
df_optimized.fillna(method='ffill', inplace=True)

# Ресурсы
X = df_optimized.drop(columns=['Response'])

# Целевая переменная
y = df_optimized['Response']
# Просмотр строк и столбцов x
X.shape

# Просмотр строк и столбцов
y.shape

# Импорт библиотек
from sklearn.model_selection import train_test_split

# Отдел обучения и тестирования
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Просмотр тренировочных данных
print("Размерность обучающего набора (X_train):", X_train.shape)

# Просмотр тестовых данных
print("Размерность целевой переменной (y_train):", y_train.shape)

# Преобразование категориальных столбцов в фиктивные переменные
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Просмотр обучающих данных
print("Просмотр строк и столбцов в обучающем наборе (X_train):", X_train.shape)

# Просмотр тестовых данных
print("Просмотр строк и столбцов в тестовом наборе (y_train):", y_train.shape)



# Импорт библиотеки моделей машинного обучения
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Импорт библиотеки для моделей машинного обучения метрик
from sklearn.metrics import accuracy_score

# Модели, подлежащие эволюции
models = [GaussianNB(), # Naive Bayes Model
          DecisionTreeClassifier(random_state=42), # Decision Tree Model
          #RandomForestClassifier(n_estimators=100, random_state=42), # Random forest model
          LogisticRegression(random_state=50), # Logistic regression model
          AdaBoostClassifier(random_state=45), # Ada Boost Model
          XGBClassifier(), # XGBoost Model Parameter tree_method='gpu_hist' for XGBoost GPU
          LGBMClassifier()] # LightGBM Model Parameter device='gpu' for LightGBM GPU

# Эволюция каждой модели
for i, model in enumerate(models):
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(model)
    print()
    print(f"Модель {i + 1}: {type(model).__name__}")
    print()
    print(f"Точность на обучающих данных: {train_accuracy}")
    print(f"Точность на тестовых данных: {test_accuracy}")
    print("------------------")

# Шаг 6: Эволюция модели
# Шаг 7: Создание прогнозов на основе тестового набора
predictions = model.predict(X_test)

# Обучение моделей, которые поддерживают важные функции

# Установка Seaborn стиля
sns.set_palette("Set2")

models_with_feature_importances = [("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
                                 #  (
                                 #  "RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=42)),
                                   ("XGBClassifier", XGBClassifier(random_state=42)),
                                   ("LGBMClassifier", LGBMClassifier(random_state=42))]

# Переборка моделей
for model_name, model in models_with_feature_importances:

    # Тренировка модели
    model.fit(X_train, y_train)

    # Представление о важности функций
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        # Если модель не содержит важных функций, переход к следующей модели.
        print(f"{model_name} не поддерживает отображение важности признаков.")
        continue

    # Создание фрейма данных для более удобного просмотра
    feature_importances_df = pd.DataFrame({'Признак': X_train.columns,
                                           'Важность': feature_importances})

    # Сортировка по важности
    feature_importances_df = feature_importances_df.sort_values(by='Важность', ascending=False)

    # Визуализация
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Важность', y='Признак', data=feature_importances_df[:10])
    plt.title(f"Топ 10 признаков - {model_name}")
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.grid(False)
    plt.show()

# plot confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

# Эволюция каждой модели
for i, model in enumerate(models):
    # Вычисление и построение матрицы ошибок
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Не откликнулся", "Откликнулся"],
                yticklabels=["Не откликнулся", "Откликнулся"])
    plt.xlabel("Предсказано")
    plt.ylabel("Факт")
    plt.title(f"Матрица ошибок - Модель {i + 1}: {type(model).__name__}")
    plt.show()
    print("------------------")

# Модели ROC-кривых
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, classification_report, confusion_matrix
)


# Эволюция каждой модели
for i, model in enumerate(models):

    # Вычисление вероятности положительного класса
    y_probs = model.predict_proba(X_test)[:, 1]

    # Вычисление кривой ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    # Вычисление площади под кривой ROC (AUC)
    auc = roc_auc_score(y_test, y_probs)

    # Plot ROC кривой
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Доля ложноположительных срабатываний')
    plt.ylabel('Доля истинноположительных срабатываний')
    plt.title(f'ROC-кривая - Модель {i + 1}: {type(model).__name__}')
    plt.legend(loc="lower right")
    plt.grid(False)
    plt.show()

    print("------------------")

# Эволюция каждой модели
for i, model in enumerate(models):

    # Оценка каждой модели и создание отчета о классификации
    report = classification_report(y_test, model.predict(X_test))
    print()
    print("Отчет о классификации:")
    print()
    print(report)
    print()

    print("=======================================")


# Просмотр набора данных train_df
train_df.head()

# Импорт библиотеки
from sklearn.preprocessing import LabelEncoder

# # Копирование исходных данных, чтобы избежать изменения исходного фрейма данных
train_df = df_cleaned_optimized_train.copy()

# Кодирование категориальных переменных
label_encoder = LabelEncoder()

# Применение LabelEncoder к каждой категориальной переменной
for col in ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Age_Bucket']:
    train_df[col] = label_encoder.fit_transform(train_df[col])

# Кодирование категориальных переменных
label_encoder = LabelEncoder()
train_df['Gender'] = label_encoder.fit_transform(train_df['Gender'])
train_df['Vehicle_Age'] = label_encoder.fit_transform(train_df['Vehicle_Age'])
train_df['Vehicle_Damage'] = label_encoder.fit_transform(train_df['Vehicle_Damage'])
# df_cleaned_optimized_train['Age_Bucket'] = label_encoder.fit_transform(df_cleaned_optimized_train['Age_Bucket'])

# Просмотр набора данных
label_encoder

# # Удаление столбца "Имя"
train_df.drop(columns=['id'], inplace=True)

# Просмотр первых записей фрейма данных после удаления столбца
train_df.head()

# Разделение данных на набор функций (X) и целевую переменную (y)

# Ресурсы
X1 = train_df.drop(columns=['Response'])

# Целевая переменная
y2 = train_df['Response']

# Разделение данных на обучающие и тестовые наборы.
X_train1, X_test1, y_train2, y_test2 = train_test_split(X1, y2, test_size=0.2, random_state=42)

# Просмотр строк и столбцов
print("Размерность обучающего набора (X_train1):", X_train1.shape)
print("Размерность целевой переменной (y_train2):", y_train2.shape)

# Преобразование категориальных столбцов в фиктивные переменные
X_train1 = pd.get_dummies(X_train1)
X_test1 = pd.get_dummies(X_test1)

# Просмотр строк и столбцов
print("Размерность обучающего набора (X_train1):", X_train1.shape)
print("Размерность тестового набора (X_test1):", X_test1.shape)


# Импорт библиотеки
from lightgbm import LGBMClassifier
import lightgbm as lgb

# Создание LGBM модели
lgbm_model = LGBMClassifier(device='gpu',
                            num_leaves=31,
                            max_depth=100,
                            learning_rate=0.1,
                            n_estimators=100)

# Обучение модели
lgbm_model.fit(X_train, y_train)

# прогнозирование модели LGBM
lgbm_model_pred = lgbm_model.predict(X_test)

# Оценка модели LightGBM
print("Оценка модели LightGBM:", lgbm_model.score(X_train, y_train))


# Plot важности объектов
# Важность функций
importance = lgbm_model.feature_importances_
feature_names = X_train.columns

# Сортировка по важности
indices = np.argsort(importance)

# Ограничение количества функций для отображения
num_features = 30  # Количество функций для отображения
top_indices = indices[-num_features:]

# Построение графика важности функций
plt.figure(figsize=(20, 10))
plt.title('Важность признаков')
plt.barh(range(len(top_indices)), importance[top_indices], align='center')
plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
plt.xlabel('Относительная важность')
plt.grid(False)
plt.show()

# Рассчет точности модели
accuracy = accuracy_score(y_test, lgbm_model_pred)
print("Точность модели LightGBM:", accuracy)

# Создание confusion matrix
conf_matrix2 = confusion_matrix(y_test, lgbm_model_pred)

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок - Классификатор LGBM')
plt.show()

# ROC кривая
y_pred_proba = lgbm_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print("Площадь под кривой ROC для LGBM (AUC):", auc)

# Plot кривой ROC
plt.plot(fpr, tpr, label='ROC-кривая - LGBM (площадь = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Доля ложноположительных срабатываний')
plt.ylabel('Доля истинноположительных срабатываний')
plt.title('ROC-кривая - LGBM')
plt.legend(loc="lower right")
plt.grid(False)
plt.show()

from xgboost import XGBClassifier

# Модель XGBoost
# Параметр tree_method='gpu_hist' для графического процессора XGBoost
model_XGBoost = XGBClassifier(tree_method='hist', use_label_encoder=False, eval_metric='logloss', random_state=42)
model_XGBoost_fit = model_XGBoost.fit(X_train, y_train)
model_XGBoost

# Оценочная модель
print("Оценка модели XGBoost:", model_XGBoost.score(X_train, y_train))

# Прогнозирование модели XGBoost
xgboost_model_pred = model_XGBoost.predict(X_test)
# Рассчет точности модели
accuracy = accuracy_score(y_test, xgboost_model_pred)
print("Точность модели XGBoost:", accuracy)

# Создание confusion matrix
conf_matrix2 = confusion_matrix(y_test, xgboost_model_pred)

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок - XGBoost')
plt.show()

# ROC кривая
y_pred_proba = model_XGBoost.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print("Площадь под кривой ROC (AUC):", auc)


# Plot кривой ROC
plt.plot(fpr, tpr, label='ROC-кривая XGBoost (площадь = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Доля ложноположительных срабатываний')
plt.ylabel('Доля истинноположительных срабатываний')
plt.title('ROC-кривая - XGBoost')
plt.legend(loc="lower right")
plt.grid(False)
plt.show()

# Модель отчета о классификации
class_report = classification_report(y_test, xgboost_model_pred)
print("Отчет о классификации - Классификатор XGBoost")
print(class_report)

# Список для сохранения результатов
results = []

# Оценка каждой модели
for i, model in enumerate(models):

    # Сохранение результатов в словарь
    results.append({'Модель': type(model).__name__,
                    'Точность на обучающих данных': train_accuracy,
                    'Точность на тестовых данных': test_accuracy})

# Преобразование списка результатов в DataFrame
results_df = pd.DataFrame(results)


# Функция выделения максимального значения в столбце
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


# Применение функции выделения к столбцу "Точность тестирования".
# Применение функции подсветки максимальных значений в столбце 'Точность на тестовых данных'
results_df.style.apply(highlight_max, subset=['Точность на тестовых данных'])


# Импорт библиотеки для моделей машинного обучения метрик
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Список для сохранения результатов
results = []

# Оценка каждой модели
for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='binary')
    test_recall = recall_score(y_test, y_test_pred, average='binary')
    test_f1 = f1_score(y_test, y_test_pred, average='binary')
    test_support = y_test.shape[0]

    # Store results in dictionary
    results.append({'Model': type(model).__name__,
                    'Accuracy': test_accuracy,
                    'Precision': test_precision,
                    'Recall': test_recall,
                    'F1-Score': test_f1,
                    'Support': test_support}
                   )

# Преобразование списка результатов в DataFrame
results_df = pd.DataFrame(results)

# Функция выделения максимального значения в каждом столбце
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow'
            if v
            else ''
            for v in is_max]


# Применение функции выделения к фрейму данных
styled_results_df = results_df.style.apply(highlight_max, subset=['Accuracy',
                                                                  'Precision',
                                                                  'Recall',
                                                                  'F1-Score'])

# Отображение стилизованного фрейма данных
styled_results_df

import pandas as pd

# Загрузка набора данных
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv ')
# Обработка пропущенных значений
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)
# Кодирование категориальных переменных
label_encoders = {}
for column in train_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_df[column] = le.fit_transform(train_df[column])
    test_df[column] = le.transform(test_df[column])
    label_encoders[column] = le

# Просмотр
le

# Разделение ресурсов и целей
X = train_df.drop(columns=['id', 'Response'])
y = train_df['Response']
X_test = test_df.drop(columns=['id'])
# Разделение данных обучения и валидации
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Установка параметров моделей для графического процессора
params = {'objective': 'binary',
          'boosting_type': 'gbdt',
          'num_leaves': 31,
          'learning_rate': 0.05,
          'feature_fraction': 0.9,
          'n_estimators': 100,
          'device': 'gpu',
          'gpu_platform_id': 0,
          'gpu_device_id': 0,
          'metric': 'auc'}

# Создание наборов данных LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Обучение модели LightGBM
lgbm_model = lgb.train(params,
                       train_data,
                       valid_sets=[val_data])

# Просмотр модели
lgbm_model

# Создание прогнозов на основе тестового набора
y_test_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
# Проверка длины тестовых данных и прогнозов
print(f"Длина тестового набора: {len(test_df)}")
print(f"Длина предсказаний: {len(y_test_pred)}")

# Проверяем соответствие размеров
if len(test_df) == len(y_test_pred):
    # Создаем DataFrame с результатами
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Response': y_test_pred
    })

    # Проверяем первые несколько строк
    print("\nПервые 5 предсказаний:")
    print(submission.head())

    # Сохраняем в файл
    submission_file = 'submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\nРезультаты успешно сохранены в файл: {submission_file}")
else:
    print("\nОшибка: количество строк в тестовых данных и предсказаниях не совпадает")
    print("Проверьте данные и модель")

# Просмотр набора данных
jf = pd.read_csv("sample_submission.csv")

# Просмотр первых 10 данных
jf.head(10)

from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from datetime import datetime
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

# Полное подавление нежелательных предупреждений
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Безопасный расчет весов классов
try:
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
except Exception as e:
    print(f"Ошибка при расчете весов классов, используется 'balanced': {str(e)}")
    class_weight_dict = 'balanced'

# Расчет scale_pos_weight с обработкой исключений
try:
    scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
except Exception as e:
    print(f"Ошибка при расчете scale_pos_weight, используется значение 1: {str(e)}")
    scale_pos_weight = 1

# Оптимизированные модели без устаревших параметров
trained_models = [
    GaussianNB(),

    DecisionTreeClassifier(
        random_state=42,
        class_weight=class_weight_dict,
        max_depth=5,
        min_samples_split=10
    ),

    LogisticRegression(
        random_state=50,
        class_weight=class_weight_dict,
        max_iter=1000,
        solver='saga',
        n_jobs=-1,
        penalty='elasticnet',
        l1_ratio=0.5
    ),

    AdaBoostClassifier(
        random_state=45,
        n_estimators=100,
        learning_rate=0.1
    ),

    XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_estimators=150,
        n_jobs=-1,
        enable_categorical=True
    ),

    LGBMClassifier(
        class_weight=class_weight_dict,
        random_state=42,
        n_estimators=120,
        verbose=-1,
        boosting_type='gbdt',
        num_leaves=31
    )
]

model_names = [type(m).__name__ for m in trained_models]

# Создание и обучение ансамбля
try:
    ensemble = VotingClassifier(
        estimators=list(zip(model_names, trained_models)),
        voting='soft',
        n_jobs=-1,
        verbose=1
    )

    print("Начало обучения ансамбля...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble.fit(X_train, y_train)
    print("Обучение завершено успешно!")

    # Сохранение модели
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_filename = f"trained_ensemble_{timestamp}.joblib"
    joblib.dump(ensemble, ensemble_filename, compress=3)

    print("\n" + "=" * 50)
    print(f"Ансамбль успешно сохранен как: {ensemble_filename}")
    print("=" * 50)

    # Рекомендации по проверке
    print("\nРекомендуемые команды для проверки:")
    print("from sklearn.metrics import classification_report, roc_auc_score")
    print("y_pred = ensemble.predict(X_test)")
    print("y_proba = ensemble.predict_proba(X_test)[:, 1]")
    print("print(classification_report(y_test, y_pred))")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

except Exception as e:
    print(f"\nОшибка при работе с ансамблем: {str(e)}")
    print("Рекомендации:")
    print("1. Проверьте размерности X_train и y_train")
    print("2. Убедитесь, что все признаки числовые")
    print("3. Проверьте наличие пропущенных значений")