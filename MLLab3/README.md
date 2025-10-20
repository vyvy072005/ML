# Лабораторная работа 3. Линейная регрессия
## Задание
Перед выполнением лабораторной работы необходимо загрузить набор данных в соответствии с вариантом на диск

1. Написать программу, которая разделяет исходную выборку на обучающую и тестовую (training set, test set). Использовать стандартные функции (train_test_split и др. нельзя).
2. С использованием библиотеки scikit-learn обучить модель линейной регрессии по обучающей выборке пример
Проверить точность модели по тестовой выборке
3. Построить модель с использованием полиномиальной функции пример. Построить графики зависимости точности на обучающей и тестовой выборке от степени полиномиальной функции.
4. Построить модель с использованием регуляризации пример. На основе экспериментов подобрать параметры для регуляризации. Построить графики зависимости точности модели на обучающей и тестовой выборках от коэффициента регуляризации.
## Вариант 18
Gas Sensor Array Drift Dataset at Different Concentrations

## Загрузка dataset
```
from ucimlrepo import fetch_ucirepo
gas_sensor_array_drift_at_different_concentrations = fetch_ucirepo(id=270)

X_df = gas_sensor_array_drift_at_different_concentrations.data.features.copy()
y_df = gas_sensor_array_drift_at_different_concentrations.data.targets.copy()
```
X_df - Features, Признаки. y_df - Targets, Цели

## Разделение исходной выборки на обучающую и тестовую в соотношении (20/80)
```
X, y = load_and_prepare_data()

# 1) Разделение на тестовую и обучающие выборки

# Общее количество образцов
n_samples = X.shape[0]

# Массив индексов от 0 до n_samples-1
indices = np.arange(n_samples)

# Перемешиваем индексы случайным образом
np.random.shuffle(indices)

# Применяем перемешанные индексы к X и y
X_shuffled = X.iloc[indices]
y_shuffled = y.iloc[indices]

# Разделяем перемешанные данные
train_size = int(n_samples * 0.8)

X_train = X_shuffled[:train_size]
X_test = X_shuffled[train_size:]

y_train = y_shuffled[:train_size]
y_test = y_shuffled[train_size:]

```
