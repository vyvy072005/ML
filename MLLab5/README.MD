# Лабораторная работа 5. Алгоритмы кластеризации данных
## Задание
Перед выполнением лабораторной работы необходимо загрузить набор данных в соответствии с вариантом на диск.

1. Произвести масштабирование признаков (scaling).
2. С использованием библиотеки scikit-learn написать программу с использованием алгоритмов кластеризации данных, позволяющую разделить исходную выборку на классы, соответствующие предложенной вариантом задаче (http://scikit-learn.org/stable/modules/clustering.html).
3. Провести эксперименты и определить наилучший алгоритм кластеризации, параметры алгоритма. Необходимо использовать не менее 3-х алгоритмов. Данные экспериментов необходимо представить в отчете (графики, ход проведения эксперимента, выводы).
## Вариант 18
AAAI 2014 Accepted Papers (Принятые документы AAAI 2014)

## Загрузка dataset и его подготовка
```
file_path = 'Papers.csv'
data = pd.read_csv(file_path)

# Выбираем текстовые столбцы для моделирования признаков
text_columns = ['keywords', 'topics', 'abstract']
texts = data[text_columns].fillna('')

# Объединяем выбранные текстовые поля в один большой текст для каждого примера
texts_combined = [' '.join(row) for row in texts.values]

# Преобразуем текст в числовое представление с помощью TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(texts_combined)

# Масштабируем признаки
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_tfidf)
```
## Визуализация данных
```
# Преобразуем разреженную матрицу в плотную перед визуализацией
X_scaled_dense = X_scaled.toarray()

# Визуализация данных после PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_dense)

plt.scatter(X_pca[:, 0], X_pca[:, 1], s=30)
plt.title("Визуализация признаков (PCA)")
plt.show()
```
![alt text](1.png)
## Функция для оценки кластеризации
```
def evaluate_clustering(model, data):
    labels = model.fit_predict(data)
    # Проверка, что есть более 1 кластера
    if len(set(labels)) > 1 and -1 in labels:
        core_labels = labels[labels != -1]
        core_data = data[labels != -1]
        score = silhouette_score(core_data, core_labels)
    elif len(set(labels)) > 1:
        score = silhouette_score(data, labels)
    else:
        score = -1
    return labels, score
```
core_data — массив данных, для которых считается метрика.
core_labels — метки кластеров для этих данных, то есть, к каким кластерам принадлежат точки.

## Метод кластеризации - KMeans

```
# 1. KMeans
kmeans_params = [2, 3, 4, 5, 6]
best_score_kmeans = -1
best_kmeans = None
best_labels_kmeans = None
best_k = None
labels_for_k = []  # список для хранения меток для каждого k

for k in kmeans_params:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels, score = evaluate_clustering(kmeans, X_scaled)
    print(f'KMeans с k={k}, Силуэтный коэффициент: {score:.3f}')
    labels_for_k.append(labels)  # сохраняем метки для каждого k
    if score > best_score_kmeans:
        best_score_kmeans = score
        best_kmeans = kmeans
        best_labels_kmeans = labels  # метки для лучшего k
        best_k = k

print(f'Лучшее число кластеров для KMeans: {best_k} с коэффициентом: {best_score_kmeans:.3f}')

# Визуализация для каждого k
fig, axes = plt.subplots(1, len(kmeans_params), figsize=(13, 4))
for i, k in enumerate(kmeans_params):
    labels = labels_for_k[i]
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'k={k}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

plt.tight_layout()
plt.show()
```
![alt text](2.png)
![alt text](console1.png)

## Метод кластеризации - Agglomerative Clustering

```
# 2. Agglomerative Clustering
agg_params = [2, 3, 4, 5, 6]
best_score_agg = -1
best_labels_agg = []
best_n_agg = None

# В список для хранения меток для каждого k
labels_list = []

for n in agg_params:
    agg = AgglomerativeClustering(n_clusters=n)
    labels, score = evaluate_clustering(agg, X_scaled_dense)
    print(f'Agglomerative с n_clusters={n}, Силуэтный коэффициент: {score:.3f}')
    labels_list.append(labels)
    if score > best_score_agg:
        best_score_agg = score
        best_labels_agg = labels
        best_n_agg = n

print(f'Лучшее число кластеров для Agglomerative: {best_n_agg} с коэффициентом: {best_score_agg:.3f}')

# Визуализация для всех вариантов
fig, axes = plt.subplots(1, len(agg_params), figsize=(13, 4))
for i, n in enumerate(agg_params):
    labels = labels_list[i]
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'n={n}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

plt.tight_layout()
plt.show()
```
![alt text](3.png)
![alt text](console2.png)



## Метод кластеризации - SpectralClustering

```
# 3. SpectralClustering
spectral_params = [2, 3, 4, 5, 6]
best_score_spectral = -1
best_labels_spectral = []
best_n_spectral = None

# Список для хранения меток для каждого k
labels_list_spectral = []

for n in spectral_params:
    spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42)
    labels, score = evaluate_clustering(spectral, X_scaled)
    print(f'SpectralClustering с n_clusters={n}, Силуэтный коэффициент: {score:.3f}')
    labels_list_spectral.append(labels)
    if score > best_score_spectral:
        best_score_spectral = score
        best_labels_spectral = labels
        best_n_spectral = n

print(f'Лучшее число кластеров для SpectralClustering: {best_n_spectral} с коэффициентом: {best_score_spectral:.3f}')

# Визуализация результатов для всех вариантов
fig, axes = plt.subplots(1, len(spectral_params), figsize=(13, 4))
for i, n in enumerate(spectral_params):
    labels = labels_list_spectral[i]
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'n={n}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

plt.tight_layout()
plt.show()
```
![alt text](4.png)
![alt text](console3.png)

## Визуализация лучших методов

```
plt.figure(figsize=(13, 5))

# KMeans
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_kmeans, cmap='viridis')
plt.title('KMeans Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Agglomerative Clustering
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_agg, cmap='viridis')
plt.title('Agglomerative Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')

# SpectralClustering
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_spectral, cmap='viridis')
plt.title('SpectralClustering')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.tight_layout()
plt.show()
```
## Определение лучшего метода

```
scores = {
    'KMeans': best_score_kmeans,
    'Agglomerative': best_score_agg,
    'SpectralClustering': best_score_spectral
}

best_method = max(scores, key=scores.get)
print(f'Лучший метод кластеризации: {best_method} с коэффициентом {scores[best_method]:.3f}')
```
![alt text](5.png)
![alt text](console4.png)


