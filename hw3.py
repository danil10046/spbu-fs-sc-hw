# # # # # # # Начнём с алгоритма k-means
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data_url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
df = pd.read_csv(data_url)

# Очистка данных
df = df.dropna()
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means
# 1. Как работает:
#    Сначаоа задаем количество кластеров (k) и случайным образом выбираем k центроидов;
#    Далее каждую точку данных назначаем к кластеру с ближайшим центроидом;
#    После пересчитываем центроиды как среднее значение всех точек, принадлежащих к каждому кластеру;
#    Повторяем шаги 2 и 3, пока центроиды не перестанут изменяться существенно или не достигнем максимального числа итераций.
#
# 2. На вход  подаются: Количество кластеров (k), максимальное количество итераций, Допуск (tolerance)
# 3. Какие параметры имеют инициализацию по умолчанию, какими значениями инициализируются и почему (предположительно), как влияют на результат:
#     Количество кластеров (k): По умолчанию инициализируется значением 8, что позволяет начать с разумного количества кластеров;
#     Максимальное количество итераций: По умолчанию задается значением 300, чтобы дать алгоритму достаточно времени для сходимости;
#     Допуск (tolerance): По умолчанию равен 0.0001, что обеспечивает точную остановку при минимальном изменении центроидов.
#
# 4. Хорошо работает, когда данные имеют четко разделенные кластеры и распределены примерно равномерно.
#    Плохо работает, когда кластеры имеют сложную форму, разные размеры или пересекаются, а также в случае значительного шума и выбросов.

# Пример с использованием значений по умолчанию
# kmeans = KMeans(random_state=42)
# df['kmeans_labels'] = kmeans.fit_predict(X_scaled)

# Зададим Количество кластеров (k) - 10, Максимальное количество итераций - 500 и Допуск (tolerance) -0.0002
kmeans = KMeans(n_clusters=10, random_state=42, tol=0.0002, max_iter=500) 
df['kmeans_labels'] = kmeans.fit_predict(X_scaled)

# Визуализация
sns.pairplot(df, hue='kmeans_labels', palette='viridis')
plt.suptitle('K-means Clustering', y=1.02)
plt.show()

# # # # # # # Далее рассмотрим алгоритм Gaussian Mixture Model (GMM)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data_url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
df = pd.read_csv(data_url)

# Очистка данных
df = df.dropna()
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# GMM
# 1. Как работает:
#    Сначала задаем количество компонент и инициализируем параметры гауссовских распределений;
#    Далее каждый объект имеет вероятность принадлежности к каждой компоненте;
#    После пересчитываем параметры гауссовских распределений, максимизируя правдоподобие;
#    Повторяем шаги 2 и 3 до сходимости или достижения максимального числа итераций.
#
# 2. На вход подается: Количество компонент (k), Максимальное количество итераций, Допуск (tolerance)
# 3. Какие параметры имеют инициализацию по умолчанию, какими значениями инициализируются и почему (предположительно), как влияют на результат:
#     Количество компонент (k): По умолчанию инициализируется значением 1, но обычно лучше начать с более высокого значения, подходящего для данных;
#     Максимальное количество итераций: По умолчанию задается значением 100, чтобы дать алгоритму достаточно времени для сходимости;
#     Допуск (tolerance): По умолчанию равен 0.001, чтобы обеспечить точную остановку при минимальном изменении параметров.
#
# 4. Хорошо работает при наличии кластеров, распределённых по Гауссу.
#    Плохо работает при сильно перекрывающихся или негауссовских кластерах.

# Пример с использованием значений по умолчанию
# gmm_default = GaussianMixture(random_state=42)
# df['gmm_labels'] = gmm_default.fit_predict(X_scaled)

# Пример с явно заданными значениями гиперпараметров
gmm_custom = GaussianMixture(n_components=5, max_iter=300, tol=0.0001, random_state=42)
df['gmm_labels'] = gmm_custom.fit_predict(X_scaled)

# Визуализация результатов с явно заданными значениями гиперпараметров
sns.pairplot(df, hue='gmm_labels', palette='viridis')
plt.suptitle('GMM Clustering (Custom Parameters)', y=1.02)
plt.show()

# # # # # # # Теперь рассмотрим Иерархическую кластеризацию
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data_url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
df = pd.read_csv(data_url)

# Очистка данных
df = df.dropna()
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Иерархическая кластеризация
# 1. Как работает:
#    Сначала каждая точка данных рассматривается как отдельный кластер;
#    Далее повторно объединяем самые близкие кластеры на каждом шаге;
#    Продолжаем слияние до достижения требуемого количества кластеров.
#
# 2. На вход подается: Количество кластеров (n_clusters), Метод слияния (linkage)
# 3. Какие параметры имеют инициализацию по умолчанию, какими значениями инициализируются и почему (предположительно), как влияют на результат:
#     Количество кластеров (n_clusters): Обычно инициализируется значением 2, но чаще всего задается явно в зависимости от задачи;
#     Метод слияния (linkage): По умолчанию используется метод "ward", который минимизирует сумму квадратов внутрикластерных расстояний.
# 4. Хорошо работает при четко выраженной иерархии кластеров.
#    Плохо работает при высоком количестве данных и отсутствии естественной иерархии.

# Пример с использованием значений по умолчанию
# agglo_default = AgglomerativeClustering()
# df['agglo_labels'] = agglo_default.fit_predict(X_scaled)

# Пример с явно заданными значениями гиперпараметров
agglo_custom = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['agglo_labels'] = agglo_custom.fit_predict(X_scaled)

# Визуализация результатов с явно заданными значениями гиперпараметров
sns.pairplot(df, hue='agglo_labels', palette='viridis')
plt.suptitle('Agglomerative Clustering (Custom Parameters)', y=1.02)
plt.show()

# # # # # # # Напоследок рассмотрим алгоритм Fuzzy C-Means
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data_url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
df = pd.read_csv(data_url)

# Очистка данных
df = df.dropna()
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fuzzy C-Means
# 1. Как работает:
#     Сначала задаем количество кластеров и начальные центроиды.
#     Далее каждая точка имеет степень принадлежности к каждому кластеру;
#     После Пересчитываем центроиды и степени принадлежности;
#     Повторяем до сходимости.
# 2. На вход подается: Количество кластеров (c), Множитель fuzziness (m), Максимальное количество итераций
# 3. Какие параметры имеют инициализацию по умолчанию, какими значениями инициализируются и почему (предположительно), как влияют на результат:
#     Количество кластеров (c): Обычно инициализируется значением 2, но чаще всего задается явно;
#     Множитель fuzziness (m): По умолчанию равен 2, что контролирует степень размытия кластеров;
#     Максимальное количество итераций: По умолчанию задается значением 150, чтобы обеспечить достаточное время для сходимости.
# 4.  Хорошо работает когда кластеры имеют пересекающиеся границы.
#     Плохо работает когда кластеры четко разделены.

# # Пример с использованием значений по умолчанию
# fcm_default = FCM()
# fcm_default.fit(X_scaled)
# df['fcm_labels'] = fcm_default.predict(X_scaled)

# Пример с явно заданными значениями гиперпараметров
fcm_custom = FCM(n_clusters=5, m=4, max_iter=350)
fcm_custom.fit(X_scaled)
df['fcm_labels'] = fcm_custom.predict(X_scaled)

# Визуализация результатов с явно заданными значениями гиперпараметров
sns.pairplot(df, hue='fcm_labels', palette='viridis')
plt.suptitle('Fuzzy C-Means Clustering (Custom Parameters)', y=1.02)
plt.show()
