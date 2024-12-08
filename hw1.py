# Начнём с треугольной функции принадлежности fuzz.trimf

# Треугольная функция принадлежности определяется массивом x (входные значения) и списком abc, который содержит три точки [a, b, c], определяющие положение вершин треугольника:
#a: левая точка, где принадлежность начинается с 0
#b: вершина треугольника, где принадлежность равна 1
#c: правая точка, где принадлежность снова становится 0

import numpy as np
import skfuzzy as fuzz

# Массив входных значений
x_new = np.arange(-5, 15, 1)
# Треугольная функция принадлежности
mf_triangular = fuzz.trimf(x_new, [2, 5, 8])
print("Треугольная функция принадлежности:", mf_triangular)

# Теперь рассмотрим трапециевидную функцию принадлежности fuzz.trapmf
# Она определяется массивом x (входные значения) и списком abcd, который содержит четыре точки [a, b, c, d], определяющие форму трапеции:
#a: левая точка, где принадлежность начинается с 0
#b: точка, где принадлежность достигает 1
#c: точка, где принадлежность начинает снижаться
#d: правая точка, где принадлежность снова становится 0

x_new = np.linspace(0, 2, 10)
# Трапециевидная функция принадлежности
mf_trapezoidal = fuzz.trapmf(x_new, [5, 10, 15, 18])
print("Трапециевидная функция принадлежности:", mf_trapezoidal)

# Далее рассмотрим гауссовскую функцию принадлежности fuzz.gaussmf
# Она определяется массивом x (входные значения), средним значением mean и стандартным отклонением sigma:
#x: Массив входных значений
#mean: Среднее значение (центр функции)
#sigma: Стандартное отклонение (определяет ширину функции)

x_new = np.arange(0, 50, 5)
# Гауссовская функция принадлежности
mf_gaussian = fuzz.gaussmf(x_new, 25, 5)
print("Гауссовская функция принадлежности:", mf_gaussian)

# Сигмоидальная функция: fuzz.sigmf
# Определяется массивом x (входные значения), параметром b (определяет положение центра функции) и параметром c (определяет крутизну функции):
# x: Массив входных значений
# b: Параметр, определяющий положение центра функции
# c: Параметр, определяющий крутизну функции

x = np.linspace(-1, 1, 10)
# Сигмоидальная функция принадлежности
mf_sigmoidal = fuzz.sigmf(x, 0, 1)
print("Сигмоидальная функция принадлежности:", mf_sigmoidal)

# Звонковая (колоколообразная) функция: fuzz.gbellmf
# Определяется массивом x (входные значения), параметрами a (ширина колокола), b (форма колокола) и c (центр колокола):
# x: Массив входных значений.
# a: Параметр, определяющий ширину колокола
# b: Параметр, определяющий форму колокола (чем выше значение, тем круче пики)
# c: Параметр, определяющий центр колокола

# Массив входных значений
x = np.linspace(-5, 5, 50)
# Звонковая функция принадлежности
mf_bell = fuzz.gbellmf(x, 2, 4, 0)
print("Звонковая функция принадлежности:", mf_bell)

# Теперь перейдём к нечетким переменнам
# Antecedent (Входная нечеткая переменная). Задаётся универсум дискруса: Диапазон входных значений; и название: Название переменной.
from skfuzzy import control as ctrl

# Входная нечеткая переменная
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
print("Входная нечеткая переменная: Temperature")

# Consequent (Выходная нечеткая переменная). Задаётся универсум дискруса: Диапазон выходных значений; и название: Название переменной.

# Выходная нечеткая переменная
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')
print("Выходная нечеткая переменная: Fan Speed")

# Рассмотрим пример создания входной и выходной нечеткой переменной и добавим функции принадлежности, чтобы лучше понять их использование

# Входная нечеткая переменная (температура)
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')

# Определение функций принадлежности для температуры
temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [10, 20, 30])
temperature['high'] = fuzz.trimf(temperature.universe, [20, 40, 40])

# Выходная нечеткая переменная (скорость вентилятора)
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

# Определение функций принадлежности для скорости вентилятора
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [25, 50, 75])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

print("Функции принадлежности для температуры:")
print(temperature.terms)
print("Функции принадлежности для скорости вентилятора:")
print(fan_speed.terms)

# Теперт перейдём к нечетким правилам
# Rule 

# Определение функций принадлежности для температуры
temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [10, 20, 30])
temperature['high'] = fuzz.trimf(temperature.universe, [20, 40, 40])

# Определение функций принадлежности для скорости вентилятора
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [25, 50, 75])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

# Создание нечетких правил
rule1 = ctrl.Rule(temperature['low'], fan_speed['low'])
rule2 = ctrl.Rule(temperature['medium'], fan_speed['medium'])
rule3 = ctrl.Rule(temperature['high'], fan_speed['high'])

print("Правило 1: Если температура low, тогда скорость вентилятора low.")
print("Если температура medium, тогда скорость вентилятора medium")
print("Если температура high, тогда скорость вентилятора high")

# При создании правил можно использовать логические операторы для соединения различных условий: AND или OR
# Дополнительное условие для влажности
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])

# Правило с использованием оператора AND
rule4 = ctrl.Rule(temperature['high'] & humidity['high'], fan_speed['high'])
print("Правило 4: Если температура high AND влажность high,  тогда скорость вентилятора high")

# Теперь перейдём к Системам управления
# ControlSystem
# Создание нечетких правил (ни уже определены выше )
# Создание системы управления и добавление правил
fan_control_system = ctrl.ControlSystem([rule1, rule2, rule3])

# ControlSystemSimulation
# Создание объекта симуляции на основе системы управления
fan_simulation = ctrl.ControlSystemSimulation(fan_control_system)
# Задание значений входных переменных
fan_simulation.input['temperature'] = 30
# Выполнение симуляции
fan_simulation.compute()
# Получение значения выходных переменных
print("Скорость вентилятора:", fan_simulation.output['fan_speed'])

