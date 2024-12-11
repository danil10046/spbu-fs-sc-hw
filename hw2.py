# Начнём с определения входных и выходных переменных
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Входные переменные
distance = ctrl.Antecedent(np.arange(0, 21, 1), 'distance')  # Расстояние до клиента
courier_load = ctrl.Antecedent(np.arange(0, 11, 1), 'courier_load')  # Загруженность курьеров
order_amount_percentile = ctrl.Antecedent(np.arange(0, 101, 1), 'order_amount_percentile')  # Процентиль суммы заказа
# Выходная переменная
priority = ctrl.Consequent(np.arange(0, 101, 1), 'priority')  # Приоритет

# Далее определим функции принадлежности
# Функции принадлежности для расстояния до клиента
distance['short'] = fuzz.trimf(distance.universe, [0, 0, 10])
distance['medium'] = fuzz.trimf(distance.universe, [5, 10, 15])
distance['long'] = fuzz.trimf(distance.universe, [10, 20, 20])

# Функции принадлежности для загруженности курьеров
courier_load['low'] = fuzz.trimf(courier_load.universe, [0, 0, 5])
courier_load['medium'] = fuzz.trimf(courier_load.universe, [3, 5, 7])
courier_load['high'] = fuzz.trimf(courier_load.universe, [5, 10, 10])

# Функции принадлежности для процентиля суммы заказа
order_amount_percentile['low'] = fuzz.trimf(order_amount_percentile.universe, [0, 0, 50])
order_amount_percentile['medium'] = fuzz.trimf(order_amount_percentile.universe, [25, 50, 75])
order_amount_percentile['high'] = fuzz.trimf(order_amount_percentile.universe, [50, 100, 100])

# Функции принадлежности для приоритета доставки
priority['low'] = fuzz.trimf(priority.universe, [0, 0, 50])
priority['medium'] = fuzz.trimf(priority.universe, [25, 50, 75])
priority['high'] = fuzz.trimf(priority.universe, [50, 100, 100])

# Теперь созданим правила
# Правила для коротких расстояний (учитывая пеших и велокурьеров)
rule1 = ctrl.Rule(distance['short'] & courier_load['low'] & order_amount_percentile['high'], priority['high'])
rule2 = ctrl.Rule(distance['short'] & courier_load['low'] & order_amount_percentile['medium'], priority['medium'])
rule3 = ctrl.Rule(distance['short'] & courier_load['low'] & order_amount_percentile['low'], priority['low'])

# Правила для средних расстояний (учитывая велокурьеров)
rule4 = ctrl.Rule(distance['medium'] & courier_load['medium'] & order_amount_percentile['high'], priority['high'])
rule5 = ctrl.Rule(distance['medium'] & courier_load['medium'] & order_amount_percentile['medium'], priority['medium'])
rule6 = ctrl.Rule(distance['medium'] & courier_load['medium'] & order_amount_percentile['low'], priority['low'])

# Правила для длинных расстояний (учитывая автокурьеров)
rule7 = ctrl.Rule(distance['long'] & courier_load['high'] & order_amount_percentile['high'], priority['medium'])
rule8 = ctrl.Rule(distance['long'] & courier_load['high'] & order_amount_percentile['medium'], priority['low'])
rule9 = ctrl.Rule(distance['long'] & courier_load['high'] & order_amount_percentile['low'], priority['low'])

#И последним шагом созданим системы управления и симуляции
# Система управления и добавление правил
delivery_priority_control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

#  Объект симуляции на основе системы управления
delivery_priority_simulation = ctrl.ControlSystemSimulation(delivery_priority_control_system)

# Задание значений входных переменных
delivery_priority_simulation.input['distance'] = 8  # Пример значения расстояния до клиента
delivery_priority_simulation.input['courier_load'] = 4  # Пример значения загруженности курьеров
delivery_priority_simulation.input['order_amount_percentile'] = 70  # Пример значения процентиля суммы заказа

# Выполнение симуляции
delivery_priority_simulation.compute()
# Получение значения выходной переменной
print("Приоритет доставки:", delivery_priority_simulation.output['priority'])

# Результат симуляции указывает на то, что приоритет доставки для заданных условий составляет примерно 69.8.
# Это означает, что доставка имеет высокий приоритет, учитывая указанные параметры (расстояние до клиента, загруженность курьеров и сумма заказа).
# Более высокий приоритет означает, что заказ должен быть выполнен в ближайшее время.
