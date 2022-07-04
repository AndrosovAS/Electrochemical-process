import numpy as np

import intvalpy as ip
ip.precision.extendedPrecisionQ = False

from data import TMES, WMES


################################################################################
################################################################################
# Введём модельную функцию у которой будем восстанавливать параметры
# Вычислим её градиенты
# W(.., t) = (x0 + x1*t) / (x2 + x3*t)

model = lambda a, x: (x[0] + K1*x[1]*a) / (x[2] + x[1]*a)

dx0 = lambda a, x: 1 / (x[2] + x[1]*a)
dx1 = lambda a, x: x[2]*a* (K1 - 1) / (x[2] + x[1]*a)**2
dx2 = lambda a, x: -1 / (x[2] + x[1]*a)**2

grad = np.array([dx0, dx1, dx2])
# количество неизвестных
N = len(grad)


################################################################################
# Поскольку все входные измерения различны (кроме финального) и независимы,
# то можно объединить в одну большую выборку

tmes = np.array([mes for exp in TMES for mes in exp])
wmes = np.array([mes for exp in WMES for mes in exp])

# Известно, что данные неточные
eps_t = 0
eps_w = 0.0255

tmes = tmes + ip.Interval(-eps_t, eps_t)
wmes = wmes + ip.Interval(-eps_w, eps_w)

# Известна связь между коэффициентами
K1 = 0.75


################################################################################
# Алгоритмически вычислим выбросы
# Для этого воспользуемся методом распознающего функционала
# Поскольку погрешность по времени отсутствует, то можно взять Tol

# Сначала нам нужно найти точку, которая в наибольшей степени согласуется с данными
# Интересно сравнить с МНК-точкой

# Для нахождения такой точки проведём глобальную оптимизацию и найдём глобальный
# максимум распознающего функционала
# В случае дробно-линейной функции функционал Tol является квазивогнутой функцией,
# поэтому риска зациклиться в локальном экстремуме нет

# Следующим шагом смотрим на образующие функционала
# Если их значение сильно отклоняется от большинства, то можно утверждать,
# что это является выбросом
# Это можно определить с помощью инквартильного метода или с помощью метода стандартного отклонения
# Возможно, некоторые выбросы скрываются за более явными выбросами, поэтому
# может возникнуть необходимость несколько раз повторить алгоритм.

# Введём функции для количественного выявление выбросов
def interquartile(data):
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5

    lower, upper = q25 - cut_off, q75 + cut_off
    return np.argwhere((data < lower) | (data > upper)).flatten()

def standard_deviations(data):
    # Set upper and lower limit to 3 standard deviation
    std, mean = np.std(data), np.mean(data)
    cut_off = std * 3

    lower, upper = mean - cut_off, mean + cut_off
    return np.argwhere((data < lower) | (data > upper)).flatten()

# Задаём начальное приближение
x0 = np.array([150, 20, 10])

# Повторяем описанный алгоритм пока не иссякнет максимальное количество итераций
# или массив выбросов не станет пустым
nit = 0
maxiter = 10

while nit < maxiter:
    success, x, f = ip.nonlinear.Tol(model, tmes, wmes, maxQ=True, grad=grad, x0=x0, tol=1e-8, maxiter=5000)
    print(success)

    tt = wmes.rad - (wmes.mid - model(tmes, x)).mag
    outliers_index = standard_deviations(tt)

    if len(outliers_index) > 0:
        index = np.delete(np.arange(tmes.shape[0]), outliers_index)
        tmes, wmes = tmes[index], wmes[index]
        print('outliers_index: ', outliers_index)
    else:
        break
    nit += 1

# Таким образом получена чистая выборка
# Посмотрим на вектор x и максимальное значение функционала
print('x: ', x)
print('f: ', f)
