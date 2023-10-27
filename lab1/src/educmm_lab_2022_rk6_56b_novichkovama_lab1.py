import numpy as np
import matplotlib.pyplot as plt
import random


# Функция вычисления значения базисного полинома Лагранжа в точке x
def l_i(i, x, x_nodes):
    result = 1
    for j in range(len(x_nodes)):
        if j != i:
            result *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
    return result


# Функция вычисления значения интерполяционного полинома Лагранжа в точке x
def L(x, x_nodes, y_nodes):
    result = 0
    for i in range(len(x_nodes)):
        result += y_nodes[i] * l_i(i, x, x_nodes)
    return result


# Функция отрисовки графика функции и интерполянта по узлам
def plots(x_nodes, f, ax):
    y_nodes = f(x_nodes)
    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.plot(x_nodes, f(x_nodes), 'go', markersize=6)
    x_for_plt = np.linspace(-1, 1, 200)
    ax.plot(x_for_plt, f(x_for_plt), '#aaa', label='$f(x)$')
    ax.plot(x_for_plt, [L(el, x_nodes, y_nodes) for el in x_for_plt], 'blue', label='$L(x)$')
    ax.legend()


# Функция, в которой задается n равномерно распределенных узлов и вызывается функция для отрисовки графика
def equi_plot(n, f):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x_equi_nodes = np.linspace(-1, 1, n)
    plots(x_equi_nodes, f, ax)
    #fig.savefig(''.join(['base_eq_', str(n), '.png']), dpi=600)


# Функция, в которой задается n чебышевских узлов и вызывается функция для отрисовки графика
def cheb_plot(n, f):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x_cheb_nodes = np.array([np.cos((2 * i - 1) / (2 * n) * np.pi) for i in range(1, n + 1)])
    plots(x_cheb_nodes, f, ax)
    # fig.savefig(''.join(['base_ch_', str(n), '.png']),dpi = 600)


# Функция выполнения заданий базовой части
def base_tsk():
    f = lambda x: 1 / (1 + 25 * x ** 2)
    n = [5, 8, 11, 14, 17, 20, 23]
    for el in n:
        equi_plot(el, f)
        cheb_plot(el, f)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    both_plots(11, f, ax)  #вывод графиков сразу на одной плоскости для наглядности
    plt.show()



# Функция генерации случайной функции
def generate_f():
    n = random.randint(7, 15)
    m = random.randint(7, 15)
    a = np.array([random.uniform(0, 1) for i in range(m + 1)])
    b = np.array([random.uniform(0, 1) for i in range(1, n + 1)])
    f = lambda x: sum([a[j] * x ** j for j in range(m + 1)]) / (1 + sum([b[k] * x ** (k + 1) for k in range(0, n)]))
    return f


#Функция для отрисовки графиков функции и двух интерполянтов на одной плоскости
def both_plots(n, f, ax):
    x_equi_nodes = np.linspace(-1, 1, n)
    x_cheb_nodes = np.array([np.cos((2 * i - 1) / (2 * n) * np.pi) for i in range(1, n + 1)])
    ax.plot(x_equi_nodes, f(x_equi_nodes), 'ro', markersize=6, label='Равномерные узлы')
    ax.plot(x_cheb_nodes, f(x_cheb_nodes), 'bo', markersize=7, label='Чебышевские узлы')
    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    x_for_plt = np.linspace(-1, 1, 200)
    ax.plot(x_for_plt, f(x_for_plt), '#aaa', label='Сгенерированная функция')
    ax.plot(x_for_plt, [L(el, x_equi_nodes, f(x_equi_nodes)) for el in x_for_plt], 'red',
            label='Интерполянт по равномерным узлам')
    ax.plot(x_for_plt, [L(el, x_cheb_nodes, f(x_cheb_nodes)) for el in x_for_plt], 'blue',
            label='Интерполянт по чебышевским узлам')
    # ax.legend() # для графиков в отчет не нужно, но для графиков вне отчета удобно



# Функция вычисления расстояния для N от 1 до 31 и построения графика
def dist(f, ax):
    norm_eq = []
    norm_ch = []
    x_for_plt = np.linspace(-1, 1, 200)
    for n_norm in range(1, 31):
        x_equi_nodes = np.linspace(-1, 1, n_norm)
        x_cheb_nodes = np.array([np.cos((2 * i - 1) / (2 * n_norm) * np.pi) for i in range(1, n_norm + 1)])
        norm_eq.append(max([abs(L(el, x_equi_nodes, f(x_equi_nodes)) - f(el)) for el in x_for_plt]))
        norm_ch.append(max([abs(L(el, x_cheb_nodes, f(x_cheb_nodes)) - f(el)) for el in x_for_plt]))
    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.semilogy([i for i in range(1, 31)], norm_eq, '#fc2847')
    ax.semilogy([i for i in range(1, 31)], norm_ch, '#007fff')


# Функция выполнения заданий продвинутой части
def advanced():
    rand_functions = []
    for i in range(100):
        f_i = generate_f()
        rand_functions.append(f_i)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    for i in range(0, 100, 25): # вывод графиков для четырех функций из 100
        both_plots(7, rand_functions[i], ax[i // 50, (i // 25) % 2])

    fig, ax = plt.subplots(1, 2, figsize=(12, 12))
    both_plots(10, rand_functions[0], ax[0]) # еще вывод графиков для 1 функции отдельно
    dist(rand_functions[0], ax[1]) # вывод расстояния для той-же функции

    fig, ax = plt.subplots(1, 2, figsize=(12, 12))
    both_plots(10, rand_functions[2], ax[0]) # еще вывод графиков для 1 функции отдельно
    dist(rand_functions[2], ax[1]) # вывод расстояния для той-же функции
    plt.show()


if (__name__ == '__main__'):
    base_tsk()
    advanced()

