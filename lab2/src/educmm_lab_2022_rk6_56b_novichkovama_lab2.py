import numpy as np
import matplotlib.pyplot as plt

def g1(x):
    return x*np.exp(x)


def g2(x):
    return x*x*np.sin(3*x)


def diff2(x_0, h, f):
    return (f(x_0+h)-f(x_0-h))/(2*h)


def err_dif(x_0, h, f):
    g1_d_real = np.exp(x_0) + x_0 * np.exp(x_0)
    g1_d = f(x_0, h, g1)
    return abs(g1_d - g1_d_real)


def base_diff():
    x_0 = 2
    h = np.logspace(-16, 0, 100)
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    ax.grid()
    ax.set_xlabel('$h$')
    ax.set_ylabel('$\Delta(g_1\'(x_0))$')
    h_for_scaling = np.logspace(-5, 0, 100)
    ax.loglog(h, [err_dif(x_0, el, diff2) for el in h ], 'ro', label = '$\Delta(g_1\'(x_0))$')
    ax.loglog(h_for_scaling, h_for_scaling ** 2, '--',label = '$O(h^2)$' )
    ax.legend()
    plt.show()


def composite_simpson(a, b, n, f):
    x = np.linspace(a, b, n)
    h = (b-a)/(n-1)
    return h/3 *(f(x[0]) + 2*np.sum(f(x[2:-1:2])) + 4*np.sum(f(x[1::2])) + f(x[-1]))


def err_int(a, b, n):
    g2_int_real = (-b ** 2 * np.cos(3 * b) + a ** 2 * np.cos(3 * a)) / 3 + (2 * b * np.sin(3 * b) - 2 * a * np.sin(3 * a)) / 9 + (2 * np.cos(3 * b) - 2 * np.cos(3 * a)) / 27
    g2_int = composite_simpson(a, b, n, g2)
    return np.abs(g2_int_real - g2_int)


def base_int():
    a = 0
    b = np.pi
    n1 = 3
    n2 = 9999
    h1 = np.log10((b - a) / (n1 - 1))
    h2 = np.log10((b - a) / (n2 - 1))
    h = np.logspace(h2, h1, 50)
    n = np.array((b - a) / h + 1, dtype=int)
    for i in range(len(n)):
        if n[i] % 2 == 0:
            n[i] += 1
    h = np.array((b - a) / (n - 1))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.grid()
    ax.set_xlabel('$h$')
    ax.set_ylabel('$\Delta(I)$')
    ax.loglog(h, [err_int(a, b, el) for el in n], 'ro', markersize=6)
    h_for_scaling = np.logspace(h1, h2/2, 50)
    ax.loglog(h_for_scaling, 0.2 * h_for_scaling ** 4,'--', label = '$O(h^4)$')
    ax.legend()
    plt.show()


def diff4(x_0, h, f):
    return (f(x_0-2*h)-8*f(x_0-h)+8*f(x_0+h)-f(x_0+2*h))/(12*h)





def advanced_diff():
    x_0 = 2
    h = np.logspace(-16, 0, 90)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.grid()
    ax.set_xlabel('$h$')
    ax.set_ylabel('$\Delta(g_1\'(x_0))$')
    h_for_scaling = np.logspace(-5, 0, 100)
    ax.loglog(h, [err_dif(x_0, el, diff2) for el in h], 'ro', label='$\Delta(g_1\'(x_0))$')
    ax.loglog(h_for_scaling, h_for_scaling ** 2, '--', label='$O(h^2)$')
    ax.loglog(h, [err_dif(x_0, el, diff4) for el in h], 'bo', label='$\Delta(g_1\'(x_0))$')
    ax.loglog(h_for_scaling, h_for_scaling ** 4, 'g--', label='$O(h^4)$')
    ax.legend()

    #подсчет оптимального шага
    M_3 = 5*np.exp(2)
    M_5 = 7*np.exp(2)
    h_opt2 = (3.*(np.finfo(np.float64).eps)/M_3)**(1./3)
    h_opt4 = (45. * (np.finfo(np.float64).eps) / (4*M_5)) ** (1./5)
    print('Оптимальный шаг дифференцирования для 2-го порядка:  ', h_opt2)
    print('Оптимальный шаг дифференцирования для 4-го порядка:  ', h_opt4)
    #подсчет минимально достижимой погрешности
    e_2 = (np.finfo(np.float64).eps) / h_opt2 + (h_opt2 ** 2) * M_3 / 6
    e_4 = (3*(np.finfo(np.float64).eps))/(12*h_opt4) + (h_opt4**4)*M_5/30
    print('Минимально-достижимая погрешность для 2-го порядка:  ', e_2)
    print('Минимально-достижимая погрешность для 4-го порядка:  ', e_4)
    plt.show()



def gauss_quad5(f):
    return (5/9)*f(-np.sqrt(3/5))+(8/9)*f(0)+(5/9)*f(np.sqrt(3/5))


def gauss_quad5_ab(f, a, b):
    x1 = (a+b)/2 + (b-a)*(-np.sqrt(3/5))/2
    x2 = (a+b)/2
    x3 = (a+b)/2 + (b-a)*(np.sqrt(3/5))/2
    return (5/9)*f(x1)+(8/9)*f(x2)+(5/9)*f(x3)


def int_exact(kofs, a, b, k):
    integral = 0
    for i in range(k+1):
        integral += (kofs[i] * b ** (i + 1) / (i + 1))
        integral -= (kofs[i] * a ** (i + 1) / (i + 1))
    return integral


def advanced_int():
    polinoms_kofs = []
    for k in range(7):
        polinoms_kofs.append([])
        for i in range(k + 1):
            polinoms_kofs[k].append(np.random.randn())

    polinoms = []
    polinoms.append(lambda x: polinoms_kofs[0][0])
    polinoms.append(lambda x: polinoms_kofs[1][0] + polinoms_kofs[1][1]*x**1)
    polinoms.append(lambda x: polinoms_kofs[2][0] + polinoms_kofs[2][1] * x ** 1 + polinoms_kofs[2][2] * x ** 2)
    polinoms.append(
        lambda x: polinoms_kofs[3][0] + polinoms_kofs[3][1] * x ** 1 + polinoms_kofs[3][2] * x ** 2 + polinoms_kofs[3][
            3] * x ** 3)
    polinoms.append(
        lambda x: polinoms_kofs[4][0] + polinoms_kofs[4][1] * x ** 1 + polinoms_kofs[4][2] * x ** 2 + polinoms_kofs[4][
            3] * x ** 3 + polinoms_kofs[4][4] * x ** 4)
    polinoms.append(
        lambda x: polinoms_kofs[5][0] + polinoms_kofs[5][1] * x ** 1 + polinoms_kofs[5][2] * x ** 2 + polinoms_kofs[5][
            3] * x ** 3 + polinoms_kofs[5][4] * x ** 4 + polinoms_kofs[5][5] * x ** 5)
    polinoms.append(
        lambda x: polinoms_kofs[6][0] + polinoms_kofs[6][1] * x ** 1 + polinoms_kofs[6][2] * x ** 2 + polinoms_kofs[6][
            3] * x ** 3 + polinoms_kofs[6][4] * x ** 4 + polinoms_kofs[6][5] * x ** 5 + polinoms_kofs[6][6] * x ** 6)

    for i in range(7):
        print('Степень Полинома: ', i)
        print('коэффициенты полинома: ', polinoms_kofs[i])
        exact_int = int_exact(polinoms_kofs[i], 0, 2, i)
        print('Точное значение интеграла на промежутке от 0 до 2: ', exact_int)
        approx_int = gauss_quad5_ab(polinoms[i], 0, 2)
        print('Вычисленное значение интеграла на промежутке от 0 до 2: ', approx_int)
        err = np.abs(exact_int - approx_int)
        print('Погрешность: ', err)






if (__name__ == '__main__'):
    print('Машинное эпсилон:  ', np.finfo(np.float64).eps)
    base_diff()
    base_int()
    advanced_diff()
    advanced_int()

