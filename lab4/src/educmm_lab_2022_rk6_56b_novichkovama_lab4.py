import os
import numpy as np
import matplotlib.pyplot as plt
from lab3_advanced import solve, lu

alpha = 3
beta = 0.002
delta = 0.0006
gamma = 0.5


def f(x):
    res = np.array([alpha* x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]])
    return res


def Yacoby(x):
    J_x = np.array([[alpha-beta*x[1], -beta*x[0]], [delta*x[1], delta*x[0]-gamma]])
    return J_x


def rk4(x0, tn, f, h):
    t = np.arange(0, tn, h)
    time = t.size
    kolvo = x0.size
    x = np.zeros((time, kolvo))
    x[0] = x0
    for k in range(time - 1):
        k1 = h * f(t[k], x[k])
        k2 = h * f(t[k] + h / 2, x[k] + k1 / 2)
        k3 = h * f(t[k] + h / 2, x[k] + k2 / 2)
        k4 = h * f(t[k] + h, x[k] + k3)
        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[k + 1] = x[k] + dx
    return x, t


def newton(x_0, f, J):
    eps = 10**-8
    x_1 = x_0
    L, U, P = lu(J(x_1), True)
    y_1 = solve(L, U, P, f(x_1))
    x_k = x_1-y_1
    iter = 1
    while np.linalg.norm(x_k-x_1, ord=np.inf)> eps:
        iter += 1
        x_1 = x_k
        L, U, P = lu(J(x_1), True)
        y_1 = solve(L, U, P, f(x_1))
        x_k = x_1 - y_1
    return x_k, iter



def g(x):
    return f(x).dot(f(x))

def count_t(x_1, z_k):
    h = lambda t: g(x_1 - t * z_k / (np.linalg.norm(z_k, ord=2)))
    t_1 = 0.
    t_3 = 1.
    iter = 0
    while h(t_1) <= h(t_3):
        iter += 1
        t_3_d = t_3 / 2.
        t_3_u = t_3 * 2.
        if (np.linalg.norm([h(t_1) - h(t_3_d)], ord=np.inf) > np.linalg.norm([h(t_1) - h(t_3_u)], ord=np.inf)):
            t_3 = t_3_u
        else:
            t_3 = t_3_d
        if (iter > 500):
            break
    t_2 = t_3 / 2.
    a = h(t_1) / ((t_1 - t_2) * (t_1 - t_3))
    b = h(t_2) / ((t_2 - t_1) * (t_2 - t_3))
    c = h(t_3) / ((t_3 - t_1) * (t_3 - t_2))
    t_k = (a * (t_2 + t_3) + b * (t_1 + t_3) + c * (t_1 + t_2)) / (2 * (a + b + c))
    return t_k

def t_speedest(x_k, z_k):
    h = lambda t: g(x_k - t * z_k / (np.linalg.norm(z_k, ord=2)))
    t = np.linspace(-9, 2, num=25)
    h_k = np.array([h(10 ** i) for i in t])
    t_res = 10 ** t[np.argmin(h_k)]
    return t_res


def tSearch(x_k, z_k):
    h = lambda t: g(x_k - t * z_k / (np.linalg.norm(z_k, ord=2)))
    t = np.linspace(-30, 30, num=60)
    h_k = np.array([h(2**i) for i in t])
    t_res = 2**t[np.argmin(h_k)]
    #print(t_res)
    return t_res



def gradient_descent(x_0, f, J):
    eps = 10 ** -8
    x_1 = x_0
    J_t = np.transpose(J(x_1))
    z_k = J_t.dot(f(x_1))
    #t_k = count_t(x_1, z_k)
    t_k = t_speedest(x_1, z_k)
    x_k = x_1 - t_k*z_k/(np.linalg.norm(z_k, ord=2))
    iter = 1
    while np.linalg.norm(x_k-x_1, ord=np.inf)> eps:
        iter += 1
        x_1 = x_k
        J_t = np.transpose(J(x_1))
        z_k = J_t.dot(f(x_1))
        t_k = t_speedest(x_1, z_k)
        #t_k = count_t(x_1, z_k)
        x_k = x_1 - t_k * z_k / (np.linalg.norm(z_k, ord=2))
    return  x_k, iter


def newton_s(x_0, f, J):
    eps = 10**-8
    x_s = np.array([gamma / delta, alpha / beta])
    x_1 = x_0
    L, U, P = lu(J(x_1), True)
    y_1 = solve(L, U, P, f(x_1))
    x_k = x_1-y_1
    iter = 1
    lambds = []
    while np.linalg.norm(x_k-x_1, ord=np.inf)> eps:
        iter += 1
        x_1 = x_k
        L, U, P = lu(J(x_1), True)
        y_1 = solve(L, U, P, f(x_1))
        x_k = x_1 - y_1
        lambd = np.abs(np.linalg.norm(x_k, ord=np.inf) - np.linalg.norm(x_s, ord=np.inf)) / np.abs(
            np.linalg.norm(x_1, ord=np.inf) - np.linalg.norm(x_s, ord=np.inf)) ** 2
        lambds.append(lambd)
    return  lambds

def gradient_descent_s(x_0, f, J):
    eps = 10 ** -8
    x_s = np.array([gamma / delta,alpha / beta])
    x_1 = x_0
    J_t = np.transpose(J(x_1))
    z_k = J_t.dot(f(x_1))
    #t_k = count_t(x_1, z_k)
    t_k = t_speedest(x_1, z_k)
    x_k = x_1 - t_k*z_k/(np.linalg.norm(z_k, ord=2))
    iter = 1
    lambds = []
    while np.linalg.norm(x_k-x_1, ord=np.inf)> eps:
        iter += 1
        x_1 = x_k
        J_t = np.transpose(J(x_1))
        z_k = J_t.dot(f(x_1))
        t_k = t_speedest(x_1, z_k)
        #t_k = count_t(x_1, z_k)
        x_k = x_1 - t_k * z_k / (np.linalg.norm(z_k, ord=2))
        lambd = np.abs(np.linalg.norm(x_k, ord=np.inf) - np.linalg.norm(x_s, ord=np.inf)) / np.abs(
            np.linalg.norm(x_1, ord=np.inf) - np.linalg.norm(x_s, ord=np.inf))
        lambds.append(lambd)
    return  lambds


def adv_f():
    y_s = alpha / beta
    x_s = gamma / delta
    lambds_g = gradient_descent_s(np.array([1700, 900]), f, Yacoby)
    print(lambds_g)
    lambds_n = newton_s(np.array([1600, 900]), f, Yacoby)
    print(lambds_g)
    x_data = np.logspace(-4, 2, len(lambds_g))
    print(x_data)
    y_data = [lambds_g[i] * x_data[i] for i in range(len(lambds_g))]
    print(y_data)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(x_data, y_data, 'bo', label=r'$y~=~\lambda x$')
    x_data_h = np.linspace(1e-4, 1e2, 30)
    y_data_h = x_data_h
    ax.plot(x_data_h, y_data_h, 'b--', label=r'$O(x)$')
    ax.legend()
    ax.grid()
    plt.loglog()


    x_data = np.logspace(-4, 2, len(lambds_n))
    y_data = [lambds_n[i] * x_data[i]**2 for i in range(len(lambds_n))]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(x_data, y_data, 'ro', label=r'$y~=~\lambda x^2$')
    x_data_h = np.linspace(1e-4, 1e2, 30)
    y_data_h = x_data_h ** 2
    ax.plot(x_data_h, y_data_h, 'b--', label=r'$O(x^2)$')
    ax.legend()
    ax.grid()
    plt.loglog()

    plt.show()


def advanced():
    n=201
    y_s = alpha / beta
    x_s = gamma / delta
    static_1 = np.array([x_s, y_s])
    nodes = np.linspace(0, 200, n)
    norms_n = np.zeros((n, n))
    norms_g = np.zeros((n, n))
    x_zeros = np.linspace(0, 3000, n)
    y_zeros = np.linspace(0, 3000, n)
    iters_n = []
    iters_g = []
    i=0
    count = 0
    for x in nodes:
        j=0
        for y in nodes:
            count += 1
            os.system('CLS')
            print(f"Осталось: {100 - count / (n ** 2) * 100 :.{1}f}%")
            x0 = np.array([15*x, 15*y])
            #print(x0)
            root_n, iter_n = newton(x0, f, Yacoby)
            iters_n.append(iter_n)
            root_g, iter_g = gradient_descent(x0, f, Yacoby)
            #print(root_g)
            iters_g.append(iter_g)
            norm_n = np.linalg.norm(root_n, ord=np.inf)  
            norm_g = np.linalg.norm(root_g, ord=np.inf)
            norms_n[i,j] = norm_n
            norms_g[i,j] = norm_g
            j+=1
        i+=1
    N_n = len(iters_n)
    N_g = len(iters_g)
    M_n = sum(iters_n)/N_n
    M_g = sum(iters_g) /N_g
    print("M_n = ", M_n)
    print("M_g = ", M_g)
    #print(iters_n)
    #print(iters_g)
    S_n = np.sqrt(1/(N_n*(N_n-1))*sum([(el - M_n)**2 for el in iters_n]))
    S_g = np.sqrt(1 / (N_g * (N_g - 1)) * sum([(el - M_g) ** 2 for el in iters_g]))
    print("S_n = ", S_n)
    print("S_g = ", S_g)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cs = ax.contourf(x_zeros, y_zeros,  norms_n, cmap ="winter")
    cbar = plt.colorbar(cs)
    #ax.contourf([nodes, nodes,], norms)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cs = ax.contourf(x_zeros, y_zeros, norms_g, cmap ="winter")
    cbar = plt.colorbar(cs)
    plt.show()
    #print('end')
    #print(iter, )


def base():
    func = lambda t, x: f(x)
    start_nodes = np.array([i * 200 for i in range(1, 11)])
    # x = [600, 200]
    # print(start_nodes)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.grid()
    ax.set_ylabel('$количевство <хищников> y$')
    ax.set_xlabel('$количевство <жертв> x$')
    for x_1 in start_nodes:
        for y_1 in start_nodes:
            x_0 = np.array([x_1, y_1])
            res, t = rk4(x_0, 8, func, 0.1)
            # print(res)
            x_nodes = res[:, 0]
            y_nodes = res[:, 1]
            ax.plot(x_nodes, y_nodes, 'r')
    y_s = alpha / beta
    x_s = gamma / delta
    ax.plot(x_s, y_s, 'bo')
    #ax.plot(0, 0, 'bo')
    #x_rep = np.array([800, 600])
    x_rep = np.array([x_s, y_s])
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    res, t = rk4(x_rep, 20, func, 0.1)
    ax.grid()
    ax.set_xlabel('$t$')
    # ax.set_ylabel('$$')
    x_nodes = res[:, 0]
    y_nodes = res[:, 1]
    #ax.plot( x_nodes, y_nodes, 'red', label='$y(t)$')
    ax.plot(t, x_nodes, 'b', label='$x(t)$')
    ax.plot(t, y_nodes, 'red', label='$y(t)$')
    ax.legend()
    plt.show()


if (__name__ == '__main__'):
    #advanced()
    adv_f()
    #base()
