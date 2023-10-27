import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def centered(X):
    m = X.shape[0]
    A = (np.eye(m) - 1/m * np.ones((m,m))) @ X
    return A


A = np.array([[1.1, 1.3, 1.5, 1.55, 1.6, 1.9, 2, 2.1],
     [2, 0.9, 0.7, 1.5, 2.6, 0.3, 0.8, 1.4],
     [-2.9, -0.5, 0.1, -1.5, -3.6, 1.3, 0.4, -0.7],
     [1.1, 0.2, 0.1, 0.6, 1.3, -0.4, -0.1, 0.4],
     [0.9, -0.4, -0.8, -0, 1, -1.6, -1.2, -0.7]
     ])
print('A')
print(A)
A=centered(A)
print('centered')
print(np.round(A, 5))

print('K')
K = A.T@A
print(K)

print('собственные вектора и числа матрицы K')
self_num, self_vec = np.linalg.eig(K)
print(np.round(self_num, 5))
print('')
print(np.round(self_vec, 5))


# indx = self_num.argsort()[::-1]
# self_num = self_num[indx]
# self_vec = self_vec[:, indx]
# print('self_num = ', self_num)
# print('self_vec = ', self_vec)

pc = self_vec.T
print('Главные компоненты :')
print(np.round(pc, 5))
self_num = np.round(self_num, 5)
print('сингулярные числа')
# for i in range(len(self_num)):
#     if self_num[i] == -0:
#         self_num[i] = 0
sing_num = np.sqrt(self_num)
print(sing_num)
print('отклонения')
otkl = np.sqrt(1/4)*sing_num
print(otkl)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.grid()
ax.set_xlabel('компонента')
ax.set_ylabel('отклонение')
ax.plot(range(1, 9), otkl,'r')

print('выбранные главные компоненты')
comp = pc[:2, :]
print(comp)

N = A @ comp.T
print(N)
plt.show()