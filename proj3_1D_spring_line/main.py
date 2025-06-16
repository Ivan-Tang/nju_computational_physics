import numpy as np
import matplotlib.pyplot as plt

N = 100  # 必须为偶数
k = 1.0
m1 = 1.0
m2 = 2.0

# 刚度矩阵 K
K = 2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
masses = np.array([m1 if i % 2 == 0 else m2 for i in range(N)])
M = np.diag(masses)
A = np.linalg.inv(M) @ K
omega2, modes = np.linalg.eigh(A)
omega = np.sqrt(omega2)

q_vals = np.arange(1, N//2+1) * np.pi / (N//2 + 1)

# 数值本征值分为声学支和光学支
omega_ac = omega[:N//2]
omega_op = omega[N//2:N]

plt.figure(figsize=(8,5))
plt.scatter(q_vals, omega_ac, c='b', s=12, label='Numerical acoustic')
plt.scatter(q_vals, omega_op, c='m', s=12, label='Numerical optical')
plt.xlabel("Wave vector $q$")
plt.ylabel("Frequency $\omega$")
plt.title("Dispersion Relation of 1D Alternating Mass Chain (Numerical)")
plt.grid()
plt.xlim(0, np.pi)
plt.xticks([0, np.pi/2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
plt.legend()
plt.savefig('tex/dispersion.png')