import numpy as np
import matplotlib.pyplot as plt

def f(x, x0, y0):
    return np.exp(-x**2) / np.sqrt((x - x0)**2 + y0**2)

def refinement_function(x, x0, A):
    return 1.0 / (1 + np.exp(abs(x - x0) / A))

# 参数
N = 101
A = 1.0
x_vals = np.linspace(-A, A, N)
h = x_vals[1] - x_vals[0]

# 网格
xaxis = np.linspace(-5, 5, 50)
yaxis = np.linspace(-5, 5, 50)
field = np.zeros((len(xaxis), len(yaxis)))  

for i, x0 in enumerate(xaxis):
    for j, y0 in enumerate(yaxis):
        integral = 0.0
        for k in range(N - 1):
            # 根据位置调整步长
            h_adjusted = refinement_function(x_vals[k], x0, A) * (x_vals[k+1] - x_vals[k])
            integral += 0.5 * (f(x_vals[k], x0, y0) + f(x_vals[k+1], x0, y0)) * h_adjusted
        field[i, j] = integral

extent = [-5, 5, -5, 5]

fig, ax = plt.subplots(figsize=(10, 5))
c = ax.imshow(field.T, extent=extent, origin="lower", cmap="rainbow", aspect="auto")
levels = np.linspace(0, np.max(field), 20)
cs = ax.contour(xaxis, yaxis, field.T, levels, linewidths=1.5, cmap="inferno", alpha=0.5)
fig.colorbar(c, ax=ax)

ax.set_title("Contour and Color Map of Temperature Field")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()