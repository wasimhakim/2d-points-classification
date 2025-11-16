import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
num_points = 200
X = np.random.randn(num_points, 2)

x1 = X[:, 0]
x2 = X[:, 1]

line_y = 2 * x1 + 1

y = (x2 > line_y).astype(int)

# Plotting  
plt.scatter(x1, x2, c=y, cmap='bwr')
plt.title("Random 2D Data Points")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()
