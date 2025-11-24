import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
num_points = 200
learning_rate = 0.001
X = np.random.randn(num_points, 2)
m,n = X.shape

x1 = X[:, 0]
x2 = X[:, 1]

line_y = 2 * x1 + 1

y = (x2 > line_y).astype(int)
w = np.zeros(2)
b = 0
dj_dw = np.zeros(2,)
dj_db = 0

def predict(x):
  f_wb = np.dot(x, w) + b
  z = 1 / (1 + np.exp(-f_wb))
  return z

# train
for a in range(100):
  for i in range(m):
    z = predict(X[i])

    for j in range(n):
      dj_dw[j] = np.sum((z - y[i]) * X[i,j])
      dj_dw[j] = dj_dw[j] / num_points
      w[j] = w[j] - learning_rate * dj_dw[j]
    dj_db = np.sum(z - y[i]) / num_points
    b = b - learning_rate * dj_db

if(predict([-2.00, 1.00]) >= 0.5):
  print(1)
else:
  print(0)


# Plotting  
plt.scatter(x1, x2, c=y, cmap='bwr')
plt.title("Random 2D Data Points")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()
