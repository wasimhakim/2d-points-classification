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
epochs = 1000
computed_cost = np.empty(epochs)

def predict(x):
  f_wb = np.dot(x, w) + b
  z = 1 / (1 + np.exp(-f_wb))
  return z

def compute_cost():
  cost = 0.0
  for i in range(m):
    z = predict(X[i])
    cost += -y[i] * np.log(z) - (1 - y[i]) * np.log(1 - z)
  cost = (1/m) * cost
  return cost

# train
for a in range(epochs):
  dj_dw = np.zeros(2,)
  dj_db = 0

  for i in range(m):
    z = predict(X[i])
    err_i = z - y[i]
    for j in range(n):
      dj_dw[j] += err_i * X[i,j]
    dj_db += err_i
  w = w - learning_rate * dj_dw
  b = b - learning_rate * dj_db
  computed_cost[a] = compute_cost()

predicted_line = (-(w[0]/w[1])*x1) - (b/w[1])

# plt.plot(range(epochs), computed_cost)
# plt.show()

# Plotting  
plt.scatter(x1, x2, c=y, cmap='bwr')
plt.plot(x1, line_y, color='blue')
plt.plot(x1, predicted_line, color='red')
plt.title("Random 2D Data Points")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()
