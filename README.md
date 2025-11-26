# Logistic Regression From Scratch — 2D Line Separation

This project implements **logistic regression from scratch** (no ML libraries) and tests it on a synthetic 2D dataset. The goal was to understand the mathematics and mechanics of logistic regression by building the full pipeline manually. The prediction, cost function, gradients, and training.

---

## 🎯 Objective

Using only:
- NumPy  
- Matplotlib  
- Pure Python logic  

We:

- Generated random 2D data points  
- Separated them into binary classes using a true line  
- Implemented logistic regression  
- Trained the model using gradient descent  
- Visualized both the **true line** and the **learned line**  

---

## 🧠 Concept

We created synthetic data points:

\[
X = [x_1, x_2]
\]

Then labeled them according to:

\[
x_2 > 2x_1 + 1
\]

This created a clean, linearly separable dataset.  
Logistic regression was then trained to **learn** the separating boundary.

The learned decision boundary:

\[
w_1 x_1 + w_2 x_2 + b = 0
\]

was visualized alongside the true line.

---

## 📉 Learning Curve

The learning curve (cost vs iterations) showed:

- Rapid decrease in cost during early iterations  
- Smooth convergence  
- Asymptotic flattening  

This confirmed gradients and cost implementation were correct.

---

## 🧪 Results

- The model successfully learned a separating line  
- The learned boundary closely matched the true underlying line  
- Accuracy approached ~100% due to clean separability  
- Visualization confirmed correct classification  

---

## 🚀 How To Run

```bash
git clone <your repo url>
cd <project-folder>
python3 src/app.py