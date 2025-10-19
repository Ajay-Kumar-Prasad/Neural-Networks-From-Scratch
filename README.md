# 🧠 Neural Network from Scratch (NumPy Implementation)

A simple **two-layer Neural Network** (built completely from scratch using NumPy) trained on the **MNIST handwritten digits dataset** (0–9).  
This project demonstrates how a neural network works **without using deep learning frameworks** like TensorFlow or PyTorch — only `NumPy`.

---

## 🚀 Project Overview

This project implements:
- A **feedforward neural network** with one hidden layer
- **ReLU** activation for the hidden layer
- **Softmax** activation for the output layer
- **Mini-batch gradient descent** for optimization
- **Cross-entropy loss** function for training
- **Accuracy evaluation** on test data

---

## 🧩 Architecture
```
Input (784 nodes: 28x28 image)
↓
Hidden Layer (128 neurons, ReLU)
↓
Output Layer (10 neurons, Softmax)
```
---
## 🧰 Project Structure
```
NN_from_scratch/
│
├── main.py          # Main training and evaluation script
├── utils.py         # Utility functions (ReLU, Softmax, loss, accuracy)
├── README.md        # Project documentation (you’re reading this)
└── requirements.txt # Dependencies (optional)
```
---
## 📊 Results
After training for 10 epochs (with hidden_size=128 and learning_rate=0.01),
the loss and accuracy came out to be:
---
![alt text](image/image.png)
---
![alt text](image/plot.png)
---

| Metric        | Result      |
| ------------- | ----------- |
| Training Loss | ~0.1197 |
| Test Accuracy | **92.41%**  |

---

## 📘 Key Learnings

- Understanding how forward and backward propagation work mathematically
- Implementing gradient descent manually
- Working with one-hot encoding for categorical targets
- Handling matrix operations efficiently in NumPy
- Building an end-to-end training loop from scratch

**⭐ If you found this helpful, give the repo a star on GitHub!**