🧠 Deep Learning Lab

A hands-on repository for implementing fundamental deep learning concepts from scratch using PyTorch.

This lab focuses on understanding:
Tensor operations
Autograd and backpropagation
Linear & logistic regression
Feedforward neural networks
Dataset and DataLoader abstraction
Optimization algorithms

The goal is not just to use high-level APIs, but to deeply understand how models learn.

📂 Repository Structure
Deep-Learning-Lab/
│
├── linear_regression/
│   ├── basic_gradient_descent.py
│   ├── nn_module_version.py
│
├── logistic_regression/
│   ├── binary_classification.py
│   ├── xor_problem.py
│
├── multiple_linear_regression/
│   ├── multi_feature_regression.py
│
├── datasets/
│   ├── logistic_data.csv
│   ├── mulRegData.csv
│
└── README.md 

This is yet to happen.

🚀 Implemented Models
1️⃣ Linear Regression

Manual gradient descent
Using nn.Module
Using Dataset and DataLoader
Mini-batch training
Loss visualization

2️⃣ Logistic Regression

Binary classification
BCELoss vs BCEWithLogitsLoss
Proper train/test split
Decision boundary understanding

3️⃣ XOR Neural Network

Demonstrates failure of logistic regression
One hidden layer neural network
Sigmoid and ReLU activation
Non-linear decision boundaries

4️⃣ Multiple Linear Regression

Multi-feature input
Matrix multiplication (x @ w)
Parameter optimization
Loss curve plotting

🔬 Core Concepts Practiced

Forward pass computation
Backpropagation with .backward()
Gradient accumulation & zeroing
Model modes: train() vs eval()
torch.no_grad() usage
Batch training via DataLoader
Non-linearity in neural networks
Why linear models fail on XOR

🛠 Technologies Used

Python
PyTorch
Pandas
Matplotlib