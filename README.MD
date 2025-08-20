# Machine Learning Algorithms from Scratch

A comprehensive collection of machine learning algorithms implemented from scratch in Python, along with educational Jupyter notebooks demonstrating core ML concepts.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms Implemented](#algorithms-implemented)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains educational implementations of fundamental machine learning algorithms built from scratch using Python and NumPy. The goal is to provide clear, well-documented code that helps understand the mathematical foundations and inner workings of these algorithms.

## ✨ Features

- **Pure Python implementations** - No ML libraries used for algorithm core logic
- **Educational focus** - Clear code with detailed comments
- **Comprehensive collection** - 8+ algorithms covering classification and regression
- **Interactive notebooks** - Jupyter notebooks with step-by-step explanations
- **Visualization** - Plots and graphs showing algorithm behavior
- **Ready-to-run examples** - Each algorithm includes working test cases

## 📁 Repository Structure

```
machine-learning/
├── ml-algorithms-scratch/          # Core algorithm implementations
│   ├── adaboost.py                # AdaBoost ensemble method
│   ├── decision_tree.py           # Decision Tree classifier
│   ├── knn.py                     # K-Nearest Neighbors
│   ├── logistic_regression.py     # Logistic Regression
│   ├── naive_bayes.py             # Naive Bayes classifier
│   ├── perceptron.py              # Single-layer Perceptron
│   ├── random_forest.py           # Random Forest ensemble
│   └── svm.py                     # Support Vector Machine
├── Gradient_Descent.ipynb         # Gradient descent optimization
├── Linear_Regression_in_One_Variable.ipynb  # Linear regression tutorial
├── Logistic_Regression.ipynb      # Logistic regression from scratch  
├── Polynomial_Regression.ipynb    # Polynomial feature engineering
├── results/                        # Output visualizations
│   ├── knn.png                    # KNN classification results
│   └── svm.png                    # SVM decision boundary
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## 🛠 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdulhakkeempa/machine-learning.git
   cd machine-learning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install numpy matplotlib scikit-learn pandas
   ```

## 🚀 Usage

### Running Individual Algorithms

Each algorithm file can be executed directly to see a demonstration:

```bash
cd ml-algorithms-scratch

# Run Logistic Regression example
python logistic_regression.py

# Run Support Vector Machine example
python svm.py

# Run K-Nearest Neighbors example
python knn.py
```

### Using Algorithms in Your Code

```python
# Example: Using the custom Logistic Regression
import sys
sys.path.append('ml-algorithms-scratch')
from logistic_regression import LogisticRegression
import numpy as np

# Create sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Train the model
lr = LogisticRegression(alpha=1, epochs=10)
lr.fit(X, y)

# Make predictions
predictions = lr.predict(X)
print(f"Predictions: {predictions}")
```

### Jupyter Notebooks

Launch Jupyter to explore the educational notebooks:

```bash
jupyter notebook
```

Then open any of the `.ipynb` files to see detailed explanations and visualizations.

## 🤖 Algorithms Implemented

### Classification Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| **Logistic Regression** | `logistic_regression.py` | Binary classification using sigmoid function |
| **Support Vector Machine** | `svm.py` | Maximum margin classifier with regularization |
| **K-Nearest Neighbors** | `knn.py` | Instance-based learning algorithm |
| **Decision Tree** | `decision_tree.py` | Tree-based classifier using information gain |
| **Random Forest** | `random_forest.py` | Ensemble of decision trees |
| **Naive Bayes** | `naive_bayes.py` | Probabilistic classifier using Bayes' theorem |
| **Perceptron** | `perceptron.py` | Single-layer neural network |
| **AdaBoost** | `adaboost.py` | Adaptive boosting ensemble method |

### Regression Algorithms (Jupyter Notebooks)

| Algorithm | Notebook | Description |
|-----------|----------|-------------|
| **Linear Regression** | `Linear_Regression_in_One_Variable.ipynb` | Simple linear regression implementation |
| **Polynomial Regression** | `Polynomial_Regression.ipynb` | Feature engineering with polynomial terms |

### Key Features of Each Implementation

- **Logistic Regression**: Gradient descent optimization, sigmoid activation
- **SVM**: Hinge loss, L2 regularization, decision boundary visualization  
- **KNN**: Euclidean distance, majority voting, configurable k value
- **Decision Tree**: Information gain splitting, configurable max depth
- **Random Forest**: Bootstrap aggregating, feature randomness
- **Naive Bayes**: Gaussian distribution assumption, Laplace smoothing
- **Perceptron**: Binary classification, linear activation
- **AdaBoost**: Weak learner combination, adaptive weights

## 📚 Jupyter Notebooks

Interactive notebooks with detailed explanations:

1. **Gradient_Descent.ipynb** - Understanding optimization fundamentals
2. **Linear_Regression_in_One_Variable.ipynb** - Simple linear regression walkthrough  
3. **Logistic_Regression.ipynb** - Binary classification from first principles
4. **Polynomial_Regression.ipynb** - Feature engineering and overfitting

Each notebook includes:
- Mathematical foundations
- Step-by-step implementation
- Visualizations and plots
- Real-world examples

## 📊 Results

The `results/` folder contains visualizations generated by the algorithms:

- **knn.png** - K-Nearest Neighbors classification boundaries
- **svm.png** - Support Vector Machine decision boundaries and support vectors

Example outputs show:
- Decision boundaries for classification problems
- Learning curves and convergence behavior
- Algorithm performance on test datasets

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/new-algorithm`)
3. **Add your algorithm** with proper documentation
4. **Include test cases** and examples
5. **Add visualizations** if applicable
6. **Submit a pull request**

### Guidelines

- Follow existing code style and structure
- Include docstrings and comments
- Add test cases in the `if __name__ == "__main__":` block
- Update README if adding new algorithms

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Abdul Hakkeem PA**

- GitHub: [@abdulhakkeempa](https://github.com/abdulhakkeempa)

## 🎓 Educational Purpose

This repository is designed for educational purposes to help students and practitioners understand machine learning algorithms from the ground up. The implementations prioritize clarity and understanding over performance optimization.

---

⭐ **Star this repository if you find it helpful!**
