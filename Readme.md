### **🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)** ###

This project implements the K-Nearest Neighbors (KNN) algorithm to classify Iris flowers into three species based on four measurable features.
KNN is a simple, distance-based ML algorithm commonly used for classification tasks.

### **📘 Project Overview** ###

The Iris dataset consists of:

150 samples

4 features

3 species:

Setosa

Versicolor

Virginica

This project:

1.Loads the dataset

2.Splits it into training and testing sets

3.Trains KNN models with different K values

4.Evaluates accuracy for each K

Prints performance results

 ### **🧠 What is KNN?** ###

K-Nearest Neighbors is a non-parametric, lazy learning algorithm.

**How it works:**

Choose a value for K (number of neighbors).

For each prediction:

Compute distance from the test point to all training points.

Pick the closest K points.

Use majority class to predict.

Output the class with the most neighbors.

Smaller K → more sensitive (possible overfitting)
Larger K → smoother (possible underfitting)

### **🔧 Technologies Used** ###

Python

Scikit-learn

NumPy

Pandas

Jupyter/Colab

**🚀 Implementation Steps**

**1. Import Libraries**

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

**2. Load Dataset**

iris = load_iris()

X = iris.data

y = iris.target

**3. Train-Test Split**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**4. Loop Through K Values**
for k in [1, 3, 5, 7, 9]:

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"K = {k} → Accuracy = {accuracy:.2f}")

### **📊 Results** ###

Accuracy will vary based on the test split, but typically:

K Value	Accuracy
1	0.96 – 1.00
3	0.95 – 0.98
5	0.94 – 0.97
7	0.93 – 0.96
9	0.92 – 0.95


### **📌 Key Observations** ###

K=3 often performs the best for this dataset.

KNN is simple yet effective for small datasets.

Accuracy decreases slightly as K increases.


