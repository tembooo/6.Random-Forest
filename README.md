# 6.Random-Forest
6.Random Forest
<br>

Implement a random forest (RF) based on decision trees in Matlab or Python. A readily available implementation for learning a single decision tree can be used, but implement the bootstrapping and classification from the collection of decision trees by yourself.
<br>
Experiment with the suggested data and compare the performance with a single decision tree. Vary the number of decision trees in the forest and the size of the training subset in the experiments to see if they make a difference.
<br>
Additional files: arrhythmia.mat (Matlab).
<br>
Hints: In Matlab, the constructor T=fitctree(X,Y) can be used.
<br>
Hints: In Python, for example, "from sklearn.tree import DecisionTreeClassifier"
<br>

```python
############################## part 1 : Import required libraries ##############################

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import scipy.io as sio

############################## part 2 : Load .mat file and extract features and labels ##########

# Path to the .mat file
file_path = r'C:\12.LUT\00.Termic Cources\2.pattern recognition\jalase6\Exersice\class excersice\arrhythmia.mat'

# Load the .mat file using scipy.io
data = sio.loadmat(file_path)

# Extract features and labels from the loaded data
# Note: Replace 'X' and 'Y' with the actual variable names in the .mat file
features = data['X']
labels = data['Y'].ravel()  # Flatten the label array

############################## part 3 : Define a function to sample data (bootstrap sampling) ###

# Function to create bootstrapped datasets by randomly sampling data with replacement
def sample_data(data_X, data_Y, n_samples):
    indices = np.random.choice(range(data_X.shape[0]), size=n_samples, replace=True)
    return data_X[indices], data_Y[indices]

############################## part 4 : Define the Random Forest class ##########################

# Custom Random Forest class that uses Decision Trees
class ForestModel:
    def __init__(self, tree_count=10, subset_ratio=0.8):
        self.tree_count = tree_count  # Number of trees in the forest
        self.subset_ratio = subset_ratio  # Proportion of data used for each tree (bootstrap sample)
        self.models = []  # List to store trained Decision Trees

    def fit(self, data_X, data_Y):
        # Determine the sample size based on the subset ratio
        sample_size = int(data_X.shape[0] * self.subset_ratio)
        # Train multiple Decision Trees
        for _ in range(self.tree_count):
            X_sample, Y_sample = sample_data(data_X, data_Y, sample_size)
            model = DecisionTreeClassifier()
            model.fit(X_sample, Y_sample)
            self.models.append(model)  # Store trained model

    def predict(self, test_X):
        # Get predictions from each tree and perform majority voting
        model_predictions = np.array([model.predict(test_X) for model in self.models])
        vote_result = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=model_predictions)
        return vote_result

############################## part 5 : Train-test split ########################################

# Split the dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

############################## part 6 : Train and evaluate a Single Decision Tree ################

# Train a single Decision Tree
single_model = DecisionTreeClassifier()
single_model.fit(X_train, y_train)

# Make predictions using the single Decision Tree
single_pred = single_model.predict(X_test)

# Calculate accuracy of the single Decision Tree
single_acc = accuracy_score(y_test, single_pred)
print(f"Accuracy of Single Decision Tree: {single_acc:.2f}")

############################## part 7 : Train and evaluate the Random Forest model ###############

# Train a Random Forest model with 10 trees and 80% subset ratio
armanGolbidi = ForestModel(tree_count=10, subset_ratio=0.8)
armanGolbidi.fit(X_train, y_train)

# Make predictions using the Random Forest model
forest_pred = armanGolbidi.predict(X_test)

# Calculate accuracy of the Random Forest model
forest_acc = accuracy_score(y_test, forest_pred)
print(f"Accuracy of Random Forest: {forest_acc:.2f}")

############################## part 8 : Experiment with varying trees and subset ratios ##########

# Experiment with different numbers of trees and subset sizes
for trees in [5, 10, 50]:
    for ratio in [0.5, 0.8, 1.0]:
        armanGolbidi = ForestModel(tree_count=trees, subset_ratio=ratio)
        armanGolbidi.fit(X_train, y_train)
        forest_pred = armanGolbidi.predict(X_test)
        forest_acc = accuracy_score(y_test, forest_pred)
        print(f"Random Forest with {trees} trees and {int(ratio * 100)}% subset size: Accuracy = {forest_acc:.2f}")

```
### code1
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
################################# Loading the Dataset
from sklearn.datasets import load_iris
# Load dataset
data = load_iris()
X, y = data.data, data.target
################################# Bootstrapping Function
def bootstrap(X, y, n_samples):
    indices = np.random.choice(range(X.shape[0]), size=n_samples, replace=True)
    return X[indices], y[indices]
################################# RandomForest Class
class RandomForest:
    def __init__(self, n_trees=10, max_samples=0.8):
        self.n_trees = n_trees  # Number of trees in the forest
        self.max_samples = max_samples  # Percentage of data to use for bootstrapping
        self.trees = []  # Store individual decision trees
################################# Fitting the Random Forest
    def fit(self, X, y):
        n_samples = int(X.shape[0] * self.max_samples)  # Calculate number of samples for bootstrapping
        for _ in range(self.n_trees):
            X_sample, y_sample = bootstrap(X, y, n_samples)  # Get bootstrapped dataset
            tree = DecisionTreeClassifier()  # Create a new decision tree
            tree.fit(X_sample, y_sample)  # Train the decision tree
            self.trees.append(tree)  # Add the tree to the list
#################################  Making Predictions
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # Get predictions from each tree
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
        return majority_vote
#################################  Train-Test Split
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#################################  Single Decision Tree
# Single Decision Tree
single_tree = DecisionTreeClassifier()
single_tree.fit(X_train, y_train)
y_pred_single = single_tree.predict(X_test)
single_tree_acc = accuracy_score(y_test, y_pred_single)
print(f"Accuracy of Single Decision Tree: {single_tree_acc:.2f}")
#################################   Random Forest Training and Prediction
# Random Forest
rf = RandomForest(n_trees=10, max_samples=0.8)  # You can vary n_trees and max_samples
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of Random Forest: {rf_acc:.2f}")
#################################  Experiments with Different Parameters
for n_trees in [5, 10, 50]:
    for subset_size in [0.5, 0.8, 1.0]:
        rf = RandomForest(n_trees=n_trees, max_samples=subset_size)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest with {n_trees} trees and {int(subset_size*100)}% subset size: Accuracy = {rf_acc:.2f}")
```
### code2
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# Load the Wine dataset
data = load_wine()
X, y = data.data, data.target

# Function to create bootstrapped datasets
def bootstrap(X, y, n_samples):
    indices = np.random.choice(range(X.shape[0]), size=n_samples, replace=True)
    return X[indices], y[indices]

# Random Forest class
class RandomForest:
    def __init__(self, n_trees=10, max_samples=0.8):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.trees = []

    def fit(self, X, y):
        n_samples = int(X.shape[0] * self.max_samples)
        for _ in range(self.n_trees):
            X_sample, y_sample = bootstrap(X, y, n_samples)
            tree = DecisionTreeClassifier()
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
        return majority_vote

# Experimenting with the Wine dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Single Decision Tree
single_tree = DecisionTreeClassifier()
single_tree.fit(X_train, y_train)
y_pred_single = single_tree.predict(X_test)
single_tree_acc = accuracy_score(y_test, y_pred_single)
print(f"Accuracy of Single Decision Tree: {single_tree_acc:.2f}")

# Random Forest
rf = RandomForest(n_trees=10, max_samples=0.8)  # You can vary n_trees and max_samples
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of Random Forest: {rf_acc:.2f}")

# Experiment with varying number of trees and subset size
for n_trees in [5, 10, 50]:
    for subset_size in [0.5, 0.8, 1.0]:
        rf = RandomForest(n_trees=n_trees, max_samples=subset_size)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest with {n_trees} trees and {int(subset_size*100)}% subset size: Accuracy = {rf_acc:.2f}")

```
