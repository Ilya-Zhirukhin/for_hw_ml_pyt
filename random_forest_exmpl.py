
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import pandas as pd

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForest():

    def __init__(self, nb_trees, max_depth, min_samples_split, max_features):
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def train(self, inputs, targets):
        for _ in range(self.nb_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            subset_inputs, subset_targets = self.get_subset(inputs, targets)
            tree.fit(subset_inputs, subset_targets)
            self.trees.append(tree)

    def get_subset(self, inputs, targets):
        indices = np.random.choice(len(inputs), size=int(0.8*len(inputs)), replace=True)
        subset_inputs = inputs[indices]
        subset_targets = targets[indices]
        return subset_inputs, subset_targets

    def get_prediction(self, inputs):
        predictions = []
        for tree in self.trees:
            prediction = tree.predict_proba(inputs)
            predictions.append(prediction)
        return np.mean(predictions, axis=0)

# Load the dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Split the dataset
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.25, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Perform Random Search and train RandomForest models
nb_models = 30
results = []

for _ in range(nb_models):
    nb_trees = np.random.randint(10, 200)
    max_depth = np.random.randint(1, 30)
    min_samples_split = np.random.randint(2, 20)
    max_features = np.random.randint(1, X_train.shape[1] + 1)

    model = RandomForest(nb_trees, max_depth, min_samples_split, max_features)
    model.train(X_train, y_train)

    y_valid_pred = model.get_prediction(X_valid).argmax(axis=1)
    y_test_pred = model.get_prediction(X_test).argmax(axis=1)

    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    results.append((valid_accuracy, test_accuracy, nb_trees, max_depth, min_samples_split, max_features))

# Sort results by validation accuracy
results.sort(key=lambda x: x[0], reverse=True)

# Plot the top 10 models
top_results = results[:10]
df = pd.DataFrame(top_results,
                  columns=['Valid Accuracy', 'Test Accuracy', 'nb_trees', 'max_depth', 'min_samples_split', 'max_features'])
fig = px.scatter(df, x=df.index, y='Valid Accuracy', hover_data=['Test Accuracy', 'nb_trees', 'max_depth', 'min_samples_split', 'max_features'],
title='Top 10 RandomForest Models')
fig.show()



# Generate the confusion matrix
best_model = RandomForest(*results[0][2:])
best_model.train(X_train, y_train)
y_test_pred = best_model.get_prediction(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_test_pred)

print(cm)

# Plot the confusion matrix using Plotly
import plotly.figure_factory as ff

target_names = digits.target_names.tolist()
fig = ff.create_annotated_heatmap(cm, x=target_names, y=target_names, colorscale='Viridis', showscale=True)
fig.update_layout(title='Confusion Matrix for the Best Model', xaxis_title='Predicted', yaxis_title='True')

fig.show()