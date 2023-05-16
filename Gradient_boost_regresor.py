import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators = []

    def fit(self, X, y):
        self.estimators = []
        residuals = y.copy()

        for _ in range(self.n_estimators):
            estimator = DecisionTreeRegressor(max_depth=self.max_depth)
            estimator.fit(X, residuals)
            self.estimators.append(estimator)

            # Update residuals
            predictions = estimator.predict(X)
            residuals = y - np.sum([self.learning_rate * est.predict(X) for est in self.estimators], axis=0)

    def predict(self, X):
        return np.sum([self.learning_rate * est.predict(X) for est in self.estimators], axis=0)






# Load the wine quality dataset
wine_quality_df = pd.read_csv("wine-quality-white-and-red.csv", delimiter=',')
wine_quality_df = wine_quality_df.drop(['type'], axis=1)
wine_quality_df['quality'] = wine_quality_df['quality'].astype(float)

X_wine_quality = wine_quality_df.drop(['quality'], axis=1).values
y_wine_quality = wine_quality_df["quality"].values

# Split the wine quality dataset into training, validation, and test sets
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine_quality, y_wine_quality, test_size=0.2, random_state=42)
X_train_wine, X_val_wine, y_train_wine, y_val_wine = train_test_split(X_train_wine, y_train_wine, test_size=0.25, random_state=42)

# Build and train the gradient boosting regressor
model = GradientBoostingRegressor()
model.fit(X_train_wine, y_train_wine)

# Make predictions on the test set
y_pred_test = model.predict(X_test_wine)

# Calculate MSE on the test set
mse_test = mean_squared_error(y_test_wine, y_pred_test)
print("MSE on test set:", mse_test)

# Visualize the predicted and actual values
plt.scatter(y_test_wine, y_pred_test, alpha=0.5)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Predicted vs Actual Quality on Test Set")
plt.show()



