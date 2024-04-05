import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Define the Crayfish Optimizer class
class CrayfishOptimizer:
    def __init__(self, objective_function, n_variables, n_crayfish=50, max_iter=1000, alpha=0.5, beta=0.5):
        self.objective_function = objective_function
        self.n_variables = n_variables
        self.n_crayfish = n_crayfish
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta

    def optimize(self):
        crayfish_positions = np.random.rand(self.n_crayfish, self.n_variables)
        best_position = None
        best_fitness = float('inf')

        for _ in range(self.max_iter):
            fitness = self.objective_function(crayfish_positions)
            sorted_indices = np.argsort(fitness)
            if fitness[sorted_indices[0]] < best_fitness:
                best_fitness = fitness[sorted_indices[0]]
                best_position = crayfish_positions[sorted_indices[0]]

            for i in range(self.n_crayfish):
                random_crayfish_index = np.random.choice(sorted_indices[:int(self.beta * self.n_crayfish)])
                crayfish_positions[i] += self.alpha * (crayfish_positions[random_crayfish_index] - crayfish_positions[i])

        return best_position

# Load stock data
Stock = pd.read_csv('AAPL[1].csv', index_col=0)
df_Stock = Stock.rename(columns={'Close(t)': 'Close'})

# Drop unnecessary columns
df_Stock = df_Stock.drop(columns='Date_col')

# Split data into features and target
features = df_Stock.drop(columns=['Close_forcast'], axis=1)
target = df_Stock['Close_forcast']

# Split data into training, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(features, target, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42)


# Define objective function for Crayfish Optimizer
def objective_function(crayfish_positions):
    fitness = np.zeros(len(crayfish_positions))
    for i, coefficients in enumerate(crayfish_positions):
        model = LinearRegression()
        model.coef_ = coefficients
        model.fit(X_train, Y_train)
        predictions = model.predict(X_val)
        fitness[i] = mean_squared_error(Y_val, predictions)
    return fitness

# Create an instance of CrayfishOptimizer
optimizer = CrayfishOptimizer(objective_function, n_variables=X_train.shape[1])

# Optimize to find the best coefficients
best_coefficients = optimizer.optimize()

# Train linear regression model with best coefficients
model = LinearRegression()
model.coef_ = best_coefficients
model.fit(X_train, Y_train)

# Evaluate model performance
Y_train_pred = model.predict(X_train)
Y_val_pred = model.predict(X_val)
Y_test_pred = model.predict(X_test)

print("Training R-squared: ", round(metrics.r2_score(Y_train, Y_train_pred), 2))
print("Training Explained Variation: ", round(metrics.explained_variance_score(Y_train, Y_train_pred), 2))
print('Training MAPE:', round(get_mape(Y_train, Y_train_pred), 2))
print('Training Mean Squared Error:', round(metrics.mean_squared_error(Y_train, Y_train_pred), 2))
print("Training RMSE: ", round(np.sqrt(metrics.mean_squared_error(Y_train, Y_train_pred)), 2))
print("Training MAE: ", round(metrics.mean_absolute_error(Y_train, Y_train_pred), 2))
print(' ')

print("Validation R-squared: ", round(metrics.r2_score(Y_val, Y_val_pred), 2))
print("Validation Explained Variation: ", round(metrics.explained_variance_score(Y_val, Y_val_pred), 2))
print('Validation MAPE:', round(get_mape(Y_val, Y_val_pred), 2))
print('Validation Mean Squared Error:', round(metrics.mean_squared_error(Y_train, Y_train_pred), 2))
print("Validation RMSE: ", round(np.sqrt(metrics.mean_squared_error(Y_val, Y_val_pred)), 2))
print("Validation MAE: ", round(metrics.mean_absolute_error(Y_val, Y_val_pred), 2))
print(' ')

print("Test R-squared: ", round(metrics.r2_score(Y_test, Y_test_pred), 2))
print("Test Explained Variation: ", round(metrics.explained_variance_score(Y_test, Y_test_pred), 2))
print('Test MAPE:', round(get_mape(Y_test, Y_test_pred), 2))
print('Test Mean Squared Error:', round(metrics.mean_squared_error(Y_test, Y_test_pred), 2))
print("Test RMSE: ", round(np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred)), 2))
print("Test MAE: ", round(metrics.mean_absolute_error(Y_test, Y_test_pred), 2))

