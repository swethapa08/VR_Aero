from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
import os

# Generate dummy data
np.random.seed(42)  # Ensure reproducibility
X = np.random.rand(100, 6)  # 6 features: Heart_Rate, Eye_Tracking, etc.
y = np.random.rand(100) * 100  # Success rate (0-100)

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
os.makedirs('E:\Swetha\VirtuAero_App\models', exist_ok=True)
model_path = 'E:\Swetha\VirtuAero_App\models\efficiency_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_path}")