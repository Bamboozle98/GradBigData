import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Step 1: Load and clean data ===
df = pd.read_csv(r"C:\Users\cbran\PycharmProjects\GradBigData\Data\mlb_batters_with_salary_2024.csv")

# Drop unnecessary columns
drop_cols = ['Player Name', 'Team', 'Position', 'Total Value', 'Average Annual', 'Normalized Name', 'Years']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Ensure salary is numeric and drop rows without it
df = df[df['Salary'].notna()]
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
df = df.dropna(subset=['Salary'])

# Drop rows with excessive missing values, then fill the rest
df = df.dropna(axis=0, thresh=int(df.shape[1] * 0.7))
df = df.fillna(df.mean(numeric_only=True))

# === Step 2: Prepare features and log-transformed target ===
X = df.drop(columns=['Salary'])
y_log = np.log2(df['Salary'].values + 1)  # log2 transform

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

# === Step 3: Train Linear Regression ===
lr = LinearRegression()
lr.fit(X_train, y_train_log)
y_pred_log = lr.predict(X_test)

# === Step 4: Inverse transform and evaluate ===
y_pred = 2**y_pred_log - 1
y_true = 2**y_test_log - 1

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Save predictions
pred_df = pd.DataFrame({
    "Actual Salary": y_true,
    "Predicted Salary": y_pred
})
pred_df.to_csv(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\Results\predictions_linear_regression.csv", index=False)
print("✅ Predictions saved to predictions_linear_regression.csv")

# Save evaluation
summary_df = pd.DataFrame([{
    "Model": "Linear Regression (Log2 Salary)",
    "R2": r2,
    "MAE": mae,
    "RMSE": rmse
}])
summary_df.to_csv(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\Results\linear_regression_summary.csv", index=False)
print("📊 Summary saved to linear_regression_summary.csv")

import matplotlib.pyplot as plt

# === Step 5: Feature Coefficients Plot ===
# Get coefficient magnitudes and feature names
coefs = lr.coef_
feature_names = X.columns

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefs
}).sort_values(by='Coefficient', key=abs, ascending=False)

# Save to CSV
coef_df.to_csv(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\Results\lr_coefficients.csv", index=False)

# Plot
plt.figure(figsize=(12, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='darkorange')
plt.axvline(0, color='black', linewidth=0.8)
plt.gca().invert_yaxis()
plt.title("Linear Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()

# Save the figure
plt.savefig(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\Results\lr_coefficients_plot.png", dpi=300)
print("📊 Coefficient plot saved as lr_coefficients_plot.png")
