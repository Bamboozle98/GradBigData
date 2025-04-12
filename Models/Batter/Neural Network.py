import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

# === Step 1: Load and clean the data ===
df = pd.read_csv(r"C:\Users\cbran\PycharmProjects\GradBigData\Data\mlb_batters_with_salary_2024.csv")

drop_cols = ['Player Name', 'Team', 'Position', 'Total Value', 'Average Annual', 'Normalized Name', 'Years']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
df = df[df['Salary'].notna()]
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
df = df.dropna(subset=['Salary'])
df = df.dropna(axis=0, thresh=int(df.shape[1] * 0.7))
df = df.fillna(df.mean(numeric_only=True))

# === Step 2: Feature Importance Scoring (Random Forest) ===
X_full = df.drop(columns=['Salary'])
y = np.log1p(df['Salary'].values)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_full, y)

feature_scores = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Save feature importances
feature_scores.to_csv(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\Results\feature_importance_rf.csv", index=False)

# Use all features instead of top 20
X = X_full

# === Step 3: Scale and Split ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Step 4: TensorBoard & Callbacks ===
log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# === Step 5: Define and compile the improved model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(1024),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Dense(64),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mae']
)

# === Step 6: Train the model ===
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=150,
    batch_size=16,
    callbacks=[tensorboard_callback, lr_schedule],
    verbose=1
)

# === Step 7: Evaluate and save ===
y_pred_log = model.predict(X_test).flatten()

y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

pred_df = pd.DataFrame({
    "Actual Salary": y_true,
    "Predicted Salary": y_pred
})
pred_df.to_csv(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\Results\predictions_neural_network.csv", index=False)
print("✅ Saved predictions to predictions_neural_network.csv")

summary_df = pd.DataFrame([{
    "Model": "Improved Neural Network (All Features + LR Scheduler)",
    "R2": r2,
    "MAE": mae,
    "RMSE": rmse
}])
summary_df.to_csv(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\Results\neural_network_summary.csv", index=False)
print("📊 Saved summary to neural_network_summary.csv")

model.save(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\ModelSaves\nn_batter_model.keras")
print("💾 Saved model to nn_batter_model.keras")

# === Step 9: NN Feature Influence (First-Layer Weights) ===
import matplotlib.pyplot as plt

first_layer_weights = model.layers[0].get_weights()[0]
raw_scores = np.sum(np.abs(first_layer_weights), axis=1)
normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

score_df = pd.DataFrame({
    'Feature': X.columns,
    'Raw Score': raw_scores,
    'Normalized Score': normalized_scores
}).sort_values(by='Normalized Score', ascending=False)

score_df.to_csv(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\Results\nn_feature_weight_scores.csv", index=False)

plt.figure(figsize=(12, 6))
plt.barh(score_df['Feature'], score_df['Normalized Score'], color='slateblue')
plt.gca().invert_yaxis()
plt.title("NN Feature Influence (Normalized First-Layer Weights)")
plt.xlabel("Normalized Score (0-1)")
plt.tight_layout()
plt.savefig(r"C:\Users\cbran\PycharmProjects\GradBigData\Models\Batter\Results\nn_feature_weight_scores.png", dpi=300)
print("📊 NN feature scores plot saved as nn_feature_weight_scores.png")

# === Step 10: TensorBoard info ===
print(f"📈 View TensorBoard with:\ntensorboard --logdir={log_dir}")
