import time
import joblib
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

from prepare_data import load_and_prepare_data

# --- Ścieżki zapisu ---

save_path_model = "model.keras"
save_path_scaler_temp = "scaler_temp.pkl"
save_path_scaler_co2 = "scaler_co2.pkl"

# --- Parametry uczenia ---

epochs = 25
batch_size = 8
teacher_forcing_ratio = 0.8

last_prediction = 0.0

# --- Parametry Early Stopping ---

best_loss = float("inf")
patience = 5
patience_counter = 0


# --- Ładujemy dane ---
df = load_and_prepare_data()
new_row = {'Date': '2025-06-01', 'Temp': 1.8, 'CO2': 450}
df.loc[len(df)] = new_row

# --- Skalowanie ---
scaler_temp = MinMaxScaler()
scaler_co2 = MinMaxScaler()
temp_scaled = scaler_temp.fit_transform(df[["Temp"]].values.astype("float32"))
co2_scaled = scaler_co2.fit_transform(df[["CO2"]].values.astype("float32"))

df.drop(df.tail(1).index,inplace = True)
temp_scaled = temp_scaled[:-1]
co2_scaled = co2_scaled[:-1]
features_scaled = np.hstack([temp_scaled, co2_scaled])


# --- Parametry ---
look_back = 24
train_end_date = pd.to_datetime("2005-01-01")
val_end_date = pd.to_datetime("2005-01-01")

# --- Indeksy podziału ---
train_idx = df[df["Date"] < train_end_date].index[-1] + 1
val_idx = df[df["Date"] < val_end_date].index[-1] + 1

# --- Podział danych ---
train = features_scaled[:train_idx]
val = features_scaled[train_idx - look_back:val_idx]  # look_back dla okna czasowego
test = features_scaled[val_idx - look_back:]

def create_dataset_multistep(dataset, look_back, forecast_horizon):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        dataX.append(dataset[i:i + look_back])
        target_sequence = dataset[i + look_back:i + look_back + forecast_horizon, 0]  # tylko Temp
        dataY.append(target_sequence)
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset_multistep(train, look_back, 1)
testX, testY = create_dataset_multistep(test, look_back, 1)

# --- Model ---

model = Sequential()
model.add(LSTM(32, activation="sigmoid", input_shape=(look_back, 2)))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])


# --- Uczenie ---

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    start_time = time.time()
    epoch_loss = []

    indices = np.arange(len(trainX))
    np.random.shuffle(indices)
    trainX_shuffled = trainX[indices]
    trainY_shuffled = trainY[indices]

    for batch_start in range(0, len(trainX), batch_size):
        batch_end = batch_start + batch_size
        batchX = trainX_shuffled[batch_start:batch_end].copy()
        batchY = trainY_shuffled[batch_start:batch_end].copy()

        for i in range(len(batchX)):
            if np.random.rand() > teacher_forcing_ratio and batch_start + i > 0:
                batchX[i, -1, 0] = last_prediction

            loss = model.train_on_batch(batchX[i][np.newaxis], batchY[i][np.newaxis])
            epoch_loss.append(loss[0])

            last_prediction = model.predict(batchX[i][np.newaxis], verbose=0)[0][0]

    avg_loss = np.mean(epoch_loss)
    print(f"Avg loss: {avg_loss:.5f}")
    print(f"Epoch time: {time.time() - start_time:.2f} sec")

    # Early stopping
    if avg_loss < best_loss - 5e-6:  # tolerancja
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

model.save(save_path_model)
joblib.dump(scaler_temp, save_path_scaler_temp)
joblib.dump(scaler_co2, save_path_scaler_co2)

# --- Predykcja ---
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict_inv = scaler_temp.inverse_transform(trainPredict)
trainY_inv = scaler_temp.inverse_transform(trainY.reshape(-1, 1))

testPredict_inv = scaler_temp.inverse_transform(testPredict)
testY_inv = scaler_temp.inverse_transform(testY.reshape(-1, 1))

# --- Ocena ---
train_rmse = math.sqrt(mean_squared_error(trainY_inv, trainPredict_inv))
test_rmse = math.sqrt(mean_squared_error(testY_inv, testPredict_inv))
print(f"Train RMSE: {train_rmse:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")

# --- Wykres predykcji ---
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Temp"], label="True Temp")

trainPredictPlot = np.full((len(features_scaled),), np.nan)
trainPredictPlot[look_back:train_idx] = trainPredict_inv.flatten()

testPredictPlot = np.full((len(features_scaled),), np.nan)
testPredictPlot[val_idx:] = testPredict_inv.flatten()

plt.plot(df["Date"], trainPredictPlot, label="Train Predict")
plt.plot(df["Date"], testPredictPlot, label="Test Predict")

plt.xlabel("Date")
plt.ylabel("Temperature difference between\nmean temperature from last 12 months and 1951-1980 average")
plt.title("LSTM Temperature Prediction with CO2 and Temp features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
