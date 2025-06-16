import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from prepare_data import *

read_path_model = "model.keras"
read_path_scaler_temp = "scaler_temp.pkl"
read_path_scaler_co2 = "scaler_co2.pkl"


def rolling_forecast_known_co2(model, temp, co2, n_steps, scaler_temp, scaler_co2, forecast_months = 0):
    if forecast_months == 0:
        forecast_months = len(co2) - len(temp)

    history_temp = list(temp["Temp"][-n_steps:])
    predictions = []

    for i in range(forecast_months):
        temp_seq = np.array(history_temp[-n_steps:]).reshape(-1, 1)
        co2_seq = np.array(co2["CO2"][len(temp) + i - n_steps : len(temp) + i]).reshape(-1, 1)

        temp_seq = scaler_temp.transform(temp_seq)
        co2_seq = scaler_co2.transform(co2_seq)

        X_input = np.hstack((temp_seq,  co2_seq)).reshape(1, n_steps, 2)

        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        pred = scaler_temp.inverse_transform(pred_scaled.reshape(-1, 1))

        predictions.append(pred[0][0])
        history_temp.append(pred[0][0])

    predictions = np.array(predictions).reshape(-1, 1)

    return predictions


def plot_forecast(temp, forecast, start_date="1880-01"):
    dates_hist = pd.date_range(start=start_date, periods=len(temp), freq="M")
    dates_fore = pd.date_range(start=dates_hist[-1] + pd.offsets.MonthBegin(), periods=len(forecast), freq="M")
    all_dates = pd.date_range(start=start_date, end=dates_fore[-1], freq="M")

    last_temp_value = temp["Temp"].iloc[-1]
    last_temp_date = dates_hist[-1]

    co2 = get_co2lagged_data(co2_lag_years=20)
    co2["Date"] = pd.to_datetime(co2["Date"])
    co2.set_index("Date", inplace=True)

    co2_all = co2.loc[(co2.index >= all_dates[0]) & (co2.index <= all_dates[-1])].copy()

    last_forecast_value = forecast[-1][0] if forecast.ndim > 1 else forecast[-1]
    last_forecast_date = dates_fore[-1]

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Oś lewa — temperatura
    ax1.plot(dates_hist, temp["Temp"], label="Historical averaged temperature")
    ax1.plot(dates_fore, forecast, label="Temperature forecast")
    ax1.axvline(dates_fore[0], color="gray", linestyle=":", label="Forecast beginning")

    # Adnotacje
    ax1.annotate(f"Last temp: {last_temp_value:.2f}°C",
                 xy=(last_temp_date, last_temp_value),
                 xytext=(last_temp_date, last_temp_value + 0.3),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10,
                 ha='center')
    ax1.annotate(f"Last forecast: {last_forecast_value:.2f}°C",
                 xy=(last_forecast_date, last_forecast_value),
                 xytext=(last_forecast_date, last_forecast_value - 0.3),
                 arrowprops=dict(facecolor='blue', arrowstyle='->'),
                 fontsize=10,
                 ha='center')

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Temperature difference\n(12-month mean vs. 1951–1980 avg)")
    ax1.grid()

    # Druga oś — CO₂ w ppm
    ax2 = ax1.twinx()
    ax2.plot(co2_all.index, co2_all["CO2"], label="CO₂ concentration from before 20 years", linestyle="--", color="red", alpha=0.4)
    ax2.set_ylabel("CO₂ concentration [ppm]", color="red")
    ax2.tick_params(axis='y', labelcolor='red')

    # Legendy z obu osi
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    # Tytuł i wykończenie
    plt.title(f"Temperature forecast from {dates_fore[0].strftime('%Y-%m')} ({len(forecast)} months)")
    plt.tight_layout()
    plt.show()


temp = get_averaged_temp()
# temp.drop(temp.tail(12).index,inplace = True) # jeśli chcemy zacząć predykcję wcześniej to usuwamy n ostatnich miesięcy podając n w .tail(n)
co2 = get_averaged_co2lagged(co2_lag_years=20)

model = load_model(read_path_model)
scaler_temp = joblib.load(read_path_scaler_temp)
scaler_co2 = joblib.load(read_path_scaler_co2)

forecast = rolling_forecast_known_co2(
    model=model,
    temp=temp,
    co2=co2,
    n_steps=24, # takie samo jak look_back w train_model
    scaler_temp=scaler_temp,
    scaler_co2=scaler_co2,
    forecast_months=186,
)

plot_forecast(temp, forecast)
