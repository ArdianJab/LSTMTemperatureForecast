import pandas as pd
import numpy as np


def get_nasa_giss_temp():
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    df = pd.read_csv(url, skiprows=1)
    df = df[pd.to_numeric(df['Year'], errors='coerce').notnull()]
    df['Year'] = df['Year'].astype(int)
    df_melted = df.melt(id_vars=["Year"],
                        value_vars=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                        var_name="Month", value_name="Temp")
    month_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                  "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    df_melted["Month"] = df_melted["Month"].map(month_dict)
    df_melted["Date"] = pd.to_datetime(dict(year=df_melted["Year"], month=df_melted["Month"], day=1))
    df_temp = df_melted[["Date", "Temp"]].sort_values("Date").reset_index(drop=True)
    last_date = df_temp["Date"].max()
    last_temp = df_temp[df_temp["Date"] == last_date]["Temp"].values[0]
    dates_to_add = pd.date_range(start=last_date + pd.offsets.MonthBegin(),
                                 end="2025-05-01", freq='MS')
    df_future = pd.DataFrame({"Date": dates_to_add, "Temp": last_temp})
    df_temp = pd.concat([df_temp, df_future], ignore_index=True)
    df_temp = df_temp[df_temp["Temp"] != "***"]
    return df_temp


def get_co2_data(start_year=1880):
    # Dane historyczne: interpolacja od (start_year) do 1957
    years_hist = list(range(start_year, 1958))
    co2_start = 285.0
    co2_1900 = 295.0
    co2_1957 = 320.0

    co2_values = []
    for y in years_hist:
        if y < 1900:
            val = co2_start + (co2_1900 - co2_start) * (y - start_year) / (1900 - start_year)
        else:
            val = co2_1900 + (co2_1957 - co2_1900) * (y - 1900) / (1957 - 1900)
        co2_values.append(val)

    rows_hist = []
    for y, val in zip(years_hist, co2_values):
        for m in range(1, 13):
            rows_hist.append({'Date': pd.Timestamp(y, m, 1), 'CO2': val})
    df_hist = pd.DataFrame(rows_hist)

    # Nowoczesne dane NOAA od 1958
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    df_modern = pd.read_csv(url, comment='#', delim_whitespace=True,
                            names=['year', 'month', 'decimal_date', 'average', 'deseasonalized',
                                   'fit', 'seasonal_cycle', 'unc'])
    df_modern['Date'] = pd.to_datetime(df_modern[['year', 'month']].assign(day=1))
    df_modern = df_modern[['Date', 'average']].rename(columns={'average': 'CO2'})
    df_modern = df_modern[df_modern['CO2'] > 0]

    # Przyszłość do 2025
    last_date = df_modern["Date"].max()
    last_co2 = df_modern[df_modern["Date"] == last_date]["CO2"].values[0]
    dates_to_add = pd.date_range(start=last_date + pd.offsets.MonthBegin(),
                                 end="2025-05-01", freq='MS')
    df_future = pd.DataFrame({"Date": dates_to_add, "CO2": last_co2})
    df_modern = pd.concat([df_modern, df_future], ignore_index=True)

    # Połączenie danych
    df_co2 = pd.concat([df_hist, df_modern], ignore_index=True)
    df_co2 = df_co2.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)

    return df_co2


def load_and_prepare_data(co2_lag_years=20):
    temp_df = get_nasa_giss_temp()

    # Zakładamy, że potrzebujemy danych od (1880 - lag) do 2025
    start_year_for_co2 = 1880 - co2_lag_years
    co2_df = get_co2_data(start_year=start_year_for_co2)

    # Przesuwamy daty CO₂ w przyszłość o lag, żeby dopasować do temp
    co2_df_lagged = co2_df.copy()
    co2_df_lagged["Date"] = co2_df_lagged["Date"] + pd.DateOffset(years=co2_lag_years)

    # Łączenie po dacie
    df = pd.merge(temp_df, co2_df_lagged, on="Date", how="inner")

    # Przycinamy do właściwego zakresu
    df = df[(df["Date"] >= "1880-01-01") & (df["Date"] <= "2025-05-01")].reset_index(drop=True)

    return get_averaged_data(df)

def get_co2lagged_data(co2_lag_years=20):
    # Zakładamy, że potrzebujemy danych od (1880 - lag) do 2025
    start_year_for_co2 = 1880 - co2_lag_years
    co2_df = get_co2_data(start_year=start_year_for_co2)

    # Przesuwamy daty CO₂ w przyszłość o lag, żeby dopasować do temp
    co2_df_lagged = co2_df.copy()
    co2_df_lagged["Date"] = co2_df_lagged["Date"] + pd.DateOffset(years=co2_lag_years)
    return co2_df_lagged


def get_averaged_data(df):
    """
    Dla każdego miesiąca oblicza średnią z aktualnego i 11 poprzednich miesięcy dla Temp i CO2.
    """
    df = df.copy()
    df["Temp_avg12"] = df["Temp"].rolling(window=12, min_periods=12).mean()
    df["CO2_avg12"] = df["CO2"].rolling(window=12, min_periods=12).mean()

    # Usuwamy wiersze z NaN (czyli pierwsze 11 miesięcy bez pełnego okna)
    df = df.dropna(subset=["Temp_avg12", "CO2_avg12"]).reset_index(drop=True)

    # Zostawiamy tylko potrzebne kolumny
    df = df[["Date", "Temp_avg12", "CO2_avg12"]]
    df = df.rename(columns={"Temp_avg12": "Temp", "CO2_avg12": "CO2"})

    return df

def get_averaged_temp():
    df_temp = get_nasa_giss_temp().copy()
    df_temp["Temp_avg12"] = df_temp["Temp"].rolling(window=12, min_periods=12).mean()
    df_temp = df_temp.dropna(subset=["Temp_avg12"]).reset_index(drop=True)
    df_temp = df_temp[["Date", "Temp_avg12"]].rename(columns={"Temp_avg12": "Temp"})
    return df_temp

def get_averaged_co2lagged(co2_lag_years=20):
    df_co2 = get_co2lagged_data(co2_lag_years=co2_lag_years).copy()
    df_co2["CO2_avg12"] = df_co2["CO2"].rolling(window=12, min_periods=12).mean()
    df_co2 = df_co2.dropna(subset=["CO2_avg12"]).reset_index(drop=True)
    df_co2 = df_co2[["Date", "CO2_avg12"]].rename(columns={"CO2_avg12": "CO2"})
    return df_co2

