# Prognozowanie zmian klimatycznych - przewidywanie zmian temperaur na podstawie ich historycznych zmian oraz stężenia CO2

### Cel:

Celem projektu jest predykcja długoterminowych (10-15 lat) zmian średnich temperatur globalnych na podstawie danych o emisjach CO₂ z opóźnieniem czasowym (lag). Model przewiduje przyszłe wartości temperatury na podstawie danych historycznych oraz prognozowanych emisji CO₂.


### Architektura:

W projekcie wykorzystano model sekwencyjny typu LSTM (Long Short-Term Memory). Model został przeszkolony na danych czasowych, gdzie wejście stanowią średnie wartości temperatury oraz emisji CO₂ sprzed 20 lat z ostatnich 24 miesięcy. Dodatkowo model używa techniki teacher-forcing.

Dodatkowe informacje:

    Dane wejściowe: 2 kanały (temperatura i CO₂), sekwencje 24-miesięczne

    Dane wyjściowe: jedna prognozowana wartość temperatury

    Dane CO₂ są przesunięte o 20 lat do przodu względem temperatury, co umożliwia uczenie zależności z opóźnieniem.

### Dane i źródła:

Temperatura

    Źródło: NASA GISS Surface Temperature Analysis (GISTEMP v4)

        GISTEMP Team, 2025: GISS Surface Temperature Analysis (GISTEMP), version 4. NASA Goddard Institute for Space Studies. Dataset accessed 2025-06-16 at https://data.giss.nasa.gov/gistemp/
        Lenssen, N., G.A. Schmidt, M. Hendrickson, P. Jacobs, M. Menne, and R. Ruedy, 2024: A GISTEMPv4 observational uncertainty ensemble. J. Geophys. Res. Atmos., 129, no. 17, e2023JD040179, doi:10.1029/2023JD040179

    Zakres: 1880-01-01 – 2025-05-01

Dwutlenek węgla (CO₂)

    Źródła:

        1880–1957: Dane szacunkowe na podstawie rdzeni lodowych –
        NOAA/GISS – "Global Mean CO₂ Mixing Ratios (based on ice core records)"

        1958–2025: Pomiary bezpośrednie –
        NOAA ESRL (Mauna Loa Observatory) – "Monthly CO₂ Measurements (Keeling Curve)"

    Zastosowano 20-letnie opóźnienie (lag), by umożliwić przewidywanie przyszłości na podstawie wcześniejszych emisji

### Wyniki:
Wytrenowany model osiągnął następujące wyniki:
- Train RMSE: 0.042
- Test RMSE: 0.035


## Spis treści

- Wymagania
- Instalacja
- Użycie
- Trening modelu
- Predykcja
- Struktura folderów
- Autorzy
- Licencja

---

## Wymagania

- python 3.10
- Biblioteki:
  - joblib==1.5.1
  - numpy==1.24.3
  - pandas==2.3.0
  - tensorflow-cpu==2.13
  - keras==2.13.1
  - scikit-learn==1.7.0
  - matplotlib==3.10.3


Wszystkie używane biblioteki znajdują się w pliku requirements.txt, który przy pomocy poniższej komendy zainstaluje wymagane biblioteki:

pip install -r requirements.txt

## Instalacja

git clone https://github.com/ArdianJab/LSTMTemperatureForecast
cd LSTMTemperatureForecast
pip install -r requirements.txt


## Użycie
Należy uruchomić:
 - train_model.py do trenowania
 - forecast.py do predykcji
## Trening modelu

Aby wytrenować model należy uruchomić train_model

Dane użyte do treningu wzięte z:
	https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
	https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt

GISTEMP Team, 2025: GISS Surface Temperature Analysis (GISTEMP), version 4. NASA Goddard Institute for Space Studies. Dataset accessed 2025-06-16 at https://data.giss.nasa.gov/gistemp/
Lenssen, N., G.A. Schmidt, M. Hendrickson, P. Jacobs, M. Menne, and R. Ruedy, 2024: A GISTEMPv4 observational uncertainty ensemble. J. Geophys. Res. Atmos., 129, no. 17, e2023JD040179, doi:10.1029/2023JD040179

## Predykcja:
Do prognozy służy funkcja rolling_forecast_known_co2 z forecast.py
Do wizualizacji prognozy służy plot_forecast z forecast.py

## Struktura folderów

/project
├── src/                # cały kod
├── forecast/ 		# robienie prognozy
├── prepare_data/	# pobieranie i modyfikowanie danych
├── train_model/	# trenowanie modelu
├── model.keras		# zapisany wytrenowany model
├── scaler_temp.pkl	# zapisany temp scaler
├── scaler_co2.pkl	# zapisany co2 scaler
├── README.md           # plik README
├── LICENSE             # plik licencji
└── requirements.txt    # lista pakietów Python

## Autor
- Adrian Jabłoński – github.com/ArdianJabx – adrian.1.jablonski@student.uj.edu.pl

## Licencja
Ten projekt jest udostępniony na licencji MIT.

