import os
import json
import sqlite3
import pandas as pd
import numpy as np
import pvlib
import xgboost as xgb
import requests
import time

# --- 1. CONFIGURACJA LOKALIZACJI I ŚCIEŻEK SYSTEMOWYCH HAOS ---
LAT, LON = 50.95444, 15.5794
SURFACE_TILT = 39
SURFACE_AZIMUTH = 224

BASE_DIR = "/config/python_scripts"
MODEL_PATH = os.path.join(BASE_DIR, "best_pv_model.json")
DB_PATH = os.path.join(BASE_DIR, "pv_predictions.db")
OUTPUT_JSON = "/config/www/prognoza_pv_7dni.json"

print("[PRODUKCJA] Inicjalizacja skryptu predykcyjnego AI PV na 7 dni (wliczając dzisiaj)...")

# --- 2. ŁADOWANIE MODELU ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Brak pliku modelu JSON w katalogu: {BASE_DIR}!")

model = xgb.Booster()
model.load_model(MODEL_PATH)

XGB_EXPECTED_FEATURES = model.feature_names
if XGB_EXPECTED_FEATURES is None:
    XGB_EXPECTED_FEATURES = [
        'GTI', 'temperature_2m', 'wind_speed_10m', 'snow_depth', 
        'cloud_cover', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'GTI_rolling_mean'
    ]

# --- 3. NATYWNE POBIERANIE PROGNOZY POGODY ---
print("[API] Pobieranie danych meteorologicznych od dnia bieżącego...")
url = "https://api.open-meteo.com/v1/forecast"

hourly_params = ",".join([
    "temperature_2m", "snow_depth", "cloud_cover", "wind_speed_10m", 
    "shortwave_radiation", "diffuse_radiation", "direct_radiation"
])

params = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": hourly_params,
    "timezone": "Europe/Warsaw",
    "past_days": 0,      # ZMIANA: Zaczynamy dokładnie od dzisiaj, bez dni historycznych
    "forecast_days": 7   # Pobieramy 7 dni wprzód (dzisiaj + 6 kolejnych dni)
}

headers = {
    "Accept": "application/json"
}

MAX_RETRIES = 5
BACKOFF_FACTOR = 3
response = None

for attempt in range(MAX_RETRIES):
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        break
    except requests.exceptions.HTTPError as e:
        print(f"[API ERROR] Serwer odrzucił zapytanie (HTTP {response.status_code}).")
        raise RuntimeError(f"Błąd parametrów API: {response.text}") from e
    except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
        wait_time = BACKOFF_FACTOR * (attempt + 1)
        print(f"[API WARN] Próba {attempt + 1}/{MAX_RETRIES} nieudana. Ponowna próba za {wait_time}s...")
        if attempt == MAX_RETRIES - 1:
            raise RuntimeError("Serwer Open-Meteo nie odpowiedział.") from e
        time.sleep(wait_time)

try:
    data = response.json()
except ValueError as e:
    raise ValueError(f"Serwer zwrócił nieoczekiwaną treść zamiast JSON: {response.text[:200]}") from e

hourly_raw = data["hourly"]
time_index = pd.to_datetime(hourly_raw["time"]).tz_localize('Europe/Warsaw', ambiguous='infer').tz_localize(None)

weather_data = {"dt": time_index}
mapping = {
    "temperature_2m": "temperature_2m",
    "snow_depth": "snow_depth",
    "cloud_cover": "cloud_cover",
    "wind_speed_10m": "wind_speed_10m",
    "shortwave_radiation": "shortwave_radiation",
    "diffuse_radiation": "diffuse_radiation",
    "direct_radiation": "direct_normal_irradiance"
}

for api_name, df_name in mapping.items():
    weather_data[df_name] = np.array(hourly_raw[api_name], dtype=np.float32)

df_weather = pd.DataFrame(data=weather_data).set_index("dt")

# --- 4. WEKTOROWA INŻYNIERIA CECH ---
print("[FEATURES] Generowanie cech geometrycznych i cyklicznych...")
lokalny_indeks_tz = df_weather.index.tz_localize('Europe/Warsaw', ambiguous='infer')
sol_pos = pvlib.solarposition.get_solarposition(time=lokalny_indeks_tz, latitude=LAT, longitude=LON)

gti_calc = pvlib.irradiance.get_total_irradiance(
    surface_tilt=SURFACE_TILT, surface_azimuth=SURFACE_AZIMUTH,
    dni=df_weather["direct_normal_irradiance"], ghi=df_weather["shortwave_radiation"], dhi=df_weather["diffuse_radiation"],
    solar_zenith=sol_pos["apparent_zenith"].tz_localize(None), solar_azimuth=sol_pos["azimuth"].tz_localize(None)
)
df_weather["GTI"] = gti_calc["poa_global"].fillna(0)

df_weather['hour'] = df_weather.index.hour
df_weather['day_of_year'] = df_weather.index.dayofyear
df_weather['hour_sin'] = np.sin(2 * np.pi * df_weather['hour'] / 24)
df_weather['hour_cos'] = np.cos(2 * np.pi * df_weather['hour'] / 24)
df_weather['day_sin'] = np.sin(2 * np.pi * df_weather['day_of_year'] / 365.25)
df_weather['day_cos'] = np.cos(2 * np.pi * df_weather['day_of_year'] / 365.25)
df_weather['GTI_rolling_mean'] = df_weather['GTI'].rolling(window=3, center=False, min_periods=1).mean()
df_weather['GTI_lag1'] = df_weather['GTI'].shift(1).fillna(0)
df_weather['GTI_lag2'] = df_weather['GTI'].shift(2).fillna(0)

# --- 5. PREDYKCJA DLA 7 DNI (WŁĄCZAJĄC DZISIAJ) ---
print("[PREDICT] Wyliczanie generacji na 7 dni od dnia dzisiejszego...")

dzisiaj = pd.Timestamp.now().normalize()
dni_prognozy = [dzisiaj + pd.Timedelta(days=i) for i in range(7)]

output_days_list = []
db_all_rows = []

# Słownik do ładnego nazywania dni w pliku JSON
nazwy_dni = ["Dzisiaj", "Jutro", "Pojutrze", "Za 3 dni", "Za 4 dni", "Za 5 dni", "Za 6 dni"]

for i, dzien in enumerate(dni_prognozy):
    str_dzien = dzien.strftime('%Y-%m-%d')
    if str_dzien not in df_weather.index.strftime('%Y-%m-%d'):
        continue
        
    df_target_day = df_weather.loc[str_dzien].copy()
    
    X_prod = df_target_day[XGB_EXPECTED_FEATURES].astype(np.float32)
    dprod = xgb.DMatrix(X_prod.values, feature_names=XGB_EXPECTED_FEATURES)
    raw_preds = model.predict(dprod)
    
    final_preds = np.where(df_target_day['GTI'] < 5, 0.0, raw_preds)
    final_preds = np.clip(final_preds, 0.0, None)
    df_target_day['predicted_generation_kW'] = np.round(final_preds, 3)
    
    # 1. Dokładne sumowanie i zaokrąglanie w Pythonie
    suma_dobowa = round(float(df_target_day['predicted_generation_kW'].sum()), 2)
    
    # 2. Tworzymy gotowy, czysty tekst, który bezpośrednio wyświetlimy jako stan w HA
    tekst_stanu = f"{suma_dobowa} kWh"
    
    day_data = {
        "indeks": i,
        "etykieta": f"{i+1}. {nazwy_dni[i]}",
        "data": str_dzien,
        "suma_dobowa_kWh": suma_dobowa,
        "stan_wyswietlacza": tekst_stanu,  # <--- GOTOWY TEKST DLA NAGŁÓWKA CARD!
        "prognoza_godzinowa": {str(ts.hour): float(val) for ts, val in zip(df_target_day.index, df_target_day['predicted_generation_kW'])}
    }
    output_days_list.append(day_data)
    
    for idx, val in zip(df_target_day.index, df_target_day['predicted_generation_kW']):
        db_all_rows.append((idx.strftime('%Y-%m-%d %H:%M:%S'), float(val)))

# --- 6. EKSPORT ZBIORCZY DO JSON ---
output_dict = {
    "timestamp_obliczen": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    "prognoza_7_dni": output_days_list
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_dict, f, indent=4, ensure_ascii=False)

# --- 7. ZAPIS DO BAZY SQLITE ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS pv_forecasts (timestamp TEXT PRIMARY KEY, predicted_kW REAL)')
cursor.executemany('INSERT OR REPLACE INTO pv_forecasts (timestamp, predicted_kW) VALUES (?, ?)', db_all_rows)
conn.commit()
conn.close()

print(f"[KONIEC] Prognoza wygenerowana pomyślnie. Zapisano plik: {OUTPUT_JSON}")
