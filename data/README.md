# Dhaka CAMS Hourly Air Quality Dataset (2012–2026)

This directory contains the continuous hourly ground-level air quality and meteorological dataset recorded by the Continuous Air Monitoring Stations (CAMS) across Dhaka, Bangladesh, from **January 1, 2012 to early 2026**.

---

## 1. Dataset Overview

- **File Name**: `dhaka_cams_hourly_2012_2026.csv`
- **File Size**: ~37.9 MB (37,902,943 bytes)
- **Total Rows**: 389,448 hourly measurements across 3 monitoring stations
- **Temporal Resolution**: 1 hour
- **Timezone**: Bangladesh Standard Time (BST, UTC+6)
- **Primary Target**: `pm2_5_ug_m3` (Particulate Matter $\le 2.5\ \mu\text{m}$)

---

## 2. Monitoring Stations

The dataset covers the three official Continuous Air Monitoring Stations (CAMS) operated by the **Department of Environment (DoE)** under the Ministry of Environment, Forest and Climate Change, Government of Bangladesh (originally supported by the World Bank CASE Project):

| Station ID | Station Name | Location | Coordinates | Environment Type |
| :--- | :--- | :--- | :--- | :--- |
| **CAMS-1** | Paribesh Bhaban | Agargaon, Dhaka | 23.777° N, 90.380° E | Urban background / Institutional |
| **CAMS-2** | Farmgate (BARC) | Farmgate, Dhaka | 23.754° N, 90.389° E | High-density traffic / Commercial corridor |
| **CAMS-3** | Darussalam (Mirpur) | Technical Mor, Mirpur | 23.788° N, 90.353° E | Mixed commercial-industrial / Transit corridor |

---

## 3. Schema & Column Descriptions

| Column | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns]` | `YYYY-MM-DD HH:MM` | Hourly measurement timestamp (local time) |
| `station_id` | `string` | Categorical | Unique station code (`CAMS-1`, `CAMS-2`, `CAMS-3`) |
| `station_name` | `string` | Text | Formal full station name and location |
| `pm2_5_ug_m3` | `float64` | $\mu\text{g}/\text{m}^3$ | Fine particulate matter ($\le 2.5\ \mu\text{m}$) |
| `pm10_ug_m3` | `float64` | $\mu\text{g}/\text{m}^3$ | Coarse particulate matter ($\le 10\ \mu\text{m}$) |
| `so2_ppb` | `float64` | $\text{ppb}$ | Sulfur dioxide |
| `no2_ppb` | `float64` | $\text{ppb}$ | Nitrogen dioxide |
| `no_ppb` | `float64` | $\text{ppb}$ | Nitric oxide |
| `nox_ppb` | `float64` | $\text{ppb}$ | Total oxides of nitrogen ($\text{NO} + \text{NO}_2$) |
| `co_ppm` | `float64` | $\text{ppm}$ | Carbon monoxide |
| `o3_ppb` | `float64` | $\text{ppb}$ | Ground-level ozone |
| `temp_c` | `float64` | $^\circ\text{C}$ | Ambient dry-bulb air temperature |
| `rh_pct` | `float64` | $\%$ | Relative humidity |
| `wind_speed_ms` | `float64` | $\text{m}/\text{s}$ | Surface horizontal wind speed |
| `wind_dir_deg` | `float64` | $\text{degrees}$ | Wind direction (0°–360°, clockwise from North) |
| `barometric_pressure_hpa` | `float64` | $\text{hPa}$ | Atmospheric surface pressure |
| `solar_rad_wm2` | `float64` | $\text{W}/\text{m}^2$ | Solar downward global radiation |
| `rain_mm` | `float64` | $\text{mm}$ | Hourly accumulated rainfall |
| `vertical_wind_speed_ms` | `float64` | $\text{m}/\text{s}$ | Vertical component of wind (sparse, ~88% NaN) |

---

## 4. Temporal Coverage & Station Completeness

Hourly data coverage varies across years and stations due to sensor servicing, power disruptions, and equipment upgrades:

```
Percentage of valid PM2.5 hours per station and year:
Year       CAMS-1       CAMS-2       CAMS-3
---------------------------------------------
2012        13.6%        11.0%        15.4%   <-- Excluded from model training (<15%)
2013        49.1%        45.5%        90.3%
2014        82.5%        29.0%        89.0%
2015        67.7%        89.2%        91.6%
2016        39.7%        40.0%        67.2%
2017        64.0%        56.8%        87.1%
2018         2.6%        81.1%        77.7%   <-- CAMS-1 sensor outage
2019        37.1%        89.9%        65.4%
2020         0.0%        84.5%        86.1%   <-- CAMS-1 offline during pandemic
2021        76.5%        87.2%        89.0%
2022        91.2%        90.4%        91.3%
2023        87.4%        90.1%        88.9%   <-- Validation Split
2024        86.6%        90.1%        78.1%   <-- Test Benchmark Split
2025–2026   Ongoing / recent observation feeds
```

### Data Partitioning Strategy
- **Training Set**: 2013-01-01 to 2022-12-31 (161,740 valid rows)
- **Validation Set**: 2023-01-01 to 2023-12-31 (22,471 valid rows, used for early stopping and threshold tuning)
- **Test Set**: 2024-01-01 to 2024-12-31 (22,315 strictly held-out rows for final benchmark evaluation)

---

## 5. Quality Assurance and Data Cleaning

1. **Negative Pollutant Removal**: Sensor calibration drift occasionally produces negative pollutant values ($< 0$). These are strictly mapped to `NaN`.
2. **Timeline Regularization**: Every station is re-indexed onto a strict 1-hour interval frequency (`freq="1h"`).
3. **Short-gap Linear Interpolation**: Minor instrument dropouts $\le 3$ hours are imputed using linear time interpolation. All imputed values are marked via boolean column `is_imputed`. Longer gaps remain `NaN` to preserve autoregressive integrity.
4. **Physical Inversion Check**: Atmospheric consistency dictates that fine particulate matter cannot exceed coarse particulate matter ($PM_{2.5} \le PM_{10} + \epsilon$). Records violating this physical constraint are isolated for sensor diagnosis.
5. **Multi-Pollutant Anomaly Filtering**: An Isolation Forest algorithm flags multivariate outliers (such as stuck voltage plateaus and unphysical concentration spikes).

---

## 6. How to Obtain or Mirror the Data

- **In Repository**: The 37.9 MB dataset is contained directly in this directory:
  ```bash
  git clone https://github.com/Hemal0651/dhaka-pm25-forecasting.git
  cd dhaka-pm25-forecasting/data
  ls -lh dhaka_cams_hourly_2012_2026.csv
  ```
- **Zenodo / Google Drive Mirror**: For cloud computing environments (e.g. Google Colab, Kaggle, AWS SageMaker), mirrors and checksums can be referenced:
  - **SHA-256 Checksum**: Can be verified locally using `certutil -hashfile dhaka_cams_hourly_2012_2026.csv SHA256` or `sha256sum dhaka_cams_hourly_2012_2026.csv`.
