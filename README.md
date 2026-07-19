# NYC MTA Ridership Analysis (2020–2025)

> **Question:** As an NJ commuter, which NYC transit mode is the least crowded — and can we predict it from calendar features alone?

Using 5 years of daily MTA ridership data (~1,776 observations), this project analyses crowdedness trends across all major transit modes and builds Random Forest classification models to identify the least-crowded option by day, season, and holiday status.

---

## Dataset

**Source:** [MTA Daily Ridership Data (NYC Open Data)](https://data.ny.gov/Transportation/MTA-Daily-Ridership-Data-Beginning-2020/vxuj-8kew)  
**Period:** March 2020 – 2025  
**Features:** Daily ridership/traffic totals + % of comparable pre-pandemic day, per mode

Modes covered: Subway, Bus, LIRR, Metro-North, Access-A-Ride, Bridges & Tunnels, Staten Island Railway (SIR)

---

## Key Findings

### Holiday Effects
Holidays suppress ridership significantly, but the impact varies sharply by mode:

| Mode | Drop on Holidays |
|---|---|
| Staten Island Railway | 59.5% |
| Metro-North | 51.1% |
| Access-A-Ride | 48.9% |
| Buses | 46.5% |
| Subways | 44.2% |
| LIRR | 44.1% |
| Bridges & Tunnels | 17.5% |

Car traffic is the least affected — drivers are far less responsive to public holidays than public transit users.

### Bridge & Tunnel Traffic Prediction
Five regression models were compared for predicting car traffic from other transit ridership:

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 72,096 | 0.724 |
| Ridge | 72,096 | 0.724 |
| Lasso | 72,096 | 0.724 |
| Decision Tree | 57,324 | 0.825 |
| **Random Forest** | **44,832** | **0.893** |

Random Forest explains 89% of variance in bridge traffic from transit ridership alone — suggesting a strong inverse relationship between public transit usage and car traffic.

### Seasonal Effects
Season alone is a poor standalone predictor — the seasonal linear regression produced negative R² on test data for most modes (e.g. Subway test R² = -1.27). Ridership variance is dominated by the pandemic recovery period rather than seasonal patterns, which limits how much season-only models can generalise.

### Least-Crowded Mode Classification

| Model | Modes | Accuracy |
|---|---|---|
| Inner-City Random Forest | Subway, Bus, Access-A-Ride, SIR | 65% |
| Outer-City Random Forest | Metro-North, LIRR, Bridges/Tunnels | 78% |

**SIR is the least crowded inner-city mode on ~90% of days.** Metro-North is the least crowded outer-city option on ~86% of days. However, accuracy figures are inflated by class imbalance — SIR dominates inner-city labels (1,610 of 1,776 days) and Metro-North dominates outer-city (1,535 days), so the models largely learn to predict the majority class.

---

## Project Structure

```
├── NYC_ridership.ipynb               # Main analysis notebook
├── MTA_Daily_Ridership_Data.csv      # Dataset
└── README.md
```

---

## Notebook Structure

1. Imports & Setup
2. Load Data
3. Data Cleaning & Feature Engineering
4. Long-Term Ridership Trends (30-day rolling averages)
5. Crowdedness by Day of Week
6. Holiday vs. Non-Holiday Ridership
7. Baseline Model: Predicting Subway Ridership from Day & Holiday
8. Predicting Bridge & Tunnel Traffic from Transit Ridership
9. Seasonal Effects (Linear Regression)
10. Predicting the Least-Crowded Mode (Random Forest)
11. Model Visualisations
12. Conclusions

---

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn scipy holidays
```

Then open `NYC_ridership.ipynb` and run all cells top to bottom.

---

## Limitations

- Calendar features alone cannot capture weather, service disruptions, major events, or school schedules
- Class imbalance means the models default toward the dominant mode rather than learning meaningful distinctions
- Retraining on post-2022 data only would remove the pandemic distortion that currently dominates variance across all models
- Pre-pandemic baselines used for normalisation may not reflect current system capacity

---

## Technologies

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `SciPy` · `holidays`
