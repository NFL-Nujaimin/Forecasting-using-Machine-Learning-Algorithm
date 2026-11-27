# Public Transport Ridership Forecasting

Forecasting daily public transport ridership in the Klang Valley, Malaysia, with two models: **linear regression** and a small **neural network**. Group project for a Machine Learning course at UNITEN (Group 7).

## Data

`ridership.csv` holds daily passenger counts per line from Malaysia's open data portal (data.gov.my): 1,826 days from January 2019, with Rapid KL bus, LRT Ampang, LRT Kelana Jaya, MRT Kajang, MRT Putrajaya, Monorail and ETS.

## Method

Both notebooks share the same preprocessing:

1. Drop lines with mostly missing data (Rapid Kuantan and Penang buses, KTM Komuter, Intercity, Tebrau).
2. Fill remaining gaps with the column mean.
3. Add a `total_passengers` target, the sum of all lines for each day.
4. Remove outliers with a z-score threshold of 3.
5. Scale features with min-max normalisation, then L2 normalisation.
6. Split 80/20 into train (1,444 days) and test (361 days), using the date as the input feature.

Models:

- **Linear regression** (scikit-learn).
- **Neural network** (Keras): one hidden layer of 10 ReLU units, Adam optimiser, 100 epochs.

## Results

Mean squared error on the test set (normalised target):

| Model | Test MSE |
|---|---|
| Linear regression | 0.0116 |
| Neural network | 0.0116 |

The two models score almost the same. With the date as the only input, both mostly learn the long-term trend. Adding day-of-week, holidays or lagged ridership would be the next step to capture weekly patterns.

## Files

| File | What it is |
|---|---|
| `linear_regression.ipynb` | Preprocessing and linear regression model |
| `neural_network.ipynb` | Same preprocessing and the Keras neural network |
| `ridership.csv` | Raw daily ridership data |

## Run it

```bash
pip install pandas numpy scipy scikit-learn matplotlib tensorflow jupyter
jupyter notebook linear_regression.ipynb
```

Tools: Python, pandas, NumPy, SciPy, scikit-learn, TensorFlow/Keras, Matplotlib.
