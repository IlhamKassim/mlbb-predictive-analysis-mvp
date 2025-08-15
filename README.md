# Mobile Legends Predictive Analysis MVP

This repository contains a **Minimum Viable Product (MVP)** for
predictive sports analysis of **Mobile Legends: Bang Bang** matches.
It demonstrates how historical match data can be used to build
machine‑learning models that predict match outcomes, estimate player
performance and recommend draft strategies.

## Project Structure

```
mlbb_mvp/
├── app.py                 # Streamlit web application
├── data_preprocessing.py  # Data loading and preprocessing utilities
├── model_training.py      # Functions for training and saving models
├── recommendation_system.py # Simple heuristic draft recommendations
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── models/                # Saved classifier and regression models (generated)
```

You will need to supply a CSV dataset containing historical Mobile
Legends matches (see below) and train the models before running the
application.

## Data

The MVP expects a dataset similar to the Kaggle dataset
“**Mobile Legends M5 World Championship Knockout Stage Results**”.  The
CSV should contain at least the following columns:

- **win**: Binary indicator (1 for a win, 0 for a loss).
- **hero** or **pick** columns: The heroes selected by each team/player.
- **ban** columns: Heroes banned in the draft (optional but recommended).
- **kills**, **deaths**, **assists**, **gold**, **time**: Player and
  match statistics used to derive KDA and other metrics.

Place your dataset at `data/mlbb_data.csv` before running the app.  If
no dataset is found, the app will still launch but hero selection and
recommendation features will be disabled.

## Installation

1. Clone this repository or download its contents.
2. Navigate to the project root and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset CSV into the `data/` directory (create the
   directory if it doesn’t exist) and name it `mlbb_data.csv`.

## Training the Models

Use the Python API provided in `data_preprocessing.py` and
`model_training.py` to train a classifier and optional regression
models.  A simple example script might look like this:

```python
import pandas as pd
from data_preprocessing import load_dataset, preprocess_dataframe, split_features_labels
from model_training import train_classification_model, train_regression_models, save_models

# Load and preprocess
df = load_dataset('data/mlbb_data.csv')
df_processed = preprocess_dataframe(df)
X, y = split_features_labels(df_processed, label_column='win')

# Train classifier
clf = train_classification_model(X, y)

# Train regression models (e.g. kills, assists, gold)
regressors = train_regression_models(X, df)

# Save models to disk
save_models(clf, regressors, directory='models')

print('Training complete.')
```

This will create a directory `models/` containing the trained
classifier (`classifier.joblib`) and regression models (`regressor_kills.joblib`,
etc.).

## Running the App

Once the models are trained and saved, launch the Streamlit app:

```bash
streamlit run mlbb_mvp/app.py
```

The app provides a user interface where you can select heroes for your
team, specify banned heroes and view the predicted win probability
together with simple draft recommendations.

## Limitations and Future Work

- **Data fidelity**:  The MVP relies on a static, historical dataset.
  Integrating live data feeds would enable real‑time analysis.
- **Feature mapping**:  The mapping from user input to model features
  is heuristic.  Future versions should save and reuse the exact
  encoders used during training.
- **Recommendation logic**:  The draft recommendations are based
  solely on overall win rates.  Incorporating synergy and counter
  relationships between heroes would yield more nuanced advice.
- **Player‑level granularity**:  At present, performance predictions
  operate at the match level.  With richer data, individual player
  predictions could be improved.

Contributions and suggestions for improvement are welcome!
