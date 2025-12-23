# code/data_cleaning.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler  # fixed import
from sklearn.preprocessing import OneHotEncoder

def clean_data():
    # read dataset (relative path)
    df = pd.read_csv('code/hidden_hunger.csv')

    target = "Hidden_Hunger_Flag"
    categorical = ["Gender", "Income_Bracket", "Education_Level"]
    # numeric = all other columns except categorical + target
    numeric = [c for c in df.columns if c not in categorical + [target]]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )

    print("Loaded dataset with shape:", df.shape)
    return df, preprocess
