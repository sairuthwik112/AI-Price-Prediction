import sys
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer

from pricer.items import Item
from pricer.evaluator import evaluate


# ============================================================
# Configuration
# ============================================================

LITE_MODE = True
USERNAME = "ed-donner"

DATASET = (
    f"{USERNAME}/items_lite"
    if LITE_MODE
    else f"{USERNAME}/items_full"
)


# ============================================================
# Load Dataset
# ============================================================

train, val, test = Item.from_hub(DATASET)

print(f"Training samples   : {len(train):,}")
print(f"Validation samples : {len(val):,}")
print(f"Test samples       : {len(test):,}")


# ============================================================
# Remove invalid samples
# ============================================================

train = [
    item
    for item in train
    if item.summary is not None
    and item.price is not None
]

print(f"Usable training samples: {len(train):,}")


# ============================================================
# Prepare Training Data
# ============================================================

documents = [item.summary for item in train]

prices = np.array(
    [float(item.price) for item in train],
    dtype=np.float32
)


# ============================================================
# Convert Text -> Numerical Features
# ============================================================

vectorizer = CountVectorizer(
    max_features=3000,
    stop_words="english",
    lowercase=True
)

X_train = vectorizer.fit_transform(documents)

print(f"Vocabulary size : {len(vectorizer.vocabulary_):,}")


# ============================================================
# Train XGBoost Model
# ============================================================

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...\n")

model.fit(X_train, prices)

print("Training Complete.\n")


# ============================================================
# Prediction Function
# ============================================================

def predict_price(item):
    """
    Predict the price of a single Item object.
    """

    X = vectorizer.transform([item.summary])

    prediction = model.predict(X)[0]

    return max(0.0, float(prediction))


# ============================================================
# Evaluate
# ============================================================

evaluate(predict_price, test)