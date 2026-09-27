"""Train the business entity matching model."""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


FEATURES = [
    "name_exact",
    "name_similarity",
    "address_exact",
    "address_similarity",
    "country_same",
    "name_token_overlap",
    "address_token_overlap",
    "name_length_similarity",
    "address_length_similarity",
]


def train_model(train_path, val_path, model_path):
    """Train Logistic Regression and save the fitted model."""

    train = pd.read_csv(train_path, sep="\t")
    val = pd.read_csv(val_path, sep="\t")

    X_train = train[FEATURES]
    y_train = train["label"]

    X_val = val[FEATURES]
    y_val = val["label"]

    print("=" * 60)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 60)

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_probability = model.predict_proba(X_val)[:, 1]
    val_prediction = (val_probability >= 0.5).astype(int)

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            val_prediction,
            digits=4,
        )
    )

    print("ROC-AUC:")
    print(roc_auc_score(y_val, val_probability))

    print("\nFeature coefficients:")

    for feature, coefficient in zip(FEATURES, model.coef_[0]):
        print(f"{feature:25s}: {coefficient:.4f}")

    joblib.dump(model, model_path)

    print("\nModel saved to:")
    print(model_path)

    return model


if __name__ == "__main__":
    print(
        "This module contains the model-training function. "
        "Pass the train, validation, and model paths to train_model()."
    )