"""Прогноз расходов с помощью active-модели."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


ROOT_DIR = Path(__file__).resolve().parent
ACTIVE_DIR = ROOT_DIR / "artifacts" / "active"
MODEL_PATH = ACTIVE_DIR / "insurance_model.keras"
PREPROCESSOR_PATH = ACTIVE_DIR / "preprocessor.pkl"


def predict_charges(age, sex, bmi, children, smoker, region):
    """Принимает исходные признаки и возвращает один прогноз charges."""
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Active-модель не найдена. Сначала выберите модель.")

    # Передаём preprocessing исходные значения без ручного кодирования категорий.
    person = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    prepared_data = preprocessor.transform(person)
    prediction = model.predict(
        np.asarray(prepared_data, dtype=np.float32), verbose=0
    )
    return float(prediction[0, 0])


if __name__ == "__main__":
    prediction = predict_charges(
        age=40,
        sex="male",
        bmi=30,
        children=2,
        smoker="yes",
        region="southeast",
    )
    print(f"Прогноз charges: {prediction:,.2f}")
