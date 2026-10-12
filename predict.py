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


def _validate_prediction_inputs(age, sex, bmi, children, smoker, region):
    integer_values = {"age": (age, 18, 100), "children": (children, 0, 10)}
    normalized = {}
    for name, (value, minimum, maximum) in integer_values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} должен быть целым числом от {minimum} до {maximum}")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} должен быть в диапазоне от {minimum} до {maximum}")
        normalized[name] = int(value)

    if isinstance(bmi, (bool, np.bool_)) or not isinstance(
        bmi, (int, float, np.integer, np.floating)
    ):
        raise ValueError("bmi должен быть конечным числом от 10 до 70")
    bmi = float(bmi)
    if not np.isfinite(bmi) or not 10 <= bmi <= 70:
        raise ValueError("bmi должен быть конечным числом от 10 до 70")

    categories = {
        "sex": (sex, {"female", "male"}),
        "smoker": (smoker, {"yes", "no"}),
        "region": (region, {"northeast", "northwest", "southeast", "southwest"}),
    }
    for name, (value, allowed) in categories.items():
        if not isinstance(value, str):
            raise ValueError(f"{name} должен быть строкой из допустимых значений")
        value = value.strip().lower()
        if value not in allowed:
            options = ", ".join(sorted(allowed))
            raise ValueError(f"Недопустимое значение {name}: {value!r}. Допустимо: {options}")
        normalized[name] = value

    normalized["bmi"] = bmi
    return normalized


def predict_charges(age, sex, bmi, children, smoker, region):
    """Принимает исходные признаки и возвращает один прогноз charges."""
    values = _validate_prediction_inputs(age, sex, bmi, children, smoker, region)
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Active-модель не найдена. Сначала выберите модель.")

    # Передаём preprocessing исходные значения без ручного кодирования категорий.
    person = pd.DataFrame(
        [
            {
                "age": values["age"],
                "sex": values["sex"],
                "bmi": values["bmi"],
                "children": values["children"],
                "smoker": values["smoker"],
                "region": values["region"],
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
