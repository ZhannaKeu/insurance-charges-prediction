# Обучение регрессионной нейронной сети для прогноза medical charges

import csv
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import PercentileClipper


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "insurance2.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
ACTIVE_DIR = ARTIFACTS_DIR / "active"
EXPERIMENTS_CSV_PATH = ARTIFACTS_DIR / "experiments.csv"

FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
NUMERIC_COLUMNS = ["age", "bmi", "children"]
CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
TARGET_COLUMN = "charges"
SEED = 42


@dataclass(frozen=True)
class TrainingConfig:
    # Параметры, которые позже сможет передавать веб-интерфейс

    epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_enabled: bool = True
    patience: Optional[int] = 30

    def validate(self):
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise ValueError("epochs должен быть целым числом")
        if self.epochs <= 0:
            raise ValueError("epochs должен быть больше 0")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ValueError("batch_size должен быть целым числом")
        if self.batch_size <= 0:
            raise ValueError("batch_size должен быть больше 0")
        if isinstance(self.learning_rate, bool) or not isinstance(
            self.learning_rate, (int, float)
        ):
            raise ValueError("learning_rate должен быть числом")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate должен быть конечным числом больше 0")
        if not isinstance(self.early_stopping_enabled, bool):
            raise ValueError("early_stopping_enabled должен быть True или False")
        if self.early_stopping_enabled:
            if isinstance(self.patience, bool) or not isinstance(self.patience, int):
                raise ValueError("patience должен быть целым числом")
            if self.patience <= 0:
                raise ValueError("patience должен быть больше 0")


ConfigInput = Union[TrainingConfig, Mapping[str, Any]]


def normalize_config(config: Optional[ConfigInput] = None):
    # Преобразует словарь веб-формы в проверенную конфигурацию
    if config is None:
        normalized = TrainingConfig()
    elif isinstance(config, TrainingConfig):
        normalized = config
    elif isinstance(config, Mapping):
        try:
            normalized = TrainingConfig(**dict(config))
        except TypeError as error:
            raise ValueError(f"Некорректный набор параметров: {error}") from error
    else:
        raise TypeError("config должен быть TrainingConfig, словарём или None")
    normalized.validate()
    return normalized


def set_random_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def load_and_check_data(path=DATA_PATH):
    # Загружает CSV, проверяет схему и выводит краткий профиль данных

    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден: {path}")
    data = pd.read_csv(path)
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют столбцы: {sorted(missing_columns)}")
    data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    print("\nПроверка исходного датасета")
    print(f"Размер: {data.shape[0]} строк, {data.shape[1]} столбцов")
    print("Типы данных:")
    print(data.dtypes.to_string())
    print("Пропуски:")
    print(data.isna().sum().to_string())
    for column in CATEGORICAL_COLUMNS:
        categories = sorted(data[column].dropna().astype(str).unique().tolist())
        print(f"Категории {column}: {categories}")
    print(f"Точные дубликаты: {int(data.duplicated().sum())}")
    return data


def split_data(data):
    # Удаляет дубликаты и делит данные на 70/15/15
    clean_data = data.drop_duplicates().reset_index(drop=True)
    X = clean_data[FEATURE_COLUMNS]
    y = clean_data[TARGET_COLUMN]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )
    return clean_data, X_train, X_validation, X_test, y_train, y_validation, y_test


def build_preprocessor():
    #Создаёт preprocessing, статистики которого обучаются только на train
    # Медиана устойчива к выбросам и подходит для числовых признаков insurance.
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    bmi_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", PercentileClipper(1.0, 99.0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, ["age", "children"]),
            ("bmi", bmi_pipeline, ["bmi"]),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )


def prepare_data(preprocessor, X_train, X_validation, X_test):
    # Обучает preprocessing на train и только преобразует validation/test
    
    processed = (
        preprocessor.fit_transform(X_train),
        preprocessor.transform(X_validation),
        preprocessor.transform(X_test),
    )
    return tuple(np.asarray(values, dtype=np.float32) for values in processed)


def build_model(input_features, learning_rate):
    # Создаёт фиксированную базовую нейронную сеть для регрессии

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_features,)),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


class ValidationR2Callback(tf.keras.callbacks.Callback):
    # Сохраняет R² на validation после каждой эпохи

    def __init__(self, X_validation, y_validation):
        super().__init__()
        self.X_validation = X_validation
        self.y_validation = np.asarray(y_validation)
        self.values = []

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.X_validation, verbose=0).reshape(-1)
        value = float(r2_score(self.y_validation, predictions))
        self.values.append(value)
        if logs is not None:
            logs["val_r2"] = value


def fit_model(model, config, X_train, y_train, X_validation, y_validation):
    # Обучает Keras-модель и возвращает историю и длительность fit
    
    r2_callback = ValidationR2Callback(X_validation, y_validation)
    callbacks = [r2_callback]
    if config.early_stopping_enabled:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
            )
        )
    started_at = time.perf_counter()
    history = model.fit(
        X_train,
        np.asarray(y_train, dtype=np.float32),
        validation_data=(X_validation, np.asarray(y_validation, dtype=np.float32)),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history, r2_callback.values, float(time.perf_counter() - started_at)


def evaluate_model(model, X_test, y_test):
    # Рассчитывает финальные метрики исключительно на test
    y_true = np.asarray(y_test, dtype=float)
    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    mse = float(mean_squared_error(y_true, y_pred))
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    print("\nМетрики на test")
    for name, value in metrics.items():
        print(f"{name.upper():>4}: {value:,.4f}")
    examples = pd.DataFrame(
        {
            "Реальное значение": y_true[:5],
            "Прогноз": y_pred[:5],
            "Абсолютная ошибка": np.abs(y_true[:5] - y_pred[:5]),
        }
    )
    print("\nПримеры прогнозов:")
    print(examples.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    return metrics, y_true, y_pred


def make_experiment_id():
    return datetime.now(timezone.utc).strftime("experiment_%Y%m%d_%H%M%S_%f")


def validate_experiment_id(experiment_id):
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", experiment_id
    ):
        raise ValueError(
            "experiment_id может содержать только буквы, цифры, '_' и '-'"
        )


def save_plots(history, validation_r2, y_true, y_pred, output_dir):
    # Сохраняет графики конкретного эксперимента
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["loss"], label="Train loss")
    plt.plot(epochs, history.history["val_loss"], label="Validation loss")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title("Ошибка обучения")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_history.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, validation_r2, color="tab:green")
    plt.xlabel("Эпоха")
    plt.ylabel("R²")
    plt.title("Validation R²")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "validation_r2.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, y_true - y_pred, alpha=0.65, s=24)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Прогноз")
    plt.ylabel("Ошибка (y_true - y_pred)")
    plt.title("Ошибки прогнозирования на test")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "test_residuals.png", dpi=150)
    plt.close()


def append_experiment_row(experiment_id, result):
    # Добавляет одну строку в простой журнал экспериментов
    
    fieldnames = [
        "experiment", "epochs_max", "epochs_actual", "batch_size",
        "learning_rate", "early_stopping", "patience", "training_time_seconds",
        "MAE", "MSE", "RMSE", "R2",
    ]
    write_header = not EXPERIMENTS_CSV_PATH.exists()
    with EXPERIMENTS_CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "experiment": experiment_id,
                "epochs_max": result["epochs_max"],
                "epochs_actual": result["epochs_actual"],
                "batch_size": result["batch_size"],
                "learning_rate": result["learning_rate"],
                "early_stopping": result["early_stopping_enabled"],
                "patience": result["patience"],
                "training_time_seconds": result["training_time_seconds"],
                "MAE": result["mae"], "MSE": result["mse"],
                "RMSE": result["rmse"], "R2": result["r2"],
            }
        )


def save_experiment(
    model, preprocessor, model_info, history, validation_r2, y_true, y_pred,
    experiment_id,
):
    # Сохраняет новый эксперимент, не изменяя active-модель.

    output_dir = EXPERIMENTS_DIR / experiment_id
    if output_dir.exists():
        raise FileExistsError(f"Эксперимент уже существует: {experiment_id}")
    output_dir.mkdir(parents=True)
    model.save(output_dir / "model.keras")
    joblib.dump(preprocessor, output_dir / "preprocessor.pkl")
    (output_dir / "model_info.json").write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(history, validation_r2, y_true, y_pred, output_dir)
    return output_dir


def activate_experiment(experiment_id):
    # Явно делает сохранённый эксперимент моделью для predict.py
    validate_experiment_id(experiment_id)
    source_dir = EXPERIMENTS_DIR / experiment_id
    required = {
        "model.keras": "insurance_model.keras",
        "preprocessor.pkl": "preprocessor.pkl",
        "model_info.json": "model_info.json",
    }
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"В эксперименте отсутствуют файлы: {missing}")
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    for source_name, active_name in required.items():
        shutil.copy2(source_dir / source_name, ACTIVE_DIR / active_name)


def train_model(config: Optional[ConfigInput] = None, experiment_id=None):
    # Выполняет полный цикл и возвращает данные для будущего интерфейса
    config = normalize_config(config)
    experiment_id = experiment_id or make_experiment_id()
    validate_experiment_id(experiment_id)
    if (EXPERIMENTS_DIR / experiment_id).exists():
        raise FileExistsError(f"Эксперимент уже существует: {experiment_id}")

    set_random_seeds()

    # Загружаем данные и делим их на train / validation / test.
    original_data = load_and_check_data()
    (
        clean_data, X_train, X_validation, X_test,
        y_train, y_validation, y_test,
    ) = split_data(original_data)
    print(
        f"\nПосле удаления дубликатов: {len(clean_data)}; "
        f"train={len(X_train)}, validation={len(X_validation)}, test={len(X_test)}"
    )

    # Обучаем preprocessing только на train.
    preprocessor = build_preprocessor()
    X_train_processed, X_validation_processed, X_test_processed = prepare_data(
        preprocessor, X_train, X_validation, X_test
    )

    # Создаём и обучаем нейронную сеть.
    model = build_model(X_train_processed.shape[1], config.learning_rate)
    history, validation_r2, training_time_seconds = fit_model(
        model, config, X_train_processed, y_train,
        X_validation_processed, y_validation,
    )

    # Оцениваем готовую модель на test.
    metrics, y_true, y_pred = evaluate_model(model, X_test_processed, y_test)

    train_loss = [float(value) for value in history.history["loss"]]
    validation_loss = [float(value) for value in history.history["val_loss"]]
    validation_r2 = [float(value) for value in validation_r2]
    epochs_actual = len(train_loss)
    result = {
        "experiment_id": experiment_id,
        "epochs_max": config.epochs,
        "epochs_actual": epochs_actual,
        "batch_size": config.batch_size,
        "learning_rate": float(config.learning_rate),
        "early_stopping_enabled": config.early_stopping_enabled,
        "patience": config.patience,
        "early_stopping_triggered": bool(
            config.early_stopping_enabled and epochs_actual < config.epochs
        ),
        "training_time_seconds": training_time_seconds,
        **metrics,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_r2": validation_r2,
    }
    model_info = {
        "experiment_id": experiment_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_rows": int(len(original_data)),
        "rows_after_deduplication": int(len(clean_data)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "preprocessed_feature_count": int(X_train_processed.shape[1]),
        **metrics,
        "training_parameters": {
            "epochs_max": config.epochs,
            "epochs_actual": epochs_actual,
            "batch_size": config.batch_size,
            "learning_rate": float(config.learning_rate),
            "early_stopping_enabled": config.early_stopping_enabled,
            "patience": config.patience,
        },
        "training_time_seconds": training_time_seconds,
        "early_stopping_triggered": result["early_stopping_triggered"],
        "minimum_validation_loss_epoch": int(np.argmin(validation_loss) + 1),
        "minimum_validation_loss": float(np.min(validation_loss)),
        "random_seed": SEED,
    }

    # Сохраняем модель, preprocessing, метрики и графики эксперимента.
    output_dir = save_experiment(
        model, preprocessor, model_info, history, validation_r2,
        y_true, y_pred, experiment_id,
    )
    result["artifact_dir"] = str(output_dir)
    append_experiment_row(experiment_id, result)
    print(f"\nЭксперимент сохранён в: {output_dir}")
    print("Active-модель не изменена.")
    return result


def main():
    """CLI-запуск использует параметры TrainingConfig по умолчанию."""
    result = train_model(TrainingConfig())
    print(
        f"Готово: {result['experiment_id']}, "
        f"эпох={result['epochs_actual']}, R²={result['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
