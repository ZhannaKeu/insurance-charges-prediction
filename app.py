"""Простой Streamlit-интерфейс для обучения модели и прогнозирования."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ACTIVE_DIR = ARTIFACTS_DIR / "active"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
EXPERIMENTS_CSV = ARTIFACTS_DIR / "experiments.csv"

st.set_page_config(page_title="Нейронная сеть: медицинские расходы", layout="wide")


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def get_training_parameters(info):
    """Поддерживает новый model_info и старые учебные эксперименты."""
    parameters = info.get("training_parameters", {})
    early_stopping = info.get("early_stopping", {})
    return {
        "epochs_max": parameters.get("epochs_max", info.get("max_epochs")),
        "epochs_actual": parameters.get("epochs_actual", info.get("epochs_completed")),
        "batch_size": parameters.get("batch_size", info.get("batch_size")),
        "learning_rate": parameters.get("learning_rate", 0.001),
        "early_stopping": parameters.get(
            "early_stopping_enabled", bool(early_stopping)
        ),
        "patience": parameters.get("patience", early_stopping.get("patience")),
    }


def number(value):
    if value is None or pd.isna(value):
        return "Нет данных"
    return f"{float(value):,.2f}".replace(",", " ")


def show_metrics(values):
    columns = st.columns(4)
    columns[0].metric("MAE", number(values.get("mae")))
    columns[1].metric("MSE", number(values.get("mse")))
    columns[2].metric("RMSE", number(values.get("rmse")))
    r2 = values.get("r2")
    columns[3].metric("R²", "Нет данных" if r2 is None else f"{float(r2):.4f}")


def activate(experiment_id):
    from train import activate_experiment

    try:
        activate_experiment(experiment_id)
        st.success("Модель успешно выбрана для прогнозирования.")
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"Не удалось выбрать модель: {error}")


def training_page():
    st.title("Обучение модели")
    st.write("Настройте параметры обучения нейронной сети и запустите новый эксперимент.")

    with st.form("training_form"):
        epochs = st.number_input(
            "Количество эпох", min_value=10, max_value=3000, value=1000, step=10
        )
        left, right = st.columns(2)
        with left:
            batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=2)
        with right:
            learning_rate = st.selectbox(
                "Learning rate",
                [0.0001, 0.0005, 0.001, 0.005, 0.01],
                index=2,
                format_func=lambda value: f"{value:g}",
            )
        use_early_stopping = st.checkbox("Early Stopping", value=True)
        patience = st.number_input(
            "Patience", min_value=5, max_value=200, value=30,
            disabled=not use_early_stopping,
        )
        start_training = st.form_submit_button("Обучить модель", type="primary")

    if start_training:
        from train import train_model

        config = {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "early_stopping_enabled": use_early_stopping,
            "patience": int(patience),
        }
        try:
            with st.spinner("Модель обучается..."):
                st.session_state["training_result"] = train_model(config)
        except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
            st.error(f"Не удалось обучить модель: {error}")

    # Session state нужен только для результата последнего обучения.
    result = st.session_state.get("training_result")
    if result is None:
        return

    st.success("Обучение завершено")
    left, middle, right = st.columns(3)
    left.write(f"**Максимум эпох:** {result['epochs_max']}")
    left.write(f"**Выполнено эпох:** {result['epochs_actual']}")
    middle.write(f"**Batch size:** {result['batch_size']}")
    middle.write(f"**Learning rate:** {result['learning_rate']:g}")
    right.write(
        "**Early Stopping:** "
        + ("включён" if result["early_stopping_enabled"] else "выключен")
    )
    right.write(f"**Patience:** {result['patience']}")
    st.write(f"**Время обучения:** {result['training_time_seconds']:.2f} с")
    show_metrics(result)

    epochs_index = range(1, result["epochs_actual"] + 1)
    loss_history = pd.DataFrame(
        {
            "Train loss": result["train_loss"],
            "Validation loss": result["validation_loss"],
        },
        index=epochs_index,
    )
    st.subheader("Train loss и Validation loss")
    st.line_chart(loss_history)
    st.subheader("Validation R²")
    st.line_chart(
        pd.DataFrame({"Validation R²": result["validation_r2"]}, index=epochs_index)
    )

    residual_plot = Path(result["artifact_dir"]) / "test_residuals.png"
    if residual_plot.exists():
        st.subheader("Ошибки прогнозирования")
        st.image(str(residual_plot))

    st.info(
        "Новая модель сохранена как эксперимент. "
        "Она ещё не используется для прогнозирования."
    )
    if st.button("Сделать модель активной"):
        activate(result["experiment_id"])


def prediction_page():
    st.title("Прогнозирование расходов")
    st.write(
        "Введите характеристики клиента. Для прогноза используется текущая активная модель."
    )

    required = [
        ACTIVE_DIR / "insurance_model.keras",
        ACTIVE_DIR / "preprocessor.pkl",
        ACTIVE_DIR / "model_info.json",
    ]
    if not all(path.exists() for path in required):
        st.warning("Активная модель не найдена. Сначала обучите или выберите модель.")
        return

    try:
        info = load_json(ACTIVE_DIR / "model_info.json")
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Не удалось прочитать active-модель: {error}")
        return

    parameters = get_training_parameters(info)
    st.subheader("Текущая active-модель")
    st.caption(info.get("experiment_id", "Идентификатор не указан"))
    columns = st.columns(5)
    columns[0].metric("R²", f"{info['r2']:.4f}")
    columns[1].metric("MAE", number(info.get("mae")))
    columns[2].metric("Эпохи", parameters["epochs_actual"])
    columns[3].metric("Batch size", parameters["batch_size"])
    columns[4].metric("Learning rate", f"{parameters['learning_rate']:g}")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Возраст", 18, 100, 40)
            sex = st.selectbox("Пол", ["Мужской", "Женский"])
            bmi = st.number_input("BMI", 10.0, 80.0, 30.0)
        with right:
            children = st.number_input("Количество детей", 0, 20, 2)
            smoker = st.selectbox("Курение", ["Нет", "Да"])
            region = st.selectbox(
                "Регион", ["northeast", "northwest", "southeast", "southwest"],
                index=2,
            )
        calculate = st.form_submit_button("Рассчитать прогноз", type="primary")

    if calculate:
        from predict import predict_charges

        try:
            prediction = predict_charges(
                int(age),
                "male" if sex == "Мужской" else "female",
                float(bmi),
                int(children),
                "yes" if smoker == "Да" else "no",
                region,
            )
            st.metric("Прогнозируемые расходы", number(prediction))
            st.caption("Это прогноз модели, а не фактическое значение расходов.")
        except (ValueError, TypeError, FileNotFoundError, OSError) as error:
            st.error(f"Не удалось выполнить прогноз: {error}")


def experiments_page():
    st.title("Эксперименты")
    if not EXPERIMENTS_CSV.exists():
        st.warning("История экспериментов пока пуста.")
        return
    try:
        experiments = pd.read_csv(EXPERIMENTS_CSV)
        active_id = load_json(ACTIVE_DIR / "model_info.json").get("experiment_id")
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as error:
        st.error(f"Не удалось прочитать историю экспериментов: {error}")
        return

    columns = [
        "experiment", "epochs_max", "epochs_actual", "batch_size",
        "learning_rate", "early_stopping", "patience",
        "training_time_seconds", "MAE", "RMSE", "R2",
    ]
    st.dataframe(experiments[columns], width="stretch", hide_index=True)

    experiment_id = st.selectbox("Выберите эксперимент", experiments["experiment"])
    row = experiments.loc[experiments["experiment"] == experiment_id].iloc[0]
    directory = EXPERIMENTS_DIR / experiment_id

    try:
        info = load_json(directory / "model_info.json")
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        st.error(f"Не удалось прочитать эксперимент: {error}")
        return

    parameters = get_training_parameters(info)
    st.subheader(experiment_id)
    if experiment_id == active_id:
        st.success("Активная модель")

    left, right = st.columns(2)
    left.write(f"**Максимум эпох:** {parameters['epochs_max'] or row['epochs_max']}")
    left.write(f"**Выполнено эпох:** {parameters['epochs_actual'] or row['epochs_actual']}")
    left.write(f"**Batch size:** {parameters['batch_size'] or row['batch_size']}")
    right.write(f"**Learning rate:** {parameters['learning_rate']}")
    right.write(f"**Early Stopping:** {parameters['early_stopping']}")
    right.write(f"**Patience:** {parameters['patience'] or row['patience']}")
    training_time = info.get("training_time_seconds", row["training_time_seconds"])
    st.write(
        "**Время обучения:** "
        + ("Нет данных" if pd.isna(training_time) else f"{training_time:.2f} с")
    )
    show_metrics(
        {
            "mae": info.get("mae", row["MAE"]),
            "mse": info.get("mse", row["MSE"]),
            "rmse": info.get("rmse", row["RMSE"]),
            "r2": info.get("r2", row["R2"]),
        }
    )

    for filename, title in [
        ("loss_history.png", "Train loss и Validation loss"),
        ("validation_r2.png", "Validation R²"),
        ("test_residuals.png", "Ошибки прогнозирования"),
    ]:
        path = directory / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path))

    complete = all(
        (directory / filename).exists()
        for filename in ["model.keras", "preprocessor.pkl", "model_info.json"]
    )
    if experiment_id == active_id:
        st.info("✓ Этот эксперимент сейчас используется для прогнозирования")
    elif not complete:
        st.warning("Этот эксперимент неполный и не может быть выбран как active-модель.")
    elif st.button("Сделать этот эксперимент активным"):
        activate(experiment_id)


page = st.sidebar.radio(
    "Раздел", ["Обучение модели", "Прогнозирование", "Эксперименты"]
)

if page == "Обучение модели":
    training_page()
elif page == "Прогнозирование":
    prediction_page()
else:
    experiments_page()
