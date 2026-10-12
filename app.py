"""Простой Streamlit-интерфейс для обучения модели и прогнозирования."""

import json
from pathlib import Path
from openai import OpenAI

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ACTIVE_DIR = ARTIFACTS_DIR / "active"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
EXPERIMENTS_CSV = ARTIFACTS_DIR / "experiments.csv"

st.set_page_config(page_title="Нейронная сеть: медицинские расходы", layout="wide")



def explain_prediction(prediction, age, sex, bmi, children, smoker, region):
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        timeout=60.0,
        max_retries=1
    )

    prompt = f"""
Ты — ИИ-помощник приложения для прогнозирования
страховых медицинских расходов.

Нейронная сеть рассчитала прогноз: ${prediction:,.2f}.

Данные клиента:
- Возраст: {age}
- Пол: {sex}
- Индекс массы тела (BMI): {bmi}
- Количество детей: {children}
- Курение: {smoker}
- Регион: {region}

Подготовь объяснение на русском языке.

Структура ответа:

### 1. Результат прогнозирования
Объясни, что означает прогнозируемая сумма.

### 2. Факторы, влияющие на расходы
Объясни возможное влияние возраста, BMI,
курения и других характеристик.

### 3. Интерпретация результата
Объясни, с какими общими закономерностями
может быть связан полученный прогноз.

### 4. Ограничения прогноза
Кратко объясни, что прогноз не является
точной стоимостью страховки.

Важные правила:
- Не придумывай точные вклады отдельных факторов.
- Не утверждай, что известны причины конкретного
  предсказания нейросети.
- Не делай медицинских диагнозов.
- Не выдавай прогноз за страховой тариф.
- Используй понятный русский язык.
- Ответ должен быть кратким.
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        reasoning={"effort": "low"},
        max_output_tokens=2000
    )

    explanation = response.output_text

    if not explanation or not explanation.strip():
        raise ValueError(
            "OpenAI вернул пустой ответ. "
            f"Статус: {response.status}. "
            f"Причина: {response.incomplete_details}"
        )

    return explanation



def chat_with_ai(messages, prediction_data=None):
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        timeout=60.0,
        max_retries=1
    )

    instructions = """
Ты — ИИ-помощник приложения для прогнозирования
медицинских страховых расходов.

Ты можешь:
- Объяснять назначение приложения.
- Рассказывать о факторах страховых расходов.
- Объяснять BMI, возрастные факторы и влияние курения.
- Рассказывать, как работают нейронные сети.
- Объяснять полученные прогнозы.

Отвечай на русском языке, понятно и кратко.
Не ставь медицинских диагнозов.
Не представляй прогноз как страховой тариф.
Не придумывай результаты расчётов нейросети.
Не утверждай, что известен точный вклад признаков.
"""

    if prediction_data is not None:
        prediction = prediction_data["prediction"]
        data = prediction_data["data"]

        instructions += f"""

Пользователь уже получил прогноз нейросети.

Прогнозируемые расходы: ${prediction:,.2f}.

Характеристики клиента:
- Возраст: {data['age']}
- Пол: {data['sex']}
- BMI: {data['bmi']}
- Количество детей: {data['children']}
- Курение: {data['smoker']}
- Регион: {data['region']}

Учитывай эти данные при ответах.
Если пользователь просит пересчитать прогноз,
объясни, что для этого нужно повторно
запустить нейросеть с изменёнными параметрами.
Не придумывай числовые результаты.
"""
    else:
        instructions += """

Пользователь ещё не выполнил прогнозирование.
Если вопрос требует результата нейросети,
предложи сначала рассчитать прогноз.
"""

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=instructions,
        input=messages,
        reasoning={"effort": "low"},
        max_output_tokens=2000
    )

    answer = response.output_text

    if not answer or not answer.strip():
        raise ValueError(
            "OpenAI вернул пустой ответ. "
            f"Статус: {response.status}. "
            f"Детали: {response.incomplete_details}"
        )

    return answer



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
    


def show_ai_chat(prediction_data=None):
    st.divider()
    st.subheader("💬 ИИ-помощник")

    st.caption(
        "Задавайте вопросы о страховых расходах, "
        "нейронных сетях и результатах прогнозирования."
    )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if prediction_data is not None:
        st.info(
            "ИИ-помощник учитывает последний "
            "рассчитанный прогноз."
        )

    if st.button("Очистить историю чата"):
        st.session_state["chat_messages"] = []
        st.rerun()

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input(
        "Напишите сообщение ИИ-помощнику...",
        key="insurance_chat_input"
    )

    if user_message:
        st.session_state["chat_messages"].append({
            "role": "user",
            "content": user_message
        })

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("ИИ отвечает..."):
                try:
                    answer = chat_with_ai(
                        st.session_state["chat_messages"],
                        prediction_data
                    )

                    st.markdown(answer)

                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:
                    st.session_state["chat_messages"].pop()
                    st.error(
                        f"Не удалось получить ответ ИИ: {error}"
                    )


def prediction_page():
    st.title("Прогнозирование расходов")
    st.write(
        "Введите характеристики клиента. "
        "Для прогноза используется текущая активная модель."
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
    columns[4].metric(
        "Learning rate",
        f"{parameters['learning_rate']:g}"
    )

    # Форма ввода данных
    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input("Возраст", 18, 100, 40)
            sex = st.selectbox("Пол", ["Мужской", "Женский"])
            bmi = st.number_input("BMI", 10.0, 70.0, 30.0)

        with right:
            children = st.number_input("Количество детей", 0, 10, 2)
            smoker = st.selectbox("Курение", ["Нет", "Да"])
            region = st.selectbox(
                "Регион",
                ["northeast", "northwest", "southeast", "southwest"],
                index=2,
            )

        calculate = st.form_submit_button(
            "Рассчитать прогноз",
            type="primary"
        )

    # Сбрасываем старый прогноз при смене модели
    active_id = info.get("experiment_id")

    if st.session_state.get("prediction_model_id") != active_id:
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("ai_explanation", None)
        st.session_state["prediction_model_id"] = active_id

    # Текущие данные формы
    current_data = {
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region,
    }

    # Сбрасываем старый прогноз при изменении данных
    if "prediction_result" in st.session_state:
        saved_data = st.session_state["prediction_result"]["data"]

        if saved_data != current_data:
            st.session_state.pop("prediction_result", None)
            st.session_state.pop("ai_explanation", None)

    # Расчёт нейросети
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

            st.session_state["prediction_result"] = {
                "prediction": float(prediction),
                "data": current_data.copy(),
            }

            # Новый прогноз — новое объяснение
            st.session_state.pop("ai_explanation", None)

        except (
            ValueError,
            TypeError,
            FileNotFoundError,
            OSError
        ) as error:
            st.session_state.pop("prediction_result", None)
            st.session_state.pop("ai_explanation", None)
            st.error(f"Не удалось выполнить прогноз: {error}")

    # Показываем сохранённый результат
    result = st.session_state.get("prediction_result")

    if result is None:
        show_ai_chat()
        return

    prediction = result["prediction"]
    data = result["data"]

    st.divider()
    st.subheader("Результат прогнозирования")

    st.metric(
        "Прогнозируемые расходы",
        f"${prediction:,.2f}"
    )

    st.caption(
        "Это прогноз модели, а не фактическое "
        "значение расходов или цена страхового полиса."
    )

    # Кнопка обращения к OpenAI
    if st.button(
        "✨ Объяснить прогноз с помощью ИИ",
        type="primary"
    ):
        with st.spinner("ИИ анализирует результат..."):
            try:
                explanation = explain_prediction(
                    prediction=prediction,
                    age=data["age"],
                    sex=data["sex"],
                    bmi=data["bmi"],
                    children=data["children"],
                    smoker=data["smoker"],
                    region=data["region"],
                )

                st.session_state["ai_explanation"] = explanation

            except Exception as error:
                st.session_state.pop("ai_explanation", None)
                st.error(f"Ошибка обращения к OpenAI: {error}")
                st.exception(error)

    # Вывод объяснения
    if st.session_state.get("ai_explanation"):
        st.subheader("Объяснение ИИ")
        st.markdown(st.session_state["ai_explanation"])


def experiments_page():
    st.title("Эксперименты")
    if not EXPERIMENTS_CSV.exists():
        st.warning("История экспериментов пока пуста.")
        return
    columns = [
        "experiment", "epochs_max", "epochs_actual", "batch_size",
        "learning_rate", "early_stopping", "patience",
        "training_time_seconds", "MAE", "MSE", "RMSE", "R2",
    ]
    try:
        experiments = pd.read_csv(EXPERIMENTS_CSV)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        st.error(f"Не удалось прочитать историю экспериментов: {error}")
        return

    missing_columns = [name for name in columns if name not in experiments.columns]
    if missing_columns:
        st.error(
            "В журнале экспериментов отсутствуют обязательные столбцы: "
            + ", ".join(missing_columns)
        )
        return
    if experiments.empty:
        st.info("История экспериментов пока пуста.")
        return

    try:
        active_id = load_json(ACTIVE_DIR / "model_info.json").get("experiment_id")
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Не удалось прочитать active-модель: {error}")
        return

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
