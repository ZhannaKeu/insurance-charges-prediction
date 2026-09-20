# Прогноз медицинских расходов

Учебный проект демонстрирует подготовку данных, обучение нейронной сети для
регрессии, оценку на отложенной выборке, сохранение экспериментов и прогноз
через Streamlit-интерфейс.

## Структура

```text
data/insurance2.csv       исходный датасет
artifacts/active/         модель, которую использует predict.py
artifacts/experiments/    отдельные результаты запусков обучения
preprocessing.py          sklearn-transformer для ограничения BMI
train.py                  полный цикл обучения и оценки
predict.py                прогноз с использованием сохранённых артефактов
requirements.txt          зависимости проекта
app.py                    веб-интерфейс Streamlit
```

## Подготовка окружения

Рекомендуемая версия Python — 3.11. Для macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Сохранённые `preprocessor.pkl` создаются средствами scikit-learn/joblib и
требуют совместимой версии scikit-learn. Для артефактов этого проекта в
`requirements.txt` зафиксирована версия scikit-learn 1.6.1.

## Обучение

```bash
python train.py
```

Скрипт использует `TrainingConfig` со значениями по умолчанию, удаляет точные
дубликаты, делит данные на train/validation/test в пропорции 70/15/15, обучает
preprocessing только на train и оценивает модель на test. Каждый запуск
сохраняется в отдельном каталоге `artifacts/experiments/<experiment_id>/`:

- `model.keras` — обученная Keras-модель;
- `preprocessor.pkl` — сохранённая предобработка;
- `model_info.json` — размеры выборок, параметры и test-метрики;
- `loss_history.png`, `validation_r2.png`, `test_residuals.png` — графики.

Из Python параметры можно передать словарём:

```python
from train import train_model

result = train_model({
    "epochs": 1000,
    "batch_size": 64,
    "learning_rate": 0.001,
    "early_stopping_enabled": True,
    "patience": 30,
})
```

Новый эксперимент не заменяет active-модель. Для явной активации сохранённого
эксперимента предусмотрена функция `activate_experiment(experiment_id)`.

## Тестовый прогноз

После обучения выполните:

```bash
python predict.py
```

`predict.py` всегда загружает файлы только из `artifacts/active/`. Функцию
`predict_charges(...)` можно импортировать и вызывать с исходными числовыми и
категориальными значениями без ручного кодирования категорий.

## Веб-приложение

```bash
source .venv/bin/activate
streamlit run app.py
```

В приложении доступны три раздела:

- «Обучение модели» — настройка параметров и запуск отдельного эксперимента;
- «Прогнозирование» — расчёт расходов с помощью active-модели;
- «Эксперименты» — история, просмотр метрик и явный выбор active-модели.
