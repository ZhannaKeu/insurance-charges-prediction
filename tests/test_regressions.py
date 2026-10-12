import ast
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import predict
import train


FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
CSV_COLUMNS = [
    "experiment", "epochs_max", "epochs_actual", "batch_size",
    "learning_rate", "early_stopping", "patience", "training_time_seconds",
    "MAE", "MSE", "RMSE", "R2",
]
ARTIFACT_FILES = [
    "model.keras", "preprocessor.pkl", "model_info.json",
    "loss_history.png", "validation_r2.png", "test_residuals.png",
]


def make_training_data():
    return pd.DataFrame(
        [
            [40, "male", 30.0, 2, "no", "southeast", 10000.0],
            [31, "female", 25.0, 0, "no", "northwest", 8000.0],
            [55, "male", 28.0, 1, "yes", "northeast", 12000.0],
        ],
        columns=FEATURE_COLUMNS + ["charges"],
    )


def write_experiment(directory, files=None):
    directory.mkdir(parents=True, exist_ok=True)
    contents = files or {
        "model.keras": b"model-v2",
        "preprocessor.pkl": b"preprocessor-v2",
        "model_info.json": json.dumps({"experiment_id": "exp_new"}).encode(),
    }
    for name, content in contents.items():
        (directory / name).write_bytes(content)


class PredictionValidationTests(unittest.TestCase):
    def test_invalid_values_raise_before_loading_artifacts(self):
        cases = [
            ("age", (-10, "male", 30, 2, "no", "southeast")),
            ("age", (True, "male", 30, 2, "no", "southeast")),
            ("bmi", (40, "male", 0, 2, "no", "southeast")),
            ("bmi", (40, "male", np.nan, 2, "no", "southeast")),
            ("bmi", (40, "male", np.inf, 2, "no", "southeast")),
            ("children", (40, "male", 30, -1, "no", "southeast")),
            ("children", (40, "male", 30, True, "no", "southeast")),
            ("sex", (40, "unknown", 30, 2, "no", "southeast")),
            ("smoker", (40, "male", 30, 2, "sometimes", "southeast")),
            ("region", (40, "male", 30, 2, "no", "middle")),
        ]
        with mock.patch.object(predict.joblib, "load") as load_preprocessor, mock.patch.object(
            predict.tf.keras.models, "load_model"
        ) as load_model:
            for field, values in cases:
                with self.subTest(field=field, values=values):
                    with self.assertRaisesRegex(ValueError, field):
                        predict.predict_charges(*values)
            load_preprocessor.assert_not_called()
            load_model.assert_not_called()

    def test_string_categories_are_normalized_for_preprocessor(self):
        class RecordingPreprocessor:
            received = None

            def transform(self, frame):
                self.received = frame.copy()
                return np.zeros((1, 1), dtype=np.float32)

        class FixedModel:
            def predict(self, values, verbose=0):
                return np.array([[123.0]])

        preprocessor = RecordingPreprocessor()
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model.keras"
            preprocessor_path = Path(temporary) / "preprocessor.pkl"
            model_path.touch()
            preprocessor_path.touch()
            with mock.patch.object(predict, "MODEL_PATH", model_path), mock.patch.object(
                predict, "PREPROCESSOR_PATH", preprocessor_path
            ), mock.patch.object(
                predict.joblib, "load", return_value=preprocessor
            ), mock.patch.object(
                predict.tf.keras.models, "load_model", return_value=FixedModel()
            ):
                result = predict.predict_charges(
                    40, " Male ", 30, 2, " YES ", " NORTHWEST "
                )

        self.assertEqual(result, 123.0)
        self.assertEqual(
            preprocessor.received.iloc[0].to_dict(),
            {
                "age": 40,
                "sex": "male",
                "bmi": 30.0,
                "children": 2,
                "smoker": "yes",
                "region": "northwest",
            },
        )


class TrainingDataValidationTests(unittest.TestCase):
    def load_modified_data(self, column, value):
        data = make_training_data()
        data.loc[0, column] = value
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.csv"
            data.to_csv(path, index=False)
            with redirect_stdout(io.StringIO()):
                return train.load_and_check_data(path)

    def test_nan_and_infinite_charges_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "charges"):
                self.load_modified_data("charges", value)

    def test_infinite_numeric_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bmi"):
            self.load_modified_data("bmi", np.inf)

    def test_feature_nan_is_preserved_for_imputer(self):
        loaded = self.load_modified_data("bmi", np.nan)
        self.assertTrue(pd.isna(loaded.loc[0, "bmi"]))

    def test_valid_small_dataset_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid.csv"
            make_training_data().to_csv(path, index=False)
            with redirect_stdout(io.StringIO()):
                loaded = train.load_and_check_data(path)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(list(loaded.columns), FEATURE_COLUMNS + ["charges"])


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.experiments = self.root / "experiments"
        self.active = self.root / "active"
        write_experiment(self.experiments / "exp_new")
        write_experiment(
            self.active,
            {
                "insurance_model.keras": b"old-model",
                "preprocessor.pkl": b"old-preprocessor",
                "model_info.json": b'{"experiment_id":"old"}',
                "old-extra": b"remove-on-success",
            },
        )
        self.paths = mock.patch.multiple(
            train, EXPERIMENTS_DIR=self.experiments, ACTIVE_DIR=self.active
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.addCleanup(self.temporary.cleanup)

    def assert_old_active_unchanged(self):
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.active.iterdir()},
            {
                "insurance_model.keras": b"old-model",
                "preprocessor.pkl": b"old-preprocessor",
                "model_info.json": b'{"experiment_id":"old"}',
                "old-extra": b"remove-on-success",
            },
        )

    def assert_no_staging_or_backup(self):
        self.assertEqual(list(self.root.glob(".active.staging-*")), [])
        self.assertEqual(list(self.root.glob(".active.backup-*")), [])

    def test_success_replaces_entire_active_bundle_and_cleans_temporary_dirs(self):
        train.activate_experiment("exp_new")
        self.assertEqual(
            {path.name for path in self.active.iterdir()},
            {"insurance_model.keras", "preprocessor.pkl", "model_info.json"},
        )
        self.assertEqual(
            (self.active / "insurance_model.keras").read_bytes(), b"model-v2"
        )
        self.assertEqual(
            json.loads((self.active / "model_info.json").read_text())["experiment_id"],
            "exp_new",
        )
        self.assert_no_staging_or_backup()

    def test_second_copy_failure_keeps_old_active_and_cleans_staging(self):
        real_copy = train.shutil.copy2
        calls = 0

        def fail_second_copy(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected copy failure")
            return real_copy(source, destination)

        with mock.patch.object(train.shutil, "copy2", side_effect=fail_second_copy):
            with self.assertRaisesRegex(OSError, "injected"):
                train.activate_experiment("exp_new")
        self.assert_old_active_unchanged()
        self.assert_no_staging_or_backup()

    def test_publication_failure_rolls_back_old_active(self):
        real_replace = train.os.replace
        failed = False

        def fail_staging_publication(source, destination):
            nonlocal failed
            if Path(destination) == self.active and not failed:
                failed = True
                raise OSError("injected publication failure")
            return real_replace(source, destination)

        with mock.patch.object(train.os, "replace", side_effect=fail_staging_publication):
            with self.assertRaisesRegex(OSError, "injected"):
                train.activate_experiment("exp_new")
        self.assertTrue(failed)
        self.assert_old_active_unchanged()
        self.assert_no_staging_or_backup()

    def test_missing_empty_and_corrupt_required_files_are_rejected(self):
        cases = [
            ("missing", {"model.keras": None}),
            ("empty", {"preprocessor.pkl": b""}),
            ("corrupt metadata", {"model_info.json": b"not-json"}),
        ]
        for label, changes in cases:
            with self.subTest(label=label):
                source = self.experiments / "exp_new"
                original = {
                    name: (source / name).read_bytes()
                    for name in ("model.keras", "preprocessor.pkl", "model_info.json")
                }
                for name, content in changes.items():
                    path = source / name
                    if content is None:
                        path.unlink()
                    else:
                        path.write_bytes(content)
                with self.assertRaises((FileNotFoundError, OSError, ValueError)):
                    train.activate_experiment("exp_new")
                self.assert_old_active_unchanged()
                self.assert_no_staging_or_backup()
                for name, content in original.items():
                    path = source / name
                    if name in changes and changes[name] is None:
                        path.write_bytes(content)
                    elif name not in changes:
                        self.assertEqual(path.read_bytes(), content)


class ExperimentPersistenceTests(unittest.TestCase):
    def make_result(self):
        return {
            "epochs_max": 2,
            "epochs_actual": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "early_stopping_enabled": True,
            "patience": 2,
            "training_time_seconds": 0.5,
            "mae": 1.0,
            "mse": 2.0,
            "rmse": float(np.sqrt(2.0)),
            "r2": 0.5,
        }

    def write_plots(self, directory):
        for name in ARTIFACT_FILES[3:]:
            (Path(directory) / name).write_bytes(b"plot")

    def patched_artifact_writers(self):
        return mock.patch.object(
            train.joblib,
            "dump",
            side_effect=lambda value, path: Path(path).write_bytes(b"preprocessor"),
        ), mock.patch.object(
            train,
            "save_plots",
            side_effect=lambda *args: self.write_plots(args[-1]),
        )

    def test_file_save_failure_does_not_publish_experiment(self):
        class FailingModel:
            def save(self, path):
                Path(path).write_bytes(b"partial-model")
                raise OSError("injected model save failure")

        with tempfile.TemporaryDirectory() as temporary:
            experiments = Path(temporary) / "experiments"
            with mock.patch.object(train, "EXPERIMENTS_DIR", experiments):
                with self.assertRaisesRegex(OSError, "injected"):
                    train.save_experiment(
                        FailingModel(), object(), {}, None, [], [], [], "exp_fail"
                    )
            self.assertFalse((experiments / "exp_fail").exists())
            self.assertEqual(list(experiments.iterdir()), [])

    def test_successful_save_publishes_all_required_files(self):
        class Model:
            def save(self, path):
                Path(path).write_bytes(b"model")

        with tempfile.TemporaryDirectory() as temporary:
            experiments = Path(temporary) / "experiments"
            dump_patch, plots_patch = self.patched_artifact_writers()
            with mock.patch.object(train, "EXPERIMENTS_DIR", experiments), dump_patch, plots_patch:
                output = train.save_experiment(
                    Model(), object(), {"experiment_id": "exp_ok"}, None,
                    [], [], [], "exp_ok",
                )
            self.assertEqual({path.name for path in output.iterdir()}, set(ARTIFACT_FILES))
            self.assertEqual(list(experiments.glob("*.staging-*")), [])
            self.assertEqual(list(experiments.glob(".exp_ok.staging-*")), [])

    def test_registration_failure_removes_only_new_experiment(self):
        class Model:
            def save(self, path):
                Path(path).write_bytes(b"model")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiments = root / "experiments"
            old_experiment = experiments / "previous"
            old_experiment.mkdir(parents=True)
            (old_experiment / "keep.txt").write_text("preserve")
            csv_path = root / "experiments.csv"
            csv_path.write_text("existing journal\n")
            data = make_training_data()
            features = data[FEATURE_COLUMNS]
            target = data["charges"]
            array = np.ones((len(data), 1), dtype=np.float32)
            history = SimpleNamespace(history={"loss": [1.0], "val_loss": [1.0]})
            metrics = {"mae": 1.0, "mse": 1.0, "rmse": 1.0, "r2": 0.5}
            model = Model()
            dump_patch, plots_patch = self.patched_artifact_writers()
            patches = [
                mock.patch.object(train, "EXPERIMENTS_DIR", experiments),
                mock.patch.object(train, "EXPERIMENTS_CSV_PATH", csv_path),
                mock.patch.object(train, "set_random_seeds"),
                mock.patch.object(train, "load_and_check_data", return_value=data),
                mock.patch.object(
                    train, "split_data",
                    return_value=(data, features, features, features, target, target, target),
                ),
                mock.patch.object(train, "build_preprocessor", return_value=object()),
                mock.patch.object(train, "prepare_data", return_value=(array, array, array)),
                mock.patch.object(train, "build_model", return_value=model),
                mock.patch.object(
                    train, "fit_model",
                    return_value=(history, [0.5], 0.1),
                ),
                mock.patch.object(
                    train, "evaluate_model",
                    return_value=(metrics, np.array([1.0]), np.array([1.0])),
                ),
                mock.patch.object(
                    train, "append_experiment_row",
                    side_effect=OSError("injected journal failure"),
                ),
                dump_patch,
                plots_patch,
            ]
            try:
                for patcher in patches:
                    patcher.start()
                with self.assertRaisesRegex(OSError, "journal failure"):
                    train.train_model(
                        train.TrainingConfig(
                            epochs=1, batch_size=1,
                            early_stopping_enabled=False, patience=None,
                        ),
                        experiment_id="new_experiment",
                    )
            finally:
                for patcher in reversed(patches):
                    patcher.stop()
            self.assertFalse((experiments / "new_experiment").exists())
            self.assertEqual((old_experiment / "keep.txt").read_text(), "preserve")
            self.assertEqual(csv_path.read_text(), "existing journal\n")
            self.assertEqual(list(experiments.glob(".new_experiment.staging-*")), [])

    def test_successful_csv_update_preserves_existing_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiments.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(dict.fromkeys(CSV_COLUMNS, "old"))
            with mock.patch.object(train, "EXPERIMENTS_CSV_PATH", path):
                train.append_experiment_row("new", self.make_result())
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["experiment"] for row in rows], ["old", "new"])
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_csv_replace_failure_preserves_previous_version_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiments.csv"
            original = b"experiment,old_column\nold,keep\n"
            path.write_bytes(original)
            with mock.patch.object(train, "EXPERIMENTS_CSV_PATH", path), mock.patch.object(
                train.os, "replace", side_effect=OSError("injected replace failure")
            ):
                with self.assertRaisesRegex(OSError, "replace failure"):
                    train.append_experiment_row("new", self.make_result())
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])


def load_experiments_page(streamlit):
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "experiments_page"
    )
    namespace = {
        "pd": pd,
        "json": json,
        "st": streamlit,
        "EXPERIMENTS_CSV": None,
        "ACTIVE_DIR": None,
        "EXPERIMENTS_DIR": None,
        "load_json": lambda path: {"experiment_id": None},
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(app_path), "exec"),
        namespace,
    )
    return namespace["experiments_page"], namespace


class FakeStreamlit:
    def __init__(self):
        self.errors = []
        self.infos = []

    def title(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, message, *args, **kwargs):
        self.errors.append(message)

    def info(self, message, *args, **kwargs):
        self.infos.append(message)


class ExperimentsPageTests(unittest.TestCase):
    def run_page(self, csv_bytes):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        csv_path = Path(temporary.name) / "experiments.csv"
        csv_path.write_bytes(csv_bytes)
        streamlit = FakeStreamlit()
        page, namespace = load_experiments_page(streamlit)
        namespace["EXPERIMENTS_CSV"] = csv_path
        namespace["ACTIVE_DIR"] = Path(temporary.name) / "active"
        namespace["EXPERIMENTS_DIR"] = Path(temporary.name) / "experiments"
        page()
        return streamlit

    def test_missing_columns_show_message_without_key_error(self):
        streamlit = self.run_page(b"experiment\none\n")
        self.assertEqual(len(streamlit.errors), 1)
        self.assertIn("отсутствуют обязательные столбцы", streamlit.errors[0])

    def test_header_only_valid_csv_is_reported_as_empty(self):
        streamlit = self.run_page((",".join(CSV_COLUMNS) + "\n").encode())
        self.assertEqual(streamlit.errors, [])
        self.assertEqual(len(streamlit.infos), 1)
        self.assertIn("пуста", streamlit.infos[0])

    def test_empty_or_malformed_csv_is_handled(self):
        cases = [b"", b"\xff\xfe\xfa"]
        for content in cases:
            with self.subTest(content=content):
                streamlit = self.run_page(content)
                self.assertEqual(len(streamlit.errors), 1)


if __name__ == "__main__":
    unittest.main()
