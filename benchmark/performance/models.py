"""The fitted models of the performance benchmark: one fixed recipe for each, fitted once and saved with skops.

Each recipe has a seed, so two machines fit the same model. The first run fits each model and saves it in the model
directory. Every later run loads the saved file, so two runs of the benchmark, and the base and the head of a pull
request, measure the same model. A saved model holds the EBM in ``<name>.skops`` and its training data, with the
target in the column ``target``, in ``<name>.parquet``.
"""

from collections.abc import Callable, Mapping
from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
import polars as pl
import skops.io
from sklearn.datasets import fetch_openml, load_diabetes, make_classification

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor

# The default directory of the saved models. Git ignores it. Raise RECIPE_VERSION when a recipe changes, so that a run
# fits the models again instead of reading the files of the old recipe.
RECIPE_VERSION = 1
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / ".models"
TARGET = "target"
SEED = 0
# skops loads only the types that the caller names, as ebaml.mcp.model_store does
_TRUSTED_TYPES = [
    f"{cls.__module__}.{cls.__qualname__}" for cls in (ExplainableBoostingClassifier, ExplainableBoostingRegressor)
]


@attrs.frozen
class TrainingSet:
    """The training data of a model.

    Attributes
    ----------
    features: pl.DataFrame
        One column for each feature.
    target: npt.NDArray[np.generic]
        The class or the value of each row.

    """

    features: pl.DataFrame
    target: npt.NDArray[np.generic]


@attrs.frozen
class FittedModel:
    """A fitted model of the benchmark, with its training data.

    Attributes
    ----------
    name: str
        The name of the recipe.
    ebm: ExplainableBoostingModel
        The fitted EBM.
    data: TrainingSet
        The rows that the EBM was fitted on.

    """

    name: str
    ebm: ExplainableBoostingModel
    data: TrainingSet

    def instance(self, row: int = 0) -> dict[str, float | str]:
        """Read one row of the training data as the instance of a local explanation.

        Parameters
        ----------
        row: int
            The number of the row.

        Returns
        -------
        dict[str, float | str]
            The value of each feature, by feature name.

        """
        return dict(self.data.features.row(row, named=True))


@attrs.frozen
class Recipe:
    """How to build one model of the benchmark.

    Attributes
    ----------
    data: Callable[[], TrainingSet]
        Builds the training data.
    estimator: Callable[[], ExplainableBoostingModel]
        Builds the EBM before the fit.

    """

    data: Callable[[], TrainingSet]
    estimator: Callable[[], ExplainableBoostingModel]


def _titanic() -> TrainingSet:
    """Read the Titanic data of the notebooks: seven features and the survival of each passenger.

    Returns
    -------
    TrainingSet
        The passengers with no missing value.

    """
    frame = pl.from_pandas(fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff").frame)
    columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    frame = frame.select([*columns, "survived"]).drop_nulls()
    features = frame.select(columns).with_columns(pl.col("sex", "embarked").cast(pl.String))
    return TrainingSet(features, frame["survived"].cast(pl.String).cast(pl.Int8).to_numpy())


def _diabetes() -> TrainingSet:
    """Read the diabetes data of scikit-learn: ten features and a measure of the disease after one year.

    Returns
    -------
    TrainingSet
        The 442 patients.

    """
    bunch = load_diabetes(as_frame=True)
    return TrainingSet(pl.from_pandas(bunch.data), bunch.target.to_numpy())


def _synthetic(feature_count: int, class_count: int = 2) -> Callable[[], TrainingSet]:
    """Make the recipe of a synthetic classification data set.

    Parameters
    ----------
    feature_count: int
        The number of continuous features.
    class_count: int
        The number of classes.

    Returns
    -------
    Callable[[], TrainingSet]
        Builds 4000 rows. Each feature has more than 1024 distinct values, so a model with 1024 bins uses them all.

    """

    def build() -> TrainingSet:
        informative = min(feature_count, 8)
        features, target = make_classification(
            n_samples=4000,
            n_features=feature_count,
            n_informative=informative,
            n_redundant=0,
            n_classes=class_count,
            n_clusters_per_class=1,
            random_state=SEED,
        )
        names = [f"x{feature_index:03d}" for feature_index in range(feature_count)]
        return TrainingSet(pl.DataFrame(features, schema=names), target)

    return build


def _categorical() -> TrainingSet:
    """Make a classification data set with categorical features of many levels.

    Returns
    -------
    TrainingSet
        4000 rows: three categorical features with 50, 200 and 500 levels, and three continuous features.

    """
    random_generator = np.random.default_rng(SEED)
    row_count = 4000
    columns: dict[str, npt.NDArray[np.generic]] = {}
    signal = np.zeros(row_count)
    for level_count in (50, 200, 500):
        codes = random_generator.integers(0, level_count, size=row_count)
        signal += random_generator.normal(size=level_count)[codes]
        columns[f"levels_{level_count}"] = np.char.add("level_", codes.astype(str))
    for number in range(3):
        values = random_generator.normal(size=row_count)
        signal += values
        columns[f"number_{number}"] = values
    target = (signal + random_generator.normal(size=row_count) > 0).astype(np.int8)
    return TrainingSet(pl.DataFrame(columns), target)


def _classifier(*, max_bins: int = 1024, outer_bags: int = 14) -> Callable[[], ExplainableBoostingModel]:
    return lambda: ExplainableBoostingClassifier(
        interactions=0,
        max_bins=max_bins,
        outer_bags=outer_bags,
        random_state=SEED,
    )


def _regressor() -> ExplainableBoostingModel:
    return ExplainableBoostingRegressor(interactions=0, random_state=SEED)


def _size_recipes() -> dict[str, Recipe]:
    """Make the synthetic models that stress size: 10, 100 and 500 features, with 32 and 1024 bins.

    Returns
    -------
    dict[str, Recipe]
        The recipe of each model, by name. Two outer bags keep the fit short: the benchmark measures EBaMl, not the
        fit.

    """
    return {
        f"features-{feature_count}-bins-{max_bins}": Recipe(
            _synthetic(feature_count),
            _classifier(max_bins=max_bins, outer_bags=2),
        )
        for feature_count in (10, 100, 500)
        for max_bins in (32, 1024)
    }


# Each model of the benchmark. A new model is a new row.
RECIPES: Mapping[str, Recipe] = {
    "titanic": Recipe(_titanic, _classifier()),
    "diabetes": Recipe(_diabetes, _regressor),
    "multiclass": Recipe(_synthetic(8, class_count=5), _classifier(max_bins=32, outer_bags=2)),
    "categorical": Recipe(_categorical, _classifier(outer_bags=2)),
    **_size_recipes(),
}


def _paths(name: str, model_dir: Path) -> tuple[Path, Path]:
    folder = model_dir / f"v{RECIPE_VERSION}"
    return folder / f"{name}.skops", folder / f"{name}.parquet"


def build_model(name: str, model_dir: Path) -> None:
    """Fit a model of the benchmark and save it, with its training data.

    Parameters
    ----------
    name: str
        The name of the recipe.
    model_dir: Path
        The directory of the saved models.

    """
    recipe = RECIPES[name]
    data = recipe.data()
    ebm = recipe.estimator().fit(data.features.to_pandas(), data.target)
    model_path, data_path = _paths(name, model_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    skops.io.dump(ebm, model_path)
    data.features.with_columns(pl.Series(TARGET, data.target)).write_parquet(data_path)


def load_model(name: str, model_dir: Path) -> FittedModel:
    """Load a saved model of the benchmark, and fit it first if no file holds it.

    Parameters
    ----------
    name: str
        The name of the recipe.
    model_dir: Path
        The directory of the saved models.

    Returns
    -------
    FittedModel
        The EBM and its training data.

    """
    model_path, data_path = _paths(name, model_dir)
    if not (model_path.exists() and data_path.exists()):
        build_model(name, model_dir)
    ebm = skops.io.load(model_path, trusted=_TRUSTED_TYPES)
    frame = pl.read_parquet(data_path)
    data = TrainingSet(frame.drop(TARGET), frame[TARGET].to_numpy())
    return FittedModel(name, ebm, data)
