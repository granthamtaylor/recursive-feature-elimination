
from pathlib import Path
import dataclasses
from functools import partial
import json

from mashumaro.mixins.json import DataClassJSONMixin
import flytekit as fk
import pandas as pd
import polars as pl
import plotly
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import fetch_california_housing


version = "0.0.4"

image = fk.ImageSpec(
    builder="unionai",
    packages=[
        "pandas",
        "scikit-learn",
        "polars",
        "plotly",
        "flytekitplugins-deck-standard",
    ],
)


image = "us-central1-docker.pkg.dev/uc-serverless-production/orgs/granthamtaylor/flytekit:xdYAO4A12ZENjcTDX08KsQ"


@dataclasses.dataclass
class Result(DataClassJSONMixin):
    
    """Serializable model pruning result
    """
    
    name: str
    value: float

@dataclasses.dataclass
class ResultHistory(DataClassJSONMixin):
    
    """Serializable collection of model pruning iteration results
    """
    
    results: list[list[Result]]
    
    @classmethod
    def new(cls) -> "ResultHistory":
        
        return cls(results=[])

@fk.task(container_image=image)
def train(encoded: str, prune: str, dataset: fk.types.file.FlyteFile) -> Result:
    """Train a model iteration given a list of features, a single feature to exclude, and a dataset

    Args:
        encoded (str): list of all remaining features
        prune (str): name of feature to exclude
        dataset (fk.types.file.FlyteFile): _description_

    Returns:
        Result: model result (includes score and name of pruned feature)
    """
    
    features = json.loads(encoded)
    
    features = [feature for feature in features if feature != prune]
    
    df = pd.read_parquet(dataset)

    fold = KFold(n_splits=5, random_state=42, shuffle=True)

    model = LinearRegression()

    scores = cross_val_score(estimator=model, X=df[features], y=df["_targets"], cv=fold, scoring='r2')

    result = Result(prune, scores.mean())

    return result


def blame(results: list[Result]) -> str:
    """Find the least important feature from a list of iteration results

    Args:
        results (list[Result]): iteration results

    Returns:
        str: name of least important feature
    """
    
    pairs: dict[str, float] = {result.name: result.value for result in results}
    
    loser = max(pairs, key=lambda key: pairs[key])
    
    return loser

@fk.task(container_image=image)
def prune(results: list[Result]) -> list[str]:
    """Take a list of iteration results and remove the least important feature from the remaining features, returning only the most important ones.

    Args:
        results (list[Result]): iteration results

    Returns:
        list[str]: remaining features after pruning least important one
    """
    
    pairs: dict[str, float] = {result.name: result.value for result in results}

    loser = blame(results)

    del pairs[loser]
    
    return list(pairs.keys())

@fk.task(container_image=image, enable_deck=True)
def plot(history: ResultHistory) -> None:
    
    """Plot history of feature pruning iterations
    """
    
    records = []
    
    for iteration, results in enumerate(history.results):
        
        loser = blame(results)
        
        for result in results:
            records.append(dict(
                iteration=iteration,
                name=result.name,
                value=result.value,
                is_loser=result.name == loser
            ))
            
    df = (
        pl.from_records(records)
        .select(
            pl.exclude('iteration'),
            iteration = pl.col('iteration').max() - pl.col('iteration') + 1
        )
    )

    models = px.scatter(df.to_pandas(), y='iteration', x='value', color='name')
    models.update_traces(marker=dict(size=16, line=dict(width=1, color="white")))

    annotations = px.scatter(
        df.filter(pl.col('is_loser')).to_pandas(),
        y='iteration',
        x='value',
        text='name'
    )

    annotations.update_traces(marker=dict(size=24, symbol="x-thin", line=dict(width=3, color="black")))

    annotations.update_traces(textposition='middle right')

    best = px.line(
        (
            df.group_by('iteration')
            .agg(pl.max('value'))
            .sort('iteration')
            .to_pandas()
        ),
        y='iteration',
        x='value',
        line_shape='spline',
    )

    best.update_traces(line=dict(color='lightgray', width=12))

    iterations = px.line(
        (
            df.group_by('iteration')
            .agg(
                min=pl.col('value').min(),
                max=pl.col('value').max()
            )
            .melt(id_vars='iteration')
            .select(
                pl.all(),
                best=pl.col('value').max().over('iteration')
            )
            .to_pandas()
        ),
        x='value',
        y='iteration',
        line_group='iteration'
    )

    iterations.update_traces(line=dict(color='lightgray', width=6, dash='dot'))

    fig = go.Figure(data = iterations.data + best.data + annotations.data + models.data)

    fig.update_layout(
        template="simple_white",
        title="Recursively Pruning Features Comes with Increasing Cost to Model Performance",
        xaxis_title="Model Performance (R²)",
        yaxis_title="Count of Remaining Features",
        legend_title="Feature Names",
    )

    fk.Deck("Result History", plotly.io.to_html(fig))


@fk.task(container_image=image)
def append(history: ResultHistory, results: list[Result]) -> ResultHistory:
    """Append recent model pruning iteration results to collection of previous model pruning iteration results

    Returns:
        ResultHistory: combined result history
    """
    
    return ResultHistory(*[history.results + [results]])

@fk.task(container_image=image)
def encode(features: list[str]) -> str:
    """Encode list of features to JSON

    Returns:
        str: encoded list of features
    """

    return json.dumps(features)


@fk.task(container_image=image)
def fetch_dataset() -> fk.types.file.FlyteFile:
    
    path = Path(fk.current_context().working_directory) / "model.onnx"
    
    x, y = fetch_california_housing(as_frame=True, return_X_y=True)
    
    columns = list(x.columns)

    x = pd.DataFrame(x, columns=columns)

    x['_targets'] = y
    
    x.to_parquet(path=path)
    
    return fk.types.file.FlyteFile(path)


@fk.task(container_image=image)
def get_features(dataset: fk.types.file.FlyteFile) -> list[str]:
    
    df = pd.read_parquet(dataset)
    
    columns = list(df.columns)
    
    return [column for column in columns if column != '_targets']
    

@fk.dynamic(container_image=image)
def recursively_prune_features(depth: int, features: list[str], dataset: fk.types.file.FlyteFile) -> ResultHistory:
    
    assert depth < len(features)
    
    history = ResultHistory.new()
    
    for _ in range(depth):
        
        encoded = encode(features=features)
        
        results = fk.map_task(partial(train, encoded=encoded, dataset=dataset), concurrency=0)(prune=features)
        
        features = prune(results=results)

        history = append(history=history, results=results)
    
    return history

@fk.task(container_image=image)
def calculate_recursion_depth(features: list[str]) -> int:
    return len(features) - 1

@fk.workflow
def workflow():
    
    dataset = fetch_dataset()
    
    features = get_features(dataset=dataset)
    
    depth = calculate_recursion_depth(features=features)
    
    history = recursively_prune_features(depth=depth, features=features, dataset=dataset)
    
    plot(history=history)
