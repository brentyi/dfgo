"""Load performance data, compute metrics across folds, and print statistics."""

import dataclasses
import pathlib
from typing import Dict, List, Optional, Tuple

import beautifultable
import numpy as onp
import termcolor

from lib import experiment_files, utils

MetricDict = Dict[str, float]


@dataclasses.dataclass
class Args:
    experiment_paths: Optional[Tuple[str, ...]]
    disaggregate: bool = False


def get_valid_experiments(experiment_paths: Optional[Tuple[str, ...]]) -> List[str]:
    """Given a set of paths, return a list of experiment names. By default, returns all
    experiment names.

    For example an input of:
        (
            "./experiments/kitti/fg/hetero/something/fold_0",
            "./experiments/kitti/fg/hetero/something/fold_1",
            "./experiments/kitti/fg/hetero/something/fold_2",
        )
    Would output:
        ["kitti/fg/hetro/something"]
    """

    experiment_root = pathlib.Path("./experiments/")
    folders: List[pathlib.Path] = []
    if experiment_paths is None:
        for pattern in ("**/fg/**/", "**/ekf/**/", "**/lstm/**/"):
            folders += experiment_root.glob(pattern)
    else:
        folders.extend(map(pathlib.Path, experiment_paths))

    output = set()
    for f in folders:
        f_str = f.relative_to(experiment_root).as_posix()
        experiment, fold_, fold_number = f_str.partition("/fold_")
        if len(fold_number) == 0:
            continue
        output.add(experiment)
    return sorted(output)


def convert_radians_to_degrees(metrics: MetricDict) -> MetricDict:
    """Convert metrics in radians to degrees, which are easier to read. Hacky.

    Example input:
        {'m_per_m': 1.0, 'rad_per_m': 3.14159265}

    Output:
        {'m_per_m': 1.0, 'deg_per_m': 180.0}
    """
    out = {}
    for k, v in metrics.items():
        if "rad" in k:
            k = k.replace("rad", "deg")
            v = v * 180.0 / onp.pi
        out[k] = v
    return out


def mean_and_std_error(values: List[float]) -> str:
    """Compute a mean and standard error given a set of floats. Format and return as a
    string."""
    return f"{onp.mean(values):.4f} ± {onp.std(values) / len(values):.4f}"


def build_table(
    metric_names: Tuple[str, ...], rows: List[List[str]]
) -> beautifultable.BeautifulTable:
    # Build header
    header: List[str] = list(
        map(
            lambda s: termcolor.colored(s, attrs=["bold"]),
            [
                "Experiment",
                "F0",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "F9",
                *metric_names,
            ],
        )
    )

    # Only include per-fold data if --disaggregate parameter is passed in
    if not args.disaggregate:
        header = header[:1] + header[11:]
        for i, row in enumerate(rows):
            rows[i] = rows[i][:1] + rows[i][11:]

    # Build and print results table
    table = beautifultable.BeautifulTable(maxwidth=300)
    table.set_style(beautifultable.STYLE_BOX_ROUNDED)
    table.columns.header = header
    for row in rows:
        table.rows.append(row)
    table.columns.alignment = beautifultable.ALIGN_LEFT
    return table


def main(args: Args) -> None:
    # Find experiments
    experiment_names = get_valid_experiments(args.experiment_paths)

    # Add row for each experiment
    MetricNames = Tuple[str, ...]
    rows_from_metric_names: Dict[MetricNames, List[List[str]]] = {}
    for experiment_name in experiment_names:
        metric_list: Dict[str, List[float]] = {}

        row: List[str] = [experiment_name]

        num_folds = 10
        for fold in range(num_folds):
            # Read evaluation metrics
            experiment = experiment_files.ExperimentFiles(
                identifier=f"{experiment_name}/fold_{fold}",
                verbose=False,
            )
            try:
                # Load metrics
                metrics: MetricDict = experiment.read_metadata("best_val_metrics", dict)
                metrics = convert_radians_to_degrees(metrics)
                # metric_list.append(metrics)
                for k, v in metrics.items():
                    if k not in metric_list:
                        metric_list[k] = []
                    metric_list[k].append(v)

                row.append(" / ".join(map(lambda x: f"{x:.4f}", metrics.values())))
            except FileNotFoundError:
                print(f"Missing metrics for {experiment_name}, fold #{fold}")
                row.append("")

        row.extend(map(mean_and_std_error, metric_list.values()))

        metric_names = tuple(metric_list.keys())
        if metric_names not in rows_from_metric_names:
            rows_from_metric_names[metric_names] = []
        rows_from_metric_names[metric_names].append(row)

    # Build a table for each set of unique metric types
    for metric_names, rows in rows_from_metric_names.items():
        print("Statistics for metrics: ", metric_names)
        print(build_table(metric_names, rows))
        print()


if __name__ == "__main__":
    args = utils.parse_args(Args)
    main(args)
