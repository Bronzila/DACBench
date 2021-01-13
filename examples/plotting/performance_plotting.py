from pathlib import Path

from seaborn import plotting_context

from dacbench.logger import load_logs, log2dataframe
from dacbench.plotting import plot_performance_per_instance, plot_performance

import matplotlib.pyplot as plt


def per_instance_example():
    file = Path("data/chainererrl_cma/PerformanceTrackingWrapper.jsonl")
    logs = load_logs(file)
    data = log2dataframe(logs, wide=True, drop_columns=["time"])
    grid = plot_performance_per_instance(
        data, title="CMA Mean Performance per Instance"
    )

    grid.savefig("output/performance_per_instance.pdf")
    plt.show()


def performance_example():
    file = Path("data/sigmoid_example/PerformanceTrackingWrapper.jsonl")
    logs = load_logs(file)
    data = log2dataframe(logs, wide=True, drop_columns=["time"])

    # overall
    grid = plot_performance(data, title="Overall Performance")
    grid.savefig("output/overall_performance.pdf")
    plt.show()

    # per instance seed (hue)
    grid = plot_performance(data, title="Overall Performance", hue="seed")
    grid.savefig("output/overall_performance_per_seed_hue.pdf")
    plt.show()

    # per instance seed (col)
    with plotting_context("poster"):
        grid = plot_performance(
            data, title="Overall Performance", col="seed", col_wrap=3
        )
        grid.fig.subplots_adjust(top=0.92)
        grid.savefig("output/overall_performance_per_seed.pdf")
        plt.show()


if __name__ == "__main__":
    per_instance_example()
    performance_example()
