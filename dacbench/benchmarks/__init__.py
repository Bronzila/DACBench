# flake8: noqa: F401
import importlib
import warnings

from dacbench.benchmarks.fast_downward_benchmark import FastDownwardBenchmark
from dacbench.benchmarks.geometric_benchmark import GeometricBenchmark
from dacbench.benchmarks.luby_benchmark import LubyBenchmark
from dacbench.benchmarks.sigmoid_benchmark import SigmoidBenchmark
from dacbench.benchmarks.toysgd_benchmark import ToySGDBenchmark
from dacbench.benchmarks.toysgd_2D_benchmark import ToySGD2DBenchmark
from dacbench.benchmarks.layerwise_sgd_benchmark import LayerwiseSGDBenchmark

__all__ = [
    "LubyBenchmark",
    "SigmoidBenchmark",
    "ToySGDBenchmark",
    "ToySGD2DBenchmark",
    "GeometricBenchmark",
    "FastDownwardBenchmark",
    "LayerwiseSGDBenchmark",
]

modcma_spec = importlib.util.find_spec("modcma")
found = modcma_spec is not None
if found:
    from dacbench.benchmarks.cma_benchmark import CMAESBenchmark

    __all__.append("CMAESBenchmark")
else:
    warnings.warn(  # noqa: B028
        "CMA-ES Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )

sgd_spec = importlib.util.find_spec("backpack")
found = sgd_spec is not None
if found:
    from dacbench.benchmarks.sgd_benchmark import SGDBenchmark

    __all__.append("SGDBenchmark")
else:
    warnings.warn(  # noqa: B028
        "SGD Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )

theory_spec = importlib.util.find_spec("uuid")
found = theory_spec is not None
if found:
    from dacbench.benchmarks.theory_benchmark import TheoryBenchmark

    __all__.append("TheoryBenchmark")
else:
    warnings.warn(  # noqa: B028
        "Theory Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )
