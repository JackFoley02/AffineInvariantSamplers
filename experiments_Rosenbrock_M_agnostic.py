"""Run the Rosenbrock benchmark with a geometry-agnostic initial ensemble.

Unlike experiments_Rosenbrock_M.py's default oracle initialization, this
entry point starts every coordinate independently from N(0, 2^2).  It does
not encode the Rosenbrock valley y ~= x**2.  Results are written beneath
RosenbrockResultsM_agnostic/ and tagged initialization_mode="agnostic".

All command-line arguments accepted by experiments_Rosenbrock_M.py are
forwarded unchanged.  Example:

    python experiments_Rosenbrock_M_agnostic.py --dim 8 --af --cond 50 \
        --n-warmup 2000 --burn-in 2000 --no-report
"""

import os
import runpy
from pathlib import Path


os.environ["ROSENBROCK_INITIALIZATION"] = "agnostic"
runpy.run_path(
    str(Path(__file__).with_name("experiments_Rosenbrock_M.py")),
    run_name="__main__",
)
