"""Run the Gaussian warmup-length sweep with NUTS and HWM only.

Results use the same condition/warmup/seed hierarchy as
``experiments_Rosenbrock_W.py``::

    GaussianResultsW/<condition>c/<warmup>w/seeds/seed_<seed>/

All ordinary Gaussian benchmark options are accepted.  In particular,
``--output-dir`` can override the default ``GaussianResultsW`` root.
"""

from pathlib import Path
import runpy
import sys


gaussian_script = Path(__file__).with_name("experiments_Gaussian.py")
sys.argv = [
    str(gaussian_script),
    "--warmup-sweep",
    "--output-dir",
    "GaussianResultsW",
    *sys.argv[1:],
]
runpy.run_path(str(gaussian_script), run_name="__main__")
