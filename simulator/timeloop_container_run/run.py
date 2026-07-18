"""In-container driver for the energy-optimal mapping search.

Runs inside the timeloop-accelergy image (invoked by `make mapping`). Does the
whole flow in Python so there's no shell script:
  1. pip-install the SC accelergy plug-in (sc_mac_inner + rng_bank estimators),
  2. register `sc_mac_inner` as a Timeloop *compute* class (patch the source
     BEFORE importing timeloopfe -- it only treats a fixed set of classes as
     arithmetic),
  3. load the generated v4 config files and run the mapper (which transpiles
     the arch, calls Accelergy to build the ERT with our plug-in, then searches
     the mapspace).

Usage: python3 run.py <config_dir> <out_dir>   (defaults /work/configs /work/out)
"""
import glob
import os
import shutil
import subprocess
import sys

CONFIG_DIR = sys.argv[1] if len(sys.argv) > 1 else "/work/configs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/work/out"
PLUGIN = "/work/plugin"

# 1. install the SC plug-in. Copy it OUT of the host mount first -- pip writes
#    build/ + egg-info/ next to setup.py as the container's root user, which the
#    host user then can't delete. Building in /tmp keeps the mount clean.
shutil.copytree(PLUGIN, "/tmp/plug", dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "/tmp/plug"], check=True)

# 2. register sc_mac_inner as arithmetic. Locate arch.py via sys.path WITHOUT
#    importing pytimeloop (so the edit takes effect on first import below).
for _sp in sys.path:
    _arch = os.path.join(_sp, "pytimeloop/timeloopfe/v4/arch.py")
    if os.path.exists(_arch):
        _s = open(_arch).read()
        if "sc_mac_inner" not in _s:
            _s = _s.replace(
                'COMPUTE_CLASSES = ("mac", "intmac", "fpmac", "compute")',
                'COMPUTE_CLASSES = ("mac", "intmac", "fpmac", "compute", "sc_mac_inner")',
            )
            open(_arch, "w").write(_s)
        break

# 3. load the v4 configs + run the mapper.
from pytimeloop.timeloopfe.v4 import Specification
from pytimeloop.timeloopfe.common.backend_calls import call_mapper

files = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
os.makedirs(OUT, exist_ok=True)
print("configs:", ", ".join(os.path.basename(f) for f in files))
try:
    spec = Specification.from_yaml_files(*files)
    call_mapper(spec, output_dir=OUT)
    print("MAPPING_OK ->", OUT)
except Exception as e:
    with open(os.path.join(OUT, "err.txt"), "w") as f:
        f.write(str(e))
    print("ERROR:", type(e).__name__, "(see out/err.txt)")
    sys.exit(1)
