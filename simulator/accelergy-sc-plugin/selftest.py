"""Stand-alone sanity check for the SC Accelergy estimators.

Stubs the Accelergy base class so this runs with no Accelergy install, and
prints the energies the plug-in reports (the same numbers Accelergy would put
in the ERT). Useful before a full container run. Run: `python3 selftest.py`.
"""
import importlib
import logging
import sys
import types

# --- stub `accelergy.plug_in_interface.estimator` so the estimators import ---
_est = types.ModuleType("accelergy.plug_in_interface.estimator")
_est.Estimator = type("Estimator", (), {"logger": logging.getLogger("selftest")})
_est.actionDynamicEnergy = lambda f: f
_est.add_estimator_path = lambda *a, **k: None
_est.remove_estimator_path = lambda *a, **k: None
for _n in ("accelergy", "accelergy.plug_in_interface"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["accelergy.plug_in_interface.estimator"] = _est

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
sc_mac = importlib.import_module("sc_mac")
rng_bank = importlib.import_module("rng_bank")

inner = sc_mac.SCMacInner(technology="45nm", mag_bits=7, datawidth=8)
sng = rng_bank.RngBank(technology="45nm", width=8, stream_length=128)

comp = inner.compute()
gen = sng.read()
print(f"sc_mac_inner.compute  = {comp * 1e12:7.3f} pJ/MAC   (inner, measured)")
print(f"sc_mac_inner.gated    = {inner.gated_compute() * 1e12:7.3f} pJ")
print(f"sc_mac_inner.leak@1ns = {inner.leak(1e-9) * 1e12:7.3f} pJ")
print(f"rng_bank.read (SNG)   = {gen * 1e12:7.3f} pJ/operand   (outer delta)")
print(f"outer PE = inner+delta = {(comp + gen) * 1e12:7.3f} pJ  ({(comp + gen) / comp:.2f}x inner)")

assert comp > 0 and gen > 0, "estimators returned non-positive energy"
print("OK: estimators load and return finite positive energies.")
