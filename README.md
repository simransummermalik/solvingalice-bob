cat qubits, from loss to readout

open `cat_qubits.ipynb`: it contains all the code, explanations, plots, and results. it runs independently of the authoring scripts. select the project's `.venv` python interpreter in your notebook editor, then run the cells in order.

run with the project's python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

the `.py` files are authoring/testing support, not the submission or dependencies of the notebook. the executed notebook is the deliverable. the small tables and figures in `artifacts/` are also embedded or displayed in it. optional matrix-keyed caches in `artifacts/checkpoints/` speed up repeat runs; removing that generated cache forces fresh integration.

the notebook follows introductory tasks 1.0–1.3, then standard tasks 2.1–2.4. time is in microseconds. every rate is converted from the prompt using 2π. the reported x/z rates are expectation-value decay rates, not single-event rates.

this is a research calculation, not a claim of experimental performance or guaranteed originality. numerical and physical limitations are discussed alongside the results.

the final addendum retains the earlier sections and adds amplitude recalibration with circuit corrections, repeated-measurement bit-decay rates, larger statistical checks, and a longer readout experiment. its computed interpretation is also saved in `artifacts/audit_findings.md` and `artifacts/audit_summary.json`. the earlier `artifacts/summary.json` remains the summary of the original resonant-pulse comparison.

after restarting the kernel, you can start directly at the first code cell under `additions after reviewing the first results`. leave `audit_load_saved = True` to display the saved addendum results without running earlier tasks or any simulations. setting it to `False` restores the model setup and frozen controls for recomputation; compatible checkpoints are reused, but missing or incompatible checkpoints require new solves.
