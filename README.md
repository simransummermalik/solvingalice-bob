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

the final methods comparison collects the attempted control designs, including infeasible and unselected candidates. it also compares four fixed classical receivers on the same saved noisy records. that section starts independently after a restart and runs without quantum solvers. keep `artifacts/method_receiver_scores.npz` and `artifacts/method_receiver_provenance.json` with the notebook; they contain the fixed record projections and their provenance. the receiver analysis reuses existing evaluation records and is exploratory, not new independent validation.

`fails.md` keeps the simple scratch record of ideas that were tested and left out, including the geometric recovery and counterdiabatic corrections. it is useful when explaining why the final pulse was selected.

the last notebook section adds a phase-locking audit: it measures the transverse readout excursion and sweeps the closed-pulse strength. in the ideal two-mode model it finds no sampled stability breakdown, so the result is kept as a negative boundary test rather than an overclaim.

the next section tests a different capstone strategy: smoothly increase the buffer stabilization drive during the selected memory readout pulse, then restore it. this temporarily expands the cat while the readout is happening. the comparison uses the same mean-signal information, simulates both stored bits, checks multiple memory/buffer cutoffs and a resonant circuit correction, and measures damage over repeated cycles. the exploratory tables, settings, plot and limitations are in the notebook and `artifacts/pump_breathing_*` files.

the final checks use paired noisy homodyne records and a separate buffer-kerr/cross-kerr/dephasing stress model. the ideal-model damage reduction is much larger than in the nonlinear stress model; the receiver error difference is not statistically resolved. the notebook reports both instead of treating the ideal-model number as a hardware prediction.

the cross-kerr-aware section adds a synchronized buffer-frequency correction to the temporary cat expansion. at the stronger `chi/2pi = 0.30 MHz` stress value, the combined control reaches the information target at both `32×8` and `36×10`, beats the matched flat baselines on endpoint damage, and reduces the extra repeated-readout decay rate by about 89% relative to the original pulse. the control has a calibration boundary and remains a numerical proposal.
