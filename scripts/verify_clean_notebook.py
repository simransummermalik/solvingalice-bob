"""run the notebook in an isolated copy with the interpreter invoking this script."""

import argparse  # a fresh virtual environment can invoke this runner directly
import hashlib  # record the exact notebook sources that were tested
import importlib.metadata as metadata  # package versions belong in the evidence
import json  # emit a small machine-readable run record
import os  # keep runtime files inside the isolated directory
from pathlib import Path  # resolve the repository from the script location
import platform  # record the tested Python version
import shutil  # copy inputs so test runs do not overwrite submitted results
import sys  # the kernel uses this interpreter, including a new virtual environment
import tempfile  # use an explicitly scoped temporary workspace
import time  # report execution duration

import nbformat  # preserve valid notebook structure
import numpy as np  # check that all frozen trajectory samples are present
import pandas as pd  # distinguish complete audit inputs from partial progress tables
from nbclient import NotebookClient  # execute real cells in a fresh kernel
from jupyter_client import KernelManager  # select an explicit kernel
from jupyter_client.kernelspec import KernelSpecManager  # avoid global kernel registration

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start-cell", type=int, default=0)
parser.add_argument("--output-name", default="clean_environment_validation")
parser.add_argument("--wait-for-final-audit", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root/"cat_qubits.ipynb", as_version=4)
record = {"python": platform.python_version(), "interpreter": sys.executable,
          "start_cell": args.start_cell, "total_cells": len(notebook.cells),
          "versions": {name: metadata.version(name) for name in
                       ["dynamiqs", "jax", "jaxlib", "qutip", "numpy", "scipy", "matplotlib", "pandas", "nbformat", "nbclient", "ipykernel"]},
          "checked_cells": [], "success": False}
started = time.monotonic()
with tempfile.TemporaryDirectory(prefix="cat-notebook-check-") as directory:
    work = Path(directory)
    shutil.copy2(root/"requirements.txt", work/"requirements.txt")
    shutil.copytree(root/"artifacts", work/"artifacts",
                    ignore=shutil.ignore_patterns("final-audit-*", "clean_environment_validation*", "clean_prefix_validation*"))
    for name, value in {"MPLCONFIGDIR": str(work/"matplotlib"),
                        "IPYTHONDIR": str(work/"ipython"),
                        "JUPYTER_RUNTIME_DIR": str(work/"runtime"),
                        "XDG_CACHE_HOME": str(work/"cache")}.items():
        os.environ[name] = value
    kernel_directory = work/"kernels"/"submission"
    kernel_directory.mkdir(parents=True)
    (kernel_directory/"kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "isolated submission check", "language": "python"}))
    kernel_manager = KernelManager(kernel_name="submission",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(work/"kernels")]))
    checked = nbformat.v4.new_notebook(cells=notebook.cells[args.start_cell:])
    client = NotebookClient(checked, km=kernel_manager, timeout=1800,
                            resources={"metadata": {"path": str(work)}})

    def refresh_final_inputs(cell, cell_index, **kwargs):
        if not args.wait_for_final_audit or cell.id != "final-validation-engine":
            return
        waiting = time.monotonic()
        while True:
            ready = False
            try:
                rates = pd.read_csv(root/"artifacts"/"final_validation_protection.csv")
                records = pd.read_csv(root/"artifacts"/"final_validation_receiver.csv")
                ready = len(rates) == 9 and len(records) == 12 and "ensemble check passed" in records
                for condition, size in [("main", 1024), ("half_step", 256), ("larger_cutoff", 256)]:
                    for protocol in ["joint", "flat"]:
                        path = root/"artifacts"/f"final_validation_scores_{condition}_{protocol}.npz"
                        with np.load(path, allow_pickle=False) as payload:
                            ready = ready and len(payload["signs"]) == size
            except (OSError, ValueError, EOFError, KeyError):
                ready = False
            if ready:
                break
            if time.monotonic()-waiting > 7200:
                raise TimeoutError("the final audit inputs did not finish within two hours")
            print("waiting for complete final audit inputs before testing their notebook section", flush=True)
            time.sleep(30)
        for path in (root/"artifacts").glob("final_validation*"):
            if path.is_file():
                shutil.copy2(path, work/"artifacts"/path.name)
        print("refreshed completed final audit inputs; continuing in the same clean kernel", flush=True)

    def completed(cell, cell_index, **kwargs):
        index = args.start_cell+cell_index
        record["checked_cells"].append({"index": index, "id": cell.id,
            "source_sha256": hashlib.sha256(cell.source.encode()).hexdigest()})
        print(f"checked cell {index+1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:70]}", flush=True)

    client.on_cell_executed = completed
    client.on_cell_start = refresh_final_inputs
    try:
        client.execute()
        record["success"] = True
    except Exception as error:
        record["error"] = repr(error)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic()-started
        (root/"artifacts"/f"{args.output_name}.json").write_text(json.dumps(record, indent=2)+"\n")
        nbformat.write(checked, root/"artifacts"/f"{args.output_name}.ipynb")
        print(json.dumps({key: record[key] for key in ["success", "start_cell", "total_cells", "elapsed_seconds"]}), flush=True)
