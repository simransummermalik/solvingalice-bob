"""turn the editable, cell-by-cell source into the notebook, then optionally run it."""

import argparse  # makes --execute an explicit choice
import os  # keeps temporary plotting files inside this project
from pathlib import Path  # handles paths without depending on the shell

import nbformat  # writes real notebook cells, not a renamed python file

root = Path(__file__).resolve().parent  # the notebook should work from this folder
os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".cache" / "jupyter"))
os.environ.setdefault("IPYTHONDIR", str(root / ".cache" / "ipython"))
os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--execute", action="store_true")  # execution can take a while
parser.add_argument("--resume-after", type=int, default=0)  # preserve a completed prefix while developing later tasks
parser.add_argument("--append-after", type=int, default=0)  # source-cell count; retain the existing notebook including manually added installation cells
args = parser.parse_args()
source = (root / "cat_qubits.py").read_text()  # the source keeps diffs easy to review
cells = []  # each # %% marker becomes one notebook cell
for block in source.split("# %%")[1:]:
    kind, body = block.split("\n", 1)  # the first line says whether this is markdown
    if "[markdown]" in kind:
        body = "\n".join(line[2:] if line.startswith("# ") else line[1:] if line == "#" else line for line in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(body.strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
notebook = nbformat.v4.new_notebook(cells=cells)  # no hidden imports from a helper module
notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
notebook.metadata.language_info = {"name": "python", "file_extension": ".py"}
target = root / "cat_qubits.ipynb"
if args.append_after:
    previous = nbformat.read(target, as_version=4)
    comparable = [cell for cell in previous.cells if not cell.source.lstrip().startswith("%pip ")]
    expected = cells[:args.append_after]
    if len(comparable) != len(expected) or any(old.cell_type != new.cell_type or old.source != new.source for old, new in zip(comparable, expected)):
        raise ValueError("the existing notebook differs from the expected source prefix; refusing to replace or duplicate any cells")
    if args.execute:
        raise ValueError("append mode preserves the existing outputs; execute the appended cells in a kernel with their dependencies")
    notebook.cells = [*previous.cells, *cells[args.append_after:]]
    notebook.metadata = previous.metadata
if args.resume_after:
    previous = nbformat.read(target, as_version=4)
    for index in range(min(args.resume_after, len(previous.cells))):
        if previous.cells[index].cell_type == "code":
            notebook.cells[index].outputs = previous.cells[index].get("outputs", [])
            notebook.cells[index].execution_count = previous.cells[index].get("execution_count")
nbformat.write(notebook, target)  # also leaves a usable notebook if a later run fails
if args.execute:
    from nbclient import NotebookClient  # only needed when actually executing

    class SavingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            result = super().process_message(msg, cell, cell_index)
            if msg["msg_type"] == "stream":
                print(msg["content"]["text"], end="", flush=True)  # show progress within long sweep cells too
                nbformat.write(notebook, target)
            return result

    def report(cell, cell_index, **kwargs):
        print(f"cell {cell_index + 1}/{len(cells)}: {cell.source.splitlines()[0][:85]}", flush=True)
        nbformat.write(notebook, target)  # preserve completed outputs between cells

    client = SavingClient(notebook, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(root)}})
    client.on_cell_start = report
    try:
        if args.resume_after:
            with client.setup_kernel():
                for index in range(1, args.resume_after):  # rebuild visible state; outputs alone cannot restore variables
                    if notebook.cells[index].cell_type != "code":
                        continue
                    client.execute_cell(notebook.cells[index], index)
                for index in range(args.resume_after, len(cells)):
                    if notebook.cells[index].cell_type != "code":
                        continue
                    client.execute_cell(notebook.cells[index], index)
        else:
            client.execute()  # raises on the first failing cell; no hidden ignored errors
    finally:
        nbformat.write(notebook, target)
print(f"saved {target}")
