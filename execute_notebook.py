"""Execute the project notebook while skipping environment/server cells."""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


SOURCE = Path("HW_transformers.ipynb")
OUTPUT = Path("HW_transformers_executed.ipynb")


class ProgressExecutePreprocessor(ExecutePreprocessor):
    def preprocess_cell(self, cell, resources, index):
        print(f"Executing cell {index + 1}/{len(self.nb.cells)}", flush=True)
        return super().preprocess_cell(cell, resources, index)


notebook = nbformat.read(SOURCE, as_version=4)
skip_markers = ("pip install", "!pip install", "%pip install", "!uvicorn")

for cell in notebook.cells:
    if cell.cell_type == "code" and cell.source.strip().startswith(skip_markers):
        cell.source = "# Skipped during automated execution."

executor = ProgressExecutePreprocessor(timeout=1800, kernel_name="python3")
executor.preprocess(notebook, {"metadata": {"path": str(Path.cwd())}})
nbformat.write(notebook, OUTPUT)
print(f"Saved executed notebook to {OUTPUT}", flush=True)
