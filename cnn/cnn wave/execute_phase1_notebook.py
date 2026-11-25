"""Execute the notebook in a fresh Jupyter kernel; save outputs after each cell."""
from pathlib import Path
import os
import sys

BASE=Path(__file__).resolve().parent
runtime=BASE/'.notebook-runtime'
if runtime.exists():
    sys.path.insert(0,str(runtime))
    os.environ['PYTHONPATH']=str(runtime)+os.pathsep+os.environ.get('PYTHONPATH','')
for key,folder in [('JUPYTER_CONFIG_DIR','config'),('JUPYTER_DATA_DIR','data'),('JUPYTER_RUNTIME_DIR','runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib')]:
    p=BASE/'.notebook-state'/folder
    p.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(p)
import nbformat
from nbclient import NotebookClient

path=BASE/'phase1_cnn_forecasting.ipynb'
nb=nbformat.read(path,as_version=4)
nbformat.validate(nb)
def saved(cell,cell_index,**kwargs):
    nbformat.write(nb,path)
    print(f'Executed cell {cell_index+1}/{len(nb.cells)}',flush=True)
client=NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(BASE)}},on_cell_executed=saved)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
nbformat.validate(nb)
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code')
print('VALIDATED: all code cells executed without errors',flush=True)
