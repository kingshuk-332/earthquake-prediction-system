from pathlib import Path
import os,sys
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
runtime=ROOT/'cnn/cnn wave/.notebook-runtime'
sys.path.insert(0,str(runtime))
os.environ['PYTHONPATH']=os.pathsep.join([str(OUT/'runtime'),str(runtime)])
os.environ['IPYTHONDIR']=str(OUT/'ipython')
os.environ['JUPYTER_RUNTIME_DIR']=str(OUT/'jupyter_runtime')
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_config')
import nbformat
from nbclient import NotebookClient
p=OUT/'01_final_attention_shap.ipynb'
nb=nbformat.read(p,as_version=4)
def progress(cell,cell_index,**kwargs):
 print('Completed cell',cell_index,flush=True)
 for output in cell.get('outputs',[]):
  if output.get('output_type')=='stream':print(output.get('text',''),flush=True)
 nbformat.write(nb,p)
client=NotebookClient(nb,timeout=7200,resources={'metadata':{'path':str(OUT)}},kernel_name='python3',on_cell_executed=progress)
try:client.execute()
finally:nbformat.write(nb,p)
print('Executed notebook saved.',flush=True)
