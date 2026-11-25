from pathlib import Path
import json
import nbformat
from PIL import Image
OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'validation_results.json').read_text())
note='''## Numerical interpretation restrictions

The detection reconstruction check is **not fully passed**: 4 of 12 records exceed the predeclared tolerance of 0.1 + 0.1*abs(output minus expected output). Its maximum absolute residual is 0.622584 logits. Detection plots are therefore approximate descriptive attributions; they must not be used as precise local additive decompositions or exact probability explanations. Residuals are displayed explicitly in the local plots and retained for every record in the NPZ files. No extra sampling or explainer tuning was performed after these results.

Changing the 48-record training background shifts the detection expected logit from -0.762527 to 1.211380 (a 1.973908-logit shift), despite the same class/magnitude stratum counts. This is material reference-distribution sensitivity, even though aggregate feature rank correlation is 0.893 and the minimum local signed-attribution cosine is 0.930. Stable feature rankings do not establish a stable reference value or exact local explanation. The independent Monte Carlo repeat has feature rank correlation 1.000 and minimum local cosine 0.991.

Magnitude passes the reconstruction tolerance for all 9 records (maximum absolute residual 0.024013 magnitude units). Its expected output changes from 3.853153 to 3.861201 across backgrounds; feature rank correlation is 0.964, with minimum local cosine 0.887. The independent repeat has feature rank correlation 1.000 and minimum local cosine 0.995. These results support approximate descriptions of these selected model outputs, subject to the paired-input dependency limitations. The magnitude reference includes noise model outputs and has no interpretation as an average physical earthquake magnitude.

**CONDITIONAL** applies to the experiment as a whole: genuine SHAP expected gradients were obtained, but detection numerical reconstruction, background sensitivity, dependent inputs, and the small stratified cohort prevent an unqualified approval. Attention weights remain separate diagnostics, never SHAP values.
'''
for name in ['shap_explainability_report.md','shap_validation_report.md','academic_shap_section.md']:
 p=OUT/name
 with p.open('a',encoding='utf-8') as f:f.write('\n\n'+note)
p=OUT/'01_final_attention_shap.ipynb';nb=nbformat.read(p,as_version=4)
nb.cells.append(nbformat.v4.new_markdown_cell(note))
nbformat.validate(nb)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
nbformat.write(nb,p)
plots=list((OUT/'plots').glob('*.png'));assert len(plots)==12
for p in plots:
 with Image.open(p) as im:
  assert all(abs(v-300)<.1 for v in im.info['dpi'])
  assert im.width>1000 and im.height>700
print('Verified six executed code cells without errors, three reports, and 12 PNG plots at 300 DPI.')
