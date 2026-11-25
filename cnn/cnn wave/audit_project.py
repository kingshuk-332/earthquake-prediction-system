"""Read-only source/data inspection; writes only new Phase 1 audit artifacts."""
from pathlib import Path
import ast
import hashlib
import json
import pandas as pd
import h5py

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
OUT = BASE / 'phase1_cnn_results' / 'reports'
OUT.mkdir(parents=True, exist_ok=True)
inventory = []
paths = sorted(set(ROOT.glob('*.csv')) | set((ROOT/'cnn').glob('*.csv')) |
               set((BASE/'data wave').rglob('*.csv')) | set((BASE/'data wave').rglob('*.h5')))
for path in paths:
    item = {'path':path.relative_to(ROOT).as_posix(), 'bytes':path.stat().st_size}
    if path.suffix == '.csv':
        try:
            d = pd.read_csv(path)
            item.update(shape=list(d.shape), columns=list(d.columns),
                        dtypes={k:str(v) for k,v in d.dtypes.items()},
                        missing={k:int(v) for k,v in d.isna().sum().items() if v},
                        duplicate_rows=int(d.duplicated().sum()))
            if 'origin_time_utc' in d:
                t = pd.to_datetime(d.origin_time_utc.astype(str).str.replace(' UTC',''), utc=True, errors='coerce')
                item.update(start=str(t.min()), end=str(t.max()), invalid_dates=int(t.isna().sum()), sorted=bool(t.is_monotonic_increasing))
        except pd.errors.EmptyDataError:
            item['empty'] = True
    else:
        item['datasets'] = {}
        with h5py.File(path, 'r') as f:
            def inspect(name, obj):
                if isinstance(obj,h5py.Dataset):
                    item['datasets'][name] = {'shape':list(obj.shape), 'dtype':str(obj.dtype)}
                    if name in ['labels','label']:
                        item['datasets'][name]['counts'] = pd.Series(obj[:]).value_counts().to_dict()
            f.visititems(inspect)
    inventory.append(item)
(OUT/'dataset_inventory.json').write_text(json.dumps(inventory,indent=2,default=str),encoding='utf-8')
sources = list((BASE/'data wave'/'synthetic_waveform_simulator').glob('*.py')) + [ROOT/'preprocess.py',ROOT/'preprocess_dl_temporal.py',ROOT/'cnn'/'cnn_1d_model.py',ROOT/'cnn'/'generate_noisy_dataset.py']
source_audit = []
for p in sources:
    content = p.read_text(encoding='utf-8-sig')
    tree = ast.parse(content)
    source_audit.append({'path':p.relative_to(ROOT).as_posix(), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
       'doc':ast.get_docstring(tree), 'symbols':[{'name':n.name,'line':n.lineno,'doc':ast.get_docstring(n)} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]})
(OUT/'source_inventory.json').write_text(json.dumps(source_audit,indent=2),encoding='utf-8')
d = pd.read_csv(ROOT/'dataset_2.csv')
t = pd.to_datetime(d.origin_time_utc.str.replace(' UTC',''),utc=True)
yearly = d.groupby(t.dt.year).agg(events=('Magnitude_Value','size'), median_magnitude=('Magnitude_Value','median'), minimum_magnitude=('Magnitude_Value','min'), m45=('Magnitude_Value',lambda x:(x>=4.5).sum()))
yearly = yearly.reindex(range(t.dt.year.min(),t.dt.year.max()+1)).fillna({'events':0,'m45':0})
yearly.to_csv(OUT/'annual_catalog_counts.csv')
d['time'] = t
identity=['origin_time_utc','Lat','Long','Depth_km','Magnitude_Value']
print('Inventory:',len(inventory),'datasets;',len(source_audit),'source files')
print('Event-key duplicates:',d.duplicated(identity).sum(),'Magnitude types:',d.Magnitude_Type.value_counts().to_dict())
print('2011+ rows:',int((t>=pd.Timestamp('2011-01-01',tz='UTC')).sum()))
for item in inventory:
    if 'datasets' in item:
        print(item['path'], {k:v['shape'] for k,v in item['datasets'].items() if 'waveform' in k or k in ['labels','label']})
print('Historical checkpoints:',[(p.name,p.stat().st_size) for p in (ROOT/'cnn').glob('*.pth')])
print('Existing notebooks:',[str(p.relative_to(ROOT)) for p in ROOT.rglob('*.ipynb') if '.notebook-runtime' not in str(p)])
