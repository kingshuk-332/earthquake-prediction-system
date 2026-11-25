"""Streaming verification of existing waveform data; never modifies source files."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import h5py
from scipy.signal import detrend
from scipy.ndimage import uniform_filter1d

BASE=Path(__file__).resolve().parent
OUT=BASE/'phase1_waveform_cnn_results'/'reports'
OUT.mkdir(parents=True,exist_ok=True)
SIM=BASE/'data wave'/'synthetic_waveform_simulator'
P=SIM/'output'
P11=P/'phase11_preprocessed_dataset'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
preserve=[BASE/'phase1_cnn_forecasting.ipynb',*filter(Path.is_file,(BASE/'phase1_cnn_results').rglob('*'))]
save('preserved_catalog_hashes.json',{p.relative_to(BASE).as_posix():sha(p) for p in preserve})
mapping=pd.read_csv(P11/'split_indices.csv')
combined=pd.read_csv(P/'phase10_classification_dataset'/'earthquake_noise_mapping.csv')
phase9=pd.read_csv(P/'phase9_full_dataset'/'synthetic_waveforms_mapping.csv')
metadata=pd.read_csv(P/'phase2_travel_time'/'metadata_with_theoretical_arrivals.csv')
assert metadata.source_id.is_unique and phase9.source_id.is_unique
metadata['origin_time_utc']=pd.to_datetime(metadata.origin_time_utc,utc=True,errors='raise')
assert mapping.combined_index.is_unique and len(mapping)==16178
assert mapping.groupby('pair_id').split.nunique().max()==1
assert mapping.groupby('pair_id').label.agg(['sum','count']).eq([1,2]).all().all()
assert mapping.merge(combined,on='combined_index',suffixes=('_s','_c')).eval('label_s == label_c').all()
norm=json.loads((P11/'normalization_parameters.json').read_text())
report={'splits':{},'normalization':norm,'metadata_fields_absent_in_h5':['origin_time_utc','Lat','Long'],
        'metadata_join_source':str((P/'phase2_travel_time'/'metadata_with_theoretical_arrivals.csv').relative_to(BASE)),
        'checks':{},'source_hashes':{}}
fingerprints={'raw':{},'standardized':{}}
all_features=[]; all_metadata=[]; train_sum=train_sumsq=0.;train_n=0
for split in ['train','validation','test']:
    path=P11/f'{split}.h5'
    report['source_hashes'][path.relative_to(BASE).as_posix()]=sha(path)
    with h5py.File(path,'r') as f:
        n=len(f['labels']); y=f['labels'][:]
        assert f['waveforms'].shape==f['waveforms_standardized'].shape==(n,10000)
        assert set(np.unique(y))=={0,1} and (y==1).sum()==(y==0).sum()
        assert f.attrs['sampling_rate_hz']==100
        manifest=mapping[mapping.split==split].sort_values('combined_index').reset_index(drop=True)
        assert len(manifest)==n and np.array_equal(manifest.label,y)
        assert np.array_equal(manifest.pair_id,f['pair_id'][:])
        for key in ['source_id','trace_name','sample_type']:
            assert np.array_equal(manifest[key].astype(str),f[key].asstr()[:])
        info=pd.DataFrame({k:f[k].asstr()[:] if f[k].dtype.kind=='O' else f[k][:] for k in f if f[k].ndim==1})
        info['split']=split; info['row']=np.arange(n)
        info=info.merge(metadata[['source_id','origin_time_utc','Lat','Long','Magnitude_Value','receiver_latitude','receiver_longitude']],on='source_id',how='left',validate='many_to_one')
        eq=info.labels.eq(1)
        assert info.loc[eq,['origin_time_utc','Lat','Long','magnitude']].notna().all().all()
        assert np.allclose(info.loc[eq,'magnitude'],info.loc[eq,'Magnitude_Value'],atol=1e-5)
        assert info.loc[~eq,'magnitude'].isna().all() and info.loc[~eq,'origin_time_utc'].isna().all()
        assert ((info.loc[eq,'p_arrival_sample']>=0)&(info.loc[eq,'s_arrival_sample']>info.loc[eq,'p_arrival_sample'])&(info.loc[eq,'s_arrival_sample']<10000)).all()
        means=[];stds=[]
        for start in range(0,n,100):
            raw=f['waveforms'][start:start+100]
            x=f['waveforms_standardized'][start:start+100]
            assert np.isfinite(raw).all() and np.isfinite(x).all()
            assert (np.std(raw,axis=1)>0).all()
            means.extend(x.mean(axis=1).tolist());stds.extend(x.std(axis=1).tolist())
            for key,arr in [('raw',raw),('standardized',x)]:
                for j,row in enumerate(arr):
                    digest=hashlib.sha256(row.tobytes()).hexdigest()
                    assert digest not in fingerprints[key],f'Exact {key} duplicate: {split}:{start+j} and {fingerprints[key].get(digest)}'
                    fingerprints[key][digest]=(split,start+j)
            if split=='train':
                dx=detrend(raw.astype(np.float64),axis=1,type='linear')
                train_sum+=dx.sum();train_sumsq+=(dx*dx).sum();train_n+=dx.size
            power=x.astype(np.float64)**2
            pre=power[:,:500].mean(axis=1); post=power[:,700:].mean(axis=1)
            sta=uniform_filter1d(power,size=100,axis=1,mode='nearest')
            lta=uniform_filter1d(power,size=1000,axis=1,mode='nearest')
            max_ratio=(sta[:,500:-500]/(lta[:,500:-500]+1e-12)).max(axis=1)
            ps=np.abs(np.fft.rfft(x,axis=1))**2; ps=ps[:,1:]
            flat=np.exp(np.log(ps+1e-12).mean(axis=1))/(ps.mean(axis=1)+1e-12)
            feat=pd.DataFrame(dict(split=split,row=np.arange(start,start+len(x)),label=y[start:start+len(x)],
                log_raw_rms=np.log(np.sqrt((raw.astype(float)**2).mean(axis=1))+1e-12),
                log_pre_rms=np.log(np.sqrt(pre)+1e-12),log_post_pre=np.log((post+1e-12)/(pre+1e-12)),
                crest=np.max(np.abs(x),axis=1),max_sta_lta=max_ratio,spectral_flatness=flat,
                peak_spectral_fraction=ps.max(axis=1)/(ps.sum(axis=1)+1e-12)))
            all_features.append(feat)
        assert np.max(np.abs(means))<1e-4
        assert np.max(np.abs(np.asarray(stds)-1))<.02
        sampled=np.random.default_rng(42).choice(n,12,replace=False)
        maxerr=0.
        for i in sampled:
            raw=f['waveforms'][i]
            dx=detrend(raw,type='linear').astype(np.float32)
            expected=(dx-dx.mean())/(dx.std()+norm['epsilon'])
            maxerr=max(maxerr,float(np.max(np.abs(expected-f['waveforms_standardized'][i]))))
            assert np.allclose(expected,f['waveforms_standardized'][i],atol=1e-4,rtol=1e-4)
            expected_global=(dx-norm['global_training_mean'])/(norm['global_training_std']+norm['epsilon'])
            assert np.allclose(expected_global,f['waveforms_global_norm'][i],atol=1e-4,rtol=1e-4)
        all_metadata.append(info)
        report['splits'][split]=dict(path=path.relative_to(BASE).as_posix(),shape=[n,10000],sampling_hz=100,
            labels={str(k):int((y==k).sum()) for k in [0,1]},pairs=int(info.pair_id.nunique()),
            keys={k:{'shape':list(f[k].shape),'dtype':str(f[k].dtype)} for k in f},
            attributes={k: v.item() if hasattr(v,'item') else v for k,v in f.attrs.items()},
            origin_min=str(info.origin_time_utc.min()),origin_max=str(info.origin_time_utc.max()),
            standardized_max_abs_mean=float(np.max(np.abs(means))),standardized_std_range=[min(stds),max(stds)],
            sampled_normalization_max_error=maxerr,
            p_arrival_range=[int(info.loc[eq,'p_arrival_sample'].min()),int(info.loc[eq,'p_arrival_sample'].max())])
        print(split,n,'verified; earthquake/noise',int(y.sum()),int(n-y.sum()),flush=True)
meta=pd.concat(all_metadata,ignore_index=True)
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not (set(meta.loc[meta.split.eq(a),'pair_id'])&set(meta.loc[meta.split.eq(b),'pair_id']))
    assert not (set(meta.loc[meta.split.eq(a),'source_id'])&set(meta.loc[meta.split.eq(b),'source_id']))
# Physical duplicate earthquake keys could bypass differing source identifiers.
eq=meta[meta.labels.eq(1)]
assert not eq.duplicated(['origin_time_utc','Lat','Long','depth_km','magnitude']).any()
train_std=float(np.sqrt(train_sumsq/train_n-(train_sum/train_n)**2))
assert np.isclose(train_std,norm['global_training_std'],rtol=1e-5)
features=pd.concat(all_features,ignore_index=True)
features.to_csv(OUT/'signal_features.csv',index=False)
meta.to_csv(OUT/'verified_metadata.csv',index=False)
with h5py.File(P/'phase10_classification_dataset'/'earthquake_noise_dataset.h5','r') as f10, h5py.File(P/'phase9_full_dataset'/'synthetic_seismic_waveforms.h5','r') as f9:
    report['phase10_shape']=list(f10['waveforms'].shape);report['phase9_shape']=list(f9['waveforms'].shape)
    for split in ['train','validation','test']:
        mm=mapping[mapping.split==split].sort_values('combined_index').reset_index(drop=True)
        with h5py.File(P11/f'{split}.h5','r') as f:
            for i in np.random.default_rng(42).choice(len(mm),12,replace=False):
                ci=int(mm.iloc[i].combined_index)
                assert np.array_equal(f['waveforms'][i],f10['waveforms'][ci])
                if mm.iloc[i].label==1:
                    assert np.array_equal(f['waveforms'][i],f9['waveforms'][ci])
report['checks']=dict(all_waveforms_finite_nonzero=True,all_labels_and_mapping_aligned=True,pairs_and_sources_disjoint=True,
    no_duplicate_event_keys=True,no_exact_raw_or_standardized_duplicates=True,all_standardized_moments_checked=True,
    normalization_formula_sampled_per_split=12,upstream_waveform_alignment_sampled_per_split=12,
    training_global_std_recomputed=train_std,origin_and_coordinate_join_complete_for_earthquakes=True,
    noise_metadata_missing_intentionally=True,near_duplicate_exhaustive_check=False)
report['feasibility']={
    'magnitude':'Verified earthquake magnitudes. Standardization removes absolute amplitude, but envelope depends on magnitude; regression would primarily invert simulator assumptions. Not implemented in this detection baseline.',
    'origin_time':'UTC timestamps join successfully but absolute time is absent from waveform input; cannot identify calendar origin time.',
    'relative_arrivals':'P/S indices verified; P is anchored near 6 s, so P prediction can be a trivial simulator prior. Separate picking benchmark deferred.',
    'location':'Lat/Long join successfully. A single vertical trace without receiver context, azimuth or multiple stations cannot uniquely resolve epicenter; S-P can constrain modeled distance only. Geographic regression deferred.'}
for path in [P11/'split_indices.csv',P11/'normalization_parameters.json',P/'phase10_classification_dataset'/'earthquake_noise_mapping.csv',P/'phase9_full_dataset'/'synthetic_waveforms_mapping.csv',P/'phase2_travel_time'/'metadata_with_theoretical_arrivals.csv']:
    report['source_hashes'][path.relative_to(BASE).as_posix()]=sha(path)
save('waveform_audit.json',report)
(BASE/'.agent-memory/checkpoint.md').write_text('# Waveform correction checkpoint — 2026-11-05\nW1 complete: numerical arrays, metadata joins, mapping, pair/source separation, exact duplicates and normalization verified.\nAudit: phase1_waveform_cnn_results/reports/waveform_audit.json.\nCatalog artifacts fingerprinted and preserved. No waveform model trained yet.\nNext: implement detection notebook; optional parameter estimation deferred for documented identifiability limits.\n',encoding='utf-8')
print('PASS waveform audit; full raw/standardized fingerprint counts:',{k:len(v) for k,v in fingerprints.items()},flush=True)
