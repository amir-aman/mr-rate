import pandas as pd, numpy as np, json, re
U='data/raw/'  # local MR-RATE tables, never committed
m=pd.read_csv(U+'metadata/batch00_metadata.csv',low_memory=False,usecols=['patient_uid','study_uid','series_id','classified_modality','acquisition_plane',"Patient'sAge","Patient'sSex",'FieldStrength_T','Manufacturer',"Manufacturer'sModelName",'MRAcquisitionType'])
r=pd.read_csv(U+'reports/batch00_reports.csv',low_memory=False)
l=pd.read_csv(U+'mrrate_labels.csv',low_memory=False)
m['age']=pd.to_numeric(m["Patient'sAge"].str.extract(r'(\d+)Y')[0],errors='coerce')
print('age units',m["Patient'sAge"].str[-1].value_counts().to_dict(),'age null',m.age.isna().sum())
m['vendor']=np.where(m.Manufacturer.str.contains('Philips'),'Philips','Siemens')
st=m.groupby('study_uid').agg(patient=('patient_uid','first'),age=('age','first'),sex=("Patient'sSex",'first'),fs=('FieldStrength_T','first'),vendor=('vendor','first'),model=("Manufacturer'sModelName",'first'),nser=('series_id','size'),nage=('age','nunique'),nfs=('FieldStrength_T','nunique'))
print('studies',len(st),'patients',st.patient.nunique(),'series',len(m),'multi age',(st.nage>1).sum(),'multi fs',(st.nfs>1).sum())
out={'batch':'batch00','n_series':int(len(m)),'n_studies':int(len(st)),'n_patients':int(st.patient.nunique()),'series_per_study_median':float(st.nser.median())}
edges=list(range(0,95,5)); 
st['bin']=pd.cut(st.age,edges+[200],right=False,labels=[f'{a}' for a in edges])
h=st.groupby(['bin','sex'],observed=False).size().unstack(fill_value=0)
out['age']={'bins':[int(a) for a in edges],'F':[int(x) for x in h['F']],'M':[int(x) for x in h['M']],'median':float(st.age.median()),'min':int(st.age.min()),'max':int(st.age.max())}
out['sex']={k:int(v) for k,v in st.sex.value_counts().items()}
out['field']={f'{k:g} T':int(v) for k,v in st.fs.value_counts().sort_index().items()}
out['scanner']=[{'name':f'{a} {b}','field':f'{c:g} T','n':int(n)} for (a,b,c),n in st.groupby(['vendor','model','fs']).size().sort_values(ascending=False).items()]
out['modality']={k:int(v) for k,v in m.classified_modality.value_counts().items()}
out['plane']={k.title():int(v) for k,v in m.acquisition_plane.value_counts().items()}
out['acq']={k:int(v) for k,v in m.MRAcquisitionType.value_counts().items()}
# studies containing each modality
out['study_has']={k:int(m[m.classified_modality==k].study_uid.nunique()) for k in out['modality']}
# reports
imp=r.impression.fillna('').astype(str).str.strip(); words=imp.str.split().str.len()
out['reports']={'n':int(len(r)),'impression_missing':int((imp=='').sum()),'impression_words_median':float(words[imp!=''].median()),'impression_words_p90':float(words[imp!=''].quantile(.9)),'impression_over_48_words_pct':round(float((words>48).mean()*100),1),'findings_words_median':float(r.findings.fillna('').str.split().str.len().median())}
ascii_ratio=r.report.astype(str).str.count(r'[çğıöşüÇĞİÖŞÜ]').gt(0).mean(); print('turkish-char share',ascii_ratio)
# labels
cols=[c for c in l.columns if c!='study_uid']; print('label values',np.unique(l[cols].values))
N=len(l); out['labels_n']=int(N)
out['labels']=[{'name':c,'n':int(l[c].sum()),'pct':round(float(l[c].mean()*100),2)} for c in sorted(cols,key=lambda c:-l[c].sum())]
out['labels_none_pct']=round(float((l[cols].sum(axis=1)==0).mean()*100),1)
j=st.join(l.set_index('study_uid'),how='inner'); print('joined',len(j))
dec=[0,20,30,40,50,60,70,80]; names=['under 20','20 to 29','30 to 39','40 to 49','50 to 59','60 to 69','70 to 79','80 and over']
j['dec']=pd.cut(j.age,dec+[200],right=False,labels=names)
tot=j.groupby('dec',observed=False).size()
out['by_age']={'bands':names,'n':[int(x) for x in tot],'n_studies':int(len(j)),'labels':{c:[int(x) for x in j.groupby('dec',observed=False)[c].sum()] for c in cols}}
json.dump(out,open('explorer/aggregates/stats_full_private.json','w'),indent=1)
o=dict(out); o.pop('by_age'); print(json.dumps(o)[:3000]); print(out['by_age']['n'], {c:out['by_age']['labels'][c] for c in ['Cerebral atrophy','Gliosis','Lacunar infarct']})
import json
s=json.load(open('explorer/aggregates/stats_full_private.json'))
b=s.pop('by_age'); out={'bands':b['bands'],'n':b['n'],'n_studies':b['n_studies'],'labels':{}}
for k,v in b['labels'].items():
    if sum(v)<40: continue
    # small-cell suppression: counts under 5 are not published
    out['labels'][k]=[None if c<5 else round(100*c/n,1) for c,n in zip(v,b['n'])]
s['by_age']=out
s['dataset']={'volumes':705254,'studies':98334,'patients':84137}
json.dump(s,open('explorer/aggregates/stats_public.json','w'))
print(len(out['labels']),'labels with age profile'); print(out['labels']['Cerebral atrophy'], out['labels']['Lacunar infarct'])
