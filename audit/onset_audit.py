"""Onset audit for XJTU-SY: which channel defines the degraded phase, and how the
window set would change under V-defined, joint and fixed-relative-life rules.
Read-only analysis of the raw CSVs; no training."""
import glob, os, json, numpy as np, pandas as pd
R=os.path.join(os.environ.get('DATA_ROOT','data'),'XJTU-SY_Bearing_Datasets')
COND={'1':'35Hz12kN','2':'37.5Hz11kN','3':'40Hz10kN'}
B={'Bearing1_1':'OR','Bearing1_2':'OR','Bearing1_3':'OR','Bearing2_1':'IR','Bearing2_2':'OR','Bearing2_4':'OR','Bearing2_5':'OR','Bearing3_1':'OR','Bearing3_3':'IR','Bearing3_4':'IR','Bearing3_5':'OR'}
def onset(k,s,kt=5.0,rm=2.0,w=5,nb=20):
    n=len(k); kr=np.convolve(k,np.ones(w)/w,'valid'); ke=np.where(kr>kt)[0]
    ko=int(ke[0]) if len(ke) else None
    nbase=min(nb,n//5); base=s[:nbase].mean() if nbase>0 else 0.5
    rr=np.convolve(s,np.ones(w)/w,'valid'); re=np.where(rr>rm*base)[0]; ro=int(re[0]) if len(re) else None
    c=[o for o in (ko,ro) if o is not None]
    if c:
        o=min(c); why='kurtosis' if o==ko and (ro is None or ko<=ro) else 'RMS'
        if ko is not None and ro is not None and ko==ro: why='both'
        return o,why,False
    return int(n*0.85),'fallback (last 15%)',True
rows=[]; detail={}
for b,cls in B.items():
    c=COND[b[7]]; fs=sorted(glob.glob(f'{R}/{c}/{b}/*.csv'),key=lambda f:int(os.path.basename(f)[:-4]))
    kH,sH,kV,sV=[],[],[],[]
    for f in fs:
        d=np.loadtxt(f,delimiter=',',skiprows=1)
        for col,(kk,ss) in enumerate([(kH,sH),(kV,sV)]):
            x=d[:,col]; m=x.mean(); sd=x.std(); ss.append(sd); kk.append(float(np.mean(((x-m)/sd)**4)) if sd>1e-10 else 3.0)
    kH,sH,kV,sV=map(np.array,(kH,sH,kV,sV)); n=len(fs)
    oH,wH,fH=onset(kH,sH); oV,wV,fV=onset(kV,sV)
    oJ=min(oH,oV)
    oF=int(n*0.85)
    wpf=32768//2048
    def jac(a,b): A=set(range(a,n)); Bs=set(range(b,n)); return len(A&Bs)/len(A|Bs)
    rows.append(dict(bearing=b,cls=cls,condition=c,n_files=n,onset_H=oH,trigger_H=wH,fallback_H=fH,windows_H=(n-oH)*wpf,
        onset_V=oV,trigger_V=wV,fallback_V=fV,windows_V=(n-oV)*wpf,onset_joint=oJ,windows_joint=(n-oJ)*wpf,
        onset_last15=oF,windows_last15=(n-oF)*wpf,jaccard_H_V=round(jac(oH,oV),3),jaccard_H_last15=round(jac(oH,oF),3)))
    detail[b]=dict(kurt_H=kH.tolist(),kurt_V=kV.tolist(),std_H=sH.tolist(),std_V=sV.tolist())
    print(b,'done',flush=True)
df=pd.DataFrame(rows); df.to_csv('onset_audit.csv',index=False); json.dump(detail,open('onset_audit_traces.json','w'))
print(df.to_string())
