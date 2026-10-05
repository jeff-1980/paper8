import sys,numpy as np,scipy.io as sio,json
sys.path.insert(0,'<REPO_ROOT>')
from bearmamba3.noise import generate_pink_noise
fs=12000;T=2048
files={'Normal':('97','X097_DE_time'),'IR007':('105','X105_DE_time'),'BA007':('118','X118_DE_time'),'OR007':('130','X130_DE_time')}
f=np.fft.rfftfreq(T,1/fs)
bands={'kinematic 30-170 Hz':(30,170),'0-300 Hz':(0,300),'2-5 kHz (resonance)':(2000,5000)}
rng=np.random.default_rng(0)
out={}
for snr in [-4,-8]:
    for nt in ['awgn','pink']:
        rows={b:[] for b in bands};frac={b:[] for b in bands}
        for name,(fn,key) in files.items():
            x=sio.loadmat(f'<HOME>/data_cwru/{fn}.mat')[key].ravel()
            for i in range(40):
                st=rng.integers(0,len(x)-T);w=x[st:st+T].astype(np.float64)
                p=np.mean(w**2)
                npow=p/10**(snr/10)
                n=(rng.standard_normal(T) if nt=='awgn' else generate_pink_noise(T,rng).astype(np.float64))*np.sqrt(npow)
                w0=w-w.mean(); n0=n-n.mean()   # z-score removes DC
                S=np.abs(np.fft.rfft(w0))**2; N=np.abs(np.fft.rfft(n0))**2
                for b,(lo,hi) in bands.items():
                    m=(f>=lo)&(f<=hi)
                    rows[b].append(10*np.log10(S[m].sum()/N[m].sum()))
                    frac[b].append(N[m].sum()/N.sum())
        out[f'{nt}@{snr}']={b:(float(np.median(rows[b])),float(np.median(frac[b]))) for b in bands}
for k,v in out.items(): print(k,{b:(round(a,1),round(c,3)) for b,(a,c) in v.items()})
json.dump(out,open('inband.json','w'))
