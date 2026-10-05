import sys, os, numpy as np, torch, json
sys.path.append(os.getcwd())
from bearmamba3.data_xjtu import XJTUDataset
DR=os.path.join(os.environ.get('DATA_ROOT','data'),'XJTU-SY_Bearing_Datasets')
out={}
for b in ['Bearing3_3','Bearing3_5']:
    ds=XJTUDataset(DR,[b],n_sensors=2)
    idx=np.linspace(0,len(ds)-1,300).astype(int); X=torch.stack([ds[i][0] for i in idx])
    for s in range(5):
        sd=torch.load(f'out/xc_cnn_ATT/checkpoints/cross_seed{s}_seed{s}_final.pt',map_location='cpu')['model_state']
        from baselines.cnn1d_attnfusion import BearCNN1DAttnFusion
        m=BearCNN1DAttnFusion(64,4,2,2,2); m.load_state_dict(sd); m.eval()
        with torch.no_grad(): w=m.channel_attn.fc(X.mean(-1))
        out[f'{b}_seed{s}']=dict(mean_w=w.mean(0).tolist(), sd_w=w.std(0).tolist(), input_mean_abs=float(X.mean(-1).abs().max()))
        print(b,s,np.round(w.mean(0).numpy(),4),np.round(w.std(0).numpy(),6))
json.dump(out,open('out/attn_weights.json','w'),indent=1)
