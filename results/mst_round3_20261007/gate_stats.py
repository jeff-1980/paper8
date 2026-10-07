"""Pre-registered validity check for the gated-fusion baseline: SD over test windows of the
per-sample gate weight on channel 1 (vertical); 'adaptive' requires SD > 0.01 in every seed."""
import sys; NAMES=sys.argv[1:]
import sys, json, yaml, glob, numpy as np, torch, importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('tr','scripts/xjtu/train_xjtu_leakfree_v3.py'); tr=importlib.util.module_from_spec(spec); sys.argv=['x']; spec.loader.exec_module(tr)
from bearmamba3.data_xjtu import XJTUDataset, make_cross_condition_split, make_lobo_folds
dev=torch.device('cuda')
def gates(m,ds):
    g=[]; X=torch.stack([torch.as_tensor(w) for w in ds._windows]).float()
    X=(X-X.mean(-1,keepdim=True))/(X.std(-1,keepdim=True)+1e-8)
    m.eval()
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(0,len(X),256):
            m(X[i:i+256].to(dev)); g.append(m.conv_embed.last_gates[:,1].float().cpu())
            if getattr(m.conv_embed,'last_token_gates',None) is not None: TOK.append(m.conv_embed.last_token_gates[...,1].float().std().item())
    return torch.cat(g).numpy()
TOK=[]
for name in NAMES:
    cfg=tr.load_config(Path(f'cfg_r3/{name}.yaml')); rd=Path(cfg['results_dir']); res={}
    if cfg['mode']=='cross':
        _,tb=make_cross_condition_split(cfg['train_condition'],cfg['test_condition'])
        ds=XJTUDataset(cfg['data_root'],tb,n_sensors=2)
        for s in cfg['seeds']:
            m=tr.build_model(cfg,dev); m.load_state_dict(torch.load(rd/'checkpoints'/f'cross_seed{s}_seed{s}_final.pt',map_location=dev)['model_state'])
            TOK.clear(); g=gates(m,ds); res[s]={'mean':float(g.mean()),'sd':float(g.std()),'token_sd_mean':float(np.mean(TOK)) if TOK else None,'p05':float(np.percentile(g,5)),'p95':float(np.percentile(g,95))}
    else:
        folds=make_lobo_folds(['Bearing3_1','Bearing3_3','Bearing3_4','Bearing3_5'])
        for f in glob.glob(str(rd/'fold*_seed*.json')):
            j=json.load(open(f)); fo=int(Path(f).name[4]); s=j['seed']
            ds=XJTUDataset(cfg['data_root'],folds[fo][1],n_sensors=2)
            m=tr.build_model(cfg,dev); m.load_state_dict(torch.load(j['checkpoint_path'],map_location=dev)['model_state_dict'])
            g=gates(m,ds); res[f'fold{fo}_seed{s}']={'mean':float(g.mean()),'sd':float(g.std())}
    res['adaptive_rule_met']=all(v['sd']>0.01 for k,v in res.items() if isinstance(v,dict))
    json.dump(res,open(rd/'gate_stats.json','w'),indent=1); print(name,res)
