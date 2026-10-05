import yaml, os, copy
R="<REPO_ROOT>/results/bm3_cwru_leakfree_20260911-leakfree"
base=yaml.safe_load(open(f"{R}/exp02_snr-4_nokin/source_snapshot/exp02_snr-4_nokin.yaml"))
WS=os.getcwd()
def mk(name, chans, snr, backbone="mamba3"):
    c=copy.deepcopy(base); c["name"]=name; c["channels"]=chans; c["noise_snr_db"]=float(snr)
    c["data_dir"]="<DATA_ROOT>/cwru_12k_de"; c["num_workers"]=0
    c["results_dir"]=f"results/{name}"; c["seeds"]=[0,1,2,3,4]; c["lambda_kin"]=0.0
    if backbone!="mamba3": c["backbone"]=backbone
    yaml.safe_dump(c, open(f"cfg/cwru_fe/{name}.yaml","w"), allow_unicode=True, sort_keys=False)
for bb in ["bm3","cnn"]:
    b="mamba3" if bb=="bm3" else "cnn1d"
    for snr in [-4,-6,-8]:
        for tag,ch in [("FE",["FE"]),("DE",["DE"]),("DEFE",["DE","FE"])]:
            mk(f"{bb}_{tag}_snr{snr}", ch, snr, b)
