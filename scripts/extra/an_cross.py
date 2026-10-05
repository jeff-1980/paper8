import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from stats import *
def load(rd, seeds):
    R = [json.load(open(f"{rd}/seed_{s}.json")) for s in seeds]
    return dict(f1=[r["final_macro_f1"] for r in R], rec=[r["final_macro_recall"] for r in R],
                orr=[r["final_per_class_recall"][0] for r in R], irr=[r["final_per_class_recall"][1] for r in R],
                wall=[r["elapsed_s"] for r in R])
