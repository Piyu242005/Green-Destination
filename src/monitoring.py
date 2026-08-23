"""Lightweight model monitoring and fairness utilities."""
import json, os
import numpy as np
import pandas as pd

def population_stability_index(expected, actual, bins=10):
    expected=np.asarray(expected,dtype=float); actual=np.asarray(actual,dtype=float)
    edges=np.unique(np.quantile(expected,np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    e=np.histogram(expected, bins=edges)[0]/len(expected); a=np.histogram(actual,bins=edges)[0]/len(actual)
    e=np.clip(e,1e-6,None); a=np.clip(a,1e-6,None)
    return float(np.sum((a-e)*np.log(a/e)))

def prediction_drift(reference, current):
    return population_stability_index(reference, current)

def group_metrics(df, group_column="Gender", target="Attrition", prediction="prediction"):
    if group_column not in df or target not in df or prediction not in df: raise ValueError("Required columns are missing")
    rows=[]
    for group, part in df.groupby(group_column):
        y=part[target].map({"Yes":1,"No":0}) if part[target].dtype==object else part[target]
        p=part[prediction].astype(int)
        rows.append({"group":group,"n":len(part),"positive_rate":float(p.mean()),"actual_attrition_rate":float(y.mean())})
    return pd.DataFrame(rows)

def save_monitoring_report(metrics, path="models/monitoring_report.json"):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
