"""Onset and crack-burst timing at full resolution (Delta t = 0.01).
onset  = first step with any element at FI >= 1
burst  = step of the largest single-step increase of the share of elements at FI >= 1
Writes timing_dt001.json for the draft figures.
Usage: [GTP_ROOT=...] python tools/paper_analysis/timing_dt001.py [dir]
dir (default results/paper_analysis) holds the dt001_* rollouts of dt001.sh."""
import numpy as np, re, json, collections, os, sys
os.chdir(os.environ.get("GTP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
S = sys.argv[1] if len(sys.argv) > 1 else "results/paper_analysis"
R="data/raw/laminate_yt_variability/geometry/plate_%03d.inp"
def deck(sid):
    t=open(R%(sid+1)).read()
    yt=float(t.split("type=LARC05PROPERTIES")[1].split("\n")[1].split(",")[6])
    m=re.search(r"\*\* Name: RIGHT Type: Displacement/Rotation\s*\n\*Boundary\s*\n([^\n]+)",t)
    return yt,float(m.group(1).split(",")[-1])
out={}
for lab in ("fixed","yt"):
    d=np.load(f"{S}/dt001_{lab}/test_results/rollout_arrays.npz",allow_pickle=True)
    gt=[np.asarray(x,float).ravel() for x in d["ground_truth_damage"]]; pr=[np.asarray(x,float).ravel() for x in d["predicted_damage"]]
    sims=np.asarray(d["sim_ids_per_frame"]).astype(int); t=np.asarray(d["times"],float).ravel()
    dt=np.median(np.diff(np.unique(np.round(t,6))))
    rec=[]; curves_g=[]; curves_p=[]
    for s in np.unique(sims):
        idx=np.where(sims==s)[0]; ts=t[idx]
        Ag=np.array([(gt[i]>=1).mean() for i in idx]); Ap=np.array([(pr[i]>=1).mean() for i in idx])
        og=ts[np.argmax(Ag>0)] if (Ag>0).any() else np.nan; op=ts[np.argmax(Ap>0)] if (Ap>0).any() else np.nan
        kg=int(np.argmax(np.diff(Ag))); kp=int(np.argmax(np.diff(Ap)))
        rec.append(dict(sim=int(s),og=og,op=op,bg=ts[kg+1],bp=ts[kp+1],jg=float(np.diff(Ag)[kg]),jp=float(np.diff(Ap)[kp]),fin=float(Ag[-1])))
        curves_g.append(Ag); curves_p.append(Ap); tt=ts
    n=min(len(c) for c in curves_g); G=np.array([c[:n] for c in curves_g]); P=np.array([c[:n] for c in curves_p])
    og=np.array([r["og"] for r in rec]); op=np.array([r["op"] for r in rec]); bg=np.array([r["bg"] for r in rec]); bp=np.array([r["bp"] for r in rec])
    lag_on=np.round((op-og)/dt).astype(int); lag_b=np.round((bp-bg)/dt).astype(int)
    print(f"== {lab}: dt={dt:.3f}, {len(rec)} sims, {n} steps")
    print(f"   onset  : bias {np.mean(op-og):+.4f}, MAE {np.mean(np.abs(op-og)):.4f}; lag steps {dict(sorted(collections.Counter(lag_on.tolist()).items()))}")
    print(f"   burst  : ref t={np.median(bg):.2f}, jump {np.median([r['jg'] for r in rec])*100:.2f}% of mesh; model t={np.median(bp):.2f}, jump {np.median([r['jp'] for r in rec])*100:.2f}%")
    print(f"            bias {np.mean(bp-bg):+.4f}, MAE {np.mean(np.abs(bp-bg)):.4f}; lag steps {dict(sorted(collections.Counter(lag_b.tolist()).items()))}")
    o=dict(dt=dt,t=tt[:n].tolist(),g_med=np.median(G,0).tolist(),p_med=np.median(P,0).tolist(),
           g_q1=np.percentile(G,25,0).tolist(),g_q3=np.percentile(G,75,0).tolist(),p_q1=np.percentile(P,25,0).tolist(),p_q3=np.percentile(P,75,0).tolist(),
           lag_on=lag_on.tolist(),lag_b=lag_b.tolist(),og=og.tolist(),op=op.tolist())
    if lab=="yt":
        yu=np.array([deck(r["sim"]) for r in rec]); yt,u=yu[:,0],yu[:,1]
        X=np.column_stack([np.ones_like(yt),yt,u]); o["yt"]=yt.tolist()
        for y,k in [(og,"gt"),(op,"pred")]:
            b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b; s2=(r@r)/(len(y)-3); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
            o[k]=dict(b0=float(b[0]+b[2]*u.mean()),b1=float(b[1]),se1=float(se[1]),adj=(y-b[2]*(u-u.mean())).tolist())
            print(f"   onset vs Y_T ({k}): shift over 18 MPa {18*b[1]:+.4f} (SE {18*se[1]:.4f})")
    out[lab]=o
json.dump(out,open(f"{S}/timing_dt001.json","w"))
