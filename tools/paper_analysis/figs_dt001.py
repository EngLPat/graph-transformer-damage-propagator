"""Draft pgfplots figures from timing_dt001.json: crack timing of the laminate
model and the material-variability figure of the Y_T model.
Usage: [GTP_ROOT=...] python tools/paper_analysis/figs_dt001.py [dir]
dir (default results/paper_analysis) holds timing_dt001.json and receives the .tex files."""
import json, collections, os, sys
os.chdir(os.environ.get("GTP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
S = sys.argv[1] if len(sys.argv) > 1 else "results/paper_analysis"
D = S + "/"
J=json.load(open(S+"/timing_dt001.json"))
HEAD=r"""\documentclass[border=4pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots,fillbetween}
\pgfplotsset{compat=1.18}
\definecolor{okabeBlue}{RGB}{0,114,178}
\definecolor{okabeOrange}{RGB}{230,159,0}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=22mm},
  width=6.2cm, height=4.8cm, scale only axis,
  tick label style={font=\scriptsize}, label style={font=\small},
  title style={font=\small}, grid=both, grid style={line width=.2pt, draw=gray!25},
  legend style={font=\scriptsize, at={(0.5,1.16)}, anchor=south, legend columns=2, draw=none, fill=none},
]
"""
TAIL="\\end{groupplot}\n\\end{tikzpicture}\n\\end{document}\n"
# ---- crack timing, fixed model
f=J["fixed"]; t=f["t"]
c=lambda k: " ".join(f"({x:.2f},{100*y:.3f})" for x,y in zip(t,f[k]))
lo=collections.Counter(f["lag_on"]); lb=collections.Counter(f["lag_b"])
ks=sorted(set(lo)|set(lb)); ks=list(range(min(ks),max(ks)+1))
bars=lambda C: " ".join(f"({k},{C.get(k,0)})" for k in ks)
tex="%% Crack onset and burst timing at full resolution (Delta t = 0.01), released fixed-material model, 60 test simulations.\n"+HEAD+r"""\nextgroupplot[title={(a) cracked area during the rollout}, xlabel={pseudo-time $t$}, ylabel={elements with $\mathrm{FI}\geq 1$ (\%)}, xmin=0, xmax=1, ymin=0]
\addplot[name path=gq1, draw=none, forget plot] coordinates {"""+c("g_q1")+r"""};
\addplot[name path=gq3, draw=none, forget plot] coordinates {"""+c("g_q3")+r"""};
\addplot[gray!35, fill opacity=0.5, forget plot] fill between[of=gq1 and gq3];
\addplot[name path=pq1, draw=none, forget plot] coordinates {"""+c("p_q1")+r"""};
\addplot[name path=pq3, draw=none, forget plot] coordinates {"""+c("p_q3")+r"""};
\addplot[okabeBlue!30, fill opacity=0.5, forget plot] fill between[of=pq1 and pq3];
\addplot[line width=1.2pt, dashed, color=gray!65!black] coordinates {"""+c("g_med")+r"""};
\addlegendentry{reference (FE)}
\addplot[line width=1.2pt, color=okabeBlue] coordinates {"""+c("p_med")+r"""};
\addlegendentry{model}
\nextgroupplot[title={(b) timing error, model minus reference}, xlabel={rollout steps ($\Delta t=0.01$)}, ylabel={test simulations}, ybar, bar width=3.5pt, ymin=0, enlarge x limits=0.04, area legend]
\addplot[fill=okabeBlue!75, draw=okabeBlue!60!black] coordinates {"""+bars(lo)+r"""};
\addlegendentry{onset}
\addplot[fill=okabeOrange!80, draw=okabeOrange!60!black] coordinates {"""+bars(lb)+r"""};
\addlegendentry{crack burst}
"""+TAIL
open(D+"fig_crack_timing_fixed_model.tex","w").write(tex)
# ---- material-variability figure, variable Y_T
y=J["yt"]; yt=y["yt"]
pts=lambda a: " ".join(f"({x:.3f},{v:.4f})" for x,v in zip(yt,a))
line=lambda k: f"(51,{y[k]['b0']+y[k]['b1']*51:.4f}) (69,{y[k]['b0']+y[k]['b1']*69:.4f})"
tex="%% Variable Y_T model. (a) teacher-forced counterfactual sweep; (b) rollout onset at Delta t = 0.01, displacement-adjusted.\n"+HEAD.replace("group style={group size=2 by 1, horizontal sep=22mm},","group style={group size=2 by 1, horizontal sep=24mm},").replace("grid style={line width=.2pt, draw=gray!25},","grid style={line width=.2pt, draw=gray!25}, xmin=50, xmax=70, xlabel={transverse tensile strength $Y_{T}$ (MPa)},")+r"""\nextgroupplot[title={(a) counterfactual sweep}, ylabel={mean predicted FI rate}]
\addplot[dashed, line width=1pt, color=gray!70!black] coordinates {(51.029,1.87997) (68.999,1.87997)};
\addlegendentry{no influence}
\addplot[line width=1.2pt, color=okabeBlue, mark=*, mark size=1.6pt] coordinates {(51.029,1.90731) (53.276,1.86773) (55.522,1.82381) (57.768,1.77129) (60.014,1.71706) (62.260,1.67518) (64.507,1.64080) (66.753,1.61299) (68.999,1.59151)};
\addlegendentry{model}
\nextgroupplot[title={(b) onset of matrix cracking, rollout}, ylabel={onset pseudo-time (displacement-adjusted)}]
\addplot[only marks, mark=o, mark size=1.5pt, color=gray!65!black] coordinates {"""+pts(y["gt"]["adj"])+r"""};
\addlegendentry{reference (FE)}
\addplot[only marks, mark=*, mark size=1.3pt, color=okabeBlue, fill opacity=0.8] coordinates {"""+pts(y["pred"]["adj"])+r"""};
\addlegendentry{model}
\addplot[line width=1pt, color=gray!65!black, forget plot] coordinates {"""+line("gt")+r"""};
\addplot[line width=1.2pt, color=okabeBlue, forget plot] coordinates {"""+line("pred")+r"""};
"""+TAIL
open(D+"fig_material_variability.tex","w").write(tex)
print("figures written")
