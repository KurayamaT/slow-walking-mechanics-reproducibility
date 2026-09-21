import csv, sys
import numpy as np
sys.path.insert(0,'code'); import revision_numerics as rn
QS=[round(x,4) for x in np.arange(0.010,0.0601,0.002)]+[0.080,0.100,0.120,0.130]
def gait(q,k,c,seed):
    ap=np.arcsin(0.5*q); om=-c*ap; P=-om*np.tan(ap)
    z0=list(seed) if seed is not None else [ap,om,(1-np.cos(2*ap))*om]
    z,T,ok=rn.find_fixed_point(z0,rn.GAM,k,P,rn.RTOL,rn.ATOL,delta=1e-7)
    if not ok: return None
    J=rn.jacobian_3d(z,rn.GAM,k,P,rn.RTOL,rn.ATOL,rn.DELTA)
    if J is None: return None
    lam=float(np.max(np.abs(rn.sorted_eigs(J)))); a=float(z[0])
    return dict(q=q,c=c,alpha_h=a,s_real=2*np.sin(a),T=T,lam_max=lam,
                margin=1-lam,N_half=-np.log(2)/np.log(lam),z=z)
base=gait(0.130,-0.16,1.04,None); anchor={1.04:base['z']}
for tgt,st in ((1.20,0.01),(1.00,-0.01),(0.80,-0.01)):
    cc=1.04; seed=base['z']
    while abs(cc-tgt)>1e-9:
        cc=round(cc+(st if abs(tgt-cc)>abs(st) else tgt-cc),4)
        g=gait(0.130,-0.16,cc,seed)
        if g is None: seed=None; break
        seed=g['z']
    if seed is not None: anchor[tgt]=seed
B={}
for c in (0.80,1.00,1.04,1.20):
    seed=anchor.get(c); b={}
    for q in sorted(QS,reverse=True):
        g=gait(q,-0.16,c,seed)
        if g is None: seed=None; continue
        seed=g['z']; b[q]=g
    B[c]=b
lo=max(min(r['s_real'] for r in b.values()) for b in B.values())
S=round(lo+1e-6,6)
print("=== push-off 枝:共通実現歩幅 s=%.6f での比較 ===" % S)
print("  %-6s %-16s %-16s %-8s %-10s" % ("c","N_half(q=0.010)","N_half(s共通)","差","s_real@q=.01"))
for c in (0.80,1.00,1.04,1.20):
    b=B[c]; vs=sorted((r['s_real'],r['N_half']) for r in b.values())
    xs=[v[0] for v in vs]; ys=[v[1] for v in vs]
    nq=b[0.010]['N_half']; nc=float(np.interp(S,xs,ys))
    print("  %-6.2f %-16.1f %-16.1f %+-8.1f %.6f" % (c,nq,nc,nc-nq,b[0.010]['s_real']))
with open('data/a5_pushoff_realised.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['c','q','alpha_h','s_real','T','lam_max','margin','N_half'])
    for c,b in B.items():
        for q in sorted(b): r=b[q]; w.writerow([c,q,r['alpha_h'],r['s_real'],r['T'],r['lam_max'],r['margin'],r['N_half']])
print("\n書き出し: data/a5_pushoff_realised.csv")
