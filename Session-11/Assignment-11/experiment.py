"""Session 11: reproduce optimizer experiments. Run: python experiment.py"""
from pathlib import Path
import csv, math, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(2)
OUT=Path(__file__).parent/"results";OUT.mkdir(exist_ok=True)
SEED=2026
def save(name,rows):
    with open(OUT/name,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def adam():
    w=1.;m=v=0.;rows=[]
    t=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64))
    opt=torch.optim.Adam([t],lr=.01)
    for k,g in enumerate([.2,-.1,.3,.05,-.2],1):
        m=.9*m+.1*g;v=.999*v+.001*g*g
        mh=m/(1-.9**k);vh=v/(1-.999**k)
        delta=.01*mh/(math.sqrt(vh)+1e-8);w-=delta
        t.grad=torch.tensor(g,dtype=torch.float64);opt.step();opt.zero_grad()
        s=opt.state[t]
        rows.append(dict(step=k,g=g,m=m,v=v,m_hat=mh,v_hat=vh,delta=delta,weight_manual=w,weight_torch=t.item(),m_torch=s["exp_avg"].item(),v_torch=s["exp_avg_sq"].item(),abs_error=abs(w-t.item())))
    save("adam_hand_vs_torch.csv",rows)
    assert max(r["abs_error"] for r in rows)<1e-12
    return max(r["abs_error"] for r in rows)

def bias():
    m=v=0.;rows=[]
    for k in range(1,12001):
        m=.9*m+.1*.1;v=.999*v+.001*.01
        corrected=.01*(m/(1-.9**k))/(math.sqrt(v/(1-.999**k))+1e-8)
        raw=.01*m/(math.sqrt(v)+1e-8)
        rows.append(dict(step=k,corrected_update=corrected,uncorrected_update=raw,relative_gap=abs(raw-corrected)/abs(corrected)))
    save("bias_correction.csv",rows)
    fig,ax=plt.subplots()
    for key,label in [("corrected_update","Corrected"),("uncorrected_update","No bias correction")]:
        ax.plot([r["step"] for r in rows[:20]],[r[key] for r in rows[:20]],label=label)
    ax.set(xlabel="Step",ylabel="Update size",title="Adam first 20 updates");ax.legend()
    fig.tight_layout();fig.savefig(OUT/"bias_correction.png");plt.close(fig)
    return {str(t):next((r["step"] for r in rows if all(q["relative_gap"]<t for q in rows[r["step"]-1:])),None) for t in (.1,.05,.01)}

def data():
    rng=np.random.default_rng(SEED)
    x=torch.tensor(rng.normal(size=(256,16)),dtype=torch.float32)
    a=torch.tensor(rng.normal(size=(16,1)),dtype=torch.float32)
    y=torch.tanh(x@a)+.05*torch.tensor(rng.normal(size=(256,1)),dtype=torch.float32)
    return x,y
class Model(torch.nn.Module):
    def __init__(self,width):
        super().__init__()
        self.fc1=torch.nn.Linear(16,width);self.fc2=torch.nn.Linear(width,1)
    def forward(self,x):return self.fc2(torch.tanh(self.fc1(x)))

def multiplier(step,kind):
    if step<=30:return step/30
    if kind=="cosine":return .1+.9*.5*(1+math.cos(math.pi*(step-30)/270))
    if kind=="wsd":return 1. if step<=240 else .1+.9*(300-step)/60
    raise ValueError(kind)

def run_schedule(kind,lr):
    torch.manual_seed(SEED);net=Model(256);x,y=data()
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.01);rows=[]
    for step in range(1,201):
        effective=lr*multiplier(step,kind)
        opt.param_groups[0]["lr"]=effective
        before={n:p.detach().clone() for n,p in net.named_parameters()}
        opt.zero_grad();loss=torch.nn.functional.mse_loss(net(x),y)
        loss.backward();opt.step()
        for n,p in net.named_parameters():
            a=before[n];rows.append(dict(schedule=kind,layer=n,step=step,lr=effective,loss_pre_step=loss.item(),ratio=(p.detach()-a).norm().item()/(a.norm().item()+1e-12)))
    with torch.no_grad():last=torch.nn.functional.mse_loss(net(x),y).item()
    return rows,last

def schedules():
    rows=[];trials=[];best={}
    for kind in ("cosine","wsd"):
        possible=[]
        for lr in (.001,.003,.01):
            r,loss=run_schedule(kind,lr);trials.append(dict(schedule=kind,lr=lr,loss_200=loss))
            possible.append((loss,lr,r))
        loss,lr,r=min(possible,key=lambda q:q[0]);rows+=r
        best[kind]={"loss":loss,"base_lr":lr}
    save("schedule_lr_tuning.csv",trials);save("schedules_layer_ratios.csv",rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kind in ("cosine","wsd"):
        r=[v for v in rows if v["schedule"]==kind and v["layer"]=="fc1.weight"]
        axes[0].plot([v["step"] for v in r],[v["loss_pre_step"] for v in r],label=kind)
        axes[1].plot([v["step"] for v in r],[v["ratio"] for v in r],label=kind)
    for ax in axes:ax.axvline(30,ls=":",color="gray");ax.legend();ax.set_xlabel("Step")
    axes[0].set_ylabel("Training MSE");axes[1].set_ylabel("Update-to-weight ratio")
    fig.tight_layout();fig.savefig(OUT/"schedules_and_ratios.png");plt.close(fig)
    return best

def sweep():
    x,y=data();rows=[];best={}
    rates=(.0001,.0003,.001,.003,.01,.03,.1)
    for width in (256,512,1024):
        group=[]
        for lr in rates:
            torch.manual_seed(SEED);net=Model(width)
            opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.01)
            for _ in range(60):
                opt.zero_grad();loss=torch.nn.functional.mse_loss(net(x),y);loss.backward();opt.step()
            with torch.no_grad():last=torch.nn.functional.mse_loss(net(x),y).item()
            group.append(dict(width=width,lr=lr,final_train_mse=last,steps=60))
        rows+=group;b=min(group,key=lambda r:r["final_train_mse"])
        best[str(width)]={"lr":b["lr"],"loss":b["final_train_mse"]}
    save("lr_sweep.csv",rows)
    fig,ax=plt.subplots()
    for width in (256,512,1024):
        r=[v for v in rows if v["width"]==width]
        ax.semilogx([v["lr"] for v in r],[v["final_train_mse"] for v in r],marker="o",label=str(width))
        b=min(r,key=lambda q:q["final_train_mse"])
        ax.scatter([b["lr"]],[b["final_train_mse"]],s=120,facecolors="none",edgecolors="black")
    ax.set(xlabel="Learning rate",ylabel="MSE after 60 steps",title="Learning rate by hidden width")
    ax.legend();fig.tight_layout();fig.savefig(OUT/"lr_sweep.png");plt.close(fig)
    return best

if __name__=="__main__":
    result={"seed":SEED,"torch_version":torch.__version__,"device":"cpu",
            "hand_max_abs_error":adam(),"bias_gap_thresholds":bias(),
            "schedule_loss_after_200":schedules(),"lr_sweep_best":sweep()}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
