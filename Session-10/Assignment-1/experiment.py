import math, struct, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_num_threads(1)
torch.manual_seed(10)
V,H=32,24
class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(V,H)
        self.ff=nn.Linear(H,H)
        self.head=nn.Linear(H,V)
    def forward(self,x,trace=False):
        e=self.emb(x); h=torch.tanh(self.ff(e)); y=self.head(h)
        if trace:
            for name,t,meaning in [
                ("tokens",x,"batch,sequence"),
                ("embedding",e,"batch,sequence,hidden"),
                ("hidden",h,"batch,sequence,hidden"),
                ("logits",y,"batch,sequence,vocab"),
                ("shifted logits",y[:,:-1],"batch,prediction positions,vocab"),
                ("targets",x[:,1:],"batch,prediction positions")]:
                print(name,tuple(t.shape),meaning)
        return y
a=torch.arange(1,10); b=torch.arange(11,15)
def objective(net,docs,broken=False):
    sums=[]; counts=[]
    for d in docs:
        logits=net(d[None])[:,:-1].reshape(-1,V)
        labels=d[1:].reshape(-1)
        sums.append(F.cross_entropy(logits,labels,reduction="sum"))
        counts.append(labels.numel())
    return sum(z/n for z,n in zip(sums,counts))/len(sums) if broken else sum(sums)/sum(counts)

m=LM().double()
print("device",next(m.parameters()).device)
m(a[None],trace=True)
print("loss shape",tuple(objective(m,[a,b]).shape),"scalar")
p=m.ff.weight
m.zero_grad(set_to_none=True)
objective(m,[a,b]).backward()
analytical=p.grad[0,0].item(); orig=p[0,0].item(); eps=1e-5
with torch.no_grad(): p[0,0]=orig+eps
plus=objective(m,[a,b]).item()
with torch.no_grad(): p[0,0]=orig-eps
minus=objective(m,[a,b]).item()
with torch.no_grad(): p[0,0]=orig
numerical=(plus-minus)/(2*eps)
print("gradient analytical",analytical,"finite_difference",numerical,"abs_error",abs(numerical-analytical))
assert abs(numerical-analytical)<1e-7
print("target counts",[len(a)-1,len(b)-1])
print("objective correct",objective(m,[a,b]).item(),"wrong",objective(m,[a,b],True).item())

seed=LM(); correct=copy.deepcopy(seed); broken=copy.deepcopy(seed)
oc=torch.optim.SGD(correct.parameters(),lr=.35)
ob=torch.optim.SGD(broken.parameters(),lr=.35)
curves=[[],[]]; norms=[]
tic=time.perf_counter()
for step in range(80):
    for j,(net,opt,wrong) in enumerate([(correct,oc,False),(broken,ob,True)]):
        opt.zero_grad(set_to_none=True)
        loss=objective(net,[a,b],wrong)
        loss.backward()
        if j==0:
            gn=math.sqrt(sum(float((q.grad**2).sum()) for q in net.parameters() if q.grad is not None))
            norms.append(gn)
            print(f"step={step:02d} before_update_loss={loss.item():.8f} grad_norm={gn:.8f}")
        curves[j].append(objective(net,[a,b]).item())
        opt.step()
seconds=time.perf_counter()-tic
print("step 0 grad norm before first update",norms[0],
      "pre-update loss",curves[0][0],"loss at next step",curves[0][1])
print("final correct",curves[0][-1],"final broken",curves[1][-1])
plt.figure(figsize=(8,4))
plt.plot(curves[0],label="Correct: weighted by target counts")
plt.plot(curves[1],label="Broken: mean of microbatch means")
plt.xlabel("Optimizer step");plt.ylabel("Token-weighted evaluation cross-entropy")
plt.title("Unequal microbatch accumulation");plt.legend();plt.grid(alpha=.2)
plt.tight_layout();plt.savefig("accumulation_comparison.png",dpi=160);plt.show()
params=sum(x.numel() for x in correct.parameters())
proxy=6*params*((len(a)-1)+(len(b)-1))*80*2
print("parameter_count",params,"wall_s",seconds,"rough_flop_proxy",proxy,"rough_flop_per_s",proxy/seconds)
print("MFU unavailable: CPU-only run; GPU MFU requires theoretical hardware peak FLOP/s and timed CUDA steps.")
print("For CUDA: MFU = estimated model FLOPs / (step_seconds * theoretical_peak_FLOPs_per_second)")
bits=int.from_bytes(struct.pack(">f",0.1),"big")
bf=(bits+0x7fff+((bits>>16)&1))>>16
for name,raw,n,val in [
    ("fp32",bits,32,struct.unpack(">f",bits.to_bytes(4,"big"))[0]),
    ("bf16",bf,16,struct.unpack(">f",(bf<<16).to_bytes(4,"big"))[0]),
    ("fp8 E4M3FN",0b00011101,8,(1+5/8)*2**(-4))]:
    print(name,format(raw,f"0{n}b"),"value",val,"abs_error",abs(val-0.1))
print("Precision choice: bf16 for training on supported hardware; fp32 for sensitive operations; fp8 needs validated scaling.")
