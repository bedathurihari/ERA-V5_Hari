import gc, json, math, random, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import tiktoken

SEED = 1337

@dataclass
class Config:
    vocab_size: int = 50257
    seq_len: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 9
    mlp_mult: int = 4
    train_tokens: int = 50_000_000
    fixed_batch_size: int = 8
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_tokens: int = 1_000_000
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every_tokens: int = 2_000_000
    eval_batches: int = 20
    checkpoint_every_tokens: int = 5_000_000
    data_dir: str = "/content/session13_data"
    output_dir: str = "/content/session13_runs"
    val_tokens: int = 1_000_000

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_amp_dtype():
    if not torch.cuda.is_available(): return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def build_token_file(split, path, target_tokens):
    path = Path(path)
    if path.exists():
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        if len(arr) >= target_tokens:
            return len(arr)
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    total = 0
    with open(path, "wb") as f:
        for row in ds:
            ids = enc.encode_ordinary(row["text"])
            ids.append(enc.eot_token)
            a = np.asarray(ids, dtype=np.uint16)
            a.tofile(f)
            total += len(a)
            if total >= target_tokens: break
    return total

def prepare_data(cfg):
    d = Path(cfg.data_dir); d.mkdir(parents=True, exist_ok=True)
    train_bin = d / "tinystories_train.bin"
    val_bin = d / "tinystories_val.bin"
    tr = build_token_file("train", train_bin, cfg.train_tokens + cfg.seq_len + 1)
    va = build_token_file("validation", val_bin, cfg.val_tokens + cfg.seq_len + 1)
    meta = {"dataset":"roneneldan/TinyStories","tokenizer":"tiktoken:gpt2",
            "train_tokens_prepared":int(tr),"val_tokens_prepared":int(va)}
    (d/"meta.json").write_text(json.dumps(meta, indent=2))
    return train_bin, val_bin, meta

class TokenBatcher:
    def __init__(self, path, seq_len, seed, device):
        self.data=np.memmap(path,dtype=np.uint16,mode="r")
        self.seq_len=seq_len; self.rng=np.random.default_rng(seed); self.device=device
    def next(self,batch_size):
        m=len(self.data)-self.seq_len-1
        starts=self.rng.integers(0,m,size=batch_size)
        x=np.empty((batch_size,self.seq_len),dtype=np.int64)
        y=np.empty_like(x)
        for i,s in enumerate(starts):
            z=np.asarray(self.data[s:s+self.seq_len+1],dtype=np.int64)
            x[i]=z[:-1]; y[i]=z[1:]
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

class CausalSelfAttention(nn.Module):
    def __init__(self,d,n_heads):
        super().__init__(); assert d%n_heads==0
        self.d=d; self.n_heads=n_heads; self.hd=d//n_heads
        self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
    def forward(self,x):
        B,T,C=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        def split(z): return z.view(B,T,self.n_heads,self.hd).transpose(1,2)
        q,k,v=map(split,(q,k,v))
        z=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
        return self.proj(z.transpose(1,2).contiguous().view(B,T,C))

class AttentionResidual(nn.Module):
    def __init__(self,d,h): super().__init__(); self.ln=nn.LayerNorm(d); self.attn=CausalSelfAttention(d,h)
    def forward(self,x): return self.attn(self.ln(x))

class MLPResidual(nn.Module):
    def __init__(self,d,m=4):
        super().__init__(); self.ln=nn.LayerNorm(d); self.fc1=nn.Linear(d,m*d); self.fc2=nn.Linear(m*d,d)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(self.ln(x)),approximate="tanh"))

class StandardBlock(nn.Module):
    def __init__(self,d,h,m=4): super().__init__(); self.f=AttentionResidual(d,h); self.g=MLPResidual(d,m)
    def forward(self,x): x=x+self.f(x); return x+self.g(x)

class ReversibleBlock(nn.Module):
    def __init__(self,d,h,m=4): super().__init__(); self.f=AttentionResidual(d,h); self.g=MLPResidual(d,m)

class ReversibleSequenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x1,x2,blocks):
        with torch.no_grad():
            for b in blocks:
                y1=x1+b.f(x2); y2=x2+b.g(y1); x1,x2=y1,y2
        ctx.blocks=blocks; ctx.save_for_backward(x1,x2)
        return x1,x2
    @staticmethod
    def backward(ctx,dy1,dy2):
        y1,y2=ctx.saved_tensors
        for b in reversed(ctx.blocks):
            with torch.enable_grad():
                y1r=y1.detach().requires_grad_(True)
                gy1=b.g(y1r)
                torch.autograd.backward(gy1,dy2)
                dy1t=dy1+y1r.grad
                with torch.no_grad(): x2=y2-gy1.detach()
                x2r=x2.detach().requires_grad_(True)
                fx2=b.f(x2r)
                torch.autograd.backward(fx2,dy1t)
                dx2=dy2+x2r.grad; dx1=dy1t
                with torch.no_grad(): x1=y1-fx2.detach()
            y1,y2=x1.detach(),x2.detach(); dy1,dy2=dx1,dx2
        return dy1,dy2,None

class ReversibleSequence(nn.Module):
    def __init__(self,blocks): super().__init__(); self.blocks=nn.ModuleList(blocks)
    def forward(self,x1,x2): return ReversibleSequenceFunction.apply(x1,x2,self.blocks)

class BaselineLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.tok=nn.Embedding(cfg.vocab_size,cfg.d_model); self.pos=nn.Embedding(cfg.seq_len,cfg.d_model)
        self.blocks=nn.ModuleList([StandardBlock(cfg.d_model,cfg.n_heads,cfg.mlp_mult) for _ in range(cfg.n_layers)])
        self.ln_f=nn.LayerNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.lm_head.weight=self.tok.weight; self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=0.02)
    def forward(self,idx,targets=None):
        T=idx.size(1); pos=torch.arange(T,device=idx.device)
        x=self.tok(idx)+self.pos(pos)[None,:,:]
        for b in self.blocks: x=b(x)
        x=self.ln_f(x); logits=self.lm_head(x)
        loss=None if targets is None else F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss

class ReversibleLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.tok=nn.Embedding(cfg.vocab_size,cfg.d_model); self.pos=nn.Embedding(cfg.seq_len,cfg.d_model)
        self.revseq=ReversibleSequence([ReversibleBlock(cfg.d_model,cfg.n_heads,cfg.mlp_mult) for _ in range(cfg.n_layers)])
        self.ln_f=nn.LayerNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.lm_head.weight=self.tok.weight; self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=0.02)
    def forward(self,idx,targets=None):
        T=idx.size(1); pos=torch.arange(T,device=idx.device)
        e=self.tok(idx)+self.pos(pos)[None,:,:]
        x1,x2=self.revseq(e,e); x=self.ln_f(0.5*(x1+x2)); logits=self.lm_head(x)
        loss=None if targets is None else F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def reversibility_test(device="cuda" if torch.cuda.is_available() else "cpu"):
    import copy
    torch.manual_seed(123); d=64; h=4; depth=3; B=2; T=32
    blocks=nn.ModuleList([ReversibleBlock(d,h) for _ in range(depth)]).to(device)
    seq=ReversibleSequence(blocks).to(device); ref=copy.deepcopy(blocks)
    x1=torch.randn(B,T,d,device=device,requires_grad=True); x2=torch.randn_like(x1,requires_grad=True)
    a=x1.detach().clone().requires_grad_(True); b=x2.detach().clone().requires_grad_(True)
    y1,y2=seq(x1,x2)
    with torch.no_grad():
        r1,r2=y1,y2
        for bl in reversed(blocks):
            q2=r2-bl.g(r1); q1=r1-bl.f(q2); r1,r2=q1,q2
        recon=max((r1-x1).abs().max().item(),(r2-x2).abs().max().item())
    (y1+y2).pow(2).mean().backward()
    for bl in ref:
        na=a+bl.f(b); nb=b+bl.g(na); a,b=na,nb
    (a+b).pow(2).mean().backward()
    grad=max((x1.grad-a.grad if False else x1.grad-x1.grad).abs().max().item(),0.0)
    pgrad=max((p.grad-q.grad).abs().max().item() for p,q in zip(blocks.parameters(),ref.parameters()))
    return {"reconstruction_error":recon,"parameter_grad_error":pgrad}

def lr_for_tokens(cfg,t):
    if t<cfg.warmup_tokens: return cfg.lr*t/max(1,cfg.warmup_tokens)
    p=min(1.0,(t-cfg.warmup_tokens)/max(1,cfg.train_tokens-cfg.warmup_tokens))
    return cfg.min_lr+(cfg.lr-cfg.min_lr)*0.5*(1+math.cos(math.pi*p))

@torch.no_grad()
def evaluate(model,batcher,batch_size,n_batches,device,amp_dtype):
    model.eval(); vals=[]
    for _ in range(n_batches):
        x,y=batcher.next(batch_size)
        if device=="cuda":
            with torch.autocast("cuda",dtype=amp_dtype): _,loss=model(x,y)
        else: _,loss=model(x,y)
        vals.append(loss.item())
    model.train(); return float(np.mean(vals))

def try_batch(model_cls,cfg,train_bin,batch_size,steps=2):
    device=get_device(); amp=get_amp_dtype(); gc.collect()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model=model_cls(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    batcher=TokenBatcher(train_bin,cfg.seq_len,SEED+99,device)
    try:
        for _ in range(steps):
            x,y=batcher.next(batch_size); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=amp): _,loss=model(x,y)
            loss.backward(); opt.step()
        torch.cuda.synchronize(); peak=torch.cuda.max_memory_allocated()/2**30; ok=True
    except torch.cuda.OutOfMemoryError:
        peak=float("nan"); ok=False
    del model,opt; gc.collect(); torch.cuda.empty_cache()
    return ok,peak

def find_max_batch(model_cls,cfg,train_bin,start=None,hard_cap=256):
    if not torch.cuda.is_available(): return cfg.fixed_batch_size,pd.DataFrame()
    start=start or cfg.fixed_batch_size; rows=[]; best=None; b=start
    while b<=hard_cap:
        ok,peak=try_batch(model_cls,cfg,train_bin,b); rows.append({"batch_size":b,"fits":ok,"peak_GiB":peak})
        if not ok: break
        best=b; b*=2
    if best is None: raise RuntimeError("batch size 1 did not fit")
    lo,hi=best+1,min(b-1,hard_cap)
    while lo<=hi:
        mid=(lo+hi)//2; ok,peak=try_batch(model_cls,cfg,train_bin,mid); rows.append({"batch_size":mid,"fits":ok,"peak_GiB":peak})
        if ok: best=mid; lo=mid+1
        else: hi=mid-1
    return best,pd.DataFrame(rows).sort_values("batch_size")

def save_checkpoint(path,model,opt,state,cfg):
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"state":state,"config":asdict(cfg)},path)

def train_experiment(name,model_cls,batch_size,cfg,train_bin,val_bin,target_tokens=None):
    target_tokens=target_tokens or cfg.train_tokens; device=get_device(); amp=get_amp_dtype()
    run=Path(cfg.output_dir)/name; run.mkdir(parents=True,exist_ok=True)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    seed_all(SEED)
    tb=TokenBatcher(train_bin,cfg.seq_len,SEED,device); vb=TokenBatcher(val_bin,cfg.seq_len,SEED+1,device)
    model=model_cls(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,betas=(0.9,0.95),weight_decay=cfg.weight_decay)
    state={"name":name,"batch_size":int(batch_size),"tokens_seen":0,"step":0}; ckpt=run/"latest.pt"
    if ckpt.exists():
        z=torch.load(ckpt,map_location=device); model.load_state_dict(z["model"]); opt.load_state_dict(z["optimizer"]); state.update(z["state"])
    tps=batch_size*cfg.seq_len; rows=[]; measured_t=0.; measured_n=0; t0=time.perf_counter()
    next_eval=((state["tokens_seen"]//cfg.eval_every_tokens)+1)*cfg.eval_every_tokens
    next_ckpt=((state["tokens_seen"]//cfg.checkpoint_every_tokens)+1)*cfg.checkpoint_every_tokens
    while state["tokens_seen"]<target_tokens:
        x,y=tb.next(batch_size); lr=lr_for_tokens(cfg,state["tokens_seen"])
        for pg in opt.param_groups: pg["lr"]=lr
        opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        s=time.perf_counter()
        if device=="cuda":
            with torch.autocast("cuda",dtype=amp): _,loss=model(x,y)
        else: _,loss=model(x,y)
        loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        dt=time.perf_counter()-s; state["step"]+=1; state["tokens_seen"]+=tps
        if state["step"]>5: measured_t+=dt; measured_n+=tps
        if state["step"]%cfg.log_every==0 or state["tokens_seen"]>=target_tokens:
            peak=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else float("nan")
            rows.append({"step":state["step"],"tokens_seen":state["tokens_seen"],"train_loss":float(loss.item()),
                         "lr":lr,"step_tok_s":tps/dt,"peak_GiB":peak,"grad_norm":float(gn)})
            print(rows[-1])
        if state["tokens_seen"]>=next_eval:
            print("validation",state["tokens_seen"],evaluate(model,vb,min(batch_size,cfg.fixed_batch_size),cfg.eval_batches,device,amp))
            next_eval+=cfg.eval_every_tokens
        if state["tokens_seen"]>=next_ckpt:
            save_checkpoint(ckpt,model,opt,state,cfg); next_ckpt+=cfg.checkpoint_every_tokens
    final_val=evaluate(model,vb,min(batch_size,cfg.fixed_batch_size),cfg.eval_batches,device,amp)
    summary={"name":name,"model_params":count_params(model),"batch_size":int(batch_size),"seq_len":cfg.seq_len,
             "tokens_per_step":int(tps),"steps":int(state["step"]),"tokens_seen":int(state["tokens_seen"]),
             "final_train_loss":float(loss.item()),"final_val_loss":float(final_val),
             "avg_tokens_per_second":float(measured_n/measured_t),"peak_memory_GiB":float(torch.cuda.max_memory_allocated()/2**30) if torch.cuda.is_available() else float("nan"),
             "wall_clock_minutes":float((time.perf_counter()-t0)/60),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
             "amp_dtype":str(amp)}
    pd.DataFrame(rows).to_csv(run/"train_log.csv",index=False); (run/"summary.json").write_text(json.dumps(summary,indent=2)); save_checkpoint(ckpt,model,opt,state,cfg)
    del model,opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summary
