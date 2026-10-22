"""V5 Session 6: deterministic, auditable miniature training data execution system."""
import hashlib, json, math, random, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'submission_artifacts'
SEED = 615
WIDTH = 16
LANES = ('indic', 'reasoning', 'agentic', 'long_context', 'general')
STAGES = [
    {'name':'foundation','weights':{'indic':.30,'reasoning':.20,'agentic':.15,'long_context':.10,'general':.25},'floor':{'indic':.20,'reasoning':.10}},
    {'name':'capability','weights':{'indic':.25,'reasoning':.25,'agentic':.20,'long_context':.15,'general':.15},'floor':{'indic':.20,'reasoning':.10}},
]
TIER = ('verified', 'unverified', 'translated', 'synthetic')

def digest(x):
    b = x if isinstance(x, bytes) else json.dumps(x,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(b).hexdigest()

def write(path, item):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(item,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def log(msg):
    with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(msg+'\n')
    print(msg)

def corpus():
    examples=[]
    for lane in LANES:
        for j in range(16):
            tier=TIER[j%4] if lane=='indic' else 'native'
            text=f'{lane} {tier} example {j} explain derive verify context action reasoning evidence result '+('step observation tool response ' if lane=='agentic' else 'source context calculation solution ')* (2+(j%4))
            examples.append({'id':f'{lane}-{j:02}', 'lane':lane,'tier':tier,'split':'train','text':text,'license':'CC0-1.0','source':f'fixture:{lane}'})
    for split in ('eval','validation'):
        for k in range(2):examples.append({'id':f'{split}-{k}','lane':'general','tier':'native','split':split,'text':'held out secret assessment '+str(k),'license':'CC0-1.0','source':'fixture:heldout'})
    return examples

def tokenizer(examples):
    vocabulary=sorted({word for x in examples if x['split']=='train' for word in x['text'].lower().split()})
    spec={'kind':'whitespace_v1','pad':0,'eos':1,'unk':2,'vocab':{x:i+3 for i,x in enumerate(vocabulary)}}
    write(OUT/'tokenizer.json',spec)
    return spec

def encode(text,tok):return [tok['vocab'].get(w,2) for w in text.lower().split()]+[1]

def build_shards(examples,tok):
    shards=[]
    for ex in examples:
        tokens=encode(ex['text'],tok)
        rel=f"{ex['split']}/{ex['id']}.json"; p=OUT/'shards'/rel
        data={'id':ex['id'],'tokens':tokens};write(p,data)
        manifest={'id':ex['id'],'path':f'shards/{rel}','source':ex['source'],'license':ex['license'],
                  'split':ex['split'],'lane':ex['lane'],'tier':ex['tier'],'languages':['hi','te'] if ex['lane']=='indic' else ['en'],
                  'token_count':len(tokens),'tokenizer_hash':digest(tok),'content_hash':digest(data),
                  'cleaning_script_hash':digest('lowercase-whitespace-v1')}
        write(OUT/'manifests'/f"{ex['id']}.json",manifest);shards.append(manifest)
    log(f'shards created: {len(shards)} immutable content-addressed manifest records')
    for m in shards:
        assert digest(read(OUT/m['path']))==m['content_hash']
        assert digest(read(OUT/'tokenizer.json'))==m['tokenizer_hash']
        assert len(read(OUT/m['path'])['tokens'])==m['token_count']
    log('manifests validated');log('[PASS] tokenizer_hash_verified')
    return shards

def firewall(shards):
    allowed=[]; blocked=[]
    for m in shards:
        if m['split']!='train':
            blocked.append({'id':m['id'],'split':m['split'],'decision':'BLOCK','reason':'heldout_firewall'})
        else:allowed.append(m)
    write(OUT/'ledgers'/'blocked_shards.json',blocked)
    assert blocked and all(m['split']=='train' for m in allowed)
    log(f'evaluation data blocked: {len(blocked)} eval/validation shards');log('[PASS] eval_shard_blocked')
    return allowed

def opus(shards):
    decisions=[]; accepted=[]
    for m in shards:
        j=int(m['id'].rsplit('-',1)[-1]); tier=m['tier']
        if tier=='synthetic' and j%3==0: decision='REJECT';reason='synthetic_quality'
        elif (tier=='unverified' and j%3==1) or (m['lane']=='general' and j%5==1):decision='DEFER';reason='needs_verification'
        else:decision='ACCEPT';reason='quality_gate'
        if m['lane']=='indic' and decision=='DEFER':decision='OVERRIDE_ACCEPT';reason='indic_protected_floor'
        decisions.append({'id':m['id'],'lane':m['lane'],'tier':tier,'decision':decision,'reason':reason})
        if decision in ('ACCEPT','OVERRIDE_ACCEPT'):accepted.append(m)
    write(OUT/'ledgers'/'opus.json',decisions)
    assert {'REJECT','DEFER','ACCEPT','OVERRIDE_ACCEPT'} <= set(d['decision'] for d in decisions)
    log('OPUS decisions recorded: accept/reject/defer/protected-floor override')
    return accepted

def schedule(shards, n=24):
    by_lane=defaultdict(list)
    for m in shards:by_lane[m['lane']].append(m)
    for lane in LANES:assert by_lane[lane],lane
    plan=[]; cursor=Counter();counts_by_stage=[]
    for stage_idx,stage in enumerate(STAGES):
        counts=Counter();target=n//len(STAGES)
        for i in range(target):
            lane=max(LANES,key=lambda k:((i+1)*stage['weights'][k]-counts[k],-LANES.index(k)))
            m=by_lane[lane][cursor[lane]%len(by_lane[lane])];cursor[lane]+=1;counts[lane]+=1
            plan.append({'batch_index':len(plan),'stage':stage['name'],'lane':lane,'shard_id':m['id']})
        for lane,floor in stage['floor'].items():assert counts[lane]/target+1e-12>=floor,(lane,counts)
        counts_by_stage.append({'stage':stage['name'],'planned':stage['weights'],'actual':{k:counts[k]/target for k in LANES},'floor':stage['floor']})
    write(OUT/'mixture.json',{'stages':counts_by_stage,'plan':plan})
    log('mixture compiled: deterministic weighted-deficit stage allocation with protected floors')
    return plan

def pack(item,lookup):
    m=lookup[item['shard_id']]; seq=read(OUT/m['path'])['tokens']
    policy='contiguous' if m['lane']=='long_context' else ('turn_boundary' if m['lane']=='agentic' else 'document_boundary')
    usable=seq[:WIDTH]; n=len(usable)
    ids=usable+[0]*(WIDTH-n)
    attn=[1]*n+[0]*(WIDTH-n)
    pos=list(range(n))+[0]*(WIDTH-n)
    loss=[0]+[1]*(n-1)+[0]*(WIDTH-n)
    causal=[[int(i<n and j<n and j<=i) for j in range(WIDTH)] for i in range(WIDTH)]
    batch={'id':f"batch-{item['batch_index']:04}",'index':item['batch_index'],'stage':item['stage'],
           'lane':m['lane'],'shard_id':m['id'],'token_span':[0,n], 'policy':policy,
           'input_ids':ids,'attention_mask':attn,'position_ids':pos,'loss_mask':loss,'causal_mask':causal,
           'content_hash':m['content_hash']}
    batch['hash']=digest(batch)
    assert sum(loss)==max(0,n-1) and sum(attn)==n
    assert all(causal[i][j]==int(i<n and j<n and j<=i) for i in range(WIDTH) for j in range(WIDTH))
    return batch

class TinyBigram:
    """Actual SGD on a two-class sampled next-token objective; no dependencies."""
    def __init__(self,vocab):self.vocab=vocab;self.w={}
    def step(self,b,lr=.08):
        losses=[]; updates=defaultdict(lambda:defaultdict(float))
        ids=b['input_ids'];mask=b['loss_mask']
        for i in range(1,WIDTH):
            if not mask[i]:continue
            src,tgt=ids[i-1],ids[i];row=self.w.get(str(src),{})
            neg=2 if tgt!=2 else 1
            a=row.get(str(tgt),0.0); c=row.get(str(neg),0.0)
            p=1/(1+math.exp(max(-40,min(40,c-a))))
            losses.append({'position':i,'source_token':src,'target_token':tgt,'loss':-math.log(max(p,1e-12))})
            updates[str(src)][str(tgt)]+=p-1;updates[str(src)][str(neg)]+=1-p
        for src,dx in updates.items():
            row=self.w.setdefault(src,{})
            for tgt,g in dx.items():row[tgt]=row.get(tgt,0.0)-lr*g
        return losses
    def state(self):return {'vocab':self.vocab,'weights':self.w}
    @classmethod
    def load(cls,state):
        o=cls(state['vocab']);o.w=state['weights'];return o

def checkpoint(model,next_idx,consumption,learning,label):
    state={'next_index':next_idx,'consumption_offset':len(consumption),'learning_offset':len(learning),
           'last_batch_hash':consumption[-1]['batch_hash'] if consumption else None,
           'model':model.state(),'rng_seed':SEED}
    state['state_hash']=digest(state)
    path=OUT/'checkpoints'/f'{label}.json';write(path,state)
    return path

def run():
    if OUT.exists():
        import shutil;shutil.rmtree(OUT)
    OUT.mkdir(parents=True);t0=time.perf_counter()
    random.seed(SEED)
    examples=corpus();tok=tokenizer(examples)
    shards=build_shards(examples,tok);allowed=firewall(shards)
    accepted=opus(allowed);plan=schedule(accepted)
    lookup={m['id']:m for m in accepted};batches=[pack(p,lookup) for p in plan]
    write(OUT/'ledgers'/'batch_catalog.json',batches)
    log(f'batches packed: {len(batches)}; width={WIDTH}')
    model=TinyBigram(len(tok['vocab'])+3);consumption=[];learning=[]
    crash_at=10
    for b in batches[:crash_at]:
        losses=model.step(b)
        consumption.append({'offset':len(consumption),'batch_id':b['id'],'batch_hash':b['hash'],'shard_id':b['shard_id'],
                            'span':b['token_span'],'lane':b['lane'],'tokens':sum(b['attention_mask']),'useful_tokens':len(losses)})
        learning.append({'offset':len(learning),'batch_id':b['id'],'batch_hash':b['hash'],
                         'shard_id':b['shard_id'],'source_content_hash':b['content_hash'],
                         'token_losses':losses,'mean_loss':sum(x['loss'] for x in losses)/len(losses)})
    cp=checkpoint(model,crash_at,consumption,learning,'precrash')
    write(OUT/'ledgers'/'consumption.json',consumption);write(OUT/'ledgers'/'learning.json',learning)
    log(f'checkpoint saved: offset={crash_at}');log('[PASS] checkpoint_saved')
    try:
        log('crash simulated: deliberate RuntimeError after atomic checkpoint and ledger writes')
        raise RuntimeError('intentional_crash')
    except RuntimeError as e:assert str(e)=='intentional_crash'
    saved=read(cp);assert digest({k:v for k,v in saved.items() if k!='state_hash'})==saved['state_hash']
    consumption=read(OUT/'ledgers'/'consumption.json');learning=read(OUT/'ledgers'/'learning.json')
    assert len(consumption)==saved['consumption_offset']==saved['next_index']
    assert len(learning)==saved['learning_offset']==saved['next_index']
    model=TinyBigram.load(saved['model']);expected=batches[saved['next_index']]
    actual=pack(plan[saved['next_index']],lookup)
    resume_match=actual['id']==expected['id'] and actual['hash']==expected['hash']
    assert resume_match;log('run resumed from committed checkpoint and ledger offsets');log('[PASS] resume_next_batch_matched')
    for b in batches[saved['next_index']:]:
        losses=model.step(b)
        consumption.append({'offset':len(consumption),'batch_id':b['id'],'batch_hash':b['hash'],'shard_id':b['shard_id'],
                            'span':b['token_span'],'lane':b['lane'],'tokens':sum(b['attention_mask']),'useful_tokens':len(losses)})
        learning.append({'offset':len(learning),'batch_id':b['id'],'batch_hash':b['hash'],
                         'shard_id':b['shard_id'],'source_content_hash':b['content_hash'],
                         'token_losses':losses,'mean_loss':sum(x['loss'] for x in losses)/len(losses)})
    write(OUT/'ledgers'/'consumption.json',consumption);write(OUT/'ledgers'/'learning.json',learning)
    final_cp=checkpoint(model,len(batches),consumption,learning,'final')
    assert [c['batch_id'] for c in consumption]==[b['id'] for b in batches]
    assert len({c['batch_id'] for c in consumption})==len(batches)
    interval=(3,8)
    replay=[pack(p,lookup) for p in plan[interval[0]:interval[1]]]
    original=batches[interval[0]:interval[1]]
    matches=all((a['id'],a['token_span'],a['hash'])==(b['id'],b['token_span'],b['hash']) for a,b in zip(replay,original))
    assert matches;write(OUT/'ledgers'/'replay.json',{'interval':interval,'original_hashes':[b['hash'] for b in original],
                  'replayed_hashes':[b['hash'] for b in replay],'matched':matches})
    log('historical stream replayed: ids, spans and hashes verified');log('[PASS] replay_hash_matched')
    branch_model=TinyBigram.load(saved['model']);branch=[]
    for b in batches[crash_at:crash_at+3]:
        ls=branch_model.step(b,lr=.03);branch.append({'batch_id':b['id'],'hash':b['hash'],'loss':sum(x['loss'] for x in ls)/len(ls)})
    write(OUT/'ledgers'/'fork.json',{'parent_checkpoint':str(cp.relative_to(OUT)),'alternative_lr':.03,'batches':branch,
                                   'branched_model_hash':digest(branch_model.state())})
    log('branch forked from precrash checkpoint with alternative learning rate')
    duration=time.perf_counter()-t0
    useful=sum(c['useful_tokens'] for c in consumption);occupied=sum(c['tokens'] for c in consumption)
    performance={'wall_seconds':duration,'batches':len(batches),'capacity_tokens':len(batches)*WIDTH,
                 'occupied_tokens':occupied,'useful_loss_tokens':useful,
                 'packing_utilization':occupied/(len(batches)*WIDTH),
                 'useful_loss_tokens_per_second':useful/duration}
    write(OUT/'performance.json',performance)
    assert sum(len(x['token_losses']) for x in learning)==useful
    assert all(c['batch_hash']==l['batch_hash'] for c,l in zip(consumption,learning))
    assert all(c['shard_id'] in lookup and lookup[c['shard_id']]['split']=='train' for c in consumption)
    log('audit completed: all batches contiguous and unique, hashes and learning lineage valid')
    log('performance measured: occupied={} useful={} wall_seconds={:.6f}'.format(occupied,useful,duration))
    checks={
        'Tokenizer integrity':(all(m['tokenizer_hash']==digest(tok) for m in shards),'manifests/; tokenizer.json'),
        'Evaluation firewall':(len(read(OUT/'ledgers'/'blocked_shards.json'))==4,'ledgers/blocked_shards.json'),
        'Packing correctness':(all(sum(b['loss_mask'])==sum(b['attention_mask'])-1 for b in batches),'ledgers/batch_catalog.json'),
        'Mixture compliance':(all(x['actual'][k]>=v for x in read(OUT/'mixture.json')['stages'] for k,v in x['floor'].items()),'mixture.json'),
        'OPUS audit trail':(len(read(OUT/'ledgers'/'opus.json'))==len(allowed),'ledgers/opus.json'),
        'Crash recovery':(resume_match and len(consumption)==len(batches),'checkpoints/precrash.json; ledgers/consumption.json'),
        'Replay':(matches,'ledgers/replay.json'),
        'Learning trace':(len(learning)==len(consumption) and useful>0,'ledgers/learning.json'),
        'Throughput':(duration>0 and occupied<=len(batches)*WIDTH,'performance.json'),
        'Fork':(len(branch)==3,'ledgers/fork.json'),
        'Checkpoint':(read(final_cp)['next_index']==len(batches),'checkpoints/final.json')
    }
    evidence={'requirements':{k:{'passed':bool(v),'result':'PASS' if v else 'FAIL','evidence':path} for k,(v,path) in checks.items()},
              'resume':{'expected_batch_id':expected['id'],'actual_batch_id':actual['id'],
                        'expected_hash':expected['hash'],'actual_hash':actual['hash']},
              'replay':{'interval':interval,'matching_hashes':matches},
              'ledger_offsets':{'consumption':len(consumption),'learning':len(learning)},
              'performance':performance,'overall_pass':all(v for v,_ in checks.values())}
    write(OUT/'evidence.json',evidence)
    md=['# V5 Training Data Execution Evidence','','Generated from the actual run, manifests and ledgers.','','| Requirement | Result | Evidence |','|---|---|---|']
    md += [f'| {k} | {"PASS" if v else "FAIL"} | `{p}` |' for k,(v,p) in checks.items()]
    md += ['',f"Resume next batch: `{expected['id']}` and SHA-256 `{expected['hash']}`.",
           f"Replay interval: batches {interval[0]}–{interval[1]-1}, all matching SHA-256 hashes.",
           f"Packing utilization: {performance['packing_utilization']:.2%}; useful loss tokens/sec: {performance['useful_loss_tokens_per_second']:.2f}."]
    (OUT/'evidence.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    log('[PASS] evidence_generated')
    assert evidence['overall_pass'];return evidence

if __name__=='__main__':run()
