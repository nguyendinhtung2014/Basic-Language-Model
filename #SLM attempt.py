#SLM attempt
step=0


debug_mode=False
load_adam=False
checkpoint_gap=10
file_path="model.json"
load=input("Do you want to load an existing model? (True/False) ").lower()=="true"
if load:
    load_path=input("input the file you want to extract the model from: ")


import math
import re
from collections import Counter
from collections import deque
import random
import json
import os
import ctypes
dll_path=os.path.join(os.path.dirname(__file__),"matmul.dll")
_lib=ctypes.CDLL(dll_path)
_lib.mul.argtypes=[
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
_lib.mul.restype=None
_lib.softmax.argtypes=[
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
_lib.softmax.restype = None
#math stuff
def sceb(probs,tid):
    grad=probs.copy()
    grad[tid]-=1.0
    return grad
def softmax(l):
    N=len(l)
    L=(ctypes.c_float * (N))(*l)
    res=(ctypes.c_float * (N))()
    _lib.softmax(L,N,res)
    return list(res)
def cent(probs:list[float],tid:int):return -math.log(probs[tid]+1e-12)
def lrelu(x:float):return x if x>0 else x*0.01
def actmat(mat):
    return [list(map(lambda x:x if x>0 else x*0.01,row)) for row in mat]
def actmatd(mat):
    return [list(map(lambda x:1.0 if x>0 else 0.01,row)) for row in mat]
def add(a,b):
    bias_row=b[0]
    return [[ax+bx for ax,bx in zip(row_a,bias_row)] for row_a in a]
def transpose(a:list[list[float]]):return list(map(list,zip(*a)))
def kindamul(a,b):
    return [[ax*bx for ax,bx in zip(row_a,row_b)] for row_a,row_b in zip(a,b)]
def sub(a,b):
    return [[ax-bx for ax,bx in zip(row_a,row_b)] for row_a,row_b in zip(a,b)]
def scale(mat:list[list[float]],s:float):
    return [[val*s for val in row] for row in mat]
def flatten(mat):
    return [x for row in mat for x in row]
def unflatten(flat, rows, cols):
    return [flat[i*cols:(i+1)*cols] for i in range(rows)]
def mul(a:list[list[float]],b:list[list[float]]):
    #b_T=list(zip(*b))
    #return [[sum(x*y for x,y in zip(row_a,col_b))for col_b in b_T] for row_a in a]
    M, K = len(a), len(a[0])
    N = len(b[0])
    assert K==len(b)
    A = (ctypes.c_float * (M*K))(*flatten(a))
    B = (ctypes.c_float * (K*N))(*flatten(b))
    C = (ctypes.c_float * (M*N))()
    _lib.mul(A, B, C, M, N, K)
    return unflatten(list(C), M, N)
def kindaadd(a,b):return [[ax+bx for ax,bx in zip(row_a,row_b)] for row_a,row_b in zip(a,b)]

def round_up(x,ndigits=10):
    if isinstance(x,float):
        return round(x,ndigits)
    return [round_up(i,ndigits) for i in x]
def randlayer(indim,outdim):
    return layer([[random.uniform(-0.1, 0.1) for k in range(outdim)] for j in range(indim)],[[random.uniform(-0.1, 0.1) for k in range(outdim)]])

def save_model():
    with open(file_path,'w') as model:
        form={"config":{"winlen":daSLM.winlen,'embdim':daSLM.embdim,"layerneu":daSLM.layerneu,'step':step,"ttid":daSLM.ttid},
                'embedding':round_up(daSLM.emb.E),
                'attention':{
                    'Q':{"weights":round_up(daSLM.att.wq.w),'biases':round_up(daSLM.att.wq.b),'mw':daSLM.att.wq.mw,"mb":daSLM.att.wq.mb,'vw':daSLM.att.wq.vw,'vb':daSLM.att.wq.vb,"t":daSLM.att.wq.t},
                    'K':{"weights":round_up(daSLM.att.wk.w),'biases':round_up(daSLM.att.wk.b),'mw':daSLM.att.wk.mw,"mb":daSLM.att.wk.mb,'vw':daSLM.att.wk.vw,'vb':daSLM.att.wk.vb,"t":daSLM.att.wk.t},
                    'V':{"weights":round_up(daSLM.att.wv.w),'biases':round_up(daSLM.att.wv.b),'mw':daSLM.att.wv.mw,"mb":daSLM.att.wv.mb,'vw':daSLM.att.wv.vw,'vb':daSLM.att.wv.vb,"t":daSLM.att.wv.t}},
                'layerneu':[{f"layer {i}":{"weights":round_up(daSLM.NN.layers[i].w),'biases':round_up(daSLM.NN.layers[i].b),'mw':daSLM.NN.layers[i].mw,'vw':daSLM.NN.layers[i].vw,'vb':daSLM.NN.layers[i].vb,'mb':daSLM.NN.layers[i].mb,"t":daSLM.NN.layers[i].t}} for i in range(len(layerneu)-1)]
                }
        json.dump(form,model,separators=(",",':'))
def load_model():
    with open(load_path,'r') as f:
        form=json.load(f)
    cfg=form["config"]
    global daSLM
    daSLM.winlen=cfg['winlen']
    daSLM.embdim=cfg["embdim"]
    daSLM.layerneu=cfg["layerneu"]
    step=cfg["step"]
    daSLM.ttid=cfg["ttid"]
    daSLM.pid=daSLM.ttid["<|PAD|>"]
    daSLM.emb.E=form['embedding']
    att=form["attention"]
    daSLM.att.wq.w=att['Q']["weights"]
    daSLM.att.wq.b=att['Q']["biases"]
    if load_adam:
        daSLM.att.wq.mw=att['Q']["mw"]
        daSLM.att.wq.vw=att['Q']["vw"]
        daSLM.att.wq.mb=att['Q']["mb"]
        daSLM.att.wq.vb=att['Q']["vb"]
        daSLM.att.wq.t=att['Q']["t"]

    daSLM.att.wk.w=att['K']["weights"]
    daSLM.att.wk.b=att['K']["biases"]
    if load_adam:
        daSLM.att.wk.mw=att['K']["mw"]
        daSLM.att.wk.vw=att['K']["vw"]
        daSLM.att.wk.mb=att['K']["mb"]
        daSLM.att.wk.vb=att['K']["vb"]
        daSLM.att.wk.t=att['K']["t"]

    daSLM.att.wv.w=att['V']["weights"]
    daSLM.att.wv.b=att['V']["biases"]
    if load_adam:
        daSLM.att.wv.mw=att['V']["mw"]
        daSLM.att.wv.vw=att['V']["vw"]
        daSLM.att.wv.mb=att['V']["mb"]
        daSLM.att.wv.vb=att['V']["vb"]
        daSLM.att.wv.t=att['V']["t"]
    nn=form['layerneu']
    for i in range(len(layerneu)-1):
        daSLM.NN.layers[i].w=nn[i][f'layer {i}']['weights']
        daSLM.NN.layers[i].b=nn[i][f'layer {i}']['biases']
        if load_adam:
            daSLM.NN.layers[i].mw=nn[i][f'layer {i}']['mw']
            daSLM.NN.layers[i].mb=nn[i][f'layer {i}']['mb']
            daSLM.NN.layers[i].vw=nn[i][f'layer {i}']['vw']
            daSLM.NN.layers[i].vb=nn[i][f'layer {i}']['vb']
            daSLM.NN.layers[i].t=nn[i][f'layer {i}']['t']
    return step

#BPE token gen
unused=[chr(i) for i in range(256,2048)]
with open("AI_tokenizer_data.txt","r",encoding="utf-8") as f:
    text=f.read()
abctmp=re.findall(r"\s+|[^\w\s]+|\w+", text)
subwords=Counter(["🗺".join(list(" "+abctmp[i])) if i>0 else "🗺".join(list(abctmp[i])) for i in range(len(abctmp))])
mapping:list[tuple[tuple[str,str],str]]=[]
def stat(subwords:dict[str,int]):
    pairs=Counter()
    for word,freq in subwords.items():
        symbols=word.split("🗺")
        for i in range(len(symbols)-1):
            pairs[(symbols[i],symbols[i+1])]+=freq
    return pairs
def merge(pairs:dict[str,int],subwords:dict[str,int]):
    best=max(pairs,key=pairs.get)
    symbol=unused.pop()
    rep="🗺".join(best)
    new_subwords={}
    for word,freq in subwords.items():
        new_word=word.replace(rep,symbol)
        new_subwords[new_word]=freq
    return new_subwords,best,symbol
for i in range(150):
    pairs=stat(subwords)
    subwords,best,symbol=merge(pairs,subwords)
    mapping.append(tuple([best,symbol]))
vocab=set()
for word in subwords.keys():
    for symbol in word.split("🗺"):
        s=symbol
        for (pair, sy) in reversed(mapping):
            rep="".join(pair)
            s=s.replace(sy, rep)
        if len(s)>1:
            vocab.add(s.lower())
if debug_mode:
    print(vocab)
#Tokenization
Tokens=list(vocab)
Tokens.sort(key=lambda x:len(x),reverse=True)
def tokenize(s:str):
    s=re.sub(r"([.,!?;:(){}\[\]])",r"\1 ",s)
    s=re.sub(r"[ \t]+"," ",s)
    tokens=Tokens
    s=" "+s.lower()
    s=re.sub(r"\s+"," ",s)
    res=[]
    i=0
    while i<len(s):
        matched=False
        for j in tokens:
            if s.startswith(j,i):
                matched=True
                i+=len(j)
                res.append(j)
                break
        if not matched:
            res.append(s[i])
            i+=1
    return res
if debug_mode:
    print(tokenize("pneumonoultramicroscopicsilicovolcanoconiosis"))
    print(tokenize("walking"))
    print(tokenize(" walking"))
    print(tokenize("ing"))
#embedding: fixed completely
class embedding:
    def __init__(self,vocab_size:int,dim:int,pid:int):
        self.vocab_size=vocab_size
        self.dim=dim
        self.pid=pid
        self.E=[[random.uniform(-0.1, 0.1) for i in range(dim)]for j in range(vocab_size)]
        self.E[pid]=[0.0 for i in range(dim)]
        self.last_tokens=None
    def forward(self,tokens:list[list[int]]):
        #print(len(self.E),len(self.E[0]))
        self.last_tokens=[i for j in tokens for i in j]
        out=[]
        for batch in tokens:
            row=[]
            for tok in batch:
                row.append(self.E[tok])
            out.append(row)
        return out
    def backward(self, dh:list[list[list[float]]], lr):
        if debug_mode:
            print(f'debug2:{len(dh)} {len(dh[0])} {len(dh[0][0])}')
        flatdh=[i for j in dh for i in j]
        if debug_mode:
            print(f'debug3:{len(flatdh)} {len(flatdh[0])}')
            print(f'debug4:{len(self.last_tokens)}')
        cnt=Counter(self.last_tokens)
        for i in range(len(self.last_tokens)):
            for j in range(self.dim):
                if self.last_tokens[i]!=self.pid:
                    self.E[self.last_tokens[i]][j]-=lr*flatdh[i][j]/cnt[self.last_tokens[i]]
#Layer class: finished
class layer:
    def __init__(self,weights:list[list[float]],biases:list[list[float]]):
        self.indim=len(weights)
        self.outdim=len(weights[0])
        self.w=weights
        self.b=biases
        self.x=None
        self.z=None
        self.h=None
        self.mw=[[0.0 for _ in row] for row in self.w]
        self.vw=[[0.0 for _ in row] for row in self.w]
        self.mb=[[0.0 for _ in row] for row in self.b]
        self.vb=[[0.0 for _ in row] for row in self.b]
        self.t=0
    def forward(self,sx:list[list[list[float]]]):
        self.x=sx
        self.z=[add(mul(self.x[i],self.w),self.b) for i in range(len(self.x))]
        self.h=[actmat(self.z[i]) for i in range(len(self.x))]
        return self.h
    def backward(self,dh:list[list[list[float]]],lr=0.001):
        batsz=len(dh)
        dz=[[[dh[b][i][j]*(1.0 if self.z[b][i][j]>0 else 0.01) for j in range(len(dh[0][0]))] for i in range(len(dh[0]))] for b in range(batsz)]
        bdw=[[0.0]*self.outdim for _ in range(self.indim)]
        bdb=[[0.0]*self.outdim]
        bdx=[]
        for b in range(batsz):
            xt=transpose(self.x[b])
            dw=mul(xt,dz[b])
            db=[[sum(dz[b][i][j] for i in range(len(dz[0])))for j in range(len(dz[0][0]))]]
            wt=transpose(self.w)
            dx=mul(dz[b],wt)
            bdw=kindaadd(bdw,dw)
            bdb=kindaadd(bdb,db)
            bdx.append(dx)
        self.optimizer(scale(bdw,1/batsz),scale(bdb,1/batsz),lr=lr)
        return bdx
    def optimizer(self,dw,db,lr=0.001,beta1=0.9,beta2=0.999,eps=1e-8):
        self.t+=1
        for i in range(len(self.w)):
            for j in range(len(self.w[0])):
                g=dw[i][j]
                self.mw[i][j]=beta1*self.mw[i][j]+(1-beta1)*g
                self.vw[i][j]=beta2*self.vw[i][j]+(1-beta2)*(g*g)
                m_hat=self.mw[i][j]/(1-beta1**self.t)
                v_hat=self.vw[i][j]/(1-beta2**self.t)
                self.w[i][j]-=lr*m_hat/(math.sqrt(v_hat)+eps)
        for j in range(len(self.b[0])):
            g=db[0][j]
            self.mb[0][j]=beta1*self.mb[0][j]+(1-beta1)*g
            self.vb[0][j]=beta2*self.vb[0][j]+(1-beta2)*(g*g)
            m_hat=self.mb[0][j]/(1-beta1**self.t)
            v_hat=self.vb[0][j]/(1-beta2**self.t)
            self.b[0][j]-=lr*m_hat/(math.sqrt(v_hat)+eps)
#NeuralNet class:finished
class NN:
    def __init__(self,layerneu:list[int],lr=1): #stands for layer neurons, a list of the layer dimentions in each layer
        self.lr=lr
        self.layers=[layer([[random.uniform(-0.1, 0.1) for k in range(layerneu[i])] for j in range(layerneu[i-1])],[[random.uniform(-0.1, 0.1) for k in range(layerneu[i])]]) for i in range(1,len(layerneu))]
    def forward(self,inp:list[list[float]]):
        for i in self.layers:
            inp=i.forward(inp)
        return inp
    def backward(self,loss:list[list[float]]):
        for i in reversed(self.layers):
            loss=i.backward(loss,lr=self.lr)
        return loss
#Attention class
class attention:
    def __init__(self,dim):
        self.wq=randlayer(dim,dim)
        self.wk=randlayer(dim,dim)
        self.wv=randlayer(dim,dim)
        self.att=None
    def forward(self,input:list[list[list[float]]]):
        b=len(input)
        out=[]
        self.att=[]
        for i in range(b):
            xb=input[i]
            pad=[all(v==0.0 for v in xb[j]) for j in range(len(xb))]
            q=self.wq.forward([xb])[0]
            k=self.wk.forward([xb])[0]
            v=self.wv.forward([xb])[0]
            t=len(xb)
            d=len(xb[0])
            score=[[float("-inf")]*t for _ in range(t)]
            for t1 in range(t):
                for j in range(t1+1):
                    if pad[j]:
                        score[t1][j]=float("-inf")
                    else:
                        score[t1][j]=sum(q[t1][d1]*k[j][d1] for d1 in range(d))/math.sqrt(d)
            att=[softmax(g) for g in score]
            outb=[[0.0]*d for g in range(t)]
            for t1 in range(t):
                for j in range(t):
                    for d1 in range(d):
                        outb[t1][d1]+=att[t1][j]*v[j][d1]
            out.append(outb)
            self.att.append(att)
        return out
    def backward(self,dout:list[list[list[float]]],lr=0.001):
        if debug_mode:
            print(f'debug:{len(dout)} {len(dout[0])} {len(dout[0][0])}')
        B=len(dout)
        bdq=[]
        bdk=[]
        bdv=[]
        for b in range(B):
            att=self.att[b]
            v=self.wv.x[b]
            k=self.wk.x[b]
            q=self.wq.x[b]
            T=len(att)
            D=len(v[0])
            dv=[[0.0]*D for _ in range(T)]
            da=[[0.0]*T for _ in range(T)]
            for t in range(T):
                for i in range(t+1):
                    for d in range(D):
                        dv[i][d]+=att[t][i]*dout[b][t][d]
                        da[t][i]+=v[i][d]*dout[b][t][d]
            ds=[[0.0]*T for _ in range(T)]
            for t in range(T):
                dot=sum(da[t][j]*att[t][j] for j in range(t+1))
                for i in range(t+1):
                    ds[t][i]=att[t][i]*(da[t][i]-dot)
            dq=[[0.0]*D for _ in range(T)]
            dk=[[0.0]*D for _ in range(T)]
            scale=1.0/math.sqrt(D)
            for t in range(T):
                for i in range(t+1):
                    for d in range(D):
                        dq[t][d]+=ds[t][i]*k[i][d]*scale
                        dk[i][d]+=ds[t][i]*q[t][d]*scale
            bdq.append(dq)
            bdk.append(dk)
            bdv.append(dv)
        dxq=self.wq.backward(bdq,lr=lr)
        dxk=self.wk.backward(bdk,lr=lr)
        dxv=self.wv.backward(bdv,lr=lr)
        dinp=[]
        D=len(self.wv.x[0][0])
        T=len(att)
        for b in range(B):
            dx=[[dxq[b][t][d]+dxk[b][t][d]+dxv[b][t][d] for d in range(D)] for t in range(T)]
            dinp.append(dx)
        return dinp


#SLM class
class SLM:
    def __init__(self,batsz:int,embdim:int,layerneu:list[int],ttid:dict[str,int],winlen:int):
        self.embdim=embdim
        self.layerneu=layerneu
        self.batsz=batsz
        self.ttid=ttid
        self.pid=ttid["<|PAD|>"]
        self.vocsz=len(ttid)
        self.emb=embedding(self.vocsz,self.embdim,self.pid)
        self.NN=NN(self.layerneu)
        self.lr=0.005
        self.att=attention(embdim)
        self.NN.lr=self.lr
        self.winlen=winlen
    def forward(self,tokens:list[list[str]]):
        tokens=[deque(i)+deque(["<|PAD|>"])*(self.winlen-len(i))for i in tokens]
        tokens=[list(map(lambda x:self.ttid[x],i)) for i in tokens]
        self.last_tokens=tokens
        inp=self.emb.forward(tokens)
        if debug_mode:
            print("emb")
        att=self.att.forward(inp)
        if debug_mode:
            print("att")
        h=self.NN.forward(att)
        return h
    def training(self,input:list[list[str]],target:list[str]):
        o=self.forward(input)
        tloss=0.0
        bdout=[[0.0]*self.vocsz for i in range(len(o[0]))]
        for b in range(len(o)):
            output=o[b]
            tmp=output[-1]
            probs=softmax(tmp)
            tid=self.ttid[target[b]]
            loss=cent(probs,tid)
            tmp2=sceb(probs,tid)
            tloss+=loss
            dout=[[0.0]*self.vocsz for i in range(len(output))]
            dout[-1]=tmp2
            bdout=kindaadd(bdout,dout)
        datt=self.NN.backward([bdout])
        demb=self.att.backward(datt)
        self.emb.backward(demb*self.batsz,self.lr)
        return math.exp(tloss/self.batsz)
    def gen(self, start: str, maxtk: int, temp: float = 0.2):
        tk = deque(tokenize(start))
        print(start, end='',flush=True)
    
        for i in range(maxtk):
            fw = self.forward([tk])[0][-1]
            max_val = max(fw)
            exp_logits = [math.exp((v - max_val) / temp) for v in fw]
            total = sum(exp_logits)
            r = random.uniform(0, total)
            upto = 0
            idx = 0
            for i, weight in enumerate(exp_logits):
                upto += weight
                if upto >= r:
                    idx = i
                    break
        
            char = actual_vocab[idx]
            if char=='<|EOT|>':break
            print(char, end='', flush=True)
            tk.append(char)
            if len(tk)>self.winlen:
                tk.popleft()
with open("token prediction data.txt","r",encoding='utf-8') as f1:
    ttext=f1.read()
ta=ttext.split("\n\n")
ta=[tokenize(i)+["<|EOT|>"] for i in ta]
actual_vocab=list(vocab|{chr(i) for i in range(1,128)}|{" "+chr(i) for i in range(1,128)}|{"<|EOT|>","<|OUT|>","<|PAD|>"})
actual_vocab=sorted(actual_vocab)
ttid={actual_vocab[i]:i for i in range(len(actual_vocab))}
if debug_mode:
    print(len(ttid))

batch_size              =4
context_window          =128
embedding_dimension     =16
layerneu                =[embedding_dimension]+[64]*3+[len(ttid)]

daSLM=SLM(batch_size,embedding_dimension,layerneu,ttid,context_window)

if load:
    step=load_model()
daSLM.pid=daSLM.ttid["<|PAD|>"]
daSLM.emb.pid=daSLM.pid
idtt={v:k for k,v in daSLM.ttid.items()}
daSLM.emb.E[daSLM.pid]=[0.0]*daSLM.embdim
ttid=daSLM.ttid
if load and input("do you want to try and generate some text? ").lower()=='true':
    daSLM.gen(input("text: "),1000)
while True:
    step+=1
    daSLM.lr=0.01/(1+step/10)
    batch=[]
    outb=[]
    for i in range(daSLM.batsz):
        rnd=random.choice(ta)
        ptr2=random.randint(1,len(rnd)-1)
        ptr1=max(0,ptr2-daSLM.winlen)
        batch.append(rnd[ptr1:ptr2])
        outb.append(rnd[ptr2])
    ppl=daSLM.training(batch,outb)
    bar=int(max(0,min(1,1-(ppl/len(daSLM.ttid))))*20)
    print('█'*bar+'-'*(20-bar),ppl)
    if step%checkpoint_gap==0:
        save_model()

