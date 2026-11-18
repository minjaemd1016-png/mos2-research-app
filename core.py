"""Geometry, extraction, append-only research records and gated learning."""
import hashlib, json, sqlite3, uuid, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.affinity import rotate, translate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

FEATURES = ['L_um','W_um','oxide_nm','layers']
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def uid(): return uuid.uuid4().hex[:12]
def cox(oxide_nm=300):
    if oxide_nm <= 0: raise ValueError('산화막 두께는 양수여야 합니다.')
    return 8.8541878128e-12 * 3.9 / (oxide_nm * 1e-9)

def validate_polygon(points, width, height):
    a=np.asarray(points,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<3 or not np.isfinite(a).all():
        raise ValueError('경계에는 유효한 x,y 좌표가 3개 이상 필요합니다.')
    if (a<0).any() or (a[:,0]>width).any() or (a[:,1]>height).any():
        raise ValueError('경계가 이미지 밖에 있습니다.')
    p=Polygon(a)
    if not p.is_valid or p.area<=0: raise ValueError('경계가 교차하거나 면적이 0입니다.')
    return p

def placement(points, scale, W=4, overlap=2, margin=1, L=10):
    if scale<=0: return {'status':'보류: 스케일 미확정','area_um2':None,'layout':None}
    if min(W,L)<=0 or min(overlap,margin)<0: raise ValueError('배치 치수를 확인하세요.')
    p=Polygon(np.asarray(points)*scale)
    if not p.is_valid: raise ValueError('잘못된 경계')
    safe=p.buffer(-margin)
    out={'area_um2':round(p.area,3),'status':'탐색 범위 내 배치 미발견','layout':None}
    if safe.is_empty: return out
    # Conservative rectangle includes BOTH contact overlaps and entire channel.
    minx,miny,maxx,maxy=safe.bounds
    centers=[(safe.representative_point().x,safe.representative_point().y)]
    centers += [(x,y) for x in np.linspace(minx,maxx,13) for y in np.linspace(miny,maxy,13)]
    rect=box(-(L/2+overlap),-W/2,L/2+overlap,W/2)
    for angle in range(0,180,5):
        r=rotate(rect,angle,origin=(0,0))
        for x,y in centers:
            q=translate(r,x,y)
            if safe.covers(q):
                parts={}
                for name,part in [('source',box(-L/2-overlap,-W/2,-L/2,W/2)),('channel',box(-L/2,-W/2,L/2,W/2)),('drain',box(L/2,-W/2,L/2+overlap,W/2))]:
                    transformed=translate(rotate(part,angle,origin=(0,0)),x,y)
                    parts[name]=(np.asarray(transformed.exterior.coords)/scale).tolist()
                out.update(status='기하학적 배치 가능 · 경계/정렬 검토 필요',layout=(np.asarray(q.exterior.coords)/scale).tolist(),parts=parts,angle_deg=angle)
                return out
    return out

def extract_mobility(df,L,W,oxide,vd,lo,hi):
    if min(L,W,oxide,vd)<=0 or lo>=hi: raise ValueError('양수 치수·Vd와 올바른 피팅 구간이 필요합니다.')
    d=df[np.isclose(df.Vd_V,vd)&df.Vg_V.between(lo,hi)].copy()
    if len(d)<4 or d.Vg_V.nunique()<4: raise ValueError('선택 Vd/구간에 서로 다른 Vg가 4점 이상 필요합니다.')
    if d.Vg_V.duplicated().any(): raise ValueError('왕복/반복 스윕이 섞여 있습니다. 한 방향의 한 스윕만 업로드하세요.')
    if not np.isfinite(d[['Vg_V','Id_A']].to_numpy()).all(): raise ValueError('NaN/무한대가 있습니다.')
    slope,intercept=np.polyfit(d.Vg_V,d.Id_A,1)
    if slope<=0: raise ValueError('양의 transconductance가 아닙니다. 극성/단위/구간을 확인하세요.')
    pred=slope*d.Vg_V+intercept
    ss=float(((d.Id_A-d.Id_A.mean())**2).sum())
    r2=1-float(((d.Id_A-pred)**2).sum())/ss if ss>0 else 0
    return {'mobility':float(slope*L/W/cox(oxide)/vd*1e4),'r2':r2,'gm_A_V':float(slope),'fit_points':len(d),'fit_Vg':[lo,hi],'Vd_V':vd,'method':'linear-regime two-terminal field-effect mobility','unit':'cm²/(V·s)'}

class Store:
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, kind TEXT, entity TEXT, at TEXT, payload TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS assignments(device TEXT PRIMARY KEY, split TEXT)')
    def connect(self):
        c=sqlite3.connect(self.root/'research.sqlite3',timeout=30); c.execute('PRAGMA journal_mode=WAL'); return c
    def add(self,kind,entity,payload):
        ident=uid()
        with self.connect() as c: c.execute('INSERT INTO events VALUES(?,?,?,?,?)',(ident,kind,entity,stamp(),json.dumps(payload,ensure_ascii=False,allow_nan=False)))
        return ident
    def events(self,kind=None):
        with self.connect() as c:
            rows=c.execute('SELECT * FROM events '+('WHERE kind=? ' if kind else '')+'ORDER BY rowid',(kind,) if kind else ()).fetchall()
        return [dict(id=r[0],kind=r[1],entity=r[2],at=r[3],payload=json.loads(r[4])) for r in rows]
    def split(self,device):
        # Assignment is fixed for the life of a device, including repeat measurements.
        with self.connect() as c:
            r=c.execute('SELECT split FROM assignments WHERE device=?',(device,)).fetchone()
            if r:return r[0]
            s='validation' if int(hashlib.sha256(device.encode()).hexdigest()[:8],16)%4==0 else 'train'
            c.execute('INSERT INTO assignments VALUES(?,?)',(device,s));return s
    def records(self):
        latest={}
        for e in self.events('measurement'):
            latest[e['entity']]=e['payload'] | {'device_id':e['entity']}
        return [r for r in latest.values() if r.get('confirmed') and r.get('source')=='실측']
    def active(self,target):
        es=[e for e in self.events('activation') if e['payload']['target']==target]
        return es[-1]['payload']['version'] if es else None
    def model(self,version):
        return joblib.load(self.root/'models'/f'{version}.joblib')

def frame(records,target):
    rows=[]
    for r in records:
        base={k:r[k] for k in FEATURES}; base['device_id']=r['device_id'];base['process']=r['process']
        if target=='mobility': rows.append(base | {'y':r['mobility']})
        else:
            for point in r['iv']:
                rows.append(base|point|{'y':float(np.arcsinh(point['Id_A']/1e-9))})
    return pd.DataFrame(rows)

def train(store,target='mobility'):
    records=store.records(); signature=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    prior=[e for e in store.events('training') if e['payload']['target']==target and e['payload'].get('signature')==signature]
    if prior:return prior[-1]['payload']
    status={'target':target,'signature':signature,'devices':len(records),'status':'학습 대기'}
    # One process cohort only: user must give identical confirmed process codes.
    if len(records)<12: return status|{'reason':'확인된 서로 다른 소자 최소 12개 필요 (개발용 기준)'}
    if len({r['process'] for r in records})!=1:return status|{'reason':'공정 코드가 혼합됨: 동일 공정 데이터만 별도 데이터 폴더에서 학습하세요.'}
    d=frame(records,target); feats=FEATURES+(['Vg_V','Vd_V'] if target=='iv' else [])
    split={r['device_id']:store.split(r['device_id']) for r in records}
    tr=d[d.device_id.map(split)=='train'];va=d[d.device_id.map(split)=='validation']
    if tr.device_id.nunique()<8 or va.device_id.nunique()<3:return status|{'reason':'고정 분할에서 학습 8개·검증 3개 이상의 소자가 필요합니다.'}
    model=RandomForestRegressor(n_estimators=120,min_samples_leaf=2,random_state=42,n_jobs=1)
    weights=1/tr.groupby('device_id').device_id.transform('count')
    model.fit(tr[feats],tr.y,sample_weight=weights)
    def score(pred):
        errs=pd.DataFrame({'id':va.device_id,'err':np.abs(va.y.to_numpy()-pred)})
        return float(errs.groupby('id').err.mean().mean())
    error=score(model.predict(va[feats])); baseline=score(np.full(len(va),np.median(tr.y)))
    old=store.active(target); olderr=score(store.model(old)['estimator'].predict(va[feats])) if old else None
    passed=error<baseline and (olderr is None or error<olderr*.95)
    version=uid(); folder=store.root/'models';folder.mkdir(exist_ok=True)
    obj={'estimator':model,'features':feats,'ranges':{f:[float(tr[f].min()),float(tr[f].max())] for f in feats},'process':records[0]['process'],'target':target,'train_ids':sorted(tr.device_id.unique()),'validation_ids':sorted(va.device_id.unique())}
    joblib.dump(obj,folder/f'{version}.joblib')
    status.update(version=version,status='적용' if passed else '미적용',mae=error,baseline_mae=baseline,previous_mae=olderr,train_devices=tr.device_id.nunique(),validation_devices=va.device_id.nunique(),metric='MAE cm²/(V·s)' if target=='mobility' else '소자별 평균 asinh(Id/1nA) MAE',reason='기준 모델보다 우수하고 기존 모델 대비 5% 이상 개선 시 적용. 개발 검증이며 외부 성능 검증 아님.')
    store.add('training',target,status)
    if passed:store.add('activation',target,{'target':target,'version':version,'reason':'validation_pass'})
    return status

def predict(store,target,values,process):
    v=store.active(target)
    if not v:return {'status':'보류: 적용된 학습 모델이 없습니다.'}
    m=store.model(v)
    if process!=m['process']:return {'status':'보류: 학습 공정과 다릅니다.'}
    for k,(lo,hi) in m['ranges'].items():
        if k not in values or not lo<=values[k]<=hi:return {'status':f'보류: {k}가 학습 범위 밖이거나 누락되었습니다.'}
    y=float(m['estimator'].predict(pd.DataFrame([values])[m['features']])[0])
    return {'status':'모델 예측 · 실측 아님','value':y if target=='mobility' else float(np.sinh(y)*1e-9),'unit':'cm²/(V·s)' if target=='mobility' else 'A','version':v,'uncertainty':'정량적 신뢰구간 미검증'}
