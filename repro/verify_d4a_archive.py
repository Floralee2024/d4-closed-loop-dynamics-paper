"""Recover exact D4a archive and independently verify its q90 extraction."""
import csv, hashlib, json, math
from pathlib import Path
from collections import defaultdict
from statistics import mean
from make_figures import svg_start,text,line,write_svg
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/d4a_formal_archive'
def read(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    raw=read(DEST/'d4_repair_raw.csv'); landmarks=read(DEST/'d4_repair_landmarks.csv')
    grouped=defaultdict(list)
    for x in raw:grouped[tuple(x[k] for k in ('split','variant','seed','K'))].append(x)
    count=0
    mapping={'rank':'rank_score','dyn':'dyn_score','dyn_null':'dyn_score_null','dyn_delta':'dyn_score_delta','util':'U_symbol','contK':'U_cont_K'}
    for lm in landmarks:
        rows=sorted(grouped[tuple(lm[k] for k in ('split','variant','seed','K'))],key=lambda x:float(x['C']))
        assert len(rows)==10
        for name,field in mapping.items():
            ys=[float(x[field]) for x in rows]; gain=max(ys)-min(ys); gate=.1 if name in ('util','contK') else .02
            q=next((float(x['C']) for x in rows if float(x[field])>=min(ys)+.9*gain),float('nan')) if gain>=gate else float('nan')
            expected=float(lm[f'C_{name}_q90'])
            assert (math.isnan(q) and math.isnan(expected)) or q==expected,(name,q,expected)
            count+=1
    alignment=read(ROOT/'results/d4a_repair_alignment.csv')
    for cell in alignment:
        sub=[x for x in landmarks if all(x[k]==cell[k] for k in ('split','variant','K'))]
        assert len(sub)==20
        for m in ('dyn','rank','dyn_null','dyn_delta'):
            assert math.isclose(mean(float(x[f'{m}_to_util_q90_err']) for x in sub),float(cell[f'{m}_to_util_q90_err_mean']),abs_tol=1e-10)
        assert math.isclose(mean(float(x['dyn_closer_q90']) for x in sub),float(cell['dyn_closer_q90_rate']),abs_tol=1e-12)
    stats=dict(raw_rows=len(raw),landmark_rows=len(landmarks),q90_checks=count,structure_splits=sorted({x['structure_split'] for x in raw}),min_usage_entropy=min(float(x['usage_entropy']) for x in raw),utility_q90_at_zero=sum(float(x['C_util_q90'])==0 for x in landmarks),dyn_q90_at_zero=sum(float(x['C_dyn_q90'])==0 for x in landmarks))
    (DEST/'verification.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    parts=svg_start(1200,1100,'D4a raw metric curves','Eight cells: thin lines are individual seeds; thick lines are seed means. R: usage/rank, D: temporal proxy, U: utility. No smoothing.')
    parts.append(text(30,30,'D4a: unsmoothed metric curves (thin: seeds; thick: means)',19))
    colors={'rank_score':'#0072B2','dyn_score':'#D55E00','U_symbol':'#009E73'}
    for i,(field,color) in enumerate(colors.items()):parts.append(text(30+i*350,55,field,13,fill=color))
    for pi,(split,variant,k) in enumerate((s,v,k) for s in ('id','ood_inverted') for v in ('original','pre_softmax') for k in ('8','16')):
        left=70+(pi%2)*590;top=100+(pi//2)*245;w=480;h=170
        xx=lambda c:left+c/24*w; yy=lambda y:top+h-(y+.5)/1.7*h
        parts.append(text(left,top-13,f'{split} / {variant} / K={k}',14))
        for y in (-.5,0,.5,1):parts.append(line(left,yy(y),left+w,yy(y),stroke='#ddd'));parts.append(text(left-8,yy(y)+4,str(y),10,anchor='end'))
        for c in (0,3,8,16,24):parts.append(text(xx(c),top+h+18,str(c),10,anchor='middle'))
        parts.append(text(left+w/2,top+h+35,'C',11,anchor='middle'))
        for field,color in colors.items():
            vals=defaultdict(list)
            for seed in range(20):
                rows=sorted(grouped[(split,variant,str(seed),k)],key=lambda x:float(x['C']))
                pts=[]
                for x in rows:
                    c=float(x['C']);y=float(x[field]);vals[c].append(y);pts.append(f'{xx(c):.2f},{yy(y):.2f}')
                parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width=".6" opacity=".13"/>')
            pts=' '.join(f'{xx(c):.2f},{yy(mean(vals[c])):.2f}' for c in sorted(vals))
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    write_svg(ROOT/'figures/d4a_raw_curves.svg',parts)
    print(stats)
if __name__=='__main__':main()
