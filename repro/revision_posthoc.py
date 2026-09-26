"""Post-hoc descriptive reanalysis; never changes archived experiment outputs."""
import csv, hashlib, json, math, random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from make_figures import svg_start, text, line, write_svg
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'revision_posthoc'

def read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def finite(x):
    return x not in ('', None) and math.isfinite(float(x))

def balanced(rows, field):
    groups=defaultdict(list)
    for row in rows:
        if finite(row[field]): groups[int(row['K'])].append(float(row[field]))
    return mean(mean(groups[k]) for k in (8,16)) if all(groups[k] for k in (8,16)) else float('nan')

def main():
    OUT.mkdir(exist_ok=True)
    rawpath=ROOT/'results/d4b_formal_gpu/curves.csv'
    sumpath=ROOT/'results/d4b_formal_gpu/cstar_summary.csv'
    raw=read(rawpath); summary=read(sumpath)
    curves=defaultdict(list)
    for row in raw:
        curves[(int(row['seed']),float(row['residual_strength']),int(row['K']),row['split'])].append(row)
    ranges=[]
    for (seed,r,k,split), rows in sorted(curves.items()):
        vals={float(x['C']):float(x['utility']) for x in rows}
        assert len(vals)==10 and 0 in vals and 24 in vals
        pos=[u for c,u in vals.items() if c>0]
        ranges.append(dict(seed=seed,residual_strength=r,K=k,split=split,utility_C0=vals[0],utility_C24=vals[24],range_full=max(vals.values())-min(vals.values()),range_positive_C=max(pos)-min(pos)))
    write('utility_ranges_by_seed_K.csv',ranges)
    groups=[]; denominators=[]
    for r in (0,.1,.25,.5,1):
        for split in ('id','ood_random','ood_inverted'):
            rr=[x for x in ranges if x['residual_strength']==r and x['split']==split]
            groups.append(dict(residual_strength=r,split=split,**{f:balanced(rr,f) for f in ('utility_C0','utility_C24','range_full','range_positive_C')}))
            ss=[x for x in summary if float(x['residual_strength'])==r and x['split']==split]
            valid=[x for x in ss if finite(x['sync_gap'])]
            d=dict(residual_strength=r,split=split,n_total=len(ss),n_valid=len(valid),n_valid_K8=sum(int(x['K'])==8 for x in valid),n_valid_K16=sum(int(x['K'])==16 for x in valid),gap_K_balanced=balanced(ss,'sync_gap'),gap_pooled=mean(float(x['sync_gap']) for x in valid))
            for name,pred in [('sync',lambda x:float(x['sync_gap'])<=1),('negative',lambda x:float(x['Delta_C_dyn_eq'])<0),('positive',lambda x:float(x['Delta_C_dyn_eq'])>0)]:
                n=sum(pred(x) for x in valid)
                d[name+'_count']=n; d[name+'_rate_all']=n/len(ss); d[name+'_rate_valid_pooled']=n/len(valid)
            denominators.append(d)
    write('utility_range_summary.csv',groups); write('denominators.csv',denominators)
    # Post-hoc percentile bootstrap: resample base seeds jointly across K and strengths.
    rng=random.Random(20260926); boot=[]
    for r in (.1,.25,.5,1):
        for split in ('id','ood_random','ood_inverted'):
            for field,source in [('sync_gap',summary),('range_positive_C',ranges)]:
                strata={v:{s:[x for x in source if int(x['seed'])==s and float(x['residual_strength'])==v and x['split']==split] for s in range(10)} for v in (0,r)}
                def contrast(seeds):
                    return balanced([x for s in seeds for x in strata[r][s]],field)-balanced([x for s in seeds for x in strata[0][s]],field)
                samples=sorted(v for _ in range(2000) if math.isfinite(v:=contrast(rng.choices(range(10),k=10))))
                boot.append(dict(residual_strength=r,split=split,metric=field,contrast_vs_zero=contrast(range(10)),ci_low=samples[int(.025*(len(samples)-1))],ci_high=samples[int(.975*(len(samples)-1))],valid_resamples=len(samples)))
    write('bootstrap_contrasts.csv',boot)
    colors=['#333333','#0072B2','#D55E00','#009E73','#CC79A7']
    for mode in ('utility','markers'):
        desc=('Unsmoothed per-C utility curves; columns K=8 and K=16; rows are evaluation splits. Thin lines are individual seeds; thick lines are seed means.' if mode=='utility' else 'Unsmoothed per-C seed means; columns K=8 and K=16; rows are evaluation splits. Solid lines are mixed-q rank proxies; dashed lines are outcome/lower-state NMI association proxies.')
        parts=svg_start(1200,900,'D4b absolute utility' if mode=='utility' else 'D4b operational proxy curves',desc)
        parts.append(text(25,27,'D4b: absolute utility' if mode=='utility' else 'D4b: normalized operational proxies',20,weight='bold'))
        for i,r in enumerate((0,.1,.25,.5,1)):
            parts.append(text(25+210*i,52,f'r = {r:g}',13,fill=colors[i]))
        parts.append(text(25,74,'Thin: individual seeds; thick: seed mean' if mode=='utility' else 'Solid: mixed-q rank proxy; dashed: outcome/lower-state NMI association proxy (seed means)',12))
        for rowi,split in enumerate(('id','ood_random','ood_inverted')):
            for coli,k in enumerate((8,16)):
                left=75+coli*590; top=115+rowi*260; w=490; h=185
                ymin,ymax=(-.15,.65) if mode=='utility' else (0,1.05)
                xx=lambda c:left+c/24*w
                yy=lambda y:top+h-(y-ymin)/(ymax-ymin)*h
                parts.append(text(left,top-13,f'{split}, K={k}',15,weight='bold'))
                for y in ((0,.2,.4,.6) if mode=='utility' else (0,.25,.5,.75,1)):
                    parts.append(line(left,yy(y),left+w,yy(y),stroke='#dddddd')); parts.append(text(left-8,yy(y)+4,f'{y:g}',11,anchor='end'))
                for c in (0,3,8,16,24): parts.append(text(xx(c),top+h+18,str(c),11,anchor='middle'))
                parts.append(text(left+w/2,top+h+37,'C',12,anchor='middle'))
                for ri,r in enumerate((0,.1,.25,.5,1)):
                    fields=['utility'] if mode=='utility' else ['rank_score','dyn_score']
                    for fi,field in enumerate(fields):
                        sets=[[x for x in raw if int(x['K'])==k and x['split']==split and float(x['residual_strength'])==r and int(x['seed'])==s] for s in range(10)]
                        if mode=='utility':
                            for ss in sets:
                                pts=' '.join(f'{xx(float(x["C"])):.2f},{yy(float(x[field])):.2f}' for x in sorted(ss,key=lambda x:float(x['C'])))
                                parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[ri]}" stroke-width="0.7" opacity="0.18"/>')
                        vals=defaultdict(list)
                        for ss in sets:
                            for x in ss: vals[float(x['C'])].append(float(x[field]))
                        pts=' '.join(f'{xx(c):.2f},{yy(mean(vals[c])):.2f}' for c in sorted(vals))
                        dash=' stroke-dasharray="6 4"' if fi else ''
                        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[ri]}" stroke-width="2"{dash}/>')
        write_svg(ROOT/f'figures/d4b_{mode}_curves.svg',parts)
    md=['# Post-hoc revision analysis','', 'Status: exploratory descriptive sensitivity analysis, 26 September 2026. No training or archived result files were changed.','', 'Ranges are computed within seed x K before equal weighting of K-specific seed means. Positive-C analysis excludes only C=0; it does not redefine or recompute the archived thresholds.','', '| r | Split | U(0) | U(24) | Full-grid range | C>0 range |','| ---: | --- | ---: | ---: | ---: | ---: |']
    for x in groups: md.append(f"| {x['residual_strength']:g} | {x['split']} | {x['utility_C0']:.6f} | {x['utility_C24']:.6f} | {x['range_full']:.6f} | {x['range_positive_C']:.6f} |")
    md+=['','`denominators.csv` preserves the archived all-row event rates and adds explicitly labeled pooled-valid rates; these do not silently replace the K-balanced published estimand. Empty thresholds are not evidence for the opposite ordering.','', '`bootstrap_contrasts.csv` reports post-hoc 95% percentile intervals from 2,000 base-seed resamples (seed 20260926), jointly retaining both K levels and both residual conditions. Intervals condition on this world, grid and valid-threshold rule; they do not correct censoring, missingness or multiple comparisons and are not confirmatory significance tests.','', 'D4a artifacts were subsequently recovered from the exact SOURCES.md archive path. See results/d4a_formal_archive and repro/verify_d4a_archive.py. The D4b analyses here do not use or reconstruct D4a data.']
    (OUT/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    manifest=dict(evidence_id='d4-revision-posthoc-20260926',status='exploratory',evidence_level='descriptive_association',inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (rawpath,sumpath)},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),bootstrap_resamples=2000,bootstrap_seed=20260926)
    (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Post-hoc tables, provenance and two SVG figures generated; archived outputs unchanged.')

if __name__=='__main__': main()
