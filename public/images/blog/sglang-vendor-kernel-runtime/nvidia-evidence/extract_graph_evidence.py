#!/usr/bin/env python3
"""Link graph GPU events to annotated target/draft steps through launch correlation IDs."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics as stats

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.run/'manifest.json').read_text())
assert manifest['status']=='passed'
records=[]
for path in sorted((a.run/'profiles/decode').glob('*.trace.json.gz')):
    rank=int(re.search(r'TP-(\d+)-DECODE',path.name)[1])
    with gzip.open(path,'rt') as f: trace=json.load(f)
    events=trace['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel']
    launches=[e for e in events if e.get('name')=='cudaGraphLaunch']
    steps=sorted([e for e in events if e.get('cat')=='user_annotation' and
                  re.match(r'step\[(DRAFT|VERIFY) ',e.get('name',''))],key=lambda e:e['ts'])
    assert len(steps)==10, (path,len(steps))
    ordinal=Counter()
    for step in steps:
        phase=re.match(r'step\[(DRAFT|VERIFY)',step['name'])[1]
        calls=[e for e in launches if step['ts']<=e['ts']<step['ts']+step['dur']]
        assert len(calls)==1,(path,step['name'],len(calls))
        corr=calls[0]['args']['correlation']
        selected=sorted([e for e in kernels if e['args'].get('correlation')==corr],key=lambda e:e['ts'])
        assert selected and all(e['args'].get('graph id',0)!=0 for e in selected)
        start=min(e['ts'] for e in selected);end=max(e['ts']+e['dur'] for e in selected)
        merged=[]
        for e in selected:
            lo,hi=e['ts'],e['ts']+e['dur']
            if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
            else:merged.append([lo,hi])
        by_name=defaultdict(list)
        for e in selected:by_name[e['name']].append(e['dur'])
        row={'rank':rank,'phase':phase,'ordinal':ordinal[phase],'annotation':step['name'],
             'graph_ids':sorted({e['args']['graph id'] for e in selected}),
             'launch_correlation':corr,'cpu_launch_us':calls[0]['dur'],
             'gpu_kernel_span_us':end-start,'gpu_kernel_sum_us':sum(e['dur'] for e in selected),
             'gpu_busy_union_us':sum(hi-lo for lo,hi in merged),'kernel_count':len(selected),
             'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'trace_file':path.name,'trace_origin_us':start,
             'kernels_by_name':[{'name':name,'count':len(durations),'sum_us':sum(durations),
                                 'median_us':stats.median(durations)} for name,durations in by_name.items()]}
        records.append(row)
        # Retain exact names, launch shapes and original graph lanes for a verified zoom.
        compact={'rank':rank,'phase':phase,'ordinal':ordinal[phase],'annotation':step['name'],
                 'origin_us':start,'events':[{'name':e['name'],'start_us':e['ts']-start,
                 'duration_us':e['dur'],'trace_lane':e['args']['stream'],
                 'grid':e['args'].get('grid'),'block':e['args'].get('block'),
                 'graph_node_id':e['args'].get('graph node id')} for e in selected]}
        (a.out/f'rank{rank}-{phase.lower()}-{ordinal[phase]}.json').write_text(json.dumps(compact)+'\n')
        ordinal[phase]+=1
    assert ordinal=={'VERIFY':5,'DRAFT':5}
assert {r['rank'] for r in records}=={0,1,2,3}
summary=[]
for rank in range(4):
    for phase in ['DRAFT','VERIFY']:
        rows=[r for r in records if r['rank']==rank and r['phase']==phase]
        summary.append({'rank':rank,'phase':phase,'steps':len(rows),
            **{key:{'median':stats.median(r[key] for r in rows),'min':min(r[key] for r in rows),
                    'max':max(r[key] for r in rows)}
               for key in ['kernel_count','gpu_kernel_span_us','gpu_kernel_sum_us','gpu_busy_union_us']}})
(a.out/'graph-evidence.json').write_text(json.dumps({'arm':manifest['arm'],'summary':summary,'steps':records,
    'method':'CPU target/draft annotation -> enclosed cudaGraphLaunch -> GPU correlation ID; full graph GPU events from all trace lanes',
    'limitations':'Graph kernel spans include waits and gaps, exclude non-graph work, and are perturbed by CUPTI profiling. Kernel sums overlap and are not additive latency. Trace lane IDs need not denote independent physical streams. Use unprofiled timings for performance claims.'},indent=2)+'\n')
print(json.dumps(summary,indent=2))
