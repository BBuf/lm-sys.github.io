#!/usr/bin/env python3
"""Measured MoE path, with a matched example selected by a declared median-distance rule."""
import argparse
import json
from pathlib import Path
import statistics as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--evidence',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
paths={}
for arm in ('off','on'):
    for f in sorted((a.evidence/arm).glob('rank*-verify-*.json')):
        d=json.loads(f.read_text());es=d['events']
        def family(pattern):return [e for e in es if pattern in e['name']]
        if arm=='off':
            groups=[family('_router_triton_kernel'),family('routingIndicesDynBlockKernel'),
                    family('bmm_MxE4m3_MxE2m1'),family('bmm_Bfloat16_MxE2m1'),
                    family('moe_finalize_all_reduce_kernel<4u, 5120u, 6u')]
            labels=['Router','Expert mapping','GEMM 1 + SwiGLU','GEMM 2','Finalize + all-reduce + mHC']
        else:
            quad=family('QuadFinalizePublishDeviceKernel');lam=family('LamportMHCDeviceKernel')
            posts=[next(e for e in lam if e['start_us']>=q['start_us']) for q in quad]
            assert len(lam)==80 and [lam.index(e) for e in posts]==list(range(1,80,2))
            groups=[family('dsv41_gate_topk'),family('dsv41_routed_expert_route'),
                    family('dsv41_routed_expert_gemm1'),family('dsv41_routed_expert_gemm2'),quad,posts]
            labels=['Fast top-k','Route / pack','GEMM 1 + SwiGLU','GEMM 2','Finalize / publish','Mailbox reduction + mHC']
        assert all(len(g)==40 for g in groups),(arm,f,[len(g) for g in groups])
        for layer in range(40):
            selected=[g[layer] for g in groups]
            start=selected[0]['start_us'];end=selected[-1]['start_us']+selected[-1]['duration_us']
            assert all(start<=e['start_us']<end for e in selected)
            paths[(arm,d['rank'],d['ordinal'],layer)]={'arm':arm,'rank':d['rank'],
                'replay_ordinal':d['ordinal'],'layer_ordinal':layer,'elapsed_us':end-start,
                'events':[{'label':label,**e,'relative_start_us':e['start_us']-start}
                          for label,e in zip(labels,selected)]}
medians={arm:stats.median(v['elapsed_us'] for key,v in paths.items() if key[0]==arm) for arm in ('off','on')}
keys=[(rank,step,layer) for rank in range(4) for step in range(5) for layer in range(40)]
selected_key=min(keys,key=lambda key:sum(abs(paths[(arm,*key)]['elapsed_us']/medians[arm]-1) for arm in ('off','on')))
selected=[paths[(arm,*selected_key)] for arm in ('off','on')]
fig,axes=plt.subplots(2,1,figsize=(10.8,8.0),sharex=True,layout='constrained')
for ax,case in zip(axes,selected):
    for i,e in enumerate(case['events']):
        color='#377eb8' if i<2 else '#21835c' if i<4 else '#bd6a28'
        ax.barh(i,e['duration_us'],left=e['relative_start_us'],height=.62,color=color,alpha=.9)
    ax.set_yticks(range(len(case['events'])),[f"{e['label']}  ({e['duration_us']:.1f} µs)" for e in case['events']],fontsize=10)
    ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
    ax.set_title(f"Plugin {case['arm']} · routed MoE path spans {case['elapsed_us']:.1f} µs",loc='left',fontsize=12,pad=12)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.axvline(case['elapsed_us'],color='#888',ls=':',lw=1)
axes[-1].set_xlabel('Time from routing start (µs) · bars preserve measured overlap')
axes[-1].set_xlim(0,max(s['elapsed_us'] for s in selected)*1.08)
rank,step,layer=selected_key
fig.suptitle('Routed MoE in a warmed target-verify CUDA graph\nDeepSeek-V4.1-Flash · 4 × GB300 · TP4 · C1 → 6 verify rows',fontsize=14)
fig.text(.01,-.075,
    f'Matched example: TP rank {rank}, layer {layer}, captured replay {step+1}/5; chosen nearest the two arm medians.\n'
    f'All 40 layers × 5 replays × 4 ranks: median path span {medians["off"]:.2f} µs off / {medians["on"]:.2f} µs on.\n'
    'Rows are operations, not CUDA streams. Concurrent shared-expert and mHC preparation are omitted.\n'
    'CUPTI diagnostic timing includes dependency waits; use unprofiled runs for end-to-end speed claims.',fontsize=9)
for ext in ('svg','png'):fig.savefig(a.out/f'nvidia-moe-timeline.{ext}',dpi=180,bbox_inches='tight')
(a.out/'profile-timeline-evidence.json').write_text(json.dumps({
    'selection':'Same rank, replay ordinal, and layer ordinal minimizing summed relative distance from each arm median path span over800 pairs',
    'median_path_span_us':medians,'selected':selected,
    'all_path_spans':[{'arm':v['arm'],'rank':v['rank'],'replay':v['replay_ordinal'],
                       'layer':v['layer_ordinal'],'span_us':v['elapsed_us']} for v in paths.values()],
    'source_mapping':{
        'layer_order':'40 ordered target MoE instances/replay; model config num_hidden_layers40; runtime/executor.py ordered layer loop',
        'on':'sgl-project/sgl-nvidia-mega@d47deb3390e14741c9420ac702954e7611a14da1: moe/step.py, sm103/moe/routed_expert_gemm.py, boundary/boundaries.py, sm103/all_reduce_mhc.py',
        'off':'SGLang@7b7015e72e4ffd6641159de0d8f3f466313f4e8b: FlashInfer TRT-LLM routed pair and SGLang fused finalize/all-reduce/mHC',
    },
    'caveats':['Kernel durations include waits; sums are not elapsed time','Only selected routed and post path shown; shared expert and mHC preparation omitted','Fixed-acceptance random-input C1 workload; no generalization to all shapes','Graph replay is instrumented; this is not an unprofiled microbenchmark']
},indent=2)+'\n')
print(json.dumps({'selected':selected_key,'medians_us':medians,'selected_spans_us':[x['elapsed_us'] for x in selected]},indent=2))
