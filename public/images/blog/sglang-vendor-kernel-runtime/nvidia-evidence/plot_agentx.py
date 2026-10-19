#!/usr/bin/env python3
"""Plot a single recovered AgentX request; never describe it as a full AgentX run."""
import argparse
import json
import statistics as stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--results', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--run-suffix', default='-v3')
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
records, manifests = [], []
for arm in ('off', 'on'):
    for launch in (1, 2):
        path = a.results / f'agentx-{arm}-r{launch}{a.run_suffix}'
        m = json.loads((path/'manifest.json').read_text())
        assert m['status'] == 'passed' and m['mode'] == 'agentx' and m['arm'] == arm
        assert m['environment']['SGLANG_SIMULATE_ACC_LEN'] == '5.4'
        manifests.append(m)
        rows = json.loads((path/'agentx-results.json').read_text())
        measured = [r for r in rows if not r['warmup']]
        assert len(measured) == 10
        for r in measured:
            assert not r['profiled'] and r['recovered_chat'] and not r['token_control']
            assert r['usage']['prompt_tokens'] == 337833 and r['usage']['completion_tokens'] == 1362
            assert r['reported_cached_tokens'] == 336128
            assert r['cache_counter_delta']['sglang:cached_tokens_total'] == 336128
            assert r['input_sha256'] == m['formatted_chat']['sha256']
            records.append({'arm':arm,'launch':launch,'repeat':r['repeat'],
                            'ttft_ms':r['ttft_ms'],'tpot_ms':r['generation_ms']/1361,
                            'generation_ms':r['generation_ms'],'e2e_ms':r['e2e_ms'],
                            'http_complete_ms':r['http_complete_ms'],
                            'input_tokens':337833,'output_tokens':1362,'cached_tokens':336128})
first = manifests[0]
for m in manifests[1:]:
    assert [x for x in m['command'] if x != '--enable-nvidia-megakernel'] == first['command']
    assert m['environment'] == first['environment']
    assert m['sglang_sha'] == first['sglang_sha'] and m['plugin_sha'] == first['plugin_sha']
    assert m['model_provenance']['revision'] == first['model_provenance']['revision']
    assert m['formatted_chat'] == first['formatted_chat']
summary = []
for arm in ('off','on'):
    selected = [r for r in records if r['arm'] == arm]
    summary.append({'arm':arm, **{metric:{
        'median':stats.median(r[metric] for r in selected),
        'min':min(r[metric] for r in selected),'max':max(r[metric] for r in selected),
        'launch_medians':[stats.median(r[metric] for r in selected if r['launch']==n) for n in (1,2)]
    } for metric in ('tpot_ms','ttft_ms','e2e_ms')}})
fig, axes = plt.subplots(1,2,figsize=(9.5,5.4),layout='constrained')
colors={'off':'#526475','on':'#167742'}
for ax,metric,label in zip(axes,('tpot_ms','ttft_ms'),('Decode TPOT (ms/token)','Time to first text (ms)')):
    for i,arm in enumerate(('off','on')):
        rows=[r for r in records if r['arm']==arm]
        for launch in (1,2):
            selected=[r for r in rows if r['launch']==launch]
            offset=-.07 if launch==1 else .07
            ax.scatter([i+offset+(k-4.5)*.008 for k in range(10)], [r[metric] for r in selected],
                       color=colors[arm],alpha=.65,s=24,marker='o' if launch==1 else 's')
        median=stats.median(r[metric] for r in rows)
        ax.plot([i-.2,i+.2],[median,median],color=colors[arm],lw=3)
        ax.annotate(f'{median:.3f}' if metric=='tpot_ms' else f'{median:.1f}',
                    (i,median),xytext=(0,10),textcoords='offset points',ha='center',weight='bold')
    ax.set_xticks([0,1],['Plugin off','Plugin on'])
    ax.set_ylabel(label+' · lower is better')
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y',alpha=.18)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('One recovered AgentX long-context request\nDeepSeek-V4.1-Flash · 4 × GB300 · TP4 · concurrency 1',fontsize=14)
fig.text(.01,-.09,
         '337,833 input tokens · 336,128 cached · 1,362 output tokens · acceptance fixed at 5.4\n'
         'Two fresh launches per arm, 10 trials each after warmup. Dots/squares identify launches; lines show medians.\n'
         'Identical host-Engram / FP8-KV configuration. This isolated replay is not a full AgentX benchmark.',fontsize=9)
for ext in ('svg','png'):
    fig.savefig(a.out/f'nvidia-agentx-replay.{ext}',dpi=180,bbox_inches='tight')
(a.out/'agentx-results.json').write_text(json.dumps({
    'scope':'One recovered long-context request, not a complete AgentX run or frontier',
    'model':'deepseek-ai/DeepSeek-V4.1-Flash','model_revision':first['model_provenance']['revision'],
    'sglang_sha':first['sglang_sha'],'plugin_sha':first['plugin_sha'],
    'command_without_plugin':first['command'],'environment':first['environment'],
    'formatted_chat':first['formatted_chat'],
    'tpot_definition':'(last nonempty text or reasoning arrival - first nonempty arrival)/(1362-1)',
    'ttft_definition':'HTTP request start to first nonempty text or reasoning arrival',
    'historical_comparison':'Not compared to historical Sep30 GPU-Engram timings; this is a new matched plugin-compatible pair',
    'summary':summary,'trials':records,
},indent=2)+'\n')
print(json.dumps(summary,indent=2))
