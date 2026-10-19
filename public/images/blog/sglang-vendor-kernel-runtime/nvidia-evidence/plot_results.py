#!/usr/bin/env python3
"""Render only complete on/off repeats. Error bars are ranges, not confidence intervals."""
import argparse
import json
import statistics as stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--results', required=True, type=Path)
ap.add_argument('--out', required=True, type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
records = []
manifests = []
prompt_hashes = {}
for arm in ('off', 'on'):
    for repeat in (1, 2):
        directory = a.results / f'static-{arm}-r{repeat}'
        m = json.loads((directory / 'manifest.json').read_text())
        assert m['status'] == 'passed', directory
        assert m['arm'] == arm and m['mode'] == 'static'
        assert m['environment']['SGLANG_SIMULATE_ACC_LEN'] == '3.51'
        info = json.loads((directory / 'server-info.json').read_text())
        assert info['random_seed'] == 496233168, 'Effective server seed differs'
        manifests.append(m)
        for p in (1, 2):
            raw = [json.loads(line) for line in (directory / f'pass{p}/requests.jsonl').read_text().splitlines()]
            for c in (1, 2, 4, 8):
                assert sum(row['conc'] == c for row in raw) == (30 if c == 1 else 5*c)
            for row in raw:
                assert row['prompt_tokens'] == 8192 and row['completion_tokens'] == 1024
                key = (p, row['prompt_key'])
                assert row['prompt_sha256'] == prompt_hashes.setdefault(key, row['prompt_sha256']), 'Mismatched input across arms'
            data = [json.loads(line) for line in (directory / f'pass{p}/summary.jsonl').read_text().splitlines()]
            assert sorted(r['conc'] for r in data) == [1, 2, 4, 8]
            for r in data:
                assert r['isl'] == 8192 and r['osl'] == 1024
                records.append({'arm': arm, 'launch': repeat, 'pass': p, **r})

def normalized_command(command):
    result = [x for x in command if x != '--enable-nvidia-megakernel']
    # First baseline resolved this seed automatically; later launches pin it.
    if '--random-seed' not in result:
        at = result.index('--host')
        result[at:at] = ['--random-seed', '496233168']
    return result

first = manifests[0]
for m in manifests[1:]:
    assert normalized_command(m['command']) == normalized_command(first['command']), 'Launch mismatch'
    assert m['environment'] == first['environment'], 'Environment mismatch'
    assert m['sglang_sha'] == first['sglang_sha'] and m['plugin_sha'] == first['plugin_sha']
    assert m['model_provenance']['revision'] == first['model_provenance']['revision']

points = []
for arm in ('off', 'on'):
    for c in (1, 2, 4, 8):
        launches = []
        for repeat in (1, 2):
            rows = [r for r in records if r['arm'] == arm and r['conc'] == c and r['launch'] == repeat]
            assert len(rows) == 2
            launches.append({key: stats.mean(r[key] for r in rows) for key in ('tps_user_p50', 'out_tok_s')})
        points.append({'arm': arm, 'concurrency': c, 'launches': launches,
                       **{key: {'mean': stats.mean(r[key] for r in launches),
                                'min': min(r[key] for r in launches),
                                'max': max(r[key] for r in launches)}
                          for key in ('tps_user_p50', 'out_tok_s')}})

fig, ax = plt.subplots(figsize=(9.5, 6.3), layout='constrained')
colors = {'off': '#526475', 'on': '#167742'}
for arm in ('off', 'on'):
    series = [p for p in points if p['arm'] == arm]
    x = [p['out_tok_s']['mean'] / 4 for p in series]
    y = [p['tps_user_p50']['mean'] for p in series]
    xerr = [[(p['out_tok_s']['mean'] - p['out_tok_s']['min']) / 4 for p in series],
            [(p['out_tok_s']['max'] - p['out_tok_s']['mean']) / 4 for p in series]]
    yerr = [[p['tps_user_p50']['mean'] - p['tps_user_p50']['min'] for p in series],
            [p['tps_user_p50']['max'] - p['tps_user_p50']['mean'] for p in series]]
    individual = [r for r in records if r['arm'] == arm]
    ax.scatter([r['out_tok_s']/4 for r in individual],
               [r['tps_user_p50'] for r in individual], color=colors[arm],
               s=16, alpha=.35, zorder=1)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, color=colors[arm], marker='o',
                capsize=4, linewidth=2, label=f'NVIDIA plugin {arm}')
    for p, px, py in zip(series, x, y):
        ax.annotate(f"c={p['concurrency']}", (px, py), xytext=(7, 7 if arm == 'on' else -15),
                    textcoords='offset points', fontsize=10, color=colors[arm])
ax.set_xlabel('Aggregate output throughput per GPU (tokens/s/GPU, including prefill)')
ax.set_ylabel('Per-user decode speed (tokens/s; mean of pass medians)')
ax.set_title('DeepSeek-V4.1-Flash: low-concurrency serving trade-off\n4 × GB300 · TP4 · DSpark block 5 · FP8 KV · FlashMLA', fontsize=14, pad=15)
ax.grid(alpha=.18)
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.text(.01, -.065,
         'Static 8K input / 1K output; simulated mean acceptance 3.51. Same effective config apart from plugin flag.\n'
         'Two fresh launches per arm; small dots show passes, bars span the two launch means (not confidence intervals).\n'
         'Points cover concurrency 1, 2, 4, 8; connecting lines are guides, not a full serving frontier.', fontsize=9)
for ext in ('svg', 'png'):
    fig.savefig(a.out / f'nvidia-low-concurrency.{ext}', dpi=180, bbox_inches='tight')
(a.out / 'static-results.json').write_text(json.dumps({
    'model': first['model_provenance']['repo'], 'model_revision': first['model_provenance']['revision'],
    'sglang_sha': first['sglang_sha'], 'plugin_sha': first['plugin_sha'],
    'command_without_plugin': normalized_command(first['command']), 'environment': first['environment'],
    'points': points, 'pass_summaries': records,
    'error_bars': 'Range of two fresh-launch means; not a confidence interval',
    'seed_note': 'Effective seed496233168 verified in each server-info. First baseline selected it automatically; subsequent launches pin it explicitly.',
}, indent=2) + '\n')
print(json.dumps(points, indent=2))
