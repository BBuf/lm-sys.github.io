#!/usr/bin/env python3
"""Report paired legacy-gate observations without claiming accuracy equivalence."""
import argparse
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--results',required=True,type=Path)
ap.add_argument('--out',required=True,type=Path)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
data={}
for arm in ('off','on'):
    path=a.results/f'accuracy-audit-{arm}-v1'
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['status']=='passed' and manifest['mode']=='accuracy-audit'
    assert not any(k.startswith('SGLANG_SIMULATE_ACC_') for k in manifest['environment'])
    assert json.loads((path/'server-info.json').read_text())['random_seed']==496233168
    summary=json.loads((path/'gsm8k-audit/summary.json').read_text())
    data[arm]={'manifest':manifest,'summary':summary,'path':path}
base=data['off']['manifest']
candidate=data['on']['manifest']
assert [x for x in candidate['command'] if x!='--enable-nvidia-megakernel']==base['command']
assert candidate['environment']==base['environment']
assert data['on']['summary']['dataset_sha256']==data['off']['summary']['dataset_sha256']
cases=[]
for case in data['off']['summary']['runs']:
    name=case['name'];questions={}
    for arm in ('off','on'):
        p=data[arm]['path']/f'gsm8k-audit/{name}-questions.json'
        questions[arm]=json.loads(p.read_text())
        assert len(questions[arm])==200
    pairs=list(zip(questions['off'],questions['on']))
    for off,on in pairs:
        assert off['index']==on['index'] and off['label']==on['label'] and off['prompt_sha256']==on['prompt_sha256']
    arms={}
    for arm in ('off','on'):
        rows=questions[arm]
        source=next(r for r in data[arm]['summary']['runs'] if r['name']==name)
        verify=sum(r['meta_info']['spec_verify_ct'] for r in rows)
        correct_drafts=sum(r['meta_info']['spec_num_correct_drafts'] for r in rows)
        proposed=sum(r['meta_info']['spec_num_proposed_drafts'] for r in rows)
        arms[arm]={'correct':sum(r['correct'] for r in rows),'n':len(rows),
                   'after_demonstration_indices_correct':sum(r['correct'] for r in rows[8:]),
                   'after_demonstration_indices_n':192,
                   'budget_reached_indices':source['budget_reached_indices'],
                   'incorrect_indices':source['incorrect_indices'],
                   'real_acceptance':{'spec_verify_calls':verify,
                       'correct_drafts':correct_drafts,'proposed_drafts':proposed,
                       'draft_accept_rate':correct_drafts/proposed,
                       'reported_accept_length_weighted_by_verify_calls':sum(
                           r['meta_info']['spec_accept_length']*r['meta_info']['spec_verify_ct'] for r in rows)/verify},
                   'per_question':[{'index':r['index'],'label':r['label'],'prediction':r['prediction'],
                       'correct':r['correct'],'completion_tokens':r['meta_info']['completion_tokens'],
                       'finish_reason':r['meta_info']['finish_reason'],
                       'prompt_sha256':r['prompt_sha256'],'answer_sha256':r['answer_sha256'],
                       **{k:r['meta_info'][k] for k in ['spec_verify_ct','spec_accept_length',
                           'spec_num_correct_drafts','spec_num_proposed_drafts']}} for r in rows],
                   'questions_sha256':hashlib.sha256((data[arm]['path']/f'gsm8k-audit/{name}-questions.json').read_bytes()).hexdigest()}
    cases.append({'name':name,'parallel':case['parallel'],'max_new_tokens':case['max_new_tokens'],
        'arms':arms,
        'off_correct_on_incorrect':[x['index'] for x,y in pairs if x['correct'] and not y['correct']],
        'off_incorrect_on_correct':[x['index'] for x,y in pairs if not x['correct'] and y['correct']],
        'different_answer_hash_count':sum(x['answer_sha256']!=y['answer_sha256'] for x,y in pairs),
        'metadata_keys':sorted({k for r in questions['off']+questions['on'] for k in r['meta_info']})})
out={'scope':'Paired follow-up of the pinned legacy regression gate, not a standard held-out GSM8K benchmark',
     'dataset_url':data['off']['summary']['dataset_url'],'dataset_sha256':data['off']['summary']['dataset_sha256'],
     'caveat':data['off']['summary']['caveat'],'acceptance':'real, no simulation',
     'seed':496233168,'temperature':0,'cases':cases,
     'conclusion_limit':'Small checks and threshold passes do not establish numerical equivalence or absence of an accuracy regression.',
     'sglang_sha':base['sglang_sha'],'plugin_sha':base['plugin_sha'],
     'command_without_plugin':base['command'],'environment':base['environment']}
(a.out/'accuracy-audit-evidence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([{**{k:v for k,v in c.items() if k!='arms'},'arms':{
    arm:{k:v for k,v in value.items() if k!='per_question'} for arm,value in c['arms'].items()}}
    for c in cases],indent=2))
