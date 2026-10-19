#!/usr/bin/env python3
"""Repeat the legacy GSM8K gate with per-question outputs and truncation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
import requests
import sglang.test.few_shot_gsm8k as gsm
from sglang.utils import download_and_cache_file, read_jsonl

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--base-url',required=True)
ap.add_argument('--out',required=True,type=Path)
a=ap.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
url='https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
data=Path(download_and_cache_file(url))
lines=list(read_jsonl(str(data)))
labels=[gsm.get_answer_value(row['answer']) for row in lines[:200]]
host=urlsplit(a.base_url)
active={}
def save_outputs(filename,states):
    rows=[]
    for index,state in enumerate(states):
        text=state['answer']
        prediction=gsm.get_answer_value(text)
        prompt=gsm.get_few_shot_examples(lines,8)+gsm.get_one_example(lines,index,False)
        rows.append({'index':index,'label':labels[index],'prediction':prediction,
                     'correct':prediction==labels[index],'answer':text,
                     'meta_info':state.get_meta_info('answer'),
                     'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                     'answer_sha256':hashlib.sha256(text.encode()).hexdigest()})
    (a.out/(active['name']+'-questions.json')).write_text(json.dumps(rows,indent=2,default=str)+'\n')
    active['rows']=rows
gsm.dump_state_text=save_outputs
manifest={'dataset_url':url,'dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
          'implementation':'Pinned sglang.test.few_shot_gsm8k.run_eval, unchanged prompt construction and scorer',
          'caveat':'Legacy gate uses first8 test examples as demonstrations and first200 as questions. This overlap makes it a regression gate, not a clean held-out GSM8K benchmark.',
          'temperature':0,'acceptance':'real, no simulation','runs':[]}
for parallel,limit in [(12,512),(16,512),(8,1024)]:
    active.clear();active['name']=f'p{parallel}-max{limit}'
    response=requests.post(a.base_url+'/flush_cache',params={'timeout':30},timeout=40)
    response.raise_for_status()
    metrics=gsm.run_eval(SimpleNamespace(num_shots=8,data_path=str(data),num_questions=200,
                        parallel=parallel,max_new_tokens=limit,host=host.hostname,port=host.port,temperature=0))
    rows=active['rows']
    manifest['runs'].append({'name':active['name'],'parallel':parallel,'max_new_tokens':limit,
        'metrics':{k:float(v) for k,v in metrics.items()},'correct':sum(r['correct'] for r in rows),
        'budget_reached_indices':[r['index'] for r in rows if r['meta_info'].get('completion_tokens',0)>=limit],
        'incorrect_indices':[r['index'] for r in rows if not r['correct']]})
    (a.out/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['runs'][-1]),flush=True)
