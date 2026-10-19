#!/usr/bin/env python3
"""Capture warmed graph execution with an exact C1 prompt from the static benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
from static_raw import ids

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--base-url', required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--name', required=True)
a=ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)

def post(path, body=None):
    req=urllib.request.Request(a.base_url+path, data=json.dumps(body or {}).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        text=response.read().decode()
    try:
        result=json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(result,dict) and result.get('success') is False:
        raise RuntimeError(result)
    return result

tokens=ids(8192, 'decode-8192-1024-c1r1-t0')
body={'input_ids':tokens, 'sampling_params':{'temperature':0,'max_new_tokens':1024,'ignore_eos':True}}
post('/flush_cache')
warm=post('/generate', body)
post('/flush_cache')
capture=a.out/'all-ranks'
capture.mkdir()
payload={'output_dir':str(capture),'num_steps':5,'activities':['CPU','GPU'],
         'profile_by_stage':True,'merge_profiles':False,'with_stack':False,
         'record_shapes':True,'detailed_annotations':True,'profile_id':a.name+'-static-c1'}
(a.out/'capture-manifest.json').write_text(json.dumps({
    'input_ids_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
    'input_tokens':8192,'output_tokens':1024,'concurrency':1,
    'warmup':'One complete matching unprofiled request after prior timing passes',
    'cache':'Flush before warmup and before captured request',
    'profiler_request':payload,'warmup_meta_info':warm.get('meta_info'),
    'timing_caveat':'Profile durations are diagnostic; primary performance comes from unprofiled runs',
},indent=2)+'\n')
post('/start_profile',payload)
try:
    result=post('/generate',body)
    (a.out/'captured-meta-info.json').write_text(json.dumps(result.get('meta_info'),indent=2)+'\n')
finally:
    try:
        post('/stop_profile')
    except Exception as error:
        (a.out/'stop-profile-note.txt').write_text(str(error))
deadline=time.monotonic()+180
while True:
    traces=[p for p in capture.glob('*.trace.json.gz') if 'decode' in p.name.lower()]
    if len(traces)==4:
        break
    if time.monotonic()>=deadline:
        raise RuntimeError(f'Expected four rank decode traces, got {[p.name for p in capture.iterdir()]}')
    time.sleep(2)
decode=a.out/'decode'
decode.mkdir()
for trace in traces:
    shutil.copy2(trace,decode/trace.name)
print(json.dumps({'decode_traces':[str(p) for p in traces]},indent=2))
