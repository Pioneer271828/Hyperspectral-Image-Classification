"""Finish only the authorized M1-J2 batch and independent evaluation, then exit."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
BASE=Path('/home/lbx/hsi-pioneer271828');records=BASE/'transfer-records'
scripts=[records/'m1j2_evaluate.py',records/'m1j2_qa.py']
hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
assert (records/'m1j2-eval-preflight01.log').read_text().count('PASSED')==2
print('WAIT_TRAINING',hashes,flush=True)
while True:
 status=json.loads((BASE/'runs/m1j2-issue127-full01/status.json').read_text())['status']
 if status=='failed':raise RuntimeError('Training batch failed; no scoring')
 if status=='training_completed':break
 time.sleep(20)
while True:
 free,util=map(int,subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free,utilization.gpu',
                                          '--format=csv,noheader,nounits']).decode().strip().split(','))
 if free>=11264 and util<=60:break
 time.sleep(20)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'7','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
for script in scripts:
 assert hashlib.sha256(script.read_bytes()).hexdigest()==hashes[script]
 log=records/(script.stem+'01.log');print('START',script.name,flush=True)
 with log.open('x') as stream:
  subprocess.run([sys.executable,str(script)],cwd=BASE/'checkouts/m1j2-issue127',env=env,
                 stdout=stream,stderr=subprocess.STDOUT,check=True)
print('AUTHORIZED_MODULE_COMPLETE',flush=True)
