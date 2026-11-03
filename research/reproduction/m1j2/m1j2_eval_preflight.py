import os,sys
from pathlib import Path
ROOT=Path('/home/lbx/hsi-pioneer271828/checkouts/m1j2-issue127')
os.chdir(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import torch
from test_calibrated_selection import items
from test_joint_training import settings
from test_joint_observation import metadata
from joint_observation_study import load_study,evaluation_bank,condition_view
from model.research_config import ResearchConfig
from joint_calibrated_selection import observations
from joint_baseline import forward
from model.bn_calibration import calibrate_denoise,CONTEXT,POLICY,state_digest
torch.set_num_threads(1)
data=items();meta=metadata(data);bank=evaluation_bank(load_study(),meta,'fixture')
for device in ('cpu','cuda'):
 model=ResearchConfig.from_dict(settings()['config']).build_model({n:i['class_num'] for n,i in data.items()}).to(device).eval()
 original=state_digest(model);count=0
 for condition,records in bank.items():
  views={n:condition_view(i,meta[n],records[n],'fixture') for n,i in data.items()}
  calibrated,receipt=calibrate_denoise(model,observations(views),mode=POLICY,context=CONTEXT,source_checkpoint_sha256='0'*64)
  assert receipt['optimizer_updates']==0 and receipt['parameters_unchanged']
  assert state_digest(model)==original
  for n,p in model.named_parameters():assert torch.equal(p,dict(calibrated.named_parameters())[n])
  with torch.no_grad():
   for n,i in views.items():
    view={k:v.to(device) if torch.is_tensor(v) else v for k,v in i.items()}
    output=forward(calibrated,n,view)
    assert len(output)==3
    for prob in output:
     assert torch.isfinite(prob).all() and torch.allclose(prob.sum(1),torch.ones(prob.shape[0],device=device),atol=2e-6,rtol=0)
    count+=1
 print('PASSED',device,count,'condition/dataset checks, unchanged source and parameters',flush=True)
