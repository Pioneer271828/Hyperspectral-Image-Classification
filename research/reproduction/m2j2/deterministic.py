"""Explicit verification context only; never used for optimizer updates."""
import os,sys,json,runpy
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
print('VERIFICATION_CONTEXT',json.dumps({'torch':torch.__version__,'deterministic':True,
 'cudnn_deterministic':True,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
 'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cublas_workspace':os.environ['CUBLAS_WORKSPACE_CONFIG']}),flush=True)
entry=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(entry,run_name='__main__')
