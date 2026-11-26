"""Wall/child-CPU accounting for one already-authorized local container command."""
import datetime,json,resource,subprocess,sys,time
from pathlib import Path
base=Path(sys.argv[1]);command=sys.argv[2:];start=time.monotonic_ns();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with base.with_suffix('.start.json').open('x') as f:json.dump({'command':command,'start_monotonic_ns':start,'remote_utc':utc},f)
with base.with_suffix('.log').open('x') as f:
 result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
end=time.monotonic_ns();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
with base.with_suffix('.timing.json').open('x') as f:json.dump({'command':command,'start_monotonic_ns':start,'end_monotonic_ns':end,'elapsed_seconds':(end-start)/1e9,'start_remote_utc':utc,'end_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':result.returncode,'child_user_cpu_seconds':usage.ru_utime,'child_system_cpu_seconds':usage.ru_stime,'resource_threshold_sleep_seconds':0,'pure_gpu_kernel_time':None},f)
base.with_suffix('.exit').write_text(str(result.returncode)+'\n');raise SystemExit(result.returncode)
