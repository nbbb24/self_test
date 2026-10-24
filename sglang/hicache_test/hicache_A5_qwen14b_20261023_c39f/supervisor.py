import subprocess,os,time,socket,json,signal,urllib.request,traceback
from pathlib import Path
p=Path('/home/sxy/hicache_A5_qwen14b_20261023_c39f'); children=[]; handles=[]
result={'status':'RUNNING','model':'/home/weights/Qwen3-14B','device':4,'protocol':'device_urma'}
def save(): (p/'run-status.json').write_text(json.dumps(result,indent=2))
def start(name,argv):
    f=open(p/(name+'.log'),'w');handles.append(f)
    c=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append((name,c));result[name+'_pid']=c.pid;save();return c
if (p/'run-status.json').exists(): raise SystemExit('Existing run: archive/copy to a fresh directory before rerunning')
save()
try:
    for port in [5006,6006,18006,31000]:
        with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
    meta=start('meta',['python3','-B','-m','sglang.srt.mem_cache.storage.npu_memcache.start_meta_service','--config_path',str(p/'meta.json')])
    for i in range(30):
        if meta.poll() is not None:raise RuntimeError('MetaService exited')
        try:
            with socket.create_connection(('127.0.0.1',5006),timeout=1):break
        except OSError:time.sleep(1)
    else:raise TimeoutError('MetaService startup timeout')
    server=start('server',['bash',str(p/'basic.sh')])
    for i in range(240):
        if server.poll() is not None:raise RuntimeError('SGLang exited: '+str(server.returncode))
        try:
            with urllib.request.urlopen('http://127.0.0.1:31000/health',timeout=2) as r:
                if r.status==200:break
        except Exception:time.sleep(2)
    else:raise TimeoutError('SGLang startup exceeded 8 minutes')
    import psutil
    maps={}
    for proc in [psutil.Process(server.pid)]+psutil.Process(server.pid).children(recursive=True):
        try:
            libs=sorted({m.path for m in proc.memory_maps() if any(x in m.path for x in ['libmf','libsmem','libhybm','_pymmc','libtorch_npu'])})
            if libs: maps[str(proc.pid)]={'name':proc.name(),'libraries':libs}
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    (p/'loaded-libraries.json').write_text(json.dumps(maps,indent=2))
    result['status']='VERIFYING';save()
    verify=start('verify',['python3','-B',str(p/'verify.py'),'--router-url','http://127.0.0.1:31000','--prefill-url','http://127.0.0.1:31000','--prompt-repeat','16','--max-tokens','16','--settle-seconds','10','--output-dir',str(p/'validation')])
    rc=verify.wait(timeout=600)
    if rc:raise RuntimeError('Verifier failed: '+str(rc))
    result['status']='PASSED'
except BaseException as e:
    result['status']='FAILED';result['error']=str(e);traceback.print_exc()
finally:
    for name,c in reversed(children):
        if c.poll() is None:
            try: os.killpg(c.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: c.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try: os.killpg(c.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                c.wait()
        result[name+'_exitcode']=c.returncode
    result['cleanup_completed']=True;save()
    for h in handles:h.close()
