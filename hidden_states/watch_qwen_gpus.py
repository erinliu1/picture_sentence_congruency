"""One-shot Qwen-235B pipeline, triggered by three idle GPUs (poll every 5 min)."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path('/Intern/Erin/tmp/qwen_gpu_monitor')
MODEL = 'qwen3_vl_235b_a22b_instruct'
WEBHOOK_FILE = RUN_DIR / 'slack_webhook_url'


def log(message):
    print(f'{datetime.now(timezone.utc).isoformat()} {message}', flush=True)


def status(state, **details):
    data = {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
    temporary = RUN_DIR / 'status.tmp'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(RUN_DIR / 'status.json')


def notify(message):
    log(message)
    with (RUN_DIR / 'alerts.log').open('a') as file:
        file.write(f'{datetime.now(timezone.utc).isoformat()} {message}\n')
    if not WEBHOOK_FILE.exists():
        return
    try:
        url = WEBHOOK_FILE.read_text().strip()
        if not url.startswith('https://hooks.slack.com/services/'):
            raise ValueError('Invalid Slack webhook URL')
        request = Request(url, data=json.dumps({'text': message}).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=15) as response:
            if response.read().strip() != b'ok':
                raise RuntimeError('Unexpected Slack response')
    except Exception as error:
        # Never expose the webhook secret in logs, including exception URLs.
        log(f'Slack notification failed ({type(error).__name__}); pipeline continues.')


def idle_gpus():
    def query(argument):
        result = subprocess.run(['nvidia-smi', argument, '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=30)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    gpus = query('--query-gpu=index,uuid,memory.free,utilization.gpu')
    processes = query('--query-compute-apps=gpu_uuid,pid')
    busy = {row[0].strip() for row in processes if row}
    # Unknown measurements fail closed. 175 GiB matches extraction's memory limit.
    return [int(index) for index, uuid, free, utilization in gpus
            if uuid.strip() not in busy and free.strip().isdigit()
            and utilization.strip().isdigit() and int(free) >= 175 * 1024
            and int(utilization) <= 1]


def run_pipeline(selected):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, selected)), QWEN_HS_MULTI_GPU='1',
               TMPDIR='/Intern/Erin/tmp', TMP='/Intern/Erin/tmp', TEMP='/Intern/Erin/tmp',
               HF_HOME='/Intern/Erin/models/huggingface', HF_HUB_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR='/Intern/Erin/tmp/matplotlib', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    stages = [
        ('extraction', 'hidden_states', ['-c', f'from hs_qwen import extract_hidden_states; extract_hidden_states({MODEL!r}, instructions=True)']),
        ('decoding', 'decoding', ['lr_main.py', MODEL]),
        ('alignment', 'alignment', ['alignment_main.py', MODEL]),
    ]
    for stage, folder, args in stages:
        status('running', stage=stage, gpus=selected, pid=os.getpid())
        notify(f'Qwen-235B: starting {stage}; selected GPUs {selected}.')
        with (RUN_DIR / 'pipeline.log').open('a') as output:
            subprocess.run([sys.executable, '-u', '-B', *args], cwd=ROOT / folder,
                           env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    # Check essential outputs before reporting completion.
    import torch
    import pandas as pd
    model_dir = ROOT / 'models' / MODEL
    files = list((model_dir / 'hidden_states_instruction').glob('*.pt'))
    if len(files) != 320:
        raise ValueError('Expected 320 hidden-state files')
    layers = None
    for file in files:
        tensor = torch.load(file, map_location='cpu', weights_only=True)
        if tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise ValueError('Invalid hidden-state tensor')
        if layers is None:
            layers = tensor.shape[0]
        if tensor.shape[0] != layers:
            raise ValueError('Inconsistent layer count')
    results = pd.read_csv(model_dir / 'decoding_instruction/results.csv')
    bootstraps = pd.read_csv(model_dir / 'decoding_instruction/all_bootstraps.csv')
    coefficients = pd.read_csv(model_dir / 'alignment_instruction/coefficients.csv')
    pairs = pd.read_csv(model_dir / 'alignment_instruction/final_layer_pairs.csv')
    if len(results) != layers * 320 or not (results.groupby('layer_index').size() == 320).all():
        raise ValueError('Incomplete decoding')
    if len(bootstraps) != layers * 10000 or not (bootstraps.groupby('layer_index').size() == 10000).all():
        raise ValueError('Incomplete bootstrap')
    if len(coefficients) != layers or coefficients.isna().any().any() or len(pairs) != 320 or pairs.isna().any().any():
        raise ValueError('Incomplete alignment')
    for folder in ['decoding', 'alignment']:
        plot = ROOT / folder / 'plots/qwen_instruction' / f'{MODEL}_instruction_comparison.png'
        if not plot.exists():
            raise ValueError('Missing comparison plot')
    status('complete', gpus=selected, layers=layers)
    notify('Qwen-235B analysis complete: extraction, LR training, 10,000 bootstraps/layer, alignment, final-layer pairs, and both comparison PNGs validated.')


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / 'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A monitor is already running.')
        if '--check' in sys.argv:
            print(json.dumps({'idle_gpus': idle_gpus()}))
            return
        state_file = RUN_DIR / 'status.json'
        if state_file.exists() and json.loads(state_file.read_text())['state'] in ('running', 'complete', 'failed'):
            raise SystemExit('Prior launch recorded; inspect status before explicitly resetting it.')
        (RUN_DIR / 'monitor.pid').write_text(str(os.getpid()) + '\n')
        while True:
            try:
                available = idle_gpus()
                log(f'Idle GPUs: {available}; need 3 (no compute processes, <=1% utilization, >=175 GiB free).')
                status('waiting', idle_gpus=available, pid=os.getpid(), interval_seconds=300)
                if len(available) >= 3:
                    selected = available[:3]
                    log(f'Rechecking GPUs {selected} in 30 seconds before launch.')
                    time.sleep(30)
                    if set(selected).issubset(idle_gpus()):
                        # Persist before notification/launch, so accidental restarts cannot duplicate jobs.
                        status('running', stage='launch', gpus=selected, pid=os.getpid())
                        notify(f'Three idle GPUs confirmed: {selected}. Automatically launching Qwen-235B instruction analysis.')
                        try:
                            run_pipeline(selected)
                        except Exception as error:
                            status('failed', error_type=type(error).__name__, gpus=selected)
                            notify(f'Qwen-235B pipeline failed ({type(error).__name__}). See {RUN_DIR}/pipeline.log; no automatic retry.')
                            raise
                        return
            except (subprocess.SubprocessError, ValueError) as error:
                if state_file.exists() and json.loads(state_file.read_text())['state'] == 'failed':
                    raise
                log(f'GPU check failed ({type(error).__name__}); will retry in 5 minutes.')
            time.sleep(300)


if __name__ == '__main__':
    main()
