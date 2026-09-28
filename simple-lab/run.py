"""Offline single run or one-parameter sweep; --check never loads torch."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import runpy
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KEYS = ('name prompt seed size num_inversion_steps num_inference_steps eta '
        'stop_timestep start_timestep gamma inversion_strength editing_strength '
        'guidance_scale').split()


def read_config():
    raw = runpy.run_path(str(HERE / 'config.py'))
    return validate({k: raw[k] for k in KEYS})


def validate(c):
    if not isinstance(c['name'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', c['name']):
        raise ValueError('name must contain only ASCII letters, digits, underscores, or hyphens.')
    if not isinstance(c['prompt'], str) or not c['prompt'].strip():
        raise ValueError('prompt must not be empty.')
    for k in ('seed', 'size', 'num_inversion_steps', 'num_inference_steps'):
        if type(c[k]) is not int or c[k] < (0 if k == 'seed' else 1):
            raise ValueError(f'{k} must be a valid integer.')
    if not 0 <= c['seed'] < 2**32:
        raise ValueError('seed must be between 0 and 2**32-1.')
    if c['size'] % 16:
        raise ValueError('size must be a multiple of 16.')
    for k in ('eta', 'gamma', 'start_timestep', 'stop_timestep',
              'inversion_strength', 'editing_strength', 'guidance_scale'):
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]):
            raise ValueError(f'{k} must be a finite number.')
    for k in ('eta', 'gamma', 'start_timestep', 'stop_timestep'):
        if not 0 <= c[k] <= 1:
            raise ValueError(f'{k} must be between 0 and 1.')
    for k in ('inversion_strength', 'editing_strength'):
        if not 0 < c[k] <= 1:
            raise ValueError(f'{k} must be greater than 0 and at most 1.')
    if c['start_timestep'] > c['stop_timestep'] or c['guidance_scale'] < 0:
        raise ValueError('Check the start/stop order and guidance_scale.')
    for steps, strength in [('num_inversion_steps', 'inversion_strength'), ('num_inference_steps', 'editing_strength')]:
        if c[steps] - int(c[steps] * (1 - c[strength])) < 2:
            raise ValueError('At least two steps must remain after schedule truncation.')
    return c


def check_files():
    pipeline = ROOT / 'pipeline_flux_rf_inversion.py'
    provenance = json.loads((ROOT / 'source.json').read_text())
    digest = hashlib.sha256(pipeline.read_bytes()).hexdigest()
    if digest != provenance['sha256']:
        raise ValueError('Pipeline hash differs from source.json. Check the source version.')
    if not (ROOT / 'input_tennis.jpg').is_file():
        raise FileNotFoundError('Missing input_tennis.jpg.')
    model = ROOT / 'FLUX.1-dev'
    required = ['model_index.json', 'scheduler/scheduler_config.json',
                'text_encoder/config.json', 'text_encoder/model.safetensors',
                'text_encoder_2/config.json', 'transformer/config.json',
                'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
                'tokenizer/vocab.json', 'tokenizer/merges.txt',
                'tokenizer_2/spiece.model', 'tokenizer_2/tokenizer_config.json']
    for folder, index in [('text_encoder_2', 'model.safetensors.index.json'),
                          ('transformer', 'diffusion_pytorch_model.safetensors.index.json')]:
        data = json.loads((model / folder / index).read_text())
        required.extend(f'{folder}/{s}' for s in set(data['weight_map'].values()))
    for rel in required:
        if not (model / rel).is_file() or not (model / rel).stat().st_size:
            raise FileNotFoundError(f'Missing or empty model file: {rel}')
    return digest


def make_plan():
    raw = runpy.run_path(str(HERE / 'config.py'))
    base = validate({k: raw[k] for k in KEYS})
    sweep = raw.get('sweep', None)
    if sweep is None:
        return base, None, [dict(base, label='baseline')]
    allowed = {'eta', 'gamma', 'stop_timestep', 'start_timestep',
               'editing_strength', 'inversion_strength', 'num_inference_steps',
               'num_inversion_steps', 'guidance_scale', 'window'}
    if sweep not in allowed:
        raise ValueError(f'sweep must be one of {sorted(allowed)}, or None.')
    values = raw.get('values')
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 32:
        raise ValueError('values must be a list or tuple of 1 to 32 entries.')
    plan = []
    for i, value in enumerate(values, 1):
        c = dict(base)
        if sweep == 'window':
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError('Each window entry must be a (start, stop) pair.')
            c['start_timestep'], c['stop_timestep'] = value
            label = f'window=[{value[0]:.4g},{value[1]:.4g})'
        else:
            c[sweep] = value
            label = f'{sweep}={value:.4g}'
        validate(c)
        c.update(name=f'{i:02d}', label=label)
        plan.append(c)
    return base, sweep, plan


def inversion_key(c):
    return tuple(c[k] for k in ('seed', 'size', 'num_inversion_steps',
                                'inversion_strength', 'gamma'))


def save_grid(out, panels, base, sweep):
    from PIL import Image, ImageDraw, ImageFont
    columns = min(4, len(panels))
    cell, label_h, header = 320, 38, 126
    canvas = Image.new('RGB', (columns * cell, header + math.ceil(len(panels) / columns) * (cell + label_h)), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        try:
            font = ImageFont.load_default(size=18)
        except TypeError:
            font = ImageFont.load_default()
    # Mark the swept fields instead of displaying their unused base values.
    def shown(k):
        if k == sweep or (sweep == 'window' and k in ('start_timestep', 'stop_timestep')):
            return 'varies'
        return str(round(base[k], 4))
    lines = [f"{base['name']} | sweep={sweep or 'none'} | seed={base['seed']} size={base['size']}",
             base['prompt'],
             f"eta={shown('eta')} gamma={shown('gamma')} start={shown('start_timestep')} stop={shown('stop_timestep')}",
             f"steps inv/edit={shown('num_inversion_steps')}/{shown('num_inference_steps')} strength inv/edit={shown('inversion_strength')}/{shown('editing_strength')}",
             f"guidance={shown('guidance_scale')} | full configuration: plan.json"]
    for i, line in enumerate(lines):
        draw.text((8, 5 + i * 23), line, fill='black', font=font)
    for i, (label, img) in enumerate(panels):
        x, y = (i % columns) * cell, header + (i // columns) * (cell + label_h)
        draw.text((x + 8, y + 9), label, fill='black', font=font)
        canvas.paste(img.resize((cell, cell), Image.Resampling.LANCZOS), (x, y + label_h))
    canvas.save(out / 'grid.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Validate files and display the experiment plan without loading the model')
    args = parser.parse_args()
    base, sweep, plan = make_plan()
    digest = check_files()
    out = HERE / 'results' / base['name']
    print(f'Planned outputs: {len(plan)} | Directory: {out}', flush=True)
    for c in plan:
        print(f"{c['name']}: {c['label']}", flush=True)
    if args.check:
        print('Configuration and required files passed. Model loading and GPU execution were not tested.')
        if out.exists():
            print('This output directory already exists. Change name before running.')
        return
    if out.exists():
        raise FileExistsError('Output directory exists. Change name in config.py to avoid overwriting results.')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    from pipeline_flux_rf_inversion import RFInversionFluxPipeline
    if importlib.metadata.version('diffusers') != '0.35.1':
        raise RuntimeError('Activate the environment with diffusers==0.35.1.')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable GPU such as an A100 is required. Use --check without a GPU.')
    out.mkdir(parents=True)
    shutil.copy2(HERE / 'config.py', out / 'config.py')
    common = dict(base=base, sweep=sweep, cases=plan, pipeline_sha256=digest,
                  input_sha256=hashlib.sha256((ROOT / 'input_tennis.jpg').read_bytes()).hexdigest(),
                  versions={p: importlib.metadata.version(p) for p in
                            ('torch', 'diffusers', 'transformers', 'accelerate')},
                  source_prompt='', source_guidance_scale=0.0,
                  max_sequence_length=512, dtype='bfloat16')
    (out / 'plan.json').write_text(json.dumps(common, indent=2, ensure_ascii=False))
    notes = '# Experiment notes\n\nCompare within this group. One input and one seed do not establish a general result.\n\n'
    for c in plan:
        notes += f"## {c['name']} {c['label']}\n\n- Prediction:\n- Subject color, shape, and size:\n- Background, position, and reflection:\n- Artifacts:\n- Observation relative to the baseline:\n\n"
    (out / 'notes.md').write_text(notes)
    source = Image.open(ROOT / 'input_tennis.jpg').convert('RGB')
    source = source.resize((base['size'], base['size']), Image.Resampling.LANCZOS)
    source.save(out / 'input.png')
    panels = [('input', source)]
    print('Loading the local model once for this group...', flush=True)
    pipe = RFInversionFluxPipeline.from_pretrained(
        str(ROOT / 'FLUX.1-dev'), torch_dtype=torch.bfloat16,
        local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    cache = {}
    def seed(c):
        random.seed(c['seed']); np.random.seed(c['seed'])
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    for c in plan:
        record_path = out / f"{c['name']}.json"
        record = dict(config=c, status='running')
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        started = time.monotonic()
        print(f"Generating {c['name']}/{len(plan)}: {c['label']}", flush=True)
        try:
            key = inversion_key(c)
            reused = key in cache
            with torch.inference_mode():
                if not reused:
                    seed(c)
                    values = pipe.invert(
                        image=source, source_prompt='', source_guidance_scale=0.0,
                        num_inversion_steps=c['num_inversion_steps'], strength=c['inversion_strength'],
                        gamma=c['gamma'], height=c['size'], width=c['size'], dtype=torch.bfloat16)
                    cache[key] = tuple(v.detach().cpu().clone() for v in values)
                    del values
                    pipe.maybe_free_model_hooks()
                inverted, original, ids = (v.to('cuda').clone() for v in cache[key])
                seed(c)
                result = pipe(
                    prompt=c['prompt'], inverted_latents=inverted, image_latents=original,
                    latent_image_ids=ids, height=c['size'], width=c['size'],
                    num_inference_steps=c['num_inference_steps'], strength=c['editing_strength'],
                    start_timestep=c['start_timestep'], stop_timestep=c['stop_timestep'],
                    eta=c['eta'], guidance_scale=c['guidance_scale'], max_sequence_length=512).images[0]
            result.save(out / f"{c['name']}.png")
            panels.append((c['label'], result))
            save_grid(out, panels, base, sweep)
            n = c['num_inference_steps']
            effective = n - int(max(n - min(n * c['editing_strength'], n), 0))
            start, stop = int(c['start_timestep'] * n), min(int(c['stop_timestep'] * n), n)
            record.update(status='complete', inversion_reused=reused,
                          editing_steps_from_schedule=effective,
                          controller_active_indices=[i for i in range(effective) if start <= i < stop and c['eta'] != 0],
                          seconds=round(time.monotonic() - started, 2))
            del inverted, original, ids
            pipe.maybe_free_model_hooks()
        except BaseException as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f'Done. Open {out / "grid.png"} and record observations in notes.md.', flush=True)


if __name__ == '__main__':
    main()
