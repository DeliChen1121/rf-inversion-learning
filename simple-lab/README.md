# Run a parameter sweep

Edit `name`, `sweep`, and `values` in `config.py`. Keep the other settings fixed. The scanned field overrides its baseline value.

```python
name = "01_eta_scan"
sweep = "eta"
values = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
```

```bash
python run.py --check
python -u run.py
```

Use a new name for each group. `sweep = None` runs one baseline case.

## Sweep recipes

Run these as separate groups, not a Cartesian product.

| Group name | `sweep` | `values` |
|---|---|---|
| `01_eta_scan` | `"eta"` | `[0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]` |
| `02_stop_scan` | `"stop_timestep"` | `[2/28, 4/28, 7/28, 14/28, 20/28, 26/28]` |
| `03_window_scan` | `"window"` | `[(0, 7/28), (7/28, 14/28), (14/28, 21/28)]` |
| `04_gamma_scan` | `"gamma"` | `[0, 0.2, 0.5, 0.7, 1.0]` |
| `05_edit_strength_scan` | `"editing_strength"` | `[0.2, 0.5, 0.75, 1.0]` |
| `06_edit_steps_scan` | `"num_inference_steps"` | `[14, 28, 56]` |

`window` pairs set `(start, stop)`. Fractions refer to requested editing steps, with an exclusive stop. At 28 steps and strength 1, the three example windows each control seven updates.

The step sweep keeps inversion at 28 steps and the cutoff fraction at 0.25; controlled update counts become 3, 7, and 14.

## What the runner saves

```text
results/<name>/
  grid.png          Input and all completed outputs
  input.png         Resized input
  01.png, 02.png…   Individual outputs
  01.json, 02.json… Per-case settings, status, schedule counts, timing
  config.py         Configuration snapshot
  plan.json         Group settings, versions, input and pipeline hashes
  notes.md          Observation template
```

The model loads once per group. Editing-only sweeps reuse the same inversion; changing gamma, inversion steps, or inversion strength recomputes it with the same seed. Latent tensors are cloned before each edit.

The grid updates after each completed case. On failure, completed outputs remain, but automatic resume is not implemented; use a new group name for a retry. Timing includes orchestration and is not a controlled performance benchmark.

API `editing_strength` truncates the schedule and is not paper controller strength. See [implementation notes](../notes/pipeline.md) before interpreting that sweep.
