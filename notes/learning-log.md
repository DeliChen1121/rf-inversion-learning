# Learning log

A retrospective by stage; exact dates are not assigned to earlier runs.

| Stage | Work | What changed in my understanding |
|---|---|---|
| Basic generation | DDPM cat images and Stable Diffusion 1.5 | Learned model loading, caching, random seeds, and sampling steps. |
| Image editing | FLUX.1-dev + RF-Inversion on AutoDL | Distinguished the base model from the inversion/editing method. |
| Controlled comparisons | Parameter sweeps on one fixed input | Preserving a scene can also preserve unwanted subject color. |
| Failure analysis | Inspected ghosting and granular artifacts | A parameter name is not enough; inspect its effect on the schedule and state. |

## Corrections worth keeping

- `eta` is a controller mixture weight, not a percentage of original pixels retained.
- `stop_timestep` stops the controller, not the entire generation process.
- The runner uses an empty source prompt.
- A 0.25 cutoff gives 3, 7, and 14 controlled updates for 14, 28, and 56 editing steps.
- Paper controller strength and API schedule strength are different parameters.

## Next entry

```text
Date / question:
Run or code section:
Observed result:
Working explanation:
What remains uncertain:
Next check:
```
