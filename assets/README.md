# Figure provenance

These PNGs are embedded figures extracted directly from the project owner's RF-Inversion observation report. They show existing AutoDL runs; no model inference or image synthesis was performed to prepare them for GitHub.

| File | Contents |
|---|---|
| `baseline.png` | Input and baseline edit |
| `eta.png` | Controller-weight sweep |
| `stop.png` | Controller-cutoff sweep |
| `window.png` | Controller-position sweep |
| `gamma.png` | Inversion-weight sweep |
| `editing-strength.png` | API schedule-strength sweep |
| `editing-steps.png` | Editing-update sweep |

The original tennis photo is linked in the [upstream example](https://github.com/LituRout/RF-Inversion#-diffusers-implementation); it is not claimed as original photography. Long cutoff labels in `stop.png` are retained from the original export and clarified in the experiment report.

Raw AutoDL configuration snapshots and per-run JSON records are not yet included. Do not treat the current example config as the exact configuration file for every historical figure.
