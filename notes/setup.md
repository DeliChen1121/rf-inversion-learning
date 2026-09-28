# Setup

Recorded environment: Python 3.12.3, PyTorch 2.8.0+cu128, Diffusers 0.35.1, A100 40GB. The runner uses BF16 and model CPU offload. This published English cleanup has not been rerun on a GPU; the figures are from earlier AutoDL runs.

## Reuse an existing AutoDL experiment

Assuming the original work is under `/root/autodl-tmp/rf-inversion-lab` and this repository is under `/root/autodl-tmp/rf-inversion-learning`, run once:

```bash
cd /root/autodl-tmp/rf-inversion-learning
ln -s /root/autodl-tmp/rf-inversion-lab/FLUX.1-dev FLUX.1-dev
cp /root/autodl-tmp/rf-inversion-lab/pipeline_flux_rf_inversion.py .
cp /root/autodl-tmp/rf-inversion-lab/input_tennis.jpg .
source /root/autodl-tmp/rf-inversion-lab/.venv/bin/activate
cd simple-lab
python run.py --check
python -u run.py
```

Adjust paths to your machine. If a destination already exists, inspect it rather than overwriting it. The model is linked, not duplicated.

## Fresh environment

1. Install a CUDA-compatible PyTorch build in a virtual environment. `requirements.txt` does not install a GPU driver or select a PyTorch CUDA build.
2. Run `python -m pip install -r requirements.txt` from the repository root.
3. Run `python scripts/prepare_pipeline.py` to fetch and SHA256-verify the pinned upstream source. It preserves the upstream copyright header. If GitHub is unreachable on the server, run this on another machine and transfer the file.
4. Obtain access to [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), download its Diffusers components, and place the directory or a symlink at `FLUX.1-dev/`.
5. Place the authors' [example photo](https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/tennis.jpg), or your own input, at `input_tennis.jpg`.
6. Run the check and generation commands above.

Recorded model revision: `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`.

`--check` validates configuration and required file presence; it does not load weights or validate GPU inference. Editing notes and reviewing results do not need a GPU.

Planning tests require only Python's standard library: `python -m unittest discover -s tests`.
