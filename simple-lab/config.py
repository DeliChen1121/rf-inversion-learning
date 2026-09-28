# Edit this file, save it, then run: python -u run.py
# A sweep overrides the corresponding baseline value below.
# Use sweep = None for a single run. Existing results are never overwritten.
name = "01_eta_scan"
sweep = "eta"
values = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

# Keep these fixed when comparing controller settings.
prompt = "a ripe red tomato"
seed = 42
size = 512
num_inversion_steps = 28  # Forward inversion updates.
num_inference_steps = 28  # Editing updates before strength truncation.

# Editing controller: mixture weight and active interval.
eta = 0.9
start_timestep = 0.0
stop_timestep = 7 / 28  # A fraction of requested editing steps, not a step count.

# Inversion controller and scheduler settings.
gamma = 0.5
inversion_strength = 1.0
editing_strength = 1.0  # Truncates the schedule; this is not eta.
guidance_scale = 3.5
