# Retained checkpoints

## Speed-slew candidate (iteration 3046)

`bidirectional_speed_slew_model_3046.pt` is the best validated intermediate controller as of 2026-09-30. It uses a 0.8 m/s² command acceleration limit and was trained with 12288 environments, seed 42. SHA-256: `ec9ea49a86eb995fd958e44429d805fa6a407a4ecc48fa48ab1142eb0f74ed6d`.

In five headless, 25-second replays (seeds 42-46), the command phases were `0, -0.4, -0.8, +0.4, +0.8 m/s`, each held for five seconds at a commanded height of 0.18 m. Averaged over the final two seconds of each phase, actual speeds were `-0.042, -0.408, -0.811, +0.338, +0.730 m/s`; body heights were `0.194, 0.190, 0.177, 0.182, 0.175 m`. The largest steady pitch or roll was about 1.4 degrees, and the largest mean left/right virtual-leg angle difference was about 4.1 degrees. The last 20 iterations of the 50-iteration training comparison ended by timeout 99.4% of the time, versus 60.8% without the speed ramp.

This is a promising intermediate model, not a complete validation. Two seeds briefly dipped below 0.12 m during the transition from reverse to +0.8 m/s, and height tracking at 0.16 m and 0.20 m remains imprecise. A 0.6 m/s² replay did not materially improve that dip, so the 0.8 m/s² setting was retained for continued training.

An additional 300 iterations were run from this checkpoint with its optimizer state preserved, producing checkpoints at iterations 3100, 3200, 3300, and 3345. The final 20 training iterations still timed out in 99.1% of episodes, but the same seed-43 replay showed a larger steady left/right leg-angle difference at +0.8 m/s: 6.0 degrees for iteration 3345 versus 0.2 degrees for iteration 3046. Its minimum height during that phase remained 0.103 m. Iteration 3046 is therefore retained as the preferred controller; further training with those settings was stopped. The continuation's logs and unselected checkpoints remain in the ignored local directory `IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat/2026-09-30_02-44-56_speed_slew_refine_300/`.

From the repository root, replay this controller with:

```bash
./play_wheel.sh --headless --num_envs 1 --seed 42 \
  --checkpoint_path ../wheel_legged_isaaclab/checkpoints/bidirectional_speed_slew_model_3046.pt \
  --fixed_command 0.0 0.0 0.18 \
  --velocity_cycle 0.0 -0.4 -0.8 0.4 0.8 --max_steps 1250
```

The full run remains in the ignored local directory `IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat/2026-09-30_02-30-12_speed_slew_termination_comparison_50/`.

## Earlier stage

`bidirectional_leg_support_stage2_model_2699.pt` is a research checkpoint, not a final controller. It was trained with 12288 environments and seed 42, continuing from stage-one `model_2200.pt`. SHA-256: `38c4885773a6f3ed1f7b004f12496687fe896426f037ef54a4f583a31a4ee769`.

In a seed-42, 0.18 m height replay, its standing segment averaged 0.173 m height, -0.001 m/s speed, 0.07° absolute pitch, 0.27° absolute roll, and 2.06° between the virtual-leg angles. Under a direct +0.8 m/s command it averaged 0.837 m/s but only 0.123 m body height, so it does not meet the high-speed stance requirement. Training's final timeout fraction was about 81%.

From the repository root, replay it with:

```bash
./play_wheel.sh --headless --num_envs 1 --seed 42 \
  --checkpoint_path ../wheel_legged_isaaclab/checkpoints/bidirectional_leg_support_stage2_model_2699.pt \
  --fixed_command 0.0 0.0 0.18 --velocity_cycle 0.0 0.5 \
  --max_steps 500
```

The full training run and TensorBoard logs remain under the ignored local directory `IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat/2026-09-30_00-25-43_bidirectional_leg_support_stage2_500/`.
