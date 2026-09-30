# Retained checkpoints

The current leg-length damping is `kd_l0=90.0 N*s/m`, selected in a frozen-model replay comparison after training model 3742. All checkpoints documented below, and the original model 3742 training run, used `kd_l0=20.0`. To reproduce their original controller behavior, add `env.kd_l0=20.0` to the replay commands below. Damping changes are controller changes, not changes to the checkpoint weights.

## Corrected VMC and height tracking (iteration 3742)

`corrected_vmc_height_model_3742.pt` is a retained intermediate model with the corrected angular Jacobian and a 75% nominal height-based leg reference. SHA-256: `8ea53fa55e2195410f639be5f49b0cc9b5261ebdf75c26a55d13e448ec1373b2`.

A continuous 105-second seed-43 replay tested 0, +0.4, +0.8, 0, -0.4, -0.8, and 0 m/s, each at 0.16/0.18/0.20 m body height. There were no terminations or timeouts. With damping increased to 90 N*s/m, the worst steady height MAE was 3.05 mm; forward endpoint speed was 0.756/0.772/0.786 m/s and reverse endpoint speed was -0.718/-0.723/-0.730 m/s. Forward leg-angle differences remain 8.56/7.35/6.00 degrees, so this is not the final controller.

At fixed checkpoint weights, increasing damping from 20 to 90 reduced the worst directional height overshoot after a command change from 22.95 to 5.25 mm. The longest time to enter and remain within 5 mm of the height command fell from 0.94 to 0.20 seconds. These are single-seed results; the initialization landing is excluded from those transition metrics.

```bash
./play_wheel.sh --device cuda:0 --num_envs 1 --seed 43 \
  --rendering_mode performance --live_stats --real-time \
  --checkpoint_path ../wheel_legged_isaaclab/checkpoints/corrected_vmc_height_model_3742.pt \
  --fixed_command 0.0 0.0 0.18 \
  --velocity_cycle 0.0 0.4 0.8 0.0 -0.4 -0.8 0.0 \
  --height_cycle 0.16 0.18 0.20 --max_steps 1750 \
  env.episode_length_s=180.0 env.kd_l0=90.0 \
  env.vmc_legacy_angular_mapping=False env.height_reference_blend=0.75 \
  env.height_feedback_gain=1.0 env.leg_length_height_offset=0.055 \
  env.forward_support_min_leg_length=0.255 env.forward_support_height_margin=0.055
```

This short visual demonstration is a different command sequence from the 105-second evaluation. The GUI panel displays simulation time, forward speed, body height, pitch/roll, both leg angles, and actual/target leg lengths. Use `--headless` and omit `--live_stats` for numerical evaluation. Avoid simultaneous long video recording during interactive playback; it adds image readback and frame-buffer memory. Raw reward totals depend on the current reward configuration and should not be compared across reward revisions.

## Height-feedback intermediate (iteration 3344)

`height_feedback_symmetry_model_3344.pt` uses the height-feedback controller before the VMC angular-mapping fix. It was trained with 12288 environments and seed 42, with a fixed learning rate of 0.0001 and stronger height, speed, and leg-symmetry penalties. SHA-256: `57c127806ea39906c333a961d2454e267768a42f91c55ee8b1cd03a4ae542648`.

Three uninterrupted seed-43 replays held speeds at 0, -0.8, and +0.8 m/s while cycling the commanded height through 0.18, 0.16, 0.20, and 0.18 m (five seconds per phase). All three completed without termination or timeout. In the final two seconds of each phase, stationary drift was 0.01-0.015 m/s; reverse leg-angle differences were 0.1-1.7 degrees. At a 0.20 m command and +0.8 m/s, body height reached 0.197 m. Remaining limitations are a 0.175 m stationary height at the 0.16 m command, forward speed of only 0.69-0.72 m/s, and forward leg-angle differences reaching 6.3 degrees in the last phase. This is an intermediate model under active improvement, not the completed controller.

```bash
./play_wheel.sh --headless --num_envs 1 --seed 43 \
  --checkpoint_path ../wheel_legged_isaaclab/checkpoints/height_feedback_symmetry_model_3344.pt \
  --fixed_command -0.8 0.0 0.18 --height_cycle 0.18 0.16 0.20 0.18 \
  --max_steps 1000 env.episode_length_s=120.0 \
  env.vmc_legacy_angular_mapping=True env.height_reference_blend=0.0 \
  env.forward_support_min_leg_length=0.25 env.forward_support_height_margin=0.05
```

Run this command from the repository root. Set the fixed speed to 0.0 or +0.8 to repeat the other cases. The training run is `IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat/2026-09-30_14-12-05_height_feedback_symmetry_v3_150/`. See [TRAINING_NOTES.md](TRAINING_NOTES.md) for the purpose and changes of each new training round. Add `--trace_csv /absolute/path/trace.csv` to retain diagnostics, then summarize them with `python3 wheel_legged_isaaclab/scripts/rsl_rl/summarize_trace.py /absolute/path/trace.csv`.

## Speed-slew baseline (iteration 3046)

`bidirectional_speed_slew_model_3046.pt` is the retained baseline for height-control improvements. It uses a 0.8 m/s² command acceleration limit and was trained with 12288 environments, seed 42. SHA-256: `ec9ea49a86eb995fd958e44429d805fa6a407a4ecc48fa48ab1142eb0f74ed6d`.

In five headless, 25-second replays (seeds 42-46), the command phases were `0, -0.4, -0.8, +0.4, +0.8 m/s`, each held for five seconds at a commanded height of 0.18 m. Averaged over the final two seconds of each phase, actual speeds were `-0.042, -0.408, -0.811, +0.338, +0.730 m/s`; body heights were `0.194, 0.190, 0.177, 0.182, 0.175 m`. The largest steady pitch or roll was about 1.4 degrees, and the largest mean left/right virtual-leg angle difference was about 4.1 degrees. The last 20 iterations of the 50-iteration training comparison ended by timeout 99.4% of the time, versus 60.8% without the speed ramp.

Correction to the original evaluation: those 25-second traces used 20-second episodes and reset immediately before the final speed phase. The previously reported 0.10 m height dips were reset transients, not evidence of a speed-transition failure. New traces explicitly record `terminated`, `time_out`, and `episode_step`; continuous evaluations use 120-second episodes. A seed-43 replay at 0.18 m with that longer episode had no resets and a minimum height of 0.166 m during the final speed phase.

The baseline still has real height-tracking errors: a continuous stationary height sequence of 0.16, 0.18, and 0.20 m produced approximately 0.188, 0.194, and 0.200 m. At +0.8 m/s and a 0.20 m height command, actual height was only 0.176 m. These are the reasons for changing the height controller and rewards. A 300-iteration continuation with the old settings increased leg-angle asymmetry and was not selected; its logs remain under `IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat/2026-09-30_02-44-56_speed_slew_refine_300/`.

From the repository root, replay this controller with:

```bash
./play_wheel.sh --headless --num_envs 1 --seed 42 \
  --checkpoint_path ../wheel_legged_isaaclab/checkpoints/bidirectional_speed_slew_model_3046.pt \
  --fixed_command 0.0 0.0 0.18 \
  --velocity_cycle 0.0 -0.4 -0.8 0.4 0.8 --max_steps 1250 \
  env.episode_length_s=120.0 env.height_feedback_gain=0.0 \
  env.vmc_legacy_angular_mapping=True env.height_reference_blend=0.0 \
  env.forward_support_min_leg_length=0.23 env.forward_support_height_margin=1.0
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
  --max_steps 500 env.height_feedback_gain=0.0 \
  env.vmc_legacy_angular_mapping=True env.height_reference_blend=0.0
```

The full training run and TensorBoard logs remain under the ignored local directory `IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat/2026-09-30_00-25-43_bidirectional_leg_support_stage2_500/`.
