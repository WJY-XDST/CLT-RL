# Retained checkpoints

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
