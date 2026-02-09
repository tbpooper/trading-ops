# SAFE MODE

If Thomas reports freezes/reboots, do NOT run long brute-force sweeps.

Allowed:
- short batches (<=25 configs)
- single-process only
- add `time.sleep(2-10)` cooldown between configs
- checkpoint after each config

Disallowed:
- infinite search loops
- large grid sweeps
- concurrent runs

Resume heavy compute only with explicit user approval.