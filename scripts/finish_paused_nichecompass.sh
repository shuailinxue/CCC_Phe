#!/usr/bin/env bash
# One-off recovery of the already paused HealthyCancer worker; no job restart.
set -euo pipefail
target_pid=3058081
target_args=$(tr '\0' ' ' < "/proc/${target_pid}/cmdline")
case "$target_args" in
  *'run_notebook_baselines.py --action worker --dataset healthy --method NicheCompass'*) ;;
  *) echo 'Target PID does not match the paused HealthyCancer NicheCompass worker.' >&2; exit 1 ;;
esac
if [ "$(id -u)" -ne 0 ]; then
  echo 'Root is needed because this system denies attaching to the existing worker.' >&2
  exit 1
fi
/usr/bin/gdb -q -nx -batch -p "$target_pid" \
  -ex 'set scheduler-locking off' \
  -ex 'call (void*)dlopen("/home/nas3/biod/xueshuailin/CCC_Phe/workspace/artifacts/nichecompass_stop/request_stop.so", 2)' \
  -ex 'detach'
kill -CONT "$target_pid"
echo 'The stop callback is queued. Training will finish its current epoch, save and export normally.'
echo 'Check nichecompass/user_stop_installed.json or user_stop_error.json in the existing HealthyCancer baseline output.'
