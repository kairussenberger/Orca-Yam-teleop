#!/bin/bash
# Launch the bimanual YAM-arms + ORCA-hands teleop sim with the right Python +
# package paths. Usage: ./run_yam.sh [sim_yam_bimanual.py args]
#   ./run_yam.sh --demo        # hands sweep, arms hold home (no camera)
#   ./run_yam.sh               # webcam -> both hands, arms hold home
#   ./run_yam.sh --swap-hands  # if hands drive the wrong sides
cd "$(dirname "$0")"
export PYTHONPATH="$HOME/Desktop/orca_sim/src:$HOME/Desktop/orca_core:$PYTHONPATH"
exec .venv/bin/mjpython sim_yam_bimanual.py "$@"
