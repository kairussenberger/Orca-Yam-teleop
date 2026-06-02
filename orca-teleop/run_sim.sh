#!/bin/bash
# Launch the ORCA sim teleop with the right Python + package paths.
# Usage: ./run_sim.sh [sim_teleop.py args]   e.g.  ./run_sim.sh --demo
cd "$(dirname "$0")"
export PYTHONPATH="$HOME/Desktop/orca_sim/src:$HOME/Desktop/orca_core:$PYTHONPATH"
exec .venv/bin/mjpython sim_teleop.py "$@"
