#!/bin/bash
# Phase-2 VR wrist teleop: Meta Quest hand-tracking wrist quaternion -> YAM arm
# wrist joints (j4/j5/j6), orientation only. Starts the MuJoCo viewer AND an
# in-process HTTPS/wss server the Quest Browser connects to over WiFi.
#
# Usage: ./run_yam_vr.sh [extra args]
#   ./run_yam_vr.sh                 # default port 8443, 3s calibration hold
#   ./run_yam_vr.sh --vr-port 9000  # different port
#   ./run_yam_vr.sh --vr-calib 5    # longer neutral-hold before calibrating
#
# Then in the Quest Browser open  https://<this-mac-ip>:8443/  (printed on start),
# accept the self-signed cert, tap "Enter teleop", and show both hands.
cd "$(dirname "$0")"
export PYTHONPATH="$HOME/Desktop/orca_sim/src:$HOME/Desktop/orca_core:$PYTHONPATH"
exec .venv/bin/mjpython sim_yam_bimanual.py --vr "$@"
