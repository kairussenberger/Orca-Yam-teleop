#!/bin/bash
# Full-pose VR teleop driven by the ORBIT Quest app over ZeroMQ.
# The Quest runs com.ORBIT.Teleoperation and PUSHes wrist pose to 127.0.0.1:8122/8123;
# adb-reverse forwards those to this Mac, where we PULL them, delta-anchor to a
# calibrated neutral, and solve full 6-DoF IK for both YAM arms.
#
# PREREQS on the headset side (see the hardware playbook):
#   adb install -r .../quest_app_v8_reprojection.apk   # once
#   adb shell pm clear com.ORBIT.Teleoperation          # clear any saved IP
#   adb shell am start -n com.ORBIT.Teleoperation/com.unity3d.player.UnityPlayerGameActivity \
#       --es orbit.network.ipAddress 127.0.0.1
#   adb reverse tcp:8122 tcp:8122 && adb reverse tcp:8123 tcp:8123   # re-run after each replug
#
# Usage:
#   ./run_yam_orbit.sh                 # defaults
#   ./run_yam_orbit.sh --vr-debug      # print live IK residuals
#   ./run_yam_orbit.sh --orbit-flip y  # chirality knob if motion is mirrored
#   ./run_yam_orbit.sh --orbit-scale 1.5   # amplify reach
cd "$(dirname "$0")"
export PYTHONPATH="$HOME/Desktop/orca_sim/src:$HOME/Desktop/orca_core:$PYTHONPATH"
exec .venv/bin/mjpython sim_yam_bimanual.py --orbit "$@"
