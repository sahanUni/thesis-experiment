"""Watch one dashboard episode in the MuJoCo viewer.

Runs as its own process. MuJoCo's viewer wants the main thread and would fight
the Dash server, so the dashboard launches this rather than embedding it.

The episode is re-simulated here rather than shipped over as data, which is safe
because the environment is deterministic given the same scenario -- the same
`scenarios.Scenario` always produces the same trace. The scenario is rebuilt
through `dashboard.InteractiveRunner.scenario_from_controls`, the same call the
charts use, so the window and the panel cannot drift apart. The metrics are
printed on startup so you can check them against the panel rather than taking
that on trust.

    python replay.py --controller ppo_direct_disturbed:11 --delay-ms 150 --noise-mm 200

Everything drawn is cosmetic and never touches physics:

  * the reference path, as flat markers;
  * the corridor edges, as two dimmer marker lines;
  * a ghost of the car where the *measured* cross-track error puts it.

The ghost is the point of this tool. Sensor noise corrupts only the e_ct the
controller reads, never the score, so on the charts it appears as a number that
is somehow catastrophic. In the window you watch the ghost tear away from the
car by a fifth of a corridor while the real car sits on the line, and the
controller steering at the ghost stops being an abstraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import config
from core import paths
from core.dynamics import fresh_model, yaw_of

MARKER_STRIDE = 10
MARKER_SIZE = (0.02, 0.02, 0.001)
PATH_RGBA = (0.55, 0.62, 0.75, 0.9)
CORRIDOR_RGBA = (0.55, 0.62, 0.75, 0.28)
# Matches the dashboard's DELAY_TINT and NOISE_TINT, so a run watched here and
# read on the charts is the same two colours in both places.
GHOST_RGBA = (1.0, 0.56, 0.64, 0.85)
GHOST_RADIUS = 0.05
GHOST_HEIGHT = 0.12
END_PAUSE_S = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        default="pid_global_robust",
        help="A dashboard controller key: pid_global_nominal, pid_global_robust, "
        "pid_custom, or <arm>:<seed> for a learned artifact.",
    )
    parser.add_argument("--path", dest="path_spec", default='{"kind": "arc"}')
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--mode", default="transient", choices=("stationary", "transient"))
    parser.add_argument("--condition", default="combined")
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--noise-mm", type=float, default=0.0)
    parser.add_argument("--event-time", type=float, default=6.0)
    parser.add_argument("--noise-seed", type=float, default=20260819.0)
    parser.add_argument("--gains", default=None, help="Kp,Ki,Kd for the pid_custom arm")
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--playback", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def simulate(args: argparse.Namespace):
    """Re-run the dashboard's episode, recording the pose at every control step.

    Imported here rather than at module scope because `dashboard` pulls in Dash
    and plotly, which this process does not otherwise need and which cost about
    a second of startup before the window can appear.
    """
    import dashboard

    runner = dashboard.InteractiveRunner(
        Path(args.models_root) if args.models_root else None, args.calibration
    )
    scenario = runner.scenario_from_controls(
        args.path_spec,
        args.speed,
        args.condition,
        args.mode,
        args.delay_ms,
        args.noise_mm,
        args.event_time,
        args.noise_seed,
    )
    gains = (
        tuple(float(part) for part in args.gains.split(","))
        if args.gains
        else (runner.calibration.nominal.kp, runner.calibration.nominal.ki,
              runner.calibration.nominal.kd)
    )

    # The frames have to come from inside the episode, so the runner's cache is
    # bypassed here: a cached result carries its metrics and trace but not the
    # pose history, and re-deriving a pose from the trace would lose the yaw.
    from calibration import PIDGains
    from env import PathFollowingEnv
    from rollout import run_episode

    poses: list[np.ndarray] = []
    if args.controller.startswith("pid_"):
        fixed = {
            "pid_global_nominal": runner.calibration.nominal,
            "pid_global_robust": runner.calibration.robust,
            "pid_custom": PIDGains.from_iterable(gains),
        }[args.controller]
        env = PathFollowingEnv(
            mode="fixed", training=False, calibration=runner.calibration,
            fixed_gains=fixed,
        )
        policy = None
        label = f"{args.controller}  Kp {fixed.kp:.2f} Ki {fixed.ki:.2f} Kd {fixed.kd:.2f}"
    else:
        entry = runner._model_entry(args.controller)
        env = PathFollowingEnv(
            mode=entry["mode"], training=False,
            calibration=(
                runner.scheduler_calibration if entry["mode"] == "scheduled"
                else runner.calibration
            ),
        )
        policy = runner._policy(entry)
        label = f"{dashboard.METHOD_LABELS[dashboard.method_of(args.controller)]} — {entry['label']}"

    def recording(observation: np.ndarray) -> np.ndarray:
        poses.append(env.data.qpos.copy())
        return np.zeros(env.action_space.shape, dtype=np.float32) if policy is None \
            else policy(observation)

    try:
        result = run_episode(env, scenario, recording)
        poses.append(env.data.qpos.copy())
    finally:
        env.close()
    return label, result, np.array(poses), scenario


def _flat_markers(viewer, points: np.ndarray, rgba, slot: int) -> int:
    for x, y in points:
        if slot >= viewer.user_scn.maxgeom - 1:
            break
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[slot],
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array(MARKER_SIZE),
            np.array([x, y, 0.002]),
            np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32),
        )
        slot += 1
    return slot


def draw_scene(viewer, built: dict) -> int:
    """Path and corridor edges. Returns the slot left free for the ghost."""
    points = built["pts"][::MARKER_STRIDE]
    slot = _flat_markers(viewer, points, PATH_RGBA, 0)

    # The corridor is what failure is defined against, so it is worth seeing.
    # Offset along the path normal, which is the tangent turned a quarter turn.
    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1) / np.maximum(norms, 1e-9)
    stride = max(1, len(points) // 120)
    for sign in (1.0, -1.0):
        edge = points[::stride] + sign * config.CONFIG.corridor_m * normals[::stride]
        slot = _flat_markers(viewer, edge, CORRIDOR_RGBA, slot)

    viewer.user_scn.ngeom = slot
    return slot


def draw_ghost(viewer, slot: int, pose: np.ndarray, offset: float) -> None:
    """Where the controller believes the car is, given the corrupted e_ct.

    Cosmetic, and approximate: the offset is applied along the car's own lateral
    axis rather than the path normal. The two agree except in a hard corner, and
    the alternative is re-deriving the nearest path point per frame for a marker
    whose only job is to show the size of the lie.
    """
    if abs(offset) < 1e-6:
        viewer.user_scn.ngeom = slot
        return
    yaw = yaw_of(pose[3:7])
    lateral = np.array([-np.sin(yaw), np.cos(yaw)])
    centre = pose[0:2] + lateral * offset
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[slot],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([GHOST_RADIUS, 0.0, 0.0]),
        np.array([centre[0], centre[1], GHOST_HEIGHT]),
        np.eye(3).flatten(),
        np.array(GHOST_RGBA, dtype=np.float32),
    )
    viewer.user_scn.ngeom = slot + 1


def main() -> None:
    args = parse_args()
    label, result, poses, scenario = simulate(args)
    metrics, trace = result.metrics, result.trace

    # The trace has one row per control step; `poses` has those plus a final
    # frame recorded after the loop. Pad so the two index together.
    offsets = [
        float(row.get("e_ct_measured", 0.0)) - float(row.get("e_ct", 0.0)) for row in trace
    ]
    offsets += [offsets[-1] if offsets else 0.0] * (len(poses) - len(offsets))

    events = ", ".join(
        f"{event.kind} {event.value:g} at {event.start_s:.1f} s" for event in scenario.events
    ) or "none"
    print(f"\n{label}")
    print(f"  scenario     : {scenario.evaluation_mode}/{scenario.condition} @ "
          f"{scenario.target_speed:g} m/s")
    print(f"  events       : {events}")
    print(f"  finished     : {metrics['finished']}  ({metrics['failure_reason'] or 'ok'})")
    print(f"  J_FA         : {metrics['failure_adjusted_error_m'] * 1000:.2f} mm")
    print(f"  mean distance: {metrics['mean_distance_m'] * 1000:.2f} mm")
    print(f"  max distance : {metrics['max_distance_m'] * 1000:.2f} mm")
    print(f"  wheel sat    : {100 * metrics['wheel_saturation_fraction']:.1f}% of steps")
    print(f"  steering TV  : {metrics['steer_total_variation_per_s']:.2f} /s")
    print(f"  duration     : {metrics['duration_s']:.2f} s over {len(poses)} frames")
    print("\nThe pink ghost is where the corrupted measurement puts the car.")
    print("Close the viewer window to exit.")

    model = fresh_model(config.NOMINAL_MASS_KG, config.NOMINAL_FRICTION, config.NOMINAL_ACTUATOR)
    data = mujoco.MjData(model)
    built = paths.build(scenario.path)
    points = built["pts"]
    frame_dt = config.CONFIG.frame_skip * float(model.opt.timestep)

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        ghost_slot = draw_scene(viewer, built)

        span = float(np.max(np.ptp(points, axis=0)))
        viewer.cam.lookat[:] = (*points.mean(axis=0), 0.0)
        viewer.cam.distance = max(2.5, span * 1.6)
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -75.0

        while viewer.is_running():
            start_wall = time.perf_counter()
            for index, pose in enumerate(poses):
                if not viewer.is_running():
                    return
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                draw_ghost(viewer, ghost_slot, pose, offsets[index])
                viewer.sync()
                target = start_wall + (index + 1) * frame_dt / max(args.playback, 0.05)
                lag = target - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
            if not args.loop:
                break
            time.sleep(END_PAUSE_S)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main())
