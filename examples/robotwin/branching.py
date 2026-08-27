"""Simulator-branching utilities for the RoboTwin cup-nesting pilot.

The RECAP-recipe consults (professor + Codex, 2026-08-01) converged on this: the
missing ingredient is *matched-state counterfactuals* — from the same pre-failure
state, run both the policy continuation (NEG) and a scripted-expert correction
(POS). That needs the SAPIEN scene saved and restored bit-for-bit at a chosen
moment, then replayed under two different controllers.

SAPIEN 3.0.1 (RoboTwin's engine) has no one-call scene serialization that also
captures velocities, so we snapshot explicitly:
  - robot articulation: qpos, qvel, qf, and per-active-joint PD drive targets
    (drive targets are load-bearing — omit them and the arm lurches to a stale
    setpoint the instant control resumes);
  - each free rigid body (the 3 cups): pose + linear/angular velocity.

Root pose is skipped: the arms load with fix_root_link=True, so the root is
static. This module is deliberately env-shape-agnostic — it discovers the robot
articulation and rigid actors from the live scene rather than hard-coding names,
so it survives the eventual VR_C2 embodiment swap.
"""

from __future__ import annotations

import numpy as np
import sapien
import sapien.physx as physx


def _rigid(entity):
    """Return an entity's PhysxRigidDynamicComponent, or None if it has none."""
    try:
        return entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
    except Exception:
        return None


def _robot_articulations(task):
    """The task's robot articulation handle(s), deduped. Stable attributes, so
    these references stay valid across the episode (unlike get_all_actors())."""
    robot = getattr(task, "robot", None)
    arts, seen = [], set()
    for attr in ("_entity", "left_entity", "right_entity"):
        art = getattr(robot, attr, None)
        if art is not None and id(art) not in seen:
            seen.add(id(art))
            arts.append(art)
    return arts


def snapshot_state(task) -> dict:
    """Capture full dynamic state of a RoboTwin task env for exact restore.

    Stores LIVE entity handles, not ids: ``scene.get_all_actors()`` returns fresh
    Python wrappers on every call (verified: id() overlap ~7/16) and the cups even
    share a name ('021_cup'), so neither id nor name is a usable cross-call key.
    Holding the handle keeps the underlying C++ entity addressable. This makes a
    snapshot valid only within the SAME live env instance (fine for in-process
    branching; disk persistence must serialise against a canonical reset ordering).
    """
    snap: dict = {"robot": [], "actors": []}

    for art in _robot_articulations(task):
        joints = art.get_active_joints()
        snap["robot"].append({
            "art": art,
            "qpos": np.asarray(art.get_qpos(), np.float64).copy(),
            "qvel": np.asarray(art.get_qvel(), np.float64).copy(),
            "qf": np.asarray(art.get_qf(), np.float64).copy(),
            "drive_targets": np.asarray(
                [j.get_drive_target() for j in joints], np.float64
            ).ravel().copy(),
        })

    for entity in task.scene.get_all_actors():
        rb = _rigid(entity)
        if rb is None or rb.kinematic:
            continue
        p = entity.get_pose()
        snap["actors"].append({
            "entity": entity,               # live handle, reused on restore
            "rb": rb,
            "pose_p": np.asarray(p.p, np.float64).copy(),
            "pose_q": np.asarray(p.q, np.float64).copy(),
            "lin_vel": np.asarray(rb.get_linear_velocity(), np.float64).copy(),
            "ang_vel": np.asarray(rb.get_angular_velocity(), np.float64).copy(),
        })
    return snap


def restore_state(task, snap: dict) -> None:
    """Restore a snapshot in place via the stored live handles. Does NOT step."""
    for rs in snap["robot"]:
        art = rs["art"]
        art.set_qpos(rs["qpos"])
        art.set_qvel(rs["qvel"])
        art.set_qf(rs["qf"])
        for j, t in zip(art.get_active_joints(), rs["drive_targets"]):
            j.set_drive_target(float(t))
    for a in snap["actors"]:
        a["entity"].set_pose(sapien.Pose(p=a["pose_p"], q=a["pose_q"]))
        a["rb"].set_linear_velocity(a["lin_vel"])
        a["rb"].set_angular_velocity(a["ang_vel"])


def state_diff(task, snap: dict) -> dict:
    """Max-abs deviation of the LIVE state from a snapshot, via stored handles."""
    d = {"robot_qpos": 0.0, "robot_qvel": 0.0, "actor_pos": 0.0, "actor_quat": 0.0}
    for rs in snap["robot"]:
        art = rs["art"]
        d["robot_qpos"] = max(d["robot_qpos"], float(np.abs(np.asarray(art.get_qpos()) - rs["qpos"]).max()))
        d["robot_qvel"] = max(d["robot_qvel"], float(np.abs(np.asarray(art.get_qvel()) - rs["qvel"]).max()))
    for a in snap["actors"]:
        p = a["entity"].get_pose()
        d["actor_pos"] = max(d["actor_pos"], float(np.abs(np.asarray(p.p) - a["pose_p"]).max()))
        d["actor_quat"] = max(d["actor_quat"], float(np.abs(np.asarray(p.q) - a["pose_q"]).max()))
    return d
