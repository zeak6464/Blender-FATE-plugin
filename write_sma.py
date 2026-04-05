from .util import *
from . import evaluate_sma
from . import read_sma
import math
import mathutils
import os
import re


DEFAULT_SMA_VERSION = "V1.0"
DEFAULT_SMA_SCALE = 0.0
DEFAULT_SMA_UNKNOWN_HEADER = 30
DEFAULT_SMA_PLAYBACK_RAW = 50
SCALE_EPSILON = 1e-4
ACTION_FRAME_SOURCE = "ACTION"

_BONE_FCURVE_PATTERN = re.compile(r'^pose\.bones\["(.+?)"\]\.')


def _get_scene_fps(scene):
    fps_base = float(scene.render.fps_base)
    if fps_base == 0.0:
        return float(scene.render.fps)
    return float(scene.render.fps) / fps_base


def _get_target_armature(context):
    active_object = context.object
    if active_object is None:
        raise ValueError("Select an armature before exporting animation data.")
    if active_object.type == "ARMATURE":
        return active_object

    armature_object = active_object.find_armature()
    if armature_object is None:
        raise ValueError("Active object is not an armature and is not bound to one.")
    return armature_object


def _get_rest_local_matrix(pose_bone):
    stored_matrix = pose_bone.bone.get("fate_rest_local_matrix")
    if stored_matrix is not None and len(stored_matrix) == 16:
        matrix = mathutils.Matrix.Identity(4)
        for row in range(4):
            for col in range(4):
                matrix[row][col] = float(stored_matrix[(row * 4) + col])
        return matrix

    rest_world_matrix = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        return rest_world_matrix
    return pose_bone.parent.bone.matrix_local.inverted_safe() @ rest_world_matrix


def _get_blender_rest_local_matrix(pose_bone):
    rest_world_matrix = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        return rest_world_matrix
    return pose_bone.parent.bone.matrix_local.inverted_safe() @ rest_world_matrix


def _iter_action_fcurves(action, action_slot=None):
    if action is None:
        return []

    if hasattr(action, "fcurves"):
        return list(action.fcurves)

    fcurves = []
    if hasattr(action, "layers") and action_slot is not None:
        for layer in action.layers:
            if not hasattr(layer, "strips"):
                continue
            for strip in layer.strips:
                if not hasattr(strip, "channelbag"):
                    continue
                try:
                    channelbag = strip.channelbag(action_slot)
                except Exception:
                    continue
                if channelbag is None or not hasattr(channelbag, "fcurves"):
                    continue
                fcurves.extend(list(channelbag.fcurves))
    return fcurves


def _get_export_bone_names(armature_object, action, action_slot=None):
    normalized_bones = {
        evaluate_sma._normalize_name(pose_bone.name): pose_bone.name
        for pose_bone in armature_object.pose.bones
    }
    action_fcurves = _iter_action_fcurves(action, action_slot)
    animated_bones = set()
    for fcurve in action_fcurves:
        match = _BONE_FCURVE_PATTERN.match(fcurve.data_path)
        if match is None:
            continue

        candidate_name = match.group(1)
        if candidate_name in armature_object.pose.bones:
            animated_bones.add(candidate_name)
            continue

        normalized_name = evaluate_sma._normalize_name(candidate_name)
        resolved_name = normalized_bones.get(normalized_name)
        if resolved_name is not None:
            animated_bones.add(resolved_name)

    ordered_names = [
        pose_bone.name
        for pose_bone in armature_object.pose.bones
        if pose_bone.name in animated_bones
    ]
    if ordered_names:
        return ordered_names

    if action is None or action_fcurves:
        return []
    return [pose_bone.name for pose_bone in armature_object.pose.bones]


def _resolve_export_header(action):
    version = DEFAULT_SMA_VERSION
    scale = DEFAULT_SMA_SCALE
    unknown_header = DEFAULT_SMA_UNKNOWN_HEADER
    playback_raw = DEFAULT_SMA_PLAYBACK_RAW

    if action is None:
        return version, scale, unknown_header, playback_raw

    source_path = action.get("fate_source_path")
    if not source_path or not str(source_path).lower().endswith(".sma"):
        return version, scale, unknown_header, playback_raw
    if not os.path.exists(source_path):
        return version, scale, unknown_header, playback_raw

    try:
        clip = read_sma.read_sma_file(source_path)
    except Exception:
        return version, scale, unknown_header, playback_raw

    clip_version = clip.version.rstrip("\0")
    if clip_version:
        version = clip_version
    scale = float(clip.scale)
    unknown_header = int(clip.unknown_header)
    playback_raw = max(
        0,
        min(255, int(round(255.0 - (float(clip.playback_scale) * 255.0)))),
    )
    return version, scale, unknown_header, playback_raw


def _get_frame_range(scene, action, frame_source):
    if frame_source == ACTION_FRAME_SOURCE:
        if action is None:
            raise ValueError("SMA export requires an active action on the target armature.")
        start_frame, end_frame = action.frame_range
    else:
        start_frame = float(scene.frame_start)
        end_frame = float(scene.frame_end)

    start_frame = float(start_frame)
    end_frame = float(end_frame)
    if end_frame < start_frame:
        raise ValueError("Animation frame range is invalid.")
    return start_frame, end_frame


def _build_sample_frames(start_frame, end_frame, clip_fps, scene_fps):
    if clip_fps <= 0.0:
        raise ValueError("Clip FPS must be greater than zero.")
    if scene_fps <= 0.0:
        raise ValueError("Scene FPS must be greater than zero.")

    scene_step = float(scene_fps) / float(clip_fps)
    duration = max(0.0, float(end_frame) - float(start_frame))
    frame_count = max(
        1,
        int(math.floor((duration / scene_step) + 1e-6)) + 1,
    )
    return [float(start_frame) + (index * scene_step) for index in range(frame_count)]


def _sample_pose_matrix_in_sms_space(pose_bone):
    blender_rest_local = _get_blender_rest_local_matrix(pose_bone)
    sms_rest_local = _get_rest_local_matrix(pose_bone)
    local_matrix = (
        blender_rest_local
        @ pose_bone.matrix_basis.copy()
        @ blender_rest_local.inverted_safe()
        @ sms_rest_local
    )

    location, rotation, scale = local_matrix.decompose()
    scale_delta = max(abs(component - 1.0) for component in scale)
    if scale_delta > SCALE_EPSILON:
        raise ValueError(
            f"SMA export does not support bone scale animation ({pose_bone.name})."
        )

    clean_matrix = rotation.to_matrix().to_4x4()
    clean_matrix.translation = location
    return clean_matrix


def _quantize_rotation_component(value):
    clamped = clamp(float(value), -1.0, 1.0)
    return max(-32767, min(32767, int(round(clamped * 32767.0))))


def _write_primary_transform(wtr, local_matrix, model_scale):
    translation = local_matrix.translation.copy()
    translation /= float(model_scale)

    rotation_matrix = local_matrix.to_quaternion().to_matrix()
    source_rows = rotation_matrix.transposed()
    row0 = source_rows[0]
    row1 = source_rows[1]
    row2 = source_rows[2]

    wtr.write_num(translation[0], FLOAT)
    wtr.write_num(translation[1], FLOAT)
    wtr.write_num(translation[2], FLOAT)

    for value in row0:
        wtr.write_num(_quantize_rotation_component(-value), SINT16)
    for value in row2:
        wtr.write_num(_quantize_rotation_component(value), SINT16)
    for value in row1:
        wtr.write_num(_quantize_rotation_component(value), SINT16)


def write_sma_file(context, filepath, clip_fps=30.0, frame_source=ACTION_FRAME_SOURCE):
    armature_object = _get_target_armature(context)
    animation_data = armature_object.animation_data
    if animation_data is None or animation_data.action is None:
        raise ValueError("SMA export requires an active action on the target armature.")
    if animation_data.action.get("fate_baked_mesh"):
        raise ValueError("SMA export does not support baked mesh animations.")

    action = animation_data.action
    action_slot = getattr(animation_data, "action_slot", None)
    bone_names = _get_export_bone_names(armature_object, action, action_slot)
    if not bone_names:
        raise ValueError("SMA export found no animated bones on the active action.")

    scene = context.scene
    start_frame, end_frame = _get_frame_range(scene, action, frame_source)
    scene_fps = _get_scene_fps(scene)
    sample_frames = _build_sample_frames(start_frame, end_frame, float(clip_fps), scene_fps)
    model_scale = float(armature_object.data.get("fate_model_scale", 1.0))
    if abs(model_scale) <= 1e-6:
        model_scale = 1.0

    version, scale, unknown_header, playback_raw = _resolve_export_header(action)
    writer = Writer(context)

    writer.write_str(version.rstrip("\0"))
    writer.write_num(scale, FLOAT)
    writer.write_num(unknown_header, UINT32)
    writer.write_num(playback_raw, UINT32)
    writer.write_num(len(bone_names), UINT32)
    writer.write_num(len(sample_frames), UINT32)
    for bone_name in bone_names:
        writer.write_str(bone_name)

    original_frame = int(scene.frame_current)
    original_subframe = float(scene.frame_subframe)
    try:
        for scene_frame in sample_frames:
            frame_int = int(math.floor(scene_frame))
            subframe = float(scene_frame) - float(frame_int)
            scene.frame_set(frame_int, subframe=subframe)
            context.view_layer.update()

            for bone_name in bone_names:
                pose_bone = armature_object.pose.bones[bone_name]
                local_matrix = _sample_pose_matrix_in_sms_space(pose_bone)
                _write_primary_transform(writer, local_matrix, model_scale)

        for frame_index in range(len(sample_frames)):
            writer.write_num(frame_index, UINT32)

        writer.write_num(0, UINT32)
    finally:
        scene.frame_set(original_frame, subframe=original_subframe)
        context.view_layer.update()

    with open(filepath, "wb") as f:
        f.write(bytes(writer.txt_data))
