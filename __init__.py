import bpy
from .util import *
from .read_material import *
from .write_material import *
from .read_mesh import *
from .write_mesh import *
from .read_sms_mesh import *
from .write_sms_mesh import *
from . import read_sma
from . import evaluate_sma
from . import write_sma
import bmesh
import os
import io
from contextlib import redirect_stdout, redirect_stderr

from bpy_extras.io_utils import ImportHelper
from bpy_extras.io_utils import ExportHelper
from bpy_extras.object_utils import AddObjectHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty
from bpy.types import Operator
import mathutils
from bpy_extras import object_utils
import math

bl_info = {
    "name": "FATE .mdl importer [2026-04-04-sync1]",
    "description": "Import and export FATE .mdl files (2026-04-04-sync1)",
    "author": "ElectricVersion",
    "blender": (2, 80, 0),
    "version": (0, 0, 2),
    "category": "Import-Export"
}

ADDON_BUILD_TAG = "2026-04-04-sync1"
BIND_MISMATCH_BAKE_THRESHOLD = 1.0

def register():
    bpy.utils.register_class(Import_MDL)
    bpy.utils.register_class(Import_SMS)
    bpy.utils.register_class(Import_SMA)
    bpy.utils.register_class(Import_Animation_DAT)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.utils.register_class(ExportMDL)
    bpy.utils.register_class(Export_SMA)
    bpy.utils.register_class(Export_SMS)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(Import_MDL)
    bpy.utils.unregister_class(Import_SMS)
    bpy.utils.unregister_class(Import_SMA)
    bpy.utils.unregister_class(Import_Animation_DAT)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(ExportMDL)
    bpy.utils.unregister_class(Export_SMA)
    bpy.utils.unregister_class(Export_SMS)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

def menu_func_import(self, context):
    self.layout.operator(Import_MDL.bl_idname, text="Import FATE mdl")
    self.layout.operator(Import_SMS.bl_idname, text="Import FATE sms")
    self.layout.operator(Import_SMA.bl_idname, text="Import FATE sma")
    self.layout.operator(Import_Animation_DAT.bl_idname, text="Import FATE animation.dat")

def menu_func_export(self, context):
    self.layout.operator(ExportMDL.bl_idname, text="Export FATE mdl")
    self.layout.operator(Export_SMA.bl_idname, text="Export FATE sma")
    self.layout.operator(Export_SMS.bl_idname, text="Export FATE sms")


def _get_target_armature(context):
    active_object = context.object
    if active_object is None:
        raise ValueError("Select an armature before importing animation data.")
    if active_object.type == "ARMATURE":
        return active_object

    armature_object = active_object.find_armature()
    if armature_object is None:
        raise ValueError("Active object is not an armature and is not bound to one.")
    return armature_object


def _get_scene_fps(context):
    fps_base = float(context.scene.render.fps_base)
    if fps_base == 0.0:
        return float(context.scene.render.fps)
    return float(context.scene.render.fps) / fps_base


def _extract_bone_axes(matrix):
    rotation = matrix.to_3x3()
    y_axis = rotation @ mathutils.Vector((0.0, 1.0, 0.0))
    z_axis = rotation @ mathutils.Vector((0.0, 0.0, 1.0))
    if y_axis.length < 1e-6:
        y_axis = mathutils.Vector((0.0, 1.0, 0.0))
    else:
        y_axis.normalize()
    if z_axis.length < 1e-6:
        z_axis = mathutils.Vector((0.0, 0.0, 1.0))
    else:
        z_axis.normalize()
    return y_axis, z_axis


def _flatten_matrix(matrix):
    return [float(matrix[row][col]) for row in range(4) for col in range(4)]


def _expand_matrix(values):
    matrix = mathutils.Matrix.Identity(4)
    if values is None or len(values) != 16:
        return matrix
    for row in range(4):
        for col in range(4):
            matrix[row][col] = float(values[(row * 4) + col])
    return matrix


def _store_rest_local_matrices(armature_object, rest_local_matrices):
    for bone_name, matrix in rest_local_matrices.items():
        if bone_name in armature_object.data.bones:
            armature_object.data.bones[bone_name]["fate_rest_local_matrix"] = _flatten_matrix(matrix)


def _get_armature_model_scale(armature_object):
    return float(armature_object.data.get("fate_model_scale", 1.0))


def _with_build_tag(message):
    return f"{message} [{ADDON_BUILD_TAG}]"


def _strip_nul(value):
    return str(value).rstrip("\0")


def _resolve_sms_texture_path(source_path, texture_name):
    cleaned_name = _strip_nul(texture_name).strip()
    if not cleaned_name:
        return None

    if os.path.isabs(cleaned_name) and os.path.exists(cleaned_name):
        return cleaned_name

    search_roots = [
        os.path.dirname(source_path),
        os.path.dirname(os.path.dirname(source_path)),
    ]
    for root in search_roots:
        candidate = os.path.normpath(os.path.join(root, cleaned_name))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(os.path.dirname(source_path), cleaned_name))


def _load_or_stub_sms_image(texture_path):
    if texture_path is None:
        return None

    image_name = os.path.basename(texture_path)
    if os.path.exists(texture_path):
        return bpy.data.images.load(texture_path, check_existing=True)

    image = bpy.data.images.get(image_name)
    if image is None:
        image = bpy.data.images.new(image_name, 1, 1)
    image.filepath = texture_path
    return image


def _build_sms_materials(mesh, filepath, mdl_data):
    material_slot_lookup = {}
    ordered_material_ids = sorted(mdl_data.material_defs.keys())
    for material_id in ordered_material_ids:
        material_info = mdl_data.material_defs[material_id]
        material_name = _strip_nul(material_info.get("name", f"material_{material_id}"))
        material = bpy.data.materials.new(material_name or f"material_{material_id}")
        material.use_nodes = True
        material["fate_material_id"] = int(material_id)

        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output_node = nodes.new("ShaderNodeOutputMaterial")
        output_node.location = (300, 0)
        bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf_node.location = (0, 0)
        links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

        diffuse_color = material_info.get("diffuse_color", (1.0, 1.0, 1.0))
        bsdf_node.inputs["Base Color"].default_value = (
            float(diffuse_color[0]),
            float(diffuse_color[1]),
            float(diffuse_color[2]),
            1.0,
        )

        texture_id = material_info.get("diffuse_map_id", -1)
        texture_name = mdl_data.texture_names.get(texture_id)
        resolved_texture_path = _resolve_sms_texture_path(filepath, texture_name) if texture_name else None
        image = _load_or_stub_sms_image(resolved_texture_path)
        if image is not None:
            texture_node = nodes.new("ShaderNodeTexImage")
            texture_node.location = (-320, 0)
            texture_node.image = image
            links.new(texture_node.outputs["Color"], bsdf_node.inputs["Base Color"])

        mesh.materials.append(material)
        material_slot_lookup[material_id] = len(mesh.materials) - 1

    return material_slot_lookup


HELPER_BONE_TRANSLATION_THRESHOLD = 10.0
HELPER_BONE_MAX_VERTEX_WEIGHT = 0.25


def _get_sms_helper_bone_indices(mdl_data):
    helper_bone_indices = set()
    for bone_index, bone in enumerate(mdl_data.bones):
        max_weight = 0.0
        usage_count = 0
        for vertex in mdl_data.vertices:
            if bone_index not in vertex.bones:
                continue
            usage_count += 1
            max_weight = max(max_weight, float(vertex.bone_weights[bone_index]))
        if (
            usage_count > 0
            and float(bone.pos.length) > HELPER_BONE_TRANSLATION_THRESHOLD
            and max_weight <= HELPER_BONE_MAX_VERTEX_WEIGHT
        ):
            helper_bone_indices.add(bone_index)
    return helper_bone_indices


def _get_sms_vertex_influences(mdl_data, vertex_index, helper_bone_indices=None):
    vertex = mdl_data.vertices[vertex_index]
    influences = [
        (bone_index, float(vertex.bone_weights[bone_index]))
        for bone_index in vertex.bones
    ]
    if not helper_bone_indices:
        return influences

    filtered_influences = [
        (bone_index, weight)
        for bone_index, weight in influences
        if bone_index not in helper_bone_indices
    ]
    if not filtered_influences:
        return influences

    total_weight = sum(weight for _bone_index, weight in filtered_influences)
    if total_weight <= 1e-6:
        return influences

    return [
        (bone_index, weight / total_weight)
        for bone_index, weight in filtered_influences
    ]


def _get_rest_local_matrix(pose_bone):
    stored_matrix = pose_bone.bone.get("fate_rest_local_matrix")
    if stored_matrix is not None:
        return _expand_matrix(stored_matrix)

    rest_world_matrix = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        return rest_world_matrix
    return pose_bone.parent.bone.matrix_local.inverted_safe() @ rest_world_matrix


def _get_blender_rest_local_matrix(pose_bone):
    rest_world_matrix = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        return rest_world_matrix
    return pose_bone.parent.bone.matrix_local.inverted_safe() @ rest_world_matrix


def _get_bind_local_matrices(armature_object):
    bind_local_matrices = {}
    for pose_bone in armature_object.pose.bones:
        bind_local_matrices[pose_bone.name] = _get_rest_local_matrix(pose_bone)
    return bind_local_matrices


def _get_bound_mesh_objects(armature_object):
    mesh_objects = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for modifier in obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == armature_object:
                mesh_objects.append(obj)
                break
    return mesh_objects


def _get_armature_bind_mismatch_max(armature_object):
    max_error = 0.0
    for pose_bone in armature_object.pose.bones:
        sms_rest_local = _get_rest_local_matrix(pose_bone)
        blender_rest_local = _get_blender_rest_local_matrix(pose_bone)
        error = 0.0
        for row in range(4):
            for col in range(4):
                delta = blender_rest_local[row][col] - sms_rest_local[row][col]
                error += delta * delta
        max_error = max(max_error, error)
    return max_error


def _should_use_baked_mesh_fallback(armature_object, bind_mismatch_max):
    if bind_mismatch_max <= BIND_MISMATCH_BAKE_THRESHOLD:
        return False
    if not armature_object.data.get("fate_source_path"):
        return False
    if not _get_bound_mesh_objects(armature_object):
        return False

    bone_names = {pose_bone.name for pose_bone in armature_object.pose.bones}
    lower_body_signature = {
        "malePlayer01 L Thigh",
        "malePlayer01 R Thigh",
        "malePlayer01 L Calf",
        "malePlayer01 R Calf",
    }
    return lower_body_signature.issubset(bone_names)


def _set_action_linear_interpolation(action, action_slot=None):
    if action is None:
        return

    fcurves = []
    if hasattr(action, "fcurves"):
        fcurves = list(action.fcurves)
    elif hasattr(action, "layers") and action_slot is not None:
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

    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'LINEAR'


def _read_sms_model(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    reader = Reader(data)
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        read_basic_info(reader)
        read_material_section(reader)
        read_mesh_section(reader)

    for bone in reader.mdl_data.bones:
        bone.name = bone.name.rstrip("\0")
    return reader.mdl_data


def _build_sms_world_matrices(sms_model, local_matrices):
    world_matrices = {}
    remaining = set(range(len(sms_model.bones)))
    while remaining:
        progressed = False
        for bone_index in list(remaining):
            bone = sms_model.bones[bone_index]
            local_matrix = local_matrices.get(bone.name)
            if local_matrix is None:
                continue
            if bone.parent is None or bone.parent < 0:
                world_matrices[bone_index] = local_matrix.copy()
                remaining.remove(bone_index)
                progressed = True
            elif bone.parent in world_matrices:
                world_matrices[bone_index] = world_matrices[bone.parent] @ local_matrix
                remaining.remove(bone_index)
                progressed = True
        if not progressed:
            missing = [sms_model.bones[index].name for index in sorted(remaining)]
            raise ValueError("Failed to resolve SMS bone world transforms for: " + ", ".join(missing))
    return world_matrices


def _compute_sms_deformed_vertices(sms_model, world_matrices):
    if not sms_model.bones:
        return []

    root_world = world_matrices.get(0)
    if root_world is None:
        root_world = next(iter(world_matrices.values()))

    vertices = []
    for vertex in sms_model.vertices:
        position = mathutils.Vector((0.0, 0.0, 0.0))
        if vertex.bones:
            for bone_index in vertex.bones:
                bone_offset = mathutils.Vector(vertex.bone_offsets[bone_index])
                position += (world_matrices[bone_index] @ bone_offset) * vertex.bone_weights[bone_index]
        else:
            position = root_world.translation.copy()
        vertices.append(position)
    return vertices


def _ensure_basis_shape_key(mesh_object):
    if mesh_object.data.shape_keys is None:
        mesh_object.shape_key_add(name="Basis", from_mix=False)


def _apply_baked_mesh_clip(context, armature_object, clip, action_name, clip_fps, runtime_evaluator, events=None):
    source_path = armature_object.data.get("fate_source_path")
    if not source_path:
        raise ValueError("Baked mesh fallback requires an SMS imported by this add-on.")

    mesh_objects = _get_bound_mesh_objects(armature_object)
    if not mesh_objects:
        raise ValueError("Baked mesh fallback requires a mesh bound to the target armature.")

    sms_model = _read_sms_model(source_path)
    sample_times = [_get_clip_sample_time(clip, frame_index) for frame_index in range(clip.frame_count)]
    scene_frames = [_clip_frame_to_scene_frame(context, sample_time, clip_fps) for sample_time in sample_times]

    created_actions = []
    for mesh_object in mesh_objects:
        _ensure_basis_shape_key(mesh_object)
        generated_shape_keys = []
        for frame_index in range(clip.frame_count):
            local_matrices = runtime_evaluator.evaluate(float(frame_index)).local_matrices
            world_matrices = _build_sms_world_matrices(sms_model, local_matrices)
            deformed_vertices = _compute_sms_deformed_vertices(sms_model, world_matrices)

            shape_key = mesh_object.shape_key_add(
                name=f"{action_name}__{frame_index:03d}",
                from_mix=False,
            )
            for vertex_index, position in enumerate(deformed_vertices):
                shape_key.data[vertex_index].co = position
            shape_key.value = 0.0
            generated_shape_keys.append(shape_key)

        key_data = mesh_object.data.shape_keys
        key_data.animation_data_create()
        action = bpy.data.actions.new(f"{action_name}_{mesh_object.name}")
        action.use_fake_user = True
        key_data.animation_data.action = action
        action["fate_source_path"] = clip.source_path
        action["fate_addon_build"] = ADDON_BUILD_TAG
        action["fate_clip_fps"] = float(clip_fps)
        action["fate_frame_count"] = int(clip.frame_count)
        action["fate_baked_mesh"] = True
        action["fate_source_mesh"] = mesh_object.name

        for current_index, scene_frame in enumerate(scene_frames):
            for shape_index, shape_key in enumerate(generated_shape_keys):
                shape_key.value = 1.0 if current_index == shape_index else 0.0
                shape_key.keyframe_insert(data_path="value", frame=scene_frame)

        for shape_key in generated_shape_keys:
            shape_key.value = 0.0

        for event in events or []:
            marker = action.pose_markers.new(event.event_name)
            marker.frame = round(_clip_frame_to_scene_frame(context, event.frame, clip_fps))

        _set_action_linear_interpolation(action, key_data.animation_data.action_slot)
        created_actions.append(action)

    armature_object.animation_data_create()
    armature_object.animation_data.action = None
    _reset_pose_bones(armature_object)
    context.view_layer.update()
    return created_actions[0]


def _build_sms_rest_armature(armature_ebones, bones, bone_order):
    for bone_index in bone_order:
        bone = bones[bone_index]
        edit_bone = armature_ebones[bone.name]
        world_matrix = bone.local_transform.copy()
        head = world_matrix.translation.copy()

        child_positions = [
            bones[child_index].local_transform.translation.copy()
            for child_index in bone.children
            if child_index is not None
        ]
        if child_positions:
            average_offset = mathutils.Vector((0.0, 0.0, 0.0))
            for child_head in child_positions:
                average_offset += (child_head - head)
            average_offset /= len(child_positions)
            length = max(average_offset.length * 0.6, 0.05)
        elif bone.parent is not None and bone.parent > -1:
            parent_head = bones[bone.parent].local_transform.translation.copy()
            parent_offset = head - parent_head
            length = max(parent_offset.length * 0.35, 0.05)
        else:
            length = 0.25

        # Use the SMS transform directly so Blender's rest basis matches the file.
        edit_bone.matrix = world_matrix
        edit_bone.length = max(length, 0.05)


def _clip_frame_to_scene_frame(context, clip_frame, clip_fps):
    if clip_fps <= 0.0:
        raise ValueError("Animation FPS must be greater than zero.")
    return 1.0 + (float(clip_frame) * (_get_scene_fps(context) / float(clip_fps)))


def _get_clip_sample_time(clip, frame_index):
    if clip is None or not getattr(clip, "frame_markers", None):
        return float(frame_index)
    if frame_index < 0 or frame_index >= len(clip.frame_markers):
        return float(frame_index)
    start_marker = float(clip.frame_markers[0])
    return float(clip.frame_markers[frame_index]) - start_marker


def _reset_pose_bones(armature_object):
    for pose_bone in armature_object.pose.bones:
        pose_bone.rotation_mode = 'QUATERNION'
        pose_bone.location = mathutils.Vector((0.0, 0.0, 0.0))
        pose_bone.rotation_quaternion = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        pose_bone.scale = mathutils.Vector((1.0, 1.0, 1.0))


def _apply_pose_channels(armature_object, local_matrices):
    for bone_name, local_matrix in local_matrices.items():
        pose_bone = armature_object.pose.bones.get(bone_name)
        if pose_bone is None:
            continue

        sms_rest_local = _get_rest_local_matrix(pose_bone)
        blender_rest_local = _get_blender_rest_local_matrix(pose_bone)
        pose_bone.matrix_basis = (
            blender_rest_local.inverted_safe()
            @ local_matrix
            @ sms_rest_local.inverted_safe()
            @ blender_rest_local
        )


def _apply_sma_clip(context, armature_object, clip, action_name, clip_fps, events=None):
    if clip.frame_count <= 0:
        raise ValueError("SMA clip has no animation frames.")

    bind_local_matrices = _get_bind_local_matrices(armature_object)
    runtime_evaluator = evaluate_sma.SmaRuntimeEvaluator(
        clip,
        bind_local_matrices,
        model_scale=_get_armature_model_scale(armature_object),
    )

    channel_group_name = runtime_evaluator.channel_group_name
    matched_channels = runtime_evaluator.channels
    missing_channels = runtime_evaluator.missing_channels
    if len(matched_channels) == 0:
        raise ValueError("No SMA channels matched bones on the selected armature.")

    armature_object.animation_data_create()
    action = bpy.data.actions.new(action_name)
    action.use_fake_user = True
    armature_object.animation_data.action = action
    action["fate_source_path"] = clip.source_path
    action["fate_addon_build"] = ADDON_BUILD_TAG
    action["fate_clip_fps"] = float(clip_fps)
    action["fate_frame_count"] = int(clip.frame_count)
    action["fate_channel_group"] = channel_group_name
    action["fate_matched_channels"] = int(len(matched_channels))
    action["fate_total_channels"] = int(len(matched_channels) + len(missing_channels))
    matched_bone_names = set(runtime_evaluator.matched_bone_names)
    animated_bone_count = sum(
        1 for pose_bone in armature_object.pose.bones if pose_bone.name in matched_bone_names
    )
    action["fate_animated_bones"] = int(animated_bone_count)
    action["fate_armature_bones"] = int(len(armature_object.pose.bones))
    if clip.trailing_bytes > 0:
        action["fate_unsupported_trailing_bytes"] = int(clip.trailing_bytes)

    bind_mismatch_max = _get_armature_bind_mismatch_max(armature_object)
    action["fate_bind_mismatch_max"] = float(bind_mismatch_max)
    if _should_use_baked_mesh_fallback(armature_object, bind_mismatch_max):
        baked_action = _apply_baked_mesh_clip(
            context,
            armature_object,
            clip,
            action_name,
            clip_fps,
            runtime_evaluator,
            events,
        )
        bpy.data.actions.remove(action)
        baked_action["fate_channel_group"] = channel_group_name
        baked_action["fate_matched_channels"] = int(len(matched_channels))
        baked_action["fate_total_channels"] = int(len(matched_channels) + len(missing_channels))
        baked_action["fate_animated_bones"] = int(animated_bone_count)
        baked_action["fate_armature_bones"] = int(len(armature_object.pose.bones))
        baked_action["fate_bind_mismatch_max"] = float(bind_mismatch_max)
        if clip.trailing_bytes > 0:
            baked_action["fate_unsupported_trailing_bytes"] = int(clip.trailing_bytes)
        return baked_action, matched_channels, missing_channels, animated_bone_count

    original_frame = context.scene.frame_current
    for frame_index in range(clip.frame_count):
        evaluated_pose = runtime_evaluator.evaluate(float(frame_index))
        local_matrices = evaluated_pose.local_matrices

        _reset_pose_bones(armature_object)
        _apply_pose_channels(armature_object, local_matrices)
        sample_time = _get_clip_sample_time(clip, frame_index)
        scene_frame = _clip_frame_to_scene_frame(context, sample_time, clip_fps)

        context.view_layer.update()
        for bone_name in matched_bone_names:
            if armature_object.pose.bones.get(bone_name) is None:
                continue
            pose_bone = armature_object.pose.bones[bone_name]
            pose_bone.keyframe_insert(
                data_path="location",
                frame=scene_frame,
            )
            pose_bone.keyframe_insert(
                data_path="rotation_quaternion",
                frame=scene_frame,
            )
            pose_bone.keyframe_insert(
                data_path="scale",
                frame=scene_frame,
            )

    for event in events or []:
        marker = action.pose_markers.new(event.event_name)
        marker.frame = round(_clip_frame_to_scene_frame(context, event.frame, clip_fps))

    _set_action_linear_interpolation(action, armature_object.animation_data.action_slot)
    _reset_pose_bones(armature_object)
    context.scene.frame_set(original_frame)
    context.view_layer.update()
    return action, matched_channels, missing_channels, animated_bone_count


def _import_animation_manifest(context, armature_object, manifest):
    imported_actions = []
    missing_files = []
    missing_channels = {}

    for entry in manifest.entries:
        clip_path = read_sma.resolve_manifest_clip_path(manifest.source_path, entry.clip_path)
        if not os.path.exists(clip_path):
            missing_files.append(clip_path)
            continue

        clip = read_sma.read_sma_file(clip_path)
        action, matched, unmatched, animated_bone_count = _apply_sma_clip(
            context,
            armature_object,
            clip,
            entry.name,
            entry.fps,
            manifest.get_events(entry.name),
        )
        imported_actions.append(action)
        if unmatched:
            missing_channels[action.name] = {
                "matched": len(matched),
                "missing": len(unmatched),
                "animated_bones": animated_bone_count,
                "armature_bones": len(armature_object.pose.bones),
                "names": unmatched,
            }

    if len(imported_actions) == 0:
        raise ValueError("No clips were imported from the manifest.")

    return imported_actions, missing_files, missing_channels


class ExportMDL(Operator, ExportHelper):
    """Import FATE model (.mdl)"""
    bl_idname = "export_mdl.mdl_data"
    bl_label = "Export FATE MDL"

    filename_ext = ".mdl"

    filter_glob: StringProperty(
        default="*.mdl",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    def execute(self, context):
        return self.write_mdl_file(context, self.filepath)
        
    def write_mdl_file(self, context, filepath):
        print("Saving MDL file.")
        #first we make a bytes object of the file contents
        writer = util.Writer(context)
        
        write_mesh.write_basic_info(writer)
        write_material.write_material_section(writer)
        write_mesh.write_mesh_section(writer)
        
        f = open(filepath, 'wb')
        f.write(bytes(writer.txt_data))
        f.close()
        
        return {'FINISHED'}

class Import_MDL(Operator, ImportHelper):
    """Import FATE model (.mdl)"""
    bl_idname = "import_mdl.mdl_data"
    bl_label = "Import FATE MDL"

    # ImportHelper mixin class uses this
    filename_ext = ".mdl"

    filter_glob: StringProperty(
        default="*.mdl",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    def execute(self, context):
        return self.read_mdl_file(context, self.filepath)
        
    def read_mdl_file(self, context, filepath):
        print("Loading MDL file.")
        f = open(filepath, 'rb')
        data = f.read()
        f.close()
        #load the data
        
        reader = util.Reader(data)
        read_mesh.read_basic_info(reader)
        read_material.read_material_section(reader)
        read_mesh.read_mesh_section(reader)
        
        mesh = bpy.data.meshes.new("Untitled FATE Object")
        
        bm = bmesh.new()
        
        print("VERTEX REFERENCES", len(reader.mdl_data.vertex_references))
        vertices = {}
        for i in range(len(reader.mdl_data.vertex_references)):
            v = reader.mdl_data.vertex_references[i]
            v_2 = reader.mdl_data.uv_references[i]
            v_3 = reader.mdl_data.normal_references[i]
            if v not in vertices:
                vertices[v] = Vertex_Data()
                vertices[v].pos = reader.mdl_data.vertices[v]
            vertices[v].uv_pos.append(reader.mdl_data.uvs[v_2])
            vertices[v].normals.append(mathutils.Vector(reader.mdl_data.normals[v_3]))
            vertices[v].references.append(i)
        
        print("VERTEX COUNT", len(vertices))
        for i in range(len(vertices)):
            vert = bm.verts.new(vertices[i].pos)
            vert.normal = vertices[i].normals[0]
            vert.normal_update()
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        
        for f in reader.mdl_data.triangles:
            face_verts = []
            for i in f:
                if bm.verts[i] not in face_verts:
                    face_verts.append(bm.verts[i])
            print(face_verts)
            if bm.faces.get(face_verts) == None and len(face_verts) == 3:
                face = bm.faces.new(face_verts)
                face.normal_update()
        
        uv_layer = bm.loops.layers.uv.verify()
        for f in bm.faces:
            for loop in f.loops:
                loop_uv = loop[uv_layer]
                current_vert = vertices[loop.vert.index]
                loop_uv.uv = current_vert.uv_pos.pop(0)
            #calculate the normal for the face from each vertex normal
            #face_verts = {}
            #for i, v in enumerate(f.verts):
            #   face_verts[i] = v
            
        bm.normal_update()
        bm.to_mesh(mesh)
        mesh.update()
        
        bm.free()
        # add the mesh as an object into the scene with this utility module
        from bpy_extras import object_utils
        object_utils.object_data_add(context, mesh)
        
        return {'FINISHED'}
        
class Import_SMS(Operator, ImportHelper):
    """Import FATE model (.sms)"""
    bl_idname = "import_sms.sms_data"
    bl_label = "Import FATE SMS"

    # ImportHelper mixin class uses this
    filename_ext = ".sms"

    filter_glob: StringProperty(
        default="*.sms",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    def execute(self, context):
        result = self.read_sms_file(context, self.filepath)
        if result == {'FINISHED'}:
            self.report(
                {'INFO'},
                _with_build_tag(f"Imported {os.path.basename(self.filepath)}."),
            )
        return result
        
    def read_sms_file(self, context, filepath):
        print("Loading SMS file.")
        f = open(filepath, 'rb')
        data = f.read()
        f.close()
        #load the data
        
        reader = util.Reader(data)
        read_sms_mesh.read_basic_info(reader)
        read_material.read_material_section(reader)
        read_sms_mesh.read_mesh_section(reader)
        
        armature = bpy.data.armatures.new("Untitled FATE Armature")
        object_utils.object_data_add(context, armature)
        override = bpy.context.copy()
        override["object"] = armature
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='EDIT')
        armature_ebones = bpy.context.object.data.edit_bones
        #armature_bones = bpy.context.object.data.bones
        
        for i in range(len(reader.mdl_data.bones)):
            v = reader.mdl_data.bones[i]
            armature_ebones.new(name=v.name)
            #armature_ebones[v.name].use_relative_parent = True
            #armature_ebones[v.name].use_inherit_rotation = True
            #bpy.context.object.data.edit_bones[v.name].use_local_location = True
        #print(len(list(armature_ebones)))

        #find the root bone
        root_bone = None
        for i in range(len(reader.mdl_data.bones)):
            v = reader.mdl_data.bones[i]
            if v.parent == -1:
                root_bone = i
            else:
                parent = reader.mdl_data.bones[v.parent]
                parent.children.append(i)
        print(root_bone)
        bones = reader.mdl_data.bones
        rest_local_matrices = {bone.name: bone.local_transform.copy() for bone in bones}
        bone_order = [root_bone]
        while len(bone_order) < len(bones):
            for i in range(len(bones)):
                v = bones[i]
                if i not in bone_order and v.parent in bone_order:
                    bone_order.append(i)
        print(bone_order)
        #for i in bone_order:
        #    v = bones[i]
        #    parent_tfm = mathutils.Matrix.Identity(4)
        #    if v.parent > -1:
        #        parent_tfm = reader.mdl_data.bones[v.parent].local_transform
        #        inverted_tfm = v.local_transform.inverted()
        #        inverted_tfm.translation = v.pos
        #        print("inverted_tfm", v.name)
        #        print(inverted_tfm)
        #        print("parent_tfm", v.name)
        #        print(parent_tfm)
        #        v.local_transform = parent_tfm @ inverted_tfm
        #        #print("TRANSFORMS")
        #        #print(i.transform)
        #        #print(parent.transform)
        #        #print(i.local_transform)
        #    else:
        #        #v.local_transform = mathutils.Matrix.Identity(4)
        #        v.local_transform.translation = v.pos
        #    print("FINAL TRANSFORM FOR", v.name)
        #    print(v.local_transform)
        #    v.local_pos = v.local_transform.to_translation()
        
        for i in bone_order:
            v = bones[i]
            if v.parent > -1:
                parent = bones[v.parent]
                parent_tfm = parent.local_transform.copy()
                v.local_transform = (parent_tfm @ v.local_transform).copy()

        _build_sms_rest_armature(armature_ebones, bones, bone_order)
        for i in range(len(reader.mdl_data.bones)):
            v = reader.mdl_data.bones[i]
            a_bone = armature_ebones[v.name]
            if v.parent > -1:
                a_bone.parent = armature_ebones[reader.mdl_data.bones[v.parent].name]
                a_bone.use_connect = False
            else:
                a_bone.parent = None
        for i in bone_order:
            v = bones[i]
            print(v.name, armature_ebones[v.name].head)
        #for i in bone_order:
        #    v = bones[i]
        #    a_bone = armature_ebones[v.name]
        #    if a_bone.length < 0.001:
        #        a_bone.tail = tfm_loc @ mathutils.Vector(v.pos.normalized())
        #for i in bone_order:
        #    v = bones[i]
        #    a_bone = armature_ebones[v.name]
        #    if len(a_bone.children) == 1:
        #        a_bone.tail = a_bone.children[0].head
        #    elif len(a_bone.children) > 1:
        #        average_pos = mathutils.Vector((0.0,0.0,0.0))
        #        for j in a_bone.children:
        #            average_pos = average_pos + j.head
        #        average_pos = average_pos / len(a_bone.children)
        #        a_bone.tail = average_pos
        #    else:
        #        a_bone.length = 0.5
        #for i in bone_order:
        #    v = bones[i]
        #    a_bone = armature_ebones[v.name]
        #    if a_bone.length < 0.001:
        #        a_bone.length = 0.1
        print(len(list(armature_ebones)))
        
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')
        armature_object = bpy.context.object
        armature_object.data["fate_model_scale"] = float(reader.mdl_data.model_scale)
        armature_object.data["fate_source_path"] = filepath
        _store_rest_local_matrices(armature_object, rest_local_matrices)
        for bone_index, bone in enumerate(reader.mdl_data.bones):
            if bone.name in armature_object.data.bones:
                armature_object.data.bones[bone.name]["fate_bone_index"] = int(bone_index)
        
        root_bone_data = None
        if root_bone is not None and 0 <= root_bone < len(bones):
            root_bone_data = bones[root_bone]
        
        
        bm = bmesh.new()
        source_index_layer = bm.verts.layers.int.new("fate_source_vertex_index")
        mesh = bpy.data.meshes.new("Untitled FATE Object")
        material_slot_lookup = _build_sms_materials(mesh, filepath, reader.mdl_data)
        helper_bone_indices = _get_sms_helper_bone_indices(reader.mdl_data)
        #print(reader.mdl_data.object_count)
        #print(len(reader.mdl_data.vertices))
        #print(len(reader.mdl_data.triangles))
        for i in range(len(reader.mdl_data.vertices)):
            #print("NEW VERTEX")
            vertex_influences = _get_sms_vertex_influences(
                reader.mdl_data,
                i,
                helper_bone_indices,
            )
            vertex_position = mathutils.Vector((0,0,0))
            for bone_index, bone_weight in vertex_influences:
                bone_tfm_local = reader.mdl_data.bones[bone_index].local_transform
                #print(bone_tfm_local)
                bone_offset =  mathutils.Vector(reader.mdl_data.vertices[i].bone_offsets[bone_index])
                #print("WEIGHT " + str(bone_weight))
                bone_offset = bone_tfm_local @ bone_offset
                vertex_position_new = bone_offset
                vertex_position_new = vertex_position_new * bone_weight
                #print(vertex_position_new)
                vertex_position = vertex_position + vertex_position_new
            if len(vertex_influences) == 0:
                if root_bone_data is not None:
                    vertex_position = root_bone_data.local_transform.translation.copy()
                else:
                    vertex_position = mathutils.Vector((0,0,0))
            vert = bm.verts.new( vertex_position )
            vert[source_index_layer] = int(i)
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        
        #print(reader.mdl_data.tag_user_data)
        
        for face_index, f in enumerate(reader.mdl_data.triangles):
            if bm.faces.get((bm.verts[f[0]], bm.verts[f[1]], bm.verts[f[2]])) == None:
                face = bm.faces.new((bm.verts[f[0]], bm.verts[f[1]], bm.verts[f[2]]))
                material_id = 0
                if face_index < len(reader.mdl_data.triangle_material_ids):
                    material_id = reader.mdl_data.triangle_material_ids[face_index]
                face.material_index = material_slot_lookup.get(material_id, 0)
        
        for i in range(len(reader.mdl_data.uvs)):
            uv = reader.mdl_data.uvs[i]
            reader.mdl_data.vertices[i].uv_pos.append(uv)
        
        uv_layer = bm.loops.layers.uv.verify()
        for f in bm.faces:
            for loop in f.loops:
                loop_uv = loop[uv_layer]
                current_vert = reader.mdl_data.vertices[loop.vert.index]
                loop_uv.uv = current_vert.uv_pos[0]
        
        bm.to_mesh(mesh)
        mesh.update()
        bm.free()
        source_index_attribute = mesh.attributes.get("fate_source_vertex_index")
        if source_index_attribute is None:
            source_index_attribute = mesh.attributes.new(
                name="fate_source_vertex_index",
                type='INT',
                domain='POINT',
            )
            for vertex_index in range(len(mesh.vertices)):
                source_index_attribute.data[vertex_index].value = int(vertex_index)
        
        object_utils.object_data_add(context, mesh)
        bpy.context.object["fate_source_path"] = filepath
        #override = bpy.context.copy()
        override["object"] = mesh
        v_groups = bpy.context.object.vertex_groups
        
        for bone in reader.mdl_data.bones:
            if v_groups.get(bone.name) == None:
                v_groups.new(name=bone.name)
        for v in range(len(mesh.vertices)):
            vertex_index = v
            vertex_influences = _get_sms_vertex_influences(
                reader.mdl_data,
                vertex_index,
                helper_bone_indices,
            )
            for bone_index, bone_weight in vertex_influences:
                bone = reader.mdl_data.bones[bone_index]
                v_groups[bone_index].add([vertex_index], bone_weight, "REPLACE")
        arm_mod = bpy.data.objects[mesh.name].modifiers.new("Armature", "ARMATURE") #add armature modifier to the mesh
        arm_mod.object = armature_object #set the modifier to use the newly created armature
        #print(reader.mdl_data.object_count)
        #print(reader.mdl_data.object_names)
        return {'FINISHED'}


class Import_SMA(Operator, ImportHelper):
    """Import FATE animation (.sma)"""
    bl_idname = "import_sma.sma_data"
    bl_label = "Import FATE SMA"

    filename_ext = ".sma"

    filter_glob: StringProperty(
        default="*.sma;*.SMA",
        options={'HIDDEN'},
        maxlen=255,
    )

    clip_fps: FloatProperty(
        name="Clip FPS",
        default=30.0,
        min=1.0,
        description="Playback rate used to place keyframes on the current scene timeline",
    )

    def execute(self, context):
        try:
            armature_object = _get_target_armature(context)
            clip = read_sma.read_sma_file(self.filepath)
            action_name = os.path.splitext(os.path.basename(self.filepath))[0]
            action, matched_channels, missing_channels, animated_bone_count = _apply_sma_clip(
                context,
                armature_object,
                clip,
                action_name,
                self.clip_fps,
            )
        except Exception as exc:
            self.report({'ERROR'}, _with_build_tag(f"SMA import failed: {exc}"))
            return {'CANCELLED'}

        if action.get("fate_baked_mesh"):
            self.report(
                {'INFO'},
                _with_build_tag(f"Imported {action.name} as baked mesh animation."),
            )
            return {'FINISHED'}

        if missing_channels:
            total_channels = len(matched_channels) + len(missing_channels)
            if animated_bone_count == len(armature_object.pose.bones):
                self.report(
                    {'INFO'},
                    _with_build_tag(
                        f"Imported {action.name}; animated all {animated_bone_count} bones on this split armature."
                    ),
                )
            elif len(matched_channels) > 0:
                self.report(
                    {'INFO'},
                    _with_build_tag(
                        f"Imported {action.name} using {len(matched_channels)}/{total_channels} channels."
                    ),
                )
            else:
                self.report(
                    {'WARNING'},
                    _with_build_tag(
                        f"Imported {action.name} with {len(missing_channels)} unmatched channels."
                    ),
                )
        elif clip.trailing_bytes > 0:
            self.report(
                {'WARNING'},
                _with_build_tag(
                    f"Imported {action.name}; {clip.trailing_bytes} trailing bytes remain undocumented."
                ),
            )
        else:
            self.report({'INFO'}, _with_build_tag(f"Imported {action.name}."))

        return {'FINISHED'}


class Import_Animation_DAT(Operator, ImportHelper):
    """Import FATE animation manifest (.dat)"""
    bl_idname = "import_animation_dat.animation_data"
    bl_label = "Import FATE animation.dat"

    filename_ext = ".dat"

    filter_glob: StringProperty(
        default="*.dat",
        options={'HIDDEN'},
        maxlen=255,
    )

    def execute(self, context):
        try:
            armature_object = _get_target_armature(context)
            manifest = read_sma.parse_animation_manifest(self.filepath)
            imported_actions, missing_files, missing_channels = _import_animation_manifest(
                context,
                armature_object,
                manifest,
            )
        except Exception as exc:
            self.report({'ERROR'}, _with_build_tag(f"Animation manifest import failed: {exc}"))
            return {'CANCELLED'}

        warning_bits = []
        if missing_files:
            warning_bits.append(f"{len(missing_files)} missing SMA files")
        if missing_channels:
            full_subset_actions = sum(
                1
                for info in missing_channels.values()
                if info["animated_bones"] == info["armature_bones"]
            )
            partial_actions = sum(
                1
                for info in missing_channels.values()
                if info["matched"] > 0 and info["animated_bones"] != info["armature_bones"]
            )
            if partial_actions:
                warning_bits.append(f"{partial_actions} partial-skeleton actions")
            elif full_subset_actions != len(missing_channels):
                warning_bits.append(f"{len(missing_channels)} actions with unmatched channels")
            else:
                warning_bits.append("extra channels were ignored for split armatures")

        if warning_bits:
            self.report(
                {'WARNING'},
                _with_build_tag(
                    f"Imported {len(imported_actions)} actions; " + ", ".join(warning_bits) + "."
                ),
            )
        else:
            self.report({'INFO'}, _with_build_tag(f"Imported {len(imported_actions)} actions."))

        return {'FINISHED'}
        

class Export_SMA(Operator, ExportHelper):
    """Export FATE animation (.sma)"""
    bl_idname = "export_sma.sma_data"
    bl_label = "Export FATE SMA"

    filename_ext = ".sma"

    filter_glob: StringProperty(
        default="*.sma",
        options={'HIDDEN'},
        maxlen=255,
    )

    clip_fps: FloatProperty(
        name="Clip FPS",
        default=30.0,
        min=1.0,
        description="Playback rate written into the exported SMA timing",
    )

    frame_source: EnumProperty(
        name="Frame Range",
        items=(
            ('ACTION', "Action", "Use the active action frame range"),
            ('SCENE', "Scene", "Use the current scene frame range"),
        ),
        default='ACTION',
    )

    def execute(self, context):
        try:
            write_sma.write_sma_file(
                context,
                self.filepath,
                clip_fps=self.clip_fps,
                frame_source=self.frame_source,
            )
        except Exception as exc:
            self.report({'ERROR'}, _with_build_tag(f"SMA export failed: {exc}"))
            return {'CANCELLED'}

        self.report({'INFO'}, _with_build_tag(f"Exported {os.path.basename(self.filepath)}."))
        return {'FINISHED'}


class Export_SMS(Operator, ExportHelper):
    """Import FATE model (.sms)"""
    bl_idname = "export_sms.sms_data"
    bl_label = "Export FATE SMS"

    filename_ext = ".sms"

    filter_glob: StringProperty(
        default="*.sms",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    def execute(self, context):
        return self.write_sms_file(context, self.filepath)
        
    def write_sms_file(self, context, filepath):
        print("Saving SMS file.")
        try:
            writer = util.Writer(context)
            
            write_sms_mesh.write_basic_info(writer)
            write_material.write_material_section(writer)
            write_sms_mesh.write_mesh_section(writer)
            
            f = open(filepath, 'wb')
            f.write(bytes(writer.txt_data))
            f.close()
        except Exception as exc:
            self.report({'ERROR'}, f"SMS export failed: {exc}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
