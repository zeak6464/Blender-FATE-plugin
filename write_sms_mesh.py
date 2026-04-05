from .util import *
import mathutils


def _expand_matrix(values):
    matrix = mathutils.Matrix.Identity(4)
    if values is None or len(values) != 16:
        return matrix
    for row in range(4):
        for col in range(4):
            matrix[row][col] = float(values[(row * 4) + col])
    return matrix


def _get_sms_bone_matrix(bone):
    stored_matrix = bone.get("fate_rest_local_matrix")
    if stored_matrix is not None:
        return _expand_matrix(stored_matrix)

    matrix = bone.matrix_local.copy()
    if bone.parent is not None:
        matrix = bone.parent.matrix_local.inverted_safe() @ matrix
    return matrix


def _get_sms_bone_world_matrix(bone, cache):
    if bone.name in cache:
        return cache[bone.name].copy()

    matrix = _get_sms_bone_matrix(bone)
    if bone.parent is not None:
        matrix = _get_sms_bone_world_matrix(bone.parent, cache) @ matrix

    cache[bone.name] = matrix.copy()
    return matrix


def _prepare_sms_export_data(wtr):
    active_object = wtr.context.active_object
    if active_object is None or active_object.type != "MESH":
        raise ValueError("Select a mesh object before exporting SMS.")

    active_mesh = active_object.to_mesh(preserve_all_data_layers=True)
    if active_mesh.uv_layers.active is None:
        raise ValueError("SMS export requires an active UV map.")

    if any(len(poly.vertices) != 3 for poly in active_mesh.polygons):
        raise ValueError("SMS export requires a triangulated mesh.")

    armature_obj = active_object.find_armature()
    if armature_obj is None:
        raise ValueError("SMS export requires the mesh to be parented to an armature.")

    wtr.mdl_data.active_object = active_object
    wtr.mdl_data.active_mesh = active_mesh
    wtr.mdl_data.armature_object = armature_obj
    wtr.mdl_data.model_scale = float(armature_obj.data.get("fate_model_scale", 1.0))
    wtr.mdl_data.vertices = list(active_mesh.vertices)
    wtr.mdl_data.triangles = list(active_mesh.polygons)
    wtr.mdl_data.materials = []
    wtr.mdl_data.textures = []

    seen_materials = set()
    for slot in active_object.material_slots.values():
        material = slot.material
        if material is None or material.name in seen_materials:
            continue
        seen_materials.add(material.name)
        wtr.mdl_data.materials.append(material)

    seen_textures = set()
    for material in wtr.mdl_data.materials:
        if material.node_tree is None:
            continue
        for node in material.node_tree.nodes.values():
            if node.bl_idname != "ShaderNodeTexImage":
                continue
            image_name = ""
            if getattr(node, "image", None) is not None:
                image_name = node.image.name
            texture_key = (material.name, image_name, node.name)
            if texture_key in seen_textures:
                continue
            seen_textures.add(texture_key)
            wtr.mdl_data.textures.append(node)


def write_basic_info(wtr):
    wtr.write_str("V2.0")
    _prepare_sms_export_data(wtr)
    if abs(wtr.mdl_data.model_scale) <= 1e-6:
        wtr.mdl_data.model_scale = 1.0

    wtr.write_num(wtr.mdl_data.model_scale, FLOAT) #model scale
    wtr.write_num(1, UINT32) #mesh count (for now only one object can be exported per file)

    wtr.write_num(len(wtr.mdl_data.vertices), UINT32) #vertex count
    # Reverse-engineering shows SMS can contain additional versioned sections.
    # This exporter currently emits only the base mesh block, so tags stay empty.
    wtr.write_num(0, UINT32)
    wtr.write_num(len(wtr.mdl_data.materials), UINT32) #materials count
    wtr.write_num(len(wtr.mdl_data.textures), UINT32) #textures count

def write_mesh_section(wtr):
    for i in range(1): # change later to be the number of objects
        mesh_offset = len(wtr.txt_data)
        mesh_object = wtr.mdl_data.active_mesh
        mesh_owner = wtr.mdl_data.active_object
        vertices = list(mesh_object.vertices)
        faces = list(mesh_object.polygons)
        mesh_name = mesh_object.name.ljust(68, '\0')
        wtr.write_str(mesh_name, False)
        wtr.write_num(len(wtr.mdl_data.textures), UINT32)
        wtr.write_num(len(vertices), UINT32)
        wtr.write_num(len(faces), UINT32) #triangle count
        ptr_header_tri_start = len(wtr.txt_data) # save this address so we can go back and write it later
        wtr.write_num(0, UINT32) #triangle section start addr (maybe unused?)
        wtr.write_num(100, UINT32) #header size, idk why we need this (always the same)
        ptr_header_uv_start = len(wtr.txt_data) # save this address so we can go back and write it later
        wtr.write_num(0, UINT32) #location of UV header
        ptr_header_vert_start = len(wtr.txt_data) # save this address so we can go back and write it later
        wtr.write_num(0, UINT32) #location of verts header
        ptr_mesh_size = len(wtr.txt_data) # save this address so we can go back and write it later
        wtr.write_num(0, UINT32) #size of mesh
        
        #start triangle list
        header_tri_start = len(wtr.txt_data)
        for j in faces:
            wtr.write_num(j.vertices[0])
            wtr.write_num(j.vertices[1])
            wtr.write_num(j.vertices[2])
            wtr.write_num(j.material_index, SINT32)
        
        #start uv list
        header_uv_start = len(wtr.txt_data)
        
        uv_loops = mesh_object.uv_layers.active.data.values()
        mesh_loops = mesh_object.loops
        uvs = {}
        for loop_index in range(len(uv_loops)):
            j = uv_loops[loop_index]
            vert_index = mesh_loops[loop_index].vertex_index
            if vert_index not in uvs:
                uvs[vert_index] = j.uv
        for vertex in vertices:
            uv = uvs.get(vertex.index, (0.0, 0.0))
            wtr.write_num(uv[0], FLOAT)
            wtr.write_num(1.0 - uv[1], FLOAT)
        
        #start vertex list
        header_vert_start = len(wtr.txt_data)
        
        real_verts = mesh_object.vertices.values()
        
        armature_object = wtr.mdl_data.armature_object
        armature = armature_object.data
        bones = list(armature.bones)
        bones_dict = {v.name: v for v in bones}
        model_scale = float(wtr.mdl_data.model_scale)
        mesh_to_armature = armature_object.matrix_world.inverted_safe() @ mesh_owner.matrix_world
        bone_world_cache = {}
        
        bone_indices = {bone.name: index for index, bone in enumerate(bones)}
        vertex_groups = list(wtr.mdl_data.active_object.vertex_groups)
        group_bones = {}
        for j in vertex_groups:
            if j.name in bone_indices:
                group_bones[j.index] = j.name
        for j in vertices:
            vert = real_verts[j.index]
            vert_position = mesh_to_armature @ vert.co
            groups = []
            for k in vert.groups:
                bone_name = group_bones.get(k.group)
                if bone_name is None or k.weight <= 0.0:
                    continue
                groups.append((bones_dict[bone_name], bone_indices[bone_name], k.weight))
            wtr.write_num(len(groups)) #bone count
            for current_bone, bone_index, weight in groups:
                wtr.write_num(bone_index)
                bone_world = _get_sms_bone_world_matrix(current_bone, bone_world_cache)
                vert_rel = bone_world.inverted_safe() @ vert_position
                wtr.write_num(vert_rel[0] / model_scale, FLOAT)
                wtr.write_num(vert_rel[1] / model_scale, FLOAT)
                wtr.write_num(vert_rel[2] / model_scale, FLOAT)
                blender_normal = vert.normal.normalized() # the normal stored by blender
                au, av = compress_normal(blender_normal[0], blender_normal[1], blender_normal[2])
                wtr.write_num(au, BYTE)
                wtr.write_num(av, BYTE)
                wtr.write_num(weight, FLOAT)
        
        # skeleton time!
        
        wtr.write_num(len(bones))
        #now write the info
        for current_bone in bones:
            bone_name = current_bone.name
            bone_parent = -1
            bone_local = _get_sms_bone_matrix(current_bone)
            if current_bone.parent:
                bone_parent = bone_indices[current_bone.parent.name]
            wtr.write_str(bone_name)
            wtr.write_num(bone_parent, SINT32)
            bone_pos = bone_local.translation
            wtr.write_num(bone_pos[0] / model_scale, FLOAT)
            wtr.write_num(bone_pos[1] / model_scale, FLOAT)
            wtr.write_num(bone_pos[2] / model_scale, FLOAT)
            for matrix_row in bone_local.to_3x3().transposed():
                for matrix_col in matrix_row:
                    wtr.write_num(max(-32767, min(32767, round(matrix_col * 32767))), SINT16)
        
        mesh_end = len(wtr.txt_data)
        wtr.seek(ptr_header_tri_start)
        wtr.write_num(header_tri_start-mesh_offset)
        wtr.seek(ptr_header_uv_start)
        wtr.write_num(header_uv_start-mesh_offset)
        wtr.seek(ptr_header_vert_start)
        wtr.write_num(header_vert_start-mesh_offset)
        wtr.seek(ptr_mesh_size)
        wtr.write_num(mesh_end-mesh_offset)
        wtr.seek(mesh_end)