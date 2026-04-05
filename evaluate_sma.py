import math
import mathutils


def _normalize_name(name):
    cleaned = []
    for char in str(name).strip().lower():
        if char.isalnum():
            cleaned.append(char)
    return "".join(cleaned)


class EvaluatedPose:
    def __init__(self, local_matrices, frame_index, next_frame_index, alpha):
        self.local_matrices = local_matrices
        self.frame_index = frame_index
        self.next_frame_index = next_frame_index
        self.alpha = alpha


class SmaRuntimeEvaluator:
    def __init__(self, clip, bind_local_matrices, model_scale=1.0):
        self.clip = clip
        self.bind_local_matrices = {
            bone_name: matrix.copy() for bone_name, matrix in bind_local_matrices.items()
        }
        self.model_scale = float(model_scale)
        self.bone_names = list(bind_local_matrices.keys())
        self.normalized_bone_names = {
            _normalize_name(bone_name): bone_name for bone_name in self.bone_names
        }
        self.frame_cache = {}

        self.channel_group_name, self.channels, self.missing_channels, self.channel_remap = self._select_channels()
        self.channel_lookup = {bone_name: channel for bone_name, channel in self.channel_remap.items()}
        self.matched_bone_names = list(self.channel_remap.keys())

    def _select_channels(self):
        channel_groups = [
            ("primary", self.clip.channels),
            ("extra", self.clip.extra_channels),
        ]
        best_name = "primary"
        best_matched = []
        best_missing = []
        bone_name_set = set(self.bone_names)

        for group_name, channels in channel_groups:
            if not channels:
                continue

            matched = []
            missing = []
            remap = {}
            for channel in channels:
                if channel.name in bone_name_set:
                    matched.append(channel)
                    remap[channel.name] = channel
                    continue

                normalized_channel_name = _normalize_name(channel.name)
                bone_name = self.normalized_bone_names.get(normalized_channel_name)
                if bone_name is not None:
                    matched.append(channel)
                    remap[bone_name] = channel
                else:
                    missing.append(channel.name)

            if len(matched) > len(best_matched):
                best_name = group_name
                best_matched = matched
                best_missing = missing
                best_remap = remap

        if "best_remap" not in locals():
            best_remap = {}
        return best_name, best_matched, best_missing, best_remap

    def _frame_index(self, frame_position):
        if self.clip.frame_count <= 0:
            return 0
        return max(0, min(int(frame_position), self.clip.frame_count - 1))

    def _materialize_frame(self, frame_index):
        local_matrices = {
            bone_name: matrix.copy()
            for bone_name, matrix in self.bind_local_matrices.items()
        }

        for channel in self.channels:
            bone_name = None
            if channel.name in self.channel_remap:
                bone_name = channel.name
            else:
                normalized_channel_name = _normalize_name(channel.name)
                bone_name = self.normalized_bone_names.get(normalized_channel_name)

            if bone_name is not None and frame_index < len(channel.frames):
                channel_matrix = channel.frames[frame_index].matrix.copy()
                channel_matrix.translation = channel_matrix.translation * self.model_scale
                local_matrices[bone_name] = channel_matrix

        return local_matrices

    def _get_frame_pose(self, frame_index):
        frame_index = self._frame_index(frame_index)
        if frame_index not in self.frame_cache:
            self.frame_cache[frame_index] = self._materialize_frame(frame_index)
        return {
            bone_name: matrix.copy()
            for bone_name, matrix in self.frame_cache[frame_index].items()
        }

    def evaluate(self, frame_position):
        if self.clip.frame_count <= 0:
            return EvaluatedPose({}, 0, 0, 0.0)

        frame_position = max(0.0, min(float(frame_position), float(self.clip.frame_count - 1)))
        frame_index = self._frame_index(math.floor(frame_position))
        next_frame_index = self._frame_index(frame_index + 1)
        alpha = frame_position - float(frame_index)

        previous_pose = self._get_frame_pose(frame_index)
        if next_frame_index == frame_index or alpha <= 0.0:
            return EvaluatedPose(previous_pose, frame_index, next_frame_index, 0.0)

        next_pose = self._get_frame_pose(next_frame_index)
        blended_pose = {}
        for bone_name in self.bone_names:
            previous_matrix = previous_pose[bone_name]
            next_matrix = next_pose[bone_name]

            previous_loc, previous_rot, previous_scale = previous_matrix.decompose()
            next_loc, next_rot, next_scale = next_matrix.decompose()

            blended_loc = previous_loc.lerp(next_loc, alpha)
            blended_rot = previous_rot.slerp(next_rot, alpha)
            blended_scale = previous_scale.lerp(next_scale, alpha)

            blended_matrix = mathutils.Matrix.LocRotScale(
                blended_loc,
                blended_rot,
                blended_scale,
            )
            blended_pose[bone_name] = blended_matrix

        return EvaluatedPose(blended_pose, frame_index, next_frame_index, alpha)
