from .util import *
import mathutils
import os


ROTATION_SCALE = 1.0 / 32767.0


class Sma_Transform:
    def __init__(self):
        self.translation = mathutils.Vector((0.0, 0.0, 0.0))
        self.rotation = mathutils.Matrix.Identity(3)
        self.matrix = mathutils.Matrix.Identity(4)


class Sma_Channel:
    def __init__(self, name=""):
        self.name = name
        self.frames = []


class Sma_Clip:
    def __init__(self):
        self.source_path = ""
        self.version = ""
        self.scale = 1.0
        self.translation_scale = 1.0
        self.unknown_header = 0
        self.playback_scale = 1.0
        self.frame_count = 0
        self.channels = []
        self.frame_markers = []
        self.extra_channels = []
        self.trailing_bytes = 0


class Animation_Manifest_Entry:
    def __init__(self, name="", clip_path="", fps=30.0):
        self.name = name
        self.clip_path = clip_path
        self.fps = fps
        self.extra_fields = []


class Animation_Event:
    def __init__(self, animation_name="", frame=0.0, event_name=""):
        self.animation_name = animation_name
        self.frame = frame
        self.event_name = event_name


class Animation_Manifest:
    def __init__(self, source_path=""):
        self.source_path = source_path
        self.entries = []
        self.events = []

    def get_events(self, animation_name):
        return [event for event in self.events if event.animation_name == animation_name]


def _make_transform(translation, row0, row1, row2):
    transform = Sma_Transform()
    rotation = mathutils.Matrix((
        row0,
        row1,
        row2,
    ))
    transform.translation = translation
    transform.rotation = rotation.transposed()
    transform.matrix = transform.rotation.to_4x4()
    transform.matrix.translation = translation
    return transform


def _read_primary_transform(rdr, scale):
    translation = mathutils.Vector((
        rdr.read_num(FLOAT) * scale,
        rdr.read_num(FLOAT) * scale,
        rdr.read_num(FLOAT) * scale,
    ))
    row0 = (
        -rdr.read_num(SINT16) * ROTATION_SCALE,
        -rdr.read_num(SINT16) * ROTATION_SCALE,
        -rdr.read_num(SINT16) * ROTATION_SCALE,
    )
    row2 = (
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
    )
    row1 = (
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
    )
    return _make_transform(translation, row0, row1, row2)


def _read_secondary_transform(rdr, scale):
    translation = mathutils.Vector((
        rdr.read_num(FLOAT) * scale,
        rdr.read_num(FLOAT) * scale,
        rdr.read_num(FLOAT) * scale,
    ))
    row0 = (
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
    )
    row1 = (
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
    )
    row2 = (
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
        rdr.read_num(SINT16) * ROTATION_SCALE,
    )
    return _make_transform(translation, row0, row1, row2)


def read_sma_file(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    rdr = Reader(data)
    rdr.show_logs = False
    clip = Sma_Clip()
    clip.source_path = filepath

    try:
        clip.version = rdr.read_str()
        clip.scale = rdr.read_num(FLOAT)
        if abs(clip.scale) > 1e-6:
            clip.translation_scale = clip.scale
        else:
            clip.translation_scale = 1.0
        clip.unknown_header = rdr.read_num(UINT32)
        playback_raw = rdr.read_num(UINT32)
        clip.playback_scale = clamp((255.0 - float(playback_raw)) / 255.0, 0.1, 1.0)

        channel_count = rdr.read_num(UINT32)
        clip.frame_count = rdr.read_num(UINT32)

        for _ in range(channel_count):
            clip.channels.append(Sma_Channel(rdr.read_str()))

        for _frame_index in range(clip.frame_count):
            for channel in clip.channels:
                channel.frames.append(_read_primary_transform(rdr, clip.translation_scale))

        for _ in range(clip.frame_count):
            clip.frame_markers.append(rdr.read_num(UINT32))

        extra_channel_count = rdr.read_num(UINT32)
        for _ in range(extra_channel_count):
            clip.extra_channels.append(Sma_Channel(rdr.read_str()))

        for _frame_index in range(clip.frame_count):
            for channel in clip.extra_channels:
                channel.frames.append(_read_secondary_transform(rdr, clip.translation_scale))

        clip.trailing_bytes = len(data) - rdr.file_position
        return clip
    except Exception as exc:
        raise ValueError(
            f"Failed to parse SMA near byte offset {rdr.file_position}: {exc}"
        ) from exc


def parse_animation_manifest(filepath):
    manifest = Animation_Manifest(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in line.split(":")]
        if not parts:
            continue

        if parts[0] == "<KEY>":
            if len(parts) < 4:
                raise ValueError(f"Malformed animation key on line {line_number}.")
            event = Animation_Event(parts[1], float(parts[2]), parts[3])
            manifest.events.append(event)
            continue

        if len(parts) < 3:
            raise ValueError(f"Malformed animation entry on line {line_number}.")

        entry = Animation_Manifest_Entry(parts[0], parts[1], float(parts[-1]))
        entry.extra_fields = parts[2:-1]
        manifest.entries.append(entry)

    return manifest


def resolve_manifest_clip_path(manifest_path, clip_path):
    manifest_dir = os.path.dirname(manifest_path)
    normalized_path = clip_path.replace("\\", os.sep).replace("/", os.sep)
    drive, tail = os.path.splitdrive(normalized_path)
    if drive == "" and tail.startswith(os.sep):
        # Some game manifests prefix relative paths with a separator, e.g. "\..\ANIMS\walk.sma".
        normalized_path = tail.lstrip(os.sep)
    return os.path.normpath(os.path.join(manifest_dir, normalized_path))
