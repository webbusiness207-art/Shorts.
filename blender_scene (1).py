
#!/usr/bin/env python3
"""Procedural blocky 3D Shorts renderer for Blender 3.6+ / 4.x.

Usage:
blender -b -t 2 --python blender_scene.py -- beats.json output.mp4
"""

import bpy
import sys
import json
import math
import os
import random
from pathlib import Path
from mathutils import Vector

# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

if len(args) < 2:
    raise SystemExit("Expected beats.json and output.mp4")

json_path = args[0]
output_path = args[1]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

beats = data["beats"]
durs = data["durations"]

FPS = int(data.get("fps", 24))
W = int(data.get("width", 1080))
H = int(data.get("height", 1920))

if not beats or not durs or len(beats) != len(durs):
    raise SystemExit("Invalid beats or durations in JSON.")

FPS = max(12, min(FPS, 60))
W = max(360, W)
H = max(640, H)

Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# RENDER SETTINGS
# ------------------------------------------------------------

scene = bpy.context.scene

if bpy.app.version >= (4, 0, 0):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
else:
    scene.render.engine = "BLENDER_EEVEE"

eevee = getattr(scene, "eevee", None)
if eevee is not None:
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 16

scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.fps = FPS

scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.render.image_settings.color_mode = "RGBA"

scene.world.color = (0.025, 0.025, 0.04)

scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1

# ------------------------------------------------------------
# CLEAR DEFAULT SCENE
# ------------------------------------------------------------

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# ------------------------------------------------------------
# MATERIALS
# ------------------------------------------------------------

def mat(name, color, rough=0.82, emission=0.0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1)
    material.use_nodes = True

    bs = material.node_tree.nodes.get("Principled BSDF")

    if bs:
        bs.inputs["Base Color"].default_value = (*color, 1)
        bs.inputs["Roughness"].default_value = rough

        emission_color = bs.inputs.get("Emission Color")
        if emission_color is None:
            emission_color = bs.inputs.get("Emission")

        if emission_color is not None:
            emission_color.default_value = (*color, 1)

        emission_strength = bs.inputs.get("Emission Strength")
        if emission_strength is not None:
            emission_strength.default_value = emission

    return material


M = {
    "kevin": (
        mat("Kevin orange", (1.0, .24, .035)),
        mat("Kevin skin", (.94, .61, .39)),
        mat("Kevin hair", (.13, .065, .025)),
        mat("Kevin pants", (.08, .13, .38)),
    ),
    "dave": (
        mat("Dave blue", (.06, .32, .92)),
        mat("Dave skin", (.72, .43, .28)),
        mat("Dave hair", (.025, .02, .02)),
        mat("Dave pants", (.10, .11, .15)),
    ),
    "cop": (
        mat("Uniform", (.04, .08, .3)),
        mat("Cop skin", (.8, .55, .37)),
        mat("Cop hair", (.08, .04, .02)),
        mat("Cop pants", (.025, .045, .15)),
    ),
    "boss": (
        mat("Boss jacket", (.22, .24, .29)),
        mat("Boss skin", (.68, .43, .29)),
        mat("Boss hair", (.42, .42, .45)),
        mat("Boss pants", (.10, .10, .12)),
    ),
    "kid": (
        mat("Kid purple", (.48, .12, .72)),
        mat("Kid skin", (.94, .69, .5)),
        mat("Kid hair", (.85, .58, .08)),
        mat("Kid pants", (.08, .09, .16)),
    ),
    "pigeon": (
        mat("Pigeon grey", (.32, .34, .4)),
        mat("Pigeon wing", (.19, .21, .28)),
        mat("Pigeon beak", (1, .48, .03)),
        mat("Pigeon feet", (.9, .25, .04)),
    ),
}

WHITE = mat("Eye white", (1, 1, 1))
BLACK = mat("Pupil", (.008, .008, .012))
FLOOR = mat("Ground", (.12, .14, .19))

# ------------------------------------------------------------
# OBJECT HELPERS
# ------------------------------------------------------------

def cube(name, loc, scale, material, bevel=0.035):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)

    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale

    bpy.ops.object.transform_apply(
        location=False,
        rotation=False,
        scale=True,
    )

    obj.data.materials.append(material)

    if bevel:
        mod = obj.modifiers.new("Soft block edges", "BEVEL")
        mod.width = bevel
        mod.segments = 1

        obj.modifiers.new(
            "Weighted corner normals",
            "WEIGHTED_NORMAL",
        )

    return obj


def uv(name, loc, scale, material):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=12,
        ring_count=8,
        location=loc,
    )

    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)

    return obj


def key_obj(obj, frame, loc=None, rot=None, scale=None):
    if loc is not None:
        obj.location = loc
        obj.keyframe_insert(
            data_path="location",
            frame=frame,
        )

    if rot is not None:
        obj.rotation_euler = rot
        obj.keyframe_insert(
            data_path="rotation_euler",
            frame=frame,
        )

    if scale is not None:
        obj.scale = scale
        obj.keyframe_insert(
            data_path="scale",
            frame=frame,
        )


# ------------------------------------------------------------
# CHARACTERS
# ------------------------------------------------------------

def add_character(who, x, start, end, action, emotion, idx):

    if who == "pigeon":
        body = uv(
            "Pigeon body",
            (x, 0.0, 1.05),
            (.52, .36, .42),
            M["pigeon"][0],
        )

        wing = cube(
            "Pigeon wing",
            (x - .12, -.34, 1.12),
            (.55, .10, .28),
            M["pigeon"][1],
        )

        head = uv(
            "Pigeon head",
            (x + .38, -.02, 1.48),
            (.25, .24, .25),
            M["pigeon"][0],
        )

        beak = cube(
            "Pigeon beak",
            (x + .62, -.04, 1.45),
            (.22, .16, .11),
            M["pigeon"][2],
        )

        eye = uv(
            "Pigeon eye",
            (x + .46, -.24, 1.55),
            (.055, .035, .055),
            WHITE,
        )

        for obj in (body, wing, head, beak, eye):
            base = obj.location.copy()

            key_obj(obj, start, loc=base)

            key_obj(
                obj,
                end,
                loc=(
                    base.x + .18,
                    base.y,
                    base.z + .08,
                ),
            )

        return

    top, skin, hair, pants = M.get(who, M["kevin"])

    torso = cube(
        who + " torso",
        (x, 0, 2.05),
        (.92, .48, 1.15),
        top,
    )

    head = cube(
        who + " head",
        (x, -.015, 2.92),
        (.78, .58, .78),
        skin,
    )

    hair_obj = cube(
        who + " hair",
        (x, .0, 3.31),
        (.82, .60, .18),
        hair,
    )

    legs = []
    arms = []

    for side in (-1, 1):

        legs.append(
            cube(
                who + " leg",
                (x + side * .23, 0, .88),
                (.32, .38, .95),
                pants,
            )
        )

        arms.append(
            cube(
                who + " arm",
                (x + side * .62, 0, 2.08),
                (.28, .34, 1.02),
                top,
            )
        )

        uv(
            who + " eye",
            (x + side * .17, -.315, 3.00),
            (.09, .045, .11),
            WHITE,
        )

        uv(
            who + " pupil",
            (x + side * .17, -.36, 3.0),
            (.04, .025, .06),
            BLACK,
        )

    mouth_height = (
        .13 if emotion in ("shock", "scared") else .045
    )

    mouth = cube(
        who + " mouth",
        (x, -.315, 2.76),
        (.19, .035, mouth_height),
        BLACK,
        0,
    )

    objs = [torso, head, hair_obj, mouth] + legs + arms

    if emotion == "angry":
        for side in (-1, 1):
            brow = cube(
                who + " brow",
                (x + side * .17, -.34, 3.14),
                (.20, .045, .055),
                hair,
                0,
            )
            brow.rotation_euler[1] = side * .25

    jump = (
        action in ("jump", "panic")
        or emotion in ("shock", "scared")
    )

    for obj in objs:

        base = obj.location.copy()
        rot = obj.rotation_euler.copy()

        key_obj(obj, start, loc=base, rot=rot)

        middle = start + max(1, (end - start) // 3)

        if obj in arms and jump:

            rot2 = rot.copy()

            rot2[1] = (
                -.9 if obj.location.x < x else .9
            )

            key_obj(
                obj,
                middle,
                loc=(base.x, base.y, base.z + .12),
                rot=rot2,
            )

            key_obj(obj, end, loc=base, rot=rot)

        elif obj in legs and action in ("jump", "panic"):

            leg_rot = rot.copy()
            leg_rot[0] = (
                .25 if obj.location.x < x else -.25
            )

            key_obj(
                obj,
                middle,
                loc=(base.x, base.y, base.z + .3),
                rot=leg_rot,
            )

            key_obj(obj, end, loc=base, rot=rot)

        else:

            bob = .12 if action in ("jump", "panic") else .035

            key_obj(
                obj,
                start + max(1, (end - start) // 2),
                loc=(base.x, base.y, base.z + bob),
                rot=rot,
            )

            key_obj(obj, end, loc=base, rot=rot)

    if who == "dave":
        for side in (-1, 1):
            bpy.ops.mesh.primitive_torus_add(
                major_radius=.105,
                minor_radius=.018,
                major_segments=12,
                minor_segments=6,
                location=(x + side * .17, -.37, 3.0),
                rotation=(math.pi / 2, 0, 0),
            )

            bpy.context.object.data.materials.append(BLACK)


# ------------------------------------------------------------
# CAMERA AND LIGHTING
# ------------------------------------------------------------

bpy.ops.object.camera_add(
    location=(0, -17, 6.1)
)

cam = bpy.context.object
cam.rotation_euler = (
    math.radians(72),
    0,
    0,
)

cam.data.type = "ORTHO"
cam.data.ortho_scale = 10.4
scene.camera = cam

bpy.ops.object.light_add(
    type="AREA",
    location=(-5, -8, 10),
)

key = bpy.context.object
key.data.energy = 1500
key.data.shape = "DISK"
key.data.size = 8

bpy.ops.object.light_add(
    type="AREA",
    location=(5, 2, 7),
)

fill = bpy.context.object
fill.data.energy = 900
fill.data.size = 7

# ------------------------------------------------------------
# ENVIRONMENTS
# ------------------------------------------------------------

SCENE_COLORS = {
    "classroom": (.24, .34, .48),
    "bedroom": (.24, .28, .48),
    "kitchen": (.52, .42, .29),
    "office": (.20, .34, .30),
    "lab": (.025, .06, .14),
    "stage": (.22, .025, .24),
    "street": (.12, .24, .39),
    "city_night": (.018, .025, .12),
    "park": (.10, .34, .16),
    "space": (.008, .008, .035),
}


def set_scene(beat, frame0, frame1, beat_i):

    name = beat.get("scene", "street")
    color = SCENE_COLORS.get(
        name,
        SCENE_COLORS["street"],
    )

    scene.world.color = color

    cube(
        "Set floor " + str(beat_i),
        (0, 1, -.05),
        (16, 8, .18),
        FLOOR,
        .01,
    )

    rng = random.Random(beat_i + 33)

    for j in range(7):

        xx = -7 + j * 2.3

        h = (
            rng.uniform(1.8, 5.2)
            if name in ("street", "city_night", "space")
            else rng.uniform(1.2, 3.0)
        )

        block_color = tuple(
            min(1, c * rng.uniform(.7, 1.25))
            for c in color
        )

        block_mat = mat(
            "BackdropMat" + str(beat_i) + "_" + str(j),
            block_color,
        )

        cube(
            "Backdrop " + str(beat_i) + " " + str(j),
            (xx, 2, h / 2),
            (1.6, .35, h),
            block_mat,
            .02,
        )

    prop = {
        "classroom": (0, 1, 1.5),
        "bedroom": (2, 1, 1.1),
        "kitchen": (-2, 1, 1.0),
        "office": (2, 1, 1.4),
        "lab": (0, 1, 1.2),
        "stage": (0, 1, 1.0),
        "street": (0, 1, 1.1),
        "city_night": (0, 1, 1.1),
        "park": (0, 1, 1.0),
        "space": (0, 1, 1.0),
    }.get(name)

    if prop:
        prop_color = (
            (.95, .35, .08)
            if name in ("lab", "kitchen")
            else (.12, .55, .78)
        )

        cube(
            "Set prop " + str(beat_i),
            prop,
            (2.2, .7, 1.0),
            mat("PropMat" + str(beat_i), prop_color),
        )

    chars = beat.get("chars") or [
        {
            "who": "kevin",
            "emotion": "shock",
            "action": "panic",
        }
    ]

    positions = {
        1: [0],
        2: [-1.65, 1.65],
        3: [-2.35, 0, 2.35],
    }.get(len(chars), [0])

    for char, x in zip(chars[:3], positions):

        add_character(
            char.get("who", "kevin"),
            x,
            frame0,
            frame1,
            char.get("action", "idle"),
            char.get("emotion", "neutral"),
            beat_i,
        )

    fx = beat.get("fx", "none")

    if fx in ("explosion", "shock", "sparkle", "fire"):

        for k in range(9):

            x = rng.uniform(-3.4, 3.4)
            z = rng.uniform(1.2, 5.0)

            fx_color = (
                (1, .24, .025)
                if fx in ("explosion", "fire")
                else (1, .85, .1)
            )

            obj = uv(
                "FX " + fx,
                (x, -.6, z),
                (.16, .16, .16),
                mat(
                    "FXmat" + str(beat_i) + "_" + str(k),
                    fx_color,
                    emission=.3,
                ),
            )

            key_obj(
                obj,
                frame0,
                scale=(.001, .001, .001),
            )

            key_obj(
                obj,
                frame0 + max(2, (frame1 - frame0) // 5),
                scale=(.35, .35, .35),
            )

            key_obj(
                obj,
                frame1,
                scale=(.001, .001, .001),
            )


# ------------------------------------------------------------
# ANIMATION
# ------------------------------------------------------------

frame = 1

for i, (beat, dur) in enumerate(zip(beats, durs)):

    count = max(12, round(float(dur) * FPS))

    start = frame
    end = frame + count

    before = set(scene.objects)

    set_scene(
        beat,
        start,
        end,
        i,
    )

    new_objects = set(scene.objects) - before

    for obj in new_objects:

        obj.hide_render = True
        obj.keyframe_insert(
            data_path="hide_render",
            frame=max(0, start - 1),
        )

        obj.hide_render = False
        obj.keyframe_insert(
            data_path="hide_render",
            frame=start,
        )

        obj.hide_render = False
        obj.keyframe_insert(
            data_path="hide_render",
            frame=end,
        )

        obj.hide_render = True
        obj.keyframe_insert(
            data_path="hide_render",
            frame=end + 1,
        )

    # Camera punch-in and subtle movement.

    cam.location = (
        random.uniform(-.12, .12),
        -17,
        6.1 + random.uniform(-.08, .08),
    )

    cam.keyframe_insert(
        data_path="location",
        frame=start,
    )

    cam.location = (
        random.uniform(-.08, .08),
        -16.1,
        6.05,
    )

    cam.keyframe_insert(
        data_path="location",
        frame=min(
            end,
            start + max(3, count // 3),
        ),
    )

    frame = end + 1


# ------------------------------------------------------------
# FINAL RENDER CONFIGURATION
# ------------------------------------------------------------

scene.frame_start = 1
scene.frame_end = frame - 1

scene.render.filepath = os.path.abspath(output_path)

scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.ffmpeg.ffmpeg_preset = "REALTIME"

scene.render.resolution_percentage = 100

# Smooth animation interpolation.

for obj in scene.objects:

    if obj.animation_data and obj.animation_data.action:

        for fc in obj.animation_data.action.fcurves:

            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"

print(
    "Rendering blocky 3D animation:",
    output_path,
    "frames:",
    scene.frame_end,
    "resolution:",
    W,
    "x",
    H,
    "FPS:",
    FPS,
)

bpy.ops.render.render(animation=True)

print("3D_RENDER_COMPLETE")
