#!/usr/bin/env python3
"""Procedural blocky 3D Shorts renderer ("Huss" format) for Blender 3.6 - 4.x.

Usage: blender -b -t 2 --python blender_scene.py -- beats.json output.mp4

Environment:
  BLENDER_ENGINE   CYCLES (default, works headless on CPU) or EEVEE
  BLENDER_SAMPLES  Cycles samples (default 24)
"""
import bpy
import sys
import json
import math
import os
import random
import subprocess
import tempfile
import shutil

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
if len(args) < 2:
    raise SystemExit('Expected beats.json and output.mp4')
json_path, output_path = args[0], args[1]
data = json.load(open(json_path, encoding='utf-8'))
beats, durs = data['beats'], data['durations']

FPS = int(data.get('fps') or 18)
W, H = 540, 960
PERIOD = 0.42                      # seconds per "huss" repetition (matches pipeline.py)
P = max(2, round(PERIOD * FPS))    # frames per repetition
ENGINE = os.environ.get('BLENDER_ENGINE', 'CYCLES').upper()

scene = bpy.context.scene


# ---------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------

def setup_render():
    if ENGINE == 'CYCLES':
        scene.render.engine = 'CYCLES'
        # EEVEE needs an OpenGL context, which headless CI runners don't have.
        # Cycles on CPU works everywhere.
        scene.cycles.device = 'CPU'
        scene.cycles.samples = int(os.environ.get('BLENDER_SAMPLES', '24'))
        try:
            scene.cycles.use_denoising = True
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        except Exception:
            pass
    else:
        # 4.2+ uses BLENDER_EEVEE_NEXT, older versions use BLENDER_EEVEE.
        for name in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
            try:
                scene.render.engine = name
                break
            except TypeError:
                continue
        if hasattr(scene, 'eevee'):
            try:
                scene.eevee.taa_render_samples = 16
            except Exception:
                pass

    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    try:
        scene.view_settings.view_transform = 'Standard'
    except Exception:
        pass


setup_render()

# ---------------------------------------------------------------------
# Clear default scene, set up world
# ---------------------------------------------------------------------

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

world = scene.world or bpy.data.worlds.new('World')
scene.world = world
world.use_nodes = True
_nt = world.node_tree
BG = _nt.nodes.get('Background')
if BG is None:
    BG = _nt.nodes.new('ShaderNodeBackground')
    out = _nt.nodes.get('World Output') or _nt.nodes.new('ShaderNodeOutputWorld')
    _nt.links.new(BG.outputs['Background'], out.inputs['Surface'])
BG.inputs['Strength'].default_value = 0.9


def iter_fcurves(action):
    """fcurves for both the classic (<4.4) and the layered (4.4+) action API."""
    if action is None:
        return
    if hasattr(action, 'fcurves'):
        try:
            for fc in action.fcurves:
                yield fc
            return
        except Exception:
            pass
    for layer in getattr(action, 'layers', []):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', []):
                for fc in bag.fcurves:
                    yield fc


# ---------------------------------------------------------------------
# Materials and primitives
# ---------------------------------------------------------------------

def mat(name, color, rough=0.82, emission=0.0):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    bs = m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*color, 1)
    bs.inputs['Roughness'].default_value = rough
    if emission:
        # Input is "Emission" on Blender < 4.0 and "Emission Color" on 4.x.
        socket = bs.inputs.get('Emission Color') or bs.inputs.get('Emission')
        socket.default_value = (*color, 1)
        bs.inputs['Emission Strength'].default_value = emission
    return m


M = {
    'kevin': (mat('Kevin orange', (1.0, .24, .035)), mat('Kevin skin', (.94, .61, .39)),
              mat('Kevin hair', (.13, .065, .025)), mat('Kevin pants', (.08, .13, .38))),
    'pigeon': (mat('Pigeon grey', (.32, .34, .4)), mat('Pigeon wing', (.19, .21, .28)),
               mat('Pigeon beak', (1, .48, .03))),
}
WHITE = mat('Eye white', (1, 1, 1))
BLACK = mat('Pupil', (0.008, .008, .012))
FLOOR = mat('Ground', (.12, .14, .19))
ORANGE = mat('Ball', (.95, .42, .08))
RED = mat('Glove', (.85, .08, .1))
DARK = mat('Plate', (.07, .07, .09))
STEEL = mat('Bar', (.55, .56, .6))
WOOD = mat('Desk', (.45, .28, .15))
SCREEN = mat('Screen', (.2, .6, 1.0), emission=1.5)
ROPE = mat('Rope', (1, .9, .25))
WHITEFX = mat('Speed line', (1, 1, 1), emission=1.0)


def cube(name, loc, scale, material, bevel=0.035):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(material)
    if bevel:
        mod = o.modifiers.new('Soft block edges', 'BEVEL')
        mod.width = bevel
        mod.segments = 1
    return o


def uv(name, loc, scale, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    o.data.materials.append(material)
    return o


def key_obj(o, frame, loc=None, rot=None, scale=None):
    if loc is not None:
        o.location = loc
        o.keyframe_insert(data_path='location', frame=frame)
    if rot is not None:
        o.rotation_euler = rot
        o.keyframe_insert(data_path='rotation_euler', frame=frame)
    if scale is not None:
        o.scale = scale
        o.keyframe_insert(data_path='scale', frame=frame)


def cycle(o, start, end, dz=0.0, drot=None, loc_delta=None):
    """Repeat a two-pose loop (rest <-> pose) every PERIOD across the beat."""
    base_loc = tuple(o.location)
    base_rot = tuple(o.rotation_euler)
    key_obj(o, start, loc=base_loc, rot=base_rot)
    k = 1
    while True:
        f = start + round(k * P / 2)
        if f >= end:
            break
        up = (k % 2 == 1)
        loc = base_loc
        rot = base_rot
        if up:
            d = loc_delta or (0, 0, dz)
            loc = (base_loc[0] + d[0], base_loc[1] + d[1], base_loc[2] + d[2])
            if drot:
                rot = (base_rot[0] + drot[0], base_rot[1] + drot[1], base_rot[2] + drot[2])
        key_obj(o, f, loc=loc, rot=rot)
        k += 1
    key_obj(o, end, loc=base_loc, rot=base_rot)


# ---------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------

def add_pigeon(x, y, start, end):
    s = M['pigeon']
    body = uv('Pigeon body', (x, y, .55), (.52, .36, .42), s[0])
    wing = cube('Pigeon wing', (x - .12, y - .34, .62), (.55, .10, .28), s[1])
    head = uv('Pigeon head', (x + .38, y - .02, .98), (.25, .24, .25), s[0])
    beak = cube('Pigeon beak', (x + .62, y - .04, .95), (.22, .16, .11), s[2])
    eye = uv('Pigeon eye', (x + .46, y - .24, 1.05), (.055, .035, .055), WHITE)
    for o in (body, wing, head, beak, eye):
        cycle(o, start, end, dz=.05)


def add_character(x, start, end, action, emotion, activity):
    top, skin, hair, pants = M['kevin']
    huss = (action == 'huss')

    # Stack (feet on z=0): legs 0-.95, torso .975-2.125, head 2.16-2.94.
    torso = cube('torso', (x, 0, 1.55), (.92, .48, 1.15), top)
    head = cube('head', (x, -.015, 2.55), (.78, .58, .78), skin)
    hair_o = cube('hair', (x, 0, 3.0), (.82, .60, .18), hair)

    legs, arms, face = [], [], []
    for side in (-1, 1):
        legs.append(cube('leg', (x + side * .23, 0, .475), (.32, .38, .95), pants))
        arms.append(cube('arm', (x + side * .62, 0, 1.62), (.28, .34, 1.0), top))
        face.append(uv('eye', (x + side * .17, -.30, 2.62), (.09, .045, .11), WHITE))
        face.append(uv('pupil', (x + side * .17, -.335, 2.62), (.04, .025, .06), BLACK))

    mouth_h = .13 if emotion in ('shock', 'scared') else .045
    face.append(cube('mouth', (x, -.31, 2.38), (.19, .035, mouth_h), BLACK, 0))

    if emotion in ('angry', 'sad'):
        for side in (-1, 1):
            tilt = .25 if emotion == 'angry' else -.25
            brow = cube('brow', (x + side * .17, -.32, 2.80), (.20, .045, .055), hair, 0)
            brow.rotation_euler[1] = side * tilt
            face.append(brow)

    body_parts = [torso, head, hair_o] + face + arms
    all_parts = body_parts + legs

    if not huss:
        # Deadpan idle: tiny breathing bob only.
        for o in all_parts:
            cycle(o, start, end, dz=.03)
        return {'arms': arms, 'legs': legs}

    amp = {'jump_rope': .45, 'running': .12}.get(activity, .10)
    for o in body_parts:
        if o in arms:
            continue
        cycle(o, start, end, dz=amp)

    # Legs stay planted except for jumping and running.
    for i, leg in enumerate(legs):
        if activity == 'jump_rope':
            cycle(leg, start, end, dz=amp)
        elif activity == 'running':
            sign = 1 if i == 0 else -1
            cycle(leg, start, end, drot=(.7 * sign, 0, 0))
        else:
            cycle(leg, start, end)

    # Arms depend on the activity.
    for i, arm in enumerate(arms):
        sign = 1 if i == 0 else -1
        if activity == 'weights':
            cycle(arm, start, end, dz=amp + .35, drot=(.0, 0, .0))
        elif activity == 'running':
            cycle(arm, start, end, dz=amp, drot=(-.7 * sign, 0, 0))
        elif activity == 'basketball' and i == 1:
            cycle(arm, start, end, dz=amp, drot=(.7, 0, 0))
        elif activity == 'typing':
            cycle(arm, start, end, dz=amp, drot=(.25 * sign, 0, 0))
        elif activity == 'jump_rope':
            cycle(arm, start, end, dz=amp, drot=(0, 0, .35 * sign))
        else:
            cycle(arm, start, end, dz=amp)

    return {'arms': arms, 'legs': legs}


def add_activity(activity, x, start, end, animate):
    """Prop for the character's skill; moves in rhythm on huss beats."""
    if activity == 'basketball':
        ball = uv('Basketball', (x + .78, -.55, .32), (.3, .3, .3), ORANGE)
        if animate:
            key_obj(ball, start, loc=(x + .78, -.55, .32))
            k = 1
            while True:
                f = start + round(k * P / 2)
                if f >= end:
                    break
                key_obj(ball, f, loc=(x + .78, -.55, 1.65 if k % 2 else .32))
                k += 1
            key_obj(ball, end, loc=(x + .78, -.55, .32))

    elif activity == 'jump_rope':
        bpy.ops.mesh.primitive_torus_add(
            major_radius=1.15, minor_radius=.04, major_segments=24, minor_segments=6,
            location=(x, -.1, 1.55), rotation=(math.pi / 2, 0, 0))
        rope = bpy.context.object
        rope.name = 'Jump rope'
        rope.data.materials.append(ROPE)
        if animate:
            key_obj(rope, start, scale=(1, 1, 1))
            k = 1
            while True:
                f = start + round(k * P / 2)
                if f >= end:
                    break
                key_obj(rope, f, scale=(1, 1, .2) if k % 2 else (1, 1, 1))
                k += 1
            key_obj(rope, end, scale=(1, 1, 1))
        else:
            rope.scale = (1, 1, .35)

    elif activity == 'weights':
        z0, z1 = 2.25, 4.0
        parts = [
            cube('Bar', (x, -.5, z0), (2.4, .14, .14), STEEL),
            cube('Plate L', (x - 1.05, -.5, z0), (.2, .55, .55), DARK),
            cube('Plate R', (x + 1.05, -.5, z0), (.2, .55, .55), DARK),
        ]
        for o in parts:
            if animate:
                cycle(o, start, end, dz=z1 - z0)

    elif activity == 'boxing':
        for i, side in enumerate((-1, 1)):
            glove = uv('Glove', (x + side * .5, -.45, 1.9), (.26, .26, .26), RED)
            if animate:
                base = tuple(glove.location)
                key_obj(glove, start, loc=base)
                k = 1
                while True:
                    f = start + round(k * P / 2)
                    if f >= end:
                        break
                    punch = (k % 2 == 1) == (i == 0)
                    key_obj(glove, f, loc=(base[0] - side * .25, base[1] - (1.0 if punch else 0), base[2] + (.25 if punch else 0)))
                    k += 1
                key_obj(glove, end, loc=base)

    elif activity == 'running':
        if animate:
            for i in range(3):
                z = .5 + i * .7
                line = cube('Speed line', (x + 2.0, -.6, z), (.9, .03, .03), WHITEFX, 0)
                key_obj(line, start, loc=(x + 2.0, -.6, z))
                key_obj(line, end, loc=(x - 2.4 - i * .3, -.6, z))

    elif activity == 'typing':
        cube('Desk', (x, -.75, 1.35), (2.3, .75, .14), WOOD)
        cube('Laptop base', (x, -.7, 1.46), (1.3, .8, .06), STEEL)
        cube('Laptop screen', (x, -.28, 2.0), (1.2, .06, .85), SCREEN)


# ---------------------------------------------------------------------
# Camera and lighting
# ---------------------------------------------------------------------

CAM_Y, CAM_Z = -18.0, 3.4
bpy.ops.object.camera_add(location=(0, CAM_Y, CAM_Z))
cam = bpy.context.object
cam.rotation_euler = (math.radians(84), 0, 0)
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 7.0       # applies to the taller (vertical) side
cam.data.clip_start = 0.1
cam.data.clip_end = 80
scene.camera = cam

bpy.ops.object.light_add(type='AREA', location=(-5, -8, 10))
key = bpy.context.object
key.data.energy = 1500
key.data.shape = 'DISK'
key.data.size = 8
bpy.ops.object.light_add(type='AREA', location=(5, -6, 7))
fill = bpy.context.object
fill.data.energy = 900
fill.data.size = 7
# Point the lights at the stage.
for lamp in (key, fill):
    c = lamp.constraints.new('TRACK_TO')
    tgt = bpy.data.objects.new('light_target', None)
    scene.collection.objects.link(tgt)
    tgt.location = (0, 0, 1.5)
    c.target = tgt
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'

# ---------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------

SCENE_COLORS = {
    'classroom': (.24, .34, .48), 'bedroom': (.24, .28, .48), 'kitchen': (.52, .42, .29),
    'office': (.20, .34, .30), 'lab': (.025, .06, .14), 'stage': (.22, .025, .24),
    'street': (.12, .24, .39), 'city_night': (.018, .025, .12), 'park': (.10, .34, .16),
    'space': (.008, .008, .035),
}
PARTY_COLORS = [(1, .1, .6), (.1, .8, 1), (1, .85, .1), (.6, .3, 1), (.2, 1, .5)]


def add_party(rng, start, end):
    """Glowing colored blocks that pulse on the beat, like a party next door."""
    for j in range(7):
        col = rng.choice(PARTY_COLORS)
        x = rng.uniform(-1.8, 1.8)
        z = rng.uniform(2.6, 5.0)
        o = cube('Party light', (x, 1.2, z), (.45, .15, .45),
                 mat('PartyMat%d_%d' % (start, j), col, emission=5.0), 0)
        key_obj(o, start, scale=(1, 1, 1))
        k = 1
        while True:
            f = start + round(k * P / 2)
            if f >= end:
                break
            on = ((k + j) % 2 == 0)
            key_obj(o, f, scale=(1.3, 1.3, 1.3) if on else (.5, .5, .5))
            k += 1
        key_obj(o, end, scale=(1, 1, 1))

    # Dancing silhouettes on the horizon.
    sil = mat('Silhouette%d' % start, (.02, .015, .04))
    for j in range(5):
        x = -1.8 + j * .9
        body = cube('Dancer', (x, 1.0, .45), (.3, .2, .8), sil, 0)
        head = uv('Dancer head', (x, 1.0, 1.05), (.17, .15, .17), sil)
        for o in (body, head):
            cycle(o, start, end, dz=.18 if (j % 2) else .1)


def set_scene(beat, frame0, frame1, beat_i):
    name = beat.get('scene', 'city_night')
    color = SCENE_COLORS.get(name, SCENE_COLORS['city_night'])

    # World color per beat (constant-interpolated after the loop).
    BG.inputs['Color'].default_value = (*color, 1)
    BG.inputs['Color'].keyframe_insert('default_value', frame=frame0)

    # Big floor: orthographic camera sits low, so the floor must reach the lens.
    cube('Set floor', (0, -7, -.09), (40, 26, .18), FLOOR, .01)

    rng = random.Random(beat_i + 33)
    tall = name in ('street', 'city_night', 'space')
    for j in range(11):
        xx = -5.5 + j * 1.1
        h = rng.uniform(1.8, 5.2) if tall else rng.uniform(1.2, 3.0)
        cube('Backdrop', (xx, 2, h / 2), (1.0, .35, h),
             mat('BackdropMat%d_%d' % (beat_i, j),
                 tuple(min(1, c * rng.uniform(.7, 1.25)) for c in color)), .02)

    if beat.get('party', True):
        add_party(rng, frame0, frame1)

    chars = beat.get('chars') or [{'who': 'kevin', 'emotion': 'neutral', 'action': 'idle'}]
    main = chars[0]
    activity = beat.get('activity', 'basketball')
    animate = main.get('action') == 'huss'

    add_character(0, frame0, frame1, main.get('action', 'idle'),
                  main.get('emotion', 'neutral'), activity)
    add_activity(activity, 0, frame0, frame1, animate)

    if any(c.get('who') == 'pigeon' for c in chars):
        add_pigeon(1.3, -1.2, frame0, frame1)

    fx = beat.get('fx', 'none')
    if fx in ('explosion', 'shock', 'sparkle', 'fire'):
        for k in range(9):
            x = rng.uniform(-1.8, 1.8)
            z = rng.uniform(.6, 4.5)
            fire = fx in ('explosion', 'fire')
            obj = uv('FX ' + fx, (x, -.8, z), (.16, .16, .16),
                     mat('FXmat%d_%d' % (beat_i, k),
                         (1, .24, .025) if fire else (1, .85, .1), emission=3.0))
            key_obj(obj, frame0, scale=(.001, .001, .001))
            key_obj(obj, frame0 + max(2, (frame1 - frame0) // 5), scale=(.35, .35, .35))
            key_obj(obj, frame1, scale=(.001, .001, .001))


# ---------------------------------------------------------------------
# Build the whole timeline
# ---------------------------------------------------------------------

frame = 1
for i, (beat, dur) in enumerate(zip(beats, durs)):
    count = max(8, round(float(dur) * FPS))
    start = frame
    end = start + count - 1          # inclusive; next beat starts at start+count
    before = set(scene.objects)
    set_scene(beat, start, end, i)

    # Each beat's set and cast exist only during that beat.
    for obj in (set(scene.objects) - before):
        obj.hide_render = True
        obj.keyframe_insert(data_path='hide_render', frame=max(0, start - 1))
        obj.hide_render = False
        obj.keyframe_insert(data_path='hide_render', frame=start)
        obj.hide_render = False
        obj.keyframe_insert(data_path='hide_render', frame=end)
        obj.hide_render = True
        obj.keyframe_insert(data_path='hide_render', frame=end + 1)

    # Punch-in on every cut (orthographic camera, so zoom = ortho_scale).
    cam.data.ortho_scale = 7.4
    cam.data.keyframe_insert('ortho_scale', frame=start)
    cam.data.ortho_scale = 6.95
    cam.data.keyframe_insert('ortho_scale', frame=min(end, start + max(3, count // 3)))

    # Camera shake on big effects.
    cam.location = (0, CAM_Y, CAM_Z)
    cam.keyframe_insert('location', frame=start)
    if beat.get('fx') in ('explosion', 'shock'):
        rs = random.Random(i)
        for f in range(start + 1, min(end, start + 8), 2):
            cam.location = (rs.uniform(-.1, .1), CAM_Y, CAM_Z + rs.uniform(-.1, .1))
            cam.keyframe_insert('location', frame=f)
        cam.location = (0, CAM_Y, CAM_Z)
        cam.keyframe_insert('location', frame=min(end, start + 9))

    frame = start + count

scene.frame_start = 1
scene.frame_end = frame - 1

# Visibility and world-color changes must snap, not blend.
for obj in scene.objects:
    if obj.animation_data and obj.animation_data.action:
        for fc in iter_fcurves(obj.animation_data.action):
            if fc.data_path == 'hide_render':
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'
if world.node_tree.animation_data and world.node_tree.animation_data.action:
    for fc in iter_fcurves(world.node_tree.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'

# ---------------------------------------------------------------------
# Render PNG sequence, then encode with system FFmpeg
# ---------------------------------------------------------------------

os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
frame_dir = tempfile.mkdtemp(prefix='shorts_blender_frames_')
scene.render.filepath = os.path.join(frame_dir, 'frame_')
print('Rendering blocky 3D animation:', output_path, 'frames:', scene.frame_end,
      'engine:', scene.render.engine, flush=True)
try:
    bpy.ops.render.render(animation=True)
    command = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-framerate', str(FPS), '-start_number', '1',
        '-i', os.path.join(frame_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output_path,
    ]
    encoded = subprocess.run(command, capture_output=True, text=True)
    if encoded.returncode != 0:
        raise RuntimeError('FFmpeg encoding failed: ' + encoded.stderr[-5000:])
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
        raise RuntimeError('FFmpeg did not create a usable MP4.')
finally:
    shutil.rmtree(frame_dir, ignore_errors=True)
print('3D_RENDER_COMPLETE', flush=True)
