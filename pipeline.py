#!/usr/bin/env python3
"""Shorts Factory v5 - blocky 3D "story explainer" Shorts.

A narrator tells a short, funny "what happens when you..." story. Blocky
characters act it out in furnished 3D rooms while red/purple consequence tags
stack up, a trouble bar changes colour and key words are highlighted.
Everything is rendered in Python (Pillow + numpy): no Blender, no GPU.
"""

import asyncio  # noqa: F401  (kept so edge-tts works on every Python 3.x)
import datetime as dt
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
TODAY = dt.date.today().isoformat()
OUT_DAY = ROOT / "output" / TODAY
STATE_FILE = ROOT / "used_topics.json"
SERIES_NAME = os.getenv("SERIES_NAME", "DUMB MOVES")


def env_int(name, default):
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def env_float(name, default):
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


BRAINROT = os.getenv("BRAINROT", "0").strip() == "1"

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}

CFG = {
    "videos_per_run": max(1, min(5, env_int("VIDEOS_PER_RUN", 1))),
    "demo": os.getenv("DEMO", "0").strip() == "1",
    # Optional extra "edit" effects (split-screen runner, zoom punches, glitch
    # hits, meme sounds). OFF by default. Set the BRAINROT variable to 1 to try.
    "brainrot": BRAINROT,
    # "signs" = "When is X too much? Sign 1 of 6..." checklist videos with phone/
    # note close-ups. FORMAT=story brings back the old COOKED/chip story style.
    "format": os.getenv("FORMAT", "").strip().lower() or "signs",
    # Natural-sounding Microsoft neural voice (free, no key). Good male
    # options: en-US-AndrewNeural, en-US-BrianNeural, en-US-GuyNeural,
    # en-US-ChristopherNeural. Female: en-US-AriaNeural, en-US-JennyNeural.
    "voice": os.getenv("VOICE", "").strip() or "en-US-AndrewNeural",
    "voice_rate": os.getenv("VOICE_RATE", "").strip() or "+10%",
    "voice_pitch": os.getenv("VOICE_PITCH", "").strip() or "-2Hz",
    # Piper is only the fallback if the neural voice cannot be reached.
    "piper_bin": os.getenv("PIPER_BIN", "piper"),
    "piper_model": os.getenv(
        "PIPER_MODEL", str(ROOT / "models" / "en_US-lessac-medium.onnx")
    ),
    "gemini_model": os.getenv("GEMINI_MODEL", "").strip(),
    "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    "music_volume": max(0, min(1, env_float("MUSIC_VOLUME", 0.06))),
    "fps": env_int("FPS", 24),
    "width": 1080,
    "height": 1920,
    # Scene is drawn at this multiple of the final size, then shrunk
    # (smooth edges). 1.0 = fastest, 1.5 = nicer, 2.0 = best/slowest.
    "supersample": env_float("SUPERSAMPLE", 1.5),
}

TW, TH, FPS = CFG["width"], CFG["height"], CFG["fps"]
PANEL_H = 1160 if BRAINROT else TH   # story panel height (runner fills the rest)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
TRENDS_URL = "https://trends.google.com/trending/rss?geo=US"

# Vocabulary shared by the story prompt, the validator and the renderer.
SCENES = [
    "bathroom", "hallway", "office", "classroom", "kitchen",
    "street", "park", "security_room", "store",
]
WHO = ["hero", "friend", "kid", "cop", "boss", "teacher", "parent", "pigeon"]
POSES = [
    "idle", "walk", "run", "panic", "shock", "point",
    "cry", "celebrate", "sit", "fall", "shrug",
]
EMOTIONS = ["neutral", "happy", "shock", "scared", "angry", "sad", "smug"]
SHOTS = ["wide", "medium", "close"]
INSERT_KINDS = ["phone", "note", "alarm", "laptop", "calendar"]
CHIP_COLORS = {
    "red": (225, 35, 55),
    "purple": (112, 52, 190),
    "blue": (40, 105, 225),
    "orange": (245, 135, 25),
    "green": (40, 165, 90),
}
TROUBLE = {
    "big": ("BIG TROUBLE", (225, 35, 55)),
    "some": ("SOME TROUBLE", (245, 135, 25)),
    "none": ("NO TROUBLE", (40, 175, 95)),
}
TROUBLE_BR = {
    "big": ("COOKED", (225, 35, 55)),
    "some": ("MID", (245, 135, 25)),
    "none": ("LOCKED IN", (40, 175, 95)),
}


# ---------------------------------------------------------------------
# UTILITIES + LLM (story text generation)
# ---------------------------------------------------------------------

def run(command, cwd=None, check=True):
    result = subprocess.run(
        [str(x) for x in command],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        error = result.stderr.decode("utf-8", "replace")
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, command))}\n{error[-4000:]}"
        )
    return result


def probe_duration(path):
    result = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    try:
        return max(0.1, float(result.stdout.decode().strip()))
    except Exception:
        return 0.1


def pick_files(folder, extensions):
    folder = Path(folder)
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_name(text):
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return text[:55] or "episode"


def load_state():
    """Load history. Accepts the old layouts that used 'used_topics'."""
    state = {"episodes": [], "topics": []}
    raw = {}
    if STATE_FILE.exists():
        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            print("Could not read topic history:", exc)
    if isinstance(raw, dict):
        state["episodes"] = [
            e for e in raw.get("episodes", []) if isinstance(e, dict)
        ]
        candidates = (
            list(raw.get("topics", []))
            + list(raw.get("used_topics", []))
            + [e.get("topic") for e in state["episodes"]]
        )
        for topic in candidates:
            if isinstance(topic, str) and topic.strip():
                topic = topic.strip()
                if topic not in state["topics"]:
                    state["topics"].append(topic)
    return state


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------
# STORY GENERATION
# ---------------------------------------------------------------------

def trending_topics():
    topics = []
    try:
        response = requests.get(TRENDS_URL, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            if title:
                topics.append(title)
    except Exception as exc:
        print("Trend lookup unavailable:", exc)
    return topics[:20]


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def discover_gemini_models(api_key):
    """Find a small number of usable Gemini Flash models."""
    try:
        response = requests.get(
            f"{GEMINI_BASE}/models", params={"key": api_key}, timeout=15
        )
        response.raise_for_status()
        models = []
        for item in response.json().get("models", []):
            name = item.get("name", "")
            methods = item.get("supportedGenerationMethods", [])
            lowered = name.lower()
            if (
                "generateContent" in methods
                and "flash" in lowered
                and not any(x in lowered for x in (
                    "image", "audio", "tts", "live", "embedding"
                ))
            ):
                models.append(name.split("/")[-1])
        models.sort(key=lambda n: (
            "preview" in n.lower() or "experimental" in n.lower(),
            "latest" not in n.lower(),
            n,
        ))
        return models[:2]
    except Exception as exc:
        print("Gemini model discovery warning:", str(exc)[:500])
        return []


def gemini_error_detail(response):
    """Google's real error message (never includes the API key)."""
    try:
        error = response.json().get("error", {})
        detail = f"HTTP {response.status_code}"
        if error.get("status"):
            detail += f" ({error['status']})"
        detail += f": {error.get('message') or response.text[:800]}"
        return detail
    except Exception:
        return f"HTTP {response.status_code}: {response.text[:800]}"


def gemini_generate(prompt):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    candidates = []
    if CFG["gemini_model"]:
        candidates.append(CFG["gemini_model"])
    candidates.extend(discover_gemini_models(api_key))
    candidates.append("gemini-2.5-flash")
    candidates = list(dict.fromkeys(candidates))[:3]

    print("Gemini models to try:", ", ".join(candidates))
    last_error = "No Gemini model succeeded."

    for model in candidates:
        print(f"Trying Gemini model: {model}")

        for attempt in range(3):
            try:
                response = requests.post(
                    f"{GEMINI_BASE}/models/{model}:generateContent",
                    params={"key": api_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.95,
                            "responseMimeType": "application/json",
                        },
                    },
                    timeout=75,
                )
                status = response.status_code

                if status == 429:
                    last_error = gemini_error_detail(response)
                    print("Gemini rate limit or quota response:", last_error)
                    if attempt == 0:
                        try:
                            delay = clamp(
                                float(response.headers.get("Retry-After")), 2, 15
                            )
                        except (TypeError, ValueError):
                            delay = 5
                        print(f"Waiting {delay:.1f}s before one retry...")
                        time.sleep(delay)
                        continue
                    # This model's quota is used up; other models have
                    # their own quota, so move on to the next one.
                    print("Quota exhausted for", model, "- trying next model.")
                    break

                if status in (500, 502, 503, 504):
                    last_error = gemini_error_detail(response)
                    print("Temporary Gemini server error:", last_error)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    break

                if status == 404:
                    last_error = gemini_error_detail(response)
                    print("Model unavailable; trying another model.")
                    break

                if not response.ok:
                    last_error = gemini_error_detail(response)
                    raise RuntimeError("Gemini rejected the request: " + last_error)

                payload = response.json()
                cands = payload.get("candidates", [])
                if not cands:
                    last_error = "no candidates: " + json.dumps(
                        payload.get("promptFeedback", {})
                    )[:500]
                    break

                parts = cands[0].get("content", {}).get("parts", [])
                text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
                if not text.strip():
                    last_error = f"{model}: empty response"
                    break

                result = extract_json(text)
                if not isinstance(result, dict):
                    last_error = f"{model}: response was not a JSON object"
                    break

                print(f"Gemini generated a response with {model}.")
                return result

            except requests.RequestException as exc:
                last_error = f"{model}: network error: {exc}"
                print(last_error)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    break

    raise RuntimeError("All Gemini attempts failed. Last error: " + last_error)


def ollama_generate(prompt):
    response = requests.post(
        CFG["ollama_url"],
        json={
            "model": CFG["ollama_model"],
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=180,
    )
    response.raise_for_status()
    return extract_json(response.json().get("response", ""))


def llm_json(prompt):
    try:
        return gemini_generate(prompt)
    except Exception as gemini_error:
        # GitHub Actions has no Ollama server; only try it if enabled.
        if os.getenv("USE_OLLAMA_FALLBACK", "0") == "1":
            print("Gemini failed; trying configured Ollama fallback.")
            try:
                return ollama_generate(prompt)
            except Exception as ollama_error:
                raise RuntimeError(
                    f"Gemini failed: {gemini_error}; Ollama failed: {ollama_error}"
                )
        raise RuntimeError(f"Gemini story generation failed: {gemini_error}")


# ---------------------------------------------------------------------
# TINY 3D ENGINE (flat-shaded blocks, painter's algorithm)
# ---------------------------------------------------------------------

NEAR = 0.5
LIGHT = np.array([-0.35, 0.8, -0.5])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mulc(col, k):
    return tuple(int(clamp(c * k, 0, 255)) for c in col)


def box(center, size, col, R=None, flat=False):
    """A block: centre, 3 half-axis vectors (rows of A), colour."""
    half = np.diag(np.array(size, float) / 2)
    m = half if R is None else R @ half
    return (np.array(center, float), m.T.copy(), col, flat)


class Camera:
    def __init__(self, pos, target, fov=45.0):
        self.pos = np.array(pos, float)
        fwd = np.array(target, float) - self.pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross([0, 1, 0], fwd)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        self.basis = np.stack([right, up, fwd])
        self.tan = math.tan(math.radians(fov) / 2)

    def to_cam(self, p):
        return (p - self.pos) @ self.basis.T


def clip_near(poly):
    """Clip a camera-space polygon against the near plane."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ain, bin_ = a[2] >= NEAR, b[2] >= NEAR
        if ain:
            out.append(a)
        if ain != bin_:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append([a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t, NEAR])
    return out


def project(poly, f, cx, cy):
    return [(cx + f * p[0] / p[2], cy - f * p[1] / p[2]) for p in poly]


class Scene:
    """Static quads/boxes of one room, plus optional animated extras."""

    def __init__(self, name):
        self.name = name
        self.quads = []      # (4 world points, colour)
        self.boxes = []
        self.bg = ((20, 20, 40), (60, 60, 90))
        self.shadow = (40, 40, 50)
        self.dynamic = None  # fn(t) -> list of boxes
        self._qv = None

    # ----- builders
    def quad(self, p, col):
        self.quads.append((np.array(p, float), col))

    def block(self, center, size, col, flat=False):
        self.boxes.append(box(center, size, col, flat=flat))

    def floor(self, a, b, tile=2.0, x0=-8, x1=8, z0=-26, z1=9):
        xs = np.arange(x0, x1, tile)
        zs = np.arange(z0, z1, tile)
        for i, x in enumerate(xs):
            for j, z in enumerate(zs):
                col = a if (i + j) % 2 == 0 else b
                self.quad([(x, 0, z), (x + tile, 0, z),
                           (x + tile, 0, z + tile), (x, 0, z + tile)], col)

    def back_wall(self, col, z=9, x0=-8, x1=8, h=11):
        self.quad([(x0, 0, z), (x1, 0, z), (x1, h, z), (x0, h, z)], col)

    def side_walls(self, col_l, col_r=None, x=8, z0=-26, z1=9, h=11):
        col_r = col_r or col_l
        self.quad([(-x, 0, z0), (-x, 0, z1), (-x, h, z1), (-x, h, z0)], col_l)
        self.quad([(x, 0, z0), (x, 0, z1), (x, h, z1), (x, h, z0)], col_r)

    def back_rect(self, x0, x1, y0, y1, col, z=8.97):
        self.quad([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], col)

    def wall_tiles(self, col_a, col_b, tile=2.0, h=10, z=8.98, x0=-8, x1=8):
        xs = np.arange(x0, x1, tile)
        for i, x in enumerate(xs):
            for j in range(int(h / tile)):
                col = col_a if (i + j) % 2 == 0 else col_b
                y = j * tile
                self.quad([(x, y, z), (x + tile, y, z),
                           (x + tile, y + tile, z), (x, y + tile, z)], col)

    def side_tiles(self, side, col_a, col_b, tile=2.0, h=10, z0=-26, z1=9):
        x = -7.98 if side < 0 else 7.98
        for i, z in enumerate(np.arange(z0, z1, tile)):
            for j in range(int(h / tile)):
                col = col_a if (i + j) % 2 == 0 else col_b
                y = j * tile
                self.quad([(x, y, z), (x, y, z + tile),
                           (x, y + tile, z + tile), (x, y + tile, z)], col)

    def side_rect(self, side, z0, z1, y0, y1, col):
        x = -7.95 if side < 0 else 7.95
        self.quad([(x, y0, z0), (x, y0, z1), (x, y1, z1), (x, y1, z0)], col)

    def finalize(self):
        self._qv = np.array([q[0] for q in self.quads]) if self.quads else None
        self._qc = [q[1] for q in self.quads]
        self.static = None
        if self.boxes:
            self.static = (
                np.array([b[0] for b in self.boxes]),
                np.array([b[1] for b in self.boxes]),
                [b[2] for b in self.boxes],
                [b[3] for b in self.boxes],
            )


def _face_geometry(C, A):
    """All 6 faces of every box: corners, centres, normals."""
    corners, centres, normals = [], [], []
    for k in range(3):
        ax = A[:, k, :]
        u = A[:, (k + 1) % 3, :]
        v = A[:, (k + 2) % 3, :]
        nrm = ax / np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-9)
        for s in (1, -1):
            fc = C + s * ax
            corners.append(np.stack([fc + u + v, fc - u + v, fc - u - v, fc + u - v], 1))
            centres.append(fc)
            normals.append(s * nrm)
    return (np.stack(corners, 1), np.stack(centres, 1), np.stack(normals, 1))


def draw_world(draw, cam, scene, dyn_boxes, RW, RH, yshift=0.0):
    f = (RH / 2) / cam.tan
    cx, cy = RW / 2, RH / 2 + yshift * RH

    # ---- static quads (room shell and wall decoration), in listed order
    if scene._qv is not None:
        cq = cam.to_cam(scene._qv)
        for i in range(len(cq)):
            poly = cq[i]
            if poly[:, 2].min() < NEAR:
                poly = clip_near(poly.tolist())
                if len(poly) < 3:
                    continue
                pts = project(poly, f, cx, cy)
            else:
                pts = project(poly.tolist(), f, cx, cy)
            col = scene._qc[i]
            draw.polygon(pts, fill=col, outline=mulc(col, 0.8))

    # ---- soft shadows under characters
    for pts, col in (dyn_boxes.get("decals", []) if isinstance(dyn_boxes, dict) else []):
        cp = cam.to_cam(np.array(pts, float))
        if cp[:, 2].min() > NEAR:
            draw.polygon(project(cp.tolist(), f, cx, cy), fill=col)

    boxes = dyn_boxes["boxes"] if isinstance(dyn_boxes, dict) else dyn_boxes
    Cs, As, cols, flats = [], [], [], []
    if scene.static is not None:
        Cs.append(scene.static[0]); As.append(scene.static[1])
        cols += scene.static[2]; flats += scene.static[3]
    if boxes:
        Cs.append(np.array([b[0] for b in boxes]))
        As.append(np.array([b[1] for b in boxes]))
        cols += [b[2] for b in boxes]; flats += [b[3] for b in boxes]
    if not Cs:
        return
    C = np.concatenate(Cs); A = np.concatenate(As)

    corners, centres, normals = _face_geometry(C, A)          # (N,6,...)
    visible = ((cam.pos - centres) * normals).sum(-1) > 0     # (N,6)
    shade = 0.55 + 0.45 * np.clip(normals @ LIGHT, 0, 1)      # (N,6)
    ccam = cam.to_cam(corners)                                # (N,6,4,3)
    fz = cam.to_cam(centres)[..., 2]                          # (N,6)
    bz = cam.to_cam(C)[:, 2]                                  # (N,)

    faces = []
    idx = np.argwhere(visible & (fz > 0.1))
    for n, k in idx:
        poly = ccam[n, k]
        if poly[:, 2].min() < NEAR:
            poly = clip_near(poly.tolist())
            if len(poly) < 3:
                continue
        else:
            poly = poly.tolist()
        col = cols[n]
        sh = 1.0 if flats[n] else shade[n, k]
        faces.append((-bz[n], -fz[n, k], poly, mulc(col, sh), mulc(col, sh * 0.6)))
    faces.sort(key=lambda t: (t[0], t[1]))
    for _, _, poly, fill, line in faces:
        draw.polygon(project(poly, f, cx, cy), fill=fill, outline=line)


# ---------------------------------------------------------------------
# CHARACTERS
# ---------------------------------------------------------------------

ROSTER = {
    "hero": dict(shirt=(240, 150, 40), pants=(55, 62, 95), skin=(240, 190, 150),
                 hair=(95, 55, 30), style="short", scale=1.0),
    "friend": dict(shirt=(70, 180, 95), pants=(60, 70, 100), skin=(190, 130, 95),
                   hair=(30, 25, 25), style="short", scale=1.0),
    "kid": dict(shirt=(225, 65, 65), pants=(50, 90, 140), skin=(245, 205, 170),
                hair=(235, 195, 70), style="short", scale=0.8),
    "cop": dict(shirt=(40, 72, 155), pants=(28, 38, 80), skin=(220, 170, 130),
                hair=(40, 30, 25), style="short", scale=1.05, extras=("cop",)),
    "boss": dict(shirt=(100, 105, 120), pants=(50, 52, 62), skin=(210, 160, 120),
                 hair=(205, 205, 210), style="grey", scale=1.05,
                 extras=("tie", "glasses")),
    "teacher": dict(shirt=(155, 85, 195), pants=(60, 50, 85), skin=(235, 185, 150),
                    hair=(125, 65, 40), style="long", scale=1.0),
    "parent": dict(shirt=(65, 125, 205), pants=(75, 75, 88), skin=(225, 175, 135),
                   hair=(65, 45, 35), style="bun", scale=1.0),
}
DARK = (30, 28, 38)
SHOE = (42, 40, 52)


def ease_out_back(t):
    t = clamp(t, 0, 1)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease(t):
    t = clamp(t, 0, 1)
    return t * t * (3 - 2 * t)


def pose_params(pose, emotion, t, u):
    """Animation state. Arms are (forward_angle, outward_angle)."""
    p = dict(al=(0.0, 0.08), ar=(0.0, 0.08), ll=0.0, lr=0.0, bob=0.0,
             lean=0.0, tilt=0.0, mouth=0.0, dx=0.0, yaw=None, tears=False,
             hyaw=0.0, sway=0.0)
    p["bob"] = math.sin(t * 2.3) * 0.035
    p["al"] = (math.sin(t * 1.6) * 0.05, 0.08)
    p["ar"] = (math.sin(t * 1.6 + 1) * 0.05, 0.08)
    if pose == "idle":
        # talking gestures: arm lifts and chops in bursts, head nods, weight shift
        g = max(0.0, math.sin(t * 1.9 + 0.6)) ** 1.5
        beat_ = math.sin(t * 8.0)
        p["ar"] = (0.25 + 1.0 * g + 0.15 * beat_ * g, 0.15 + 0.2 * g)
        g2 = max(0.0, math.sin(t * 1.3 + 2.4)) ** 2
        p["al"] = (0.1 + 0.7 * g2, 0.12 + 0.15 * g2)
        p["tilt"] = math.sin(t * 3.1) * 0.05 + 0.05 * g
        p["sway"] = math.sin(t * 1.1) * 0.12
        p["mouth"] = 0.0
    if pose in ("walk", "run"):
        spd = 7.0 if pose == "walk" else 12.0
        amp = 0.65 if pose == "walk" else 1.0
        s = math.sin(t * spd)
        p["ll"], p["lr"] = s * amp, -s * amp
        p["al"], p["ar"] = (-s * amp * 0.9, 0.08), (s * amp * 0.9, 0.08)
        p["bob"] = abs(math.cos(t * spd)) * (0.12 if pose == "walk" else 0.3)
        p["dx"] = (u - 0.5) * (7.0 if pose == "walk" else 10.0)
        p["yaw"] = math.pi / 2
        p["lean"] = 0.0 if pose == "walk" else 0.22
    elif pose == "panic":
        p["bob"] = abs(math.sin(t * 9)) * 0.45
        flap = math.sin(t * 16) * 0.25
        p["al"], p["ar"] = (0.2, 2.5 + flap), (0.2, 2.5 - flap)
        p["ll"], p["lr"] = math.sin(t * 9) * 0.5, -math.sin(t * 9) * 0.5
        p["mouth"] = 1.0
    elif pose == "shock":
        jump = math.sin(math.pi * clamp(t / 0.3, 0, 1)) * 0.9
        p["bob"] = jump
        p["al"], p["ar"] = (0.3, 0.9), (0.3, 0.9)
        p["mouth"] = 1.0
        p["bob"] += math.sin(t * 40) * 0.015
    elif pose == "point":
        p["ar"] = (1.5, 0.12)
        p["tilt"] = 0.05
        if emotion == "angry":
            p["mouth"] = 0.5 + 0.5 * math.sin(t * 13)
            p["bob"] += abs(math.sin(t * 6.5)) * 0.08
    elif pose == "cry":
        p["al"], p["ar"] = (2.25, 0.25), (2.25, 0.25)
        p["tilt"] = 0.28
        p["bob"] = math.sin(t * 14) * 0.07
        p["dx"] = math.sin(t * 26) * 0.04
        p["tears"] = True
        p["mouth"] = 0.6
    elif pose == "celebrate":
        p["bob"] = abs(math.sin(t * 6)) * 0.7
        p["al"] = (0.2, 2.7 + math.sin(t * 12) * 0.3)
        p["ar"] = (0.2, 2.7 - math.sin(t * 12) * 0.3)
        p["ll"], p["lr"] = math.sin(t * 6) * 0.3, -math.sin(t * 6) * 0.3
    elif pose == "sit":
        p["bob"] = -1.05
        p["ll"] = p["lr"] = 1.55
        p["al"], p["ar"] = (0.9 + math.sin(t * 9) * 0.08, 0.1), (0.9 - math.sin(t * 9) * 0.08, 0.1)
    elif pose == "fall":
        k = clamp(t / 0.35, 0, 1)
        p["lean"] = -1.45 * ease(k)
        p["bob"] = 0.55 * abs(math.sin(p["lean"]))
        p["al"], p["ar"] = (0.4, 1.6), (0.4, 1.6)
        p["mouth"] = 1.0
    elif pose == "sprint":
        sw = math.sin(t * 13)
        p["ll"], p["lr"] = sw, -sw
        p["al"], p["ar"] = (-sw, 0.1), (sw, 0.1)
        p["bob"] = abs(math.cos(t * 13)) * 0.22 + u
        p["lean"] = 0.2
        if u > 0.15:                      # mid-jump: tuck legs, arms up
            p["ll"] = p["lr"] = 0.7
            p["al"] = p["ar"] = (0.5, 1.3)
    elif pose == "shrug":
        p["al"], p["ar"] = (0.3, 0.75), (0.3, 0.75)
        p["tilt"] = 0.18 + math.sin(t * 2) * 0.04

    p["hyaw"] = math.sin(t * 0.8) * 0.12
    if emotion == "sad" and pose in ("idle", "walk"):
        p["tilt"] = 0.25
        p["lean"] += 0.06
    if emotion in ("shock", "scared") and not p["mouth"]:
        p["mouth"] = 0.8
    if emotion == "scared":
        p["dx"] += math.sin(t * 30) * 0.035
    if emotion == "angry" and not p["mouth"] and pose == "idle":
        p["mouth"] = 0.0
    return p


def build_character(who, x, z, yaw, pose, emotion, t, u, look=0.0, enter=None):
    """Return list of boxes for one blocky character.
    look  = extra head turn (radians) toward whoever they're talking to.
    enter = (from_dx, progress 0..1): walk in from the side, then settle."""
    spec = ROSTER.get(who, ROSTER["hero"])
    s = spec["scale"]
    p = pose_params(pose, emotion, t, u)
    if enter is not None and enter[1] < 1.0:
        fx, k = enter
        sp = 7.0
        sw = math.sin(t * sp) * 0.65
        p["ll"], p["lr"] = sw, -sw
        p["al"], p["ar"] = (-sw * 0.9, 0.08), (sw * 0.9, 0.08)
        p["bob"] = abs(math.cos(t * sp)) * 0.12
        p["dx"] = fx * (1.0 - ease(k))
        p["yaw"] = math.pi / 2 * (1 if fx < 0 else -1)
        p["lean"] = 0.0
        p["tilt"] = 0.0
        p["mouth"] = 0.0
    elif enter is not None and enter[1] < 1.35 and pose == "idle":
        p["hyaw"] += 0.0
    if p["yaw"] is not None:
        yaw = p["yaw"]
    Rg = rot_y(yaw) @ rot_z(p["sway"]) @ rot_x(p["lean"])
    origin = np.array([x + p["dx"], 0.0, z])
    out = []

    def part(center, size, col, R=None, pivot=None, flat=False):
        c = np.array(center, float)
        Rl = np.eye(3)
        if R is not None:
            piv = np.array(pivot if pivot is not None else center, float)
            c = piv + R @ (c - piv)
            Rl = R
        c = c + np.array([0, p["bob"], 0])
        cw = origin + s * (Rg @ c)
        out.append(box(cw, np.array(size, float) * s, col, R=Rg @ Rl, flat=flat))

    shirt, pants, skin, hair = spec["shirt"], spec["pants"], spec["skin"], spec["hair"]

    # legs + shoes
    for side, ang in ((-1, p["ll"]), (1, p["lr"])):
        R = rot_x(-ang)
        piv = (side * 0.5, 2.0, 0)
        part((side * 0.5, 1.0, 0), (0.92, 2.0, 0.92), pants, R, piv)
        part((side * 0.5, 0.18, 0.12), (0.98, 0.36, 1.2), SHOE, R, piv)

    # torso
    part((0, 3.0, 0), (2.0, 2.0, 1.0), shirt)
    if "tie" in spec.get("extras", ()):
        part((0, 3.25, 0.52), (0.3, 1.3, 0.05), (190, 30, 45))
        part((0, 3.95, 0.52), (0.9, 0.22, 0.05), (235, 235, 240))
    if "cop" in spec.get("extras", ()):
        part((-0.55, 3.45, 0.52), (0.38, 0.38, 0.05), (245, 200, 60))
        part((0, 2.12, 0.0), (2.04, 0.22, 1.04), (30, 30, 36))

    # arms (upper sleeve + hand)
    for side, (fw, out_a) in ((-1, p["al"]), (1, p["ar"])):
        R = rot_z(side * out_a) @ rot_x(-fw)
        piv = (side * 1.45, 3.9, 0)
        part((side * 1.45, 3.2, 0), (0.92, 1.5, 0.92), shirt, R, piv)
        part((side * 1.45, 2.15, 0), (0.92, 0.7, 0.92), skin, R, piv)

    # head group (nods / tilts about the neck)
    Rh = rot_y(p["hyaw"] + look) @ rot_x(p["tilt"])
    neck = (0, 4.0, 0)

    def head_part(center, size, col, R=None, flat=False):
        c = np.array(center, float)
        if R is None:
            part(c, size, col, Rh, neck, flat)
        else:
            c1 = np.array(neck) + Rh @ (c - np.array(neck))
            part(c1, size, col, Rh @ R, c1, flat)

    hy = 4.9
    head_part((0, hy, 0), (1.6, 1.6, 1.6), skin)

    # hair
    style = spec["style"]
    head_part((0, hy + 0.78, 0), (1.72, 0.42, 1.72), hair)
    head_part((0, hy + 0.6, 0.78), (1.72, 0.3, 0.3), hair)
    head_part((0, hy + 0.1, -0.8), (1.72, 1.4, 0.3), hair)
    if style == "long":
        for sx in (-1, 1):
            head_part((sx * 0.9, hy - 0.5, -0.1), (0.3, 2.2, 1.6), hair)
    elif style == "bun":
        head_part((0, hy + 1.25, -0.2), (0.7, 0.6, 0.7), hair)
    elif style == "grey":
        pass
    if "cop" in spec.get("extras", ()):
        head_part((0, hy + 0.95, 0), (1.95, 0.55, 1.95), (30, 50, 110))
        head_part((0, hy + 0.78, 1.0), (1.6, 0.12, 0.7), (20, 28, 60))
        head_part((0, hy + 1.0, 0.99), (0.4, 0.3, 0.05), (245, 200, 60))

    # face
    blink = ((t + sum(map(ord, who)) % 7) % 3.4) < 0.12
    big = emotion in ("shock", "scared") or pose in ("shock", "panic", "fall")
    ew = 0.3 if big else 0.22
    eh = 0.08 if blink else (0.5 if big else 0.34)
    ey = hy + 0.12
    for sx in (-1, 1):
        if big:
            head_part((sx * 0.4, ey, 0.82), (0.44, 0.56, 0.05), (250, 250, 250))
        head_part((sx * 0.4, ey, 0.85 if big else 0.82), (ew, eh, 0.06), DARK)
    if emotion == "angry":
        for sx in (-1, 1):
            head_part((sx * 0.4, ey + 0.38, 0.84), (0.5, 0.1, 0.05), DARK,
                      R=rot_z(-sx * 0.45))
    elif emotion in ("sad", "scared"):
        for sx in (-1, 1):
            head_part((sx * 0.4, ey + 0.4, 0.84), (0.46, 0.09, 0.05), DARK,
                      R=rot_z(sx * 0.35))
    elif emotion == "smug":
        for sx in (-1, 1):
            head_part((sx * 0.4, ey + 0.28, 0.84), (0.5, 0.09, 0.05), DARK,
                      R=rot_z(-sx * 0.12))

    my = hy - 0.42
    if p["mouth"] > 0.05:
        head_part((0, my - 0.1 * p["mouth"], 0.82),
                  (0.5 + 0.1 * p["mouth"], 0.12 + 0.45 * p["mouth"], 0.05), (80, 15, 28))
    elif emotion in ("happy", "smug") or pose == "celebrate":
        head_part((0, my - 0.06, 0.82), (0.62, 0.1, 0.05), DARK)
        for sx in (-1, 1):
            head_part((sx * 0.36, my + 0.04, 0.82), (0.1, 0.18, 0.05), DARK)
    elif emotion in ("sad", "angry"):
        head_part((0, my - 0.05, 0.82), (0.5, 0.1, 0.05), DARK)
        for sx in (-1, 1):
            head_part((sx * 0.28, my - 0.14, 0.82), (0.1, 0.16, 0.05), DARK)
    else:
        head_part((0, my, 0.82), (0.5, 0.09, 0.05), DARK)

    if "glasses" in spec.get("extras", ()):
        for sx in (-1, 1):
            head_part((sx * 0.4, ey, 0.9), (0.62, 0.52, 0.04), (35, 35, 45))
            head_part((sx * 0.4, ey, 0.92), (0.5, 0.4, 0.05), (190, 225, 245))
        head_part((0, ey + 0.1, 0.9), (0.3, 0.06, 0.04), (35, 35, 45))

    if p["tears"]:
        for sx in (-1, 1):
            ty = ey - 0.35 - ((t * 1.6 + (0.5 if sx > 0 else 0)) % 1.0) * 0.5
            head_part((sx * 0.45, ty, 0.84), (0.14, 0.3, 0.05), (90, 170, 255))
    return out, origin


def build_pigeon(x, z, t, yaw=math.pi):
    R = rot_y(yaw)
    o = np.array([x, 0.0, z])
    bob = abs(math.sin(t * 5)) * 0.08
    peck = max(0.0, math.sin(t * 2.2)) ** 6 * 0.35
    out = []

    def part(c, size, col):
        out.append(box(o + R @ (np.array(c, float) + [0, bob, 0]), size, col, R=R))

    part((0, 1.3, 0), (1.5, 1.2, 2.1), (130, 140, 160))
    part((0.82, 1.3, -0.1), (0.2, 0.8, 1.2), (95, 105, 128))
    part((-0.82, 1.3, -0.1), (0.2, 0.8, 1.2), (95, 105, 128))
    part((0, 2.15 - peck, 0.9 + peck), (0.85, 0.85, 0.85), (120, 135, 160))
    part((0, 2.1 - peck, 1.45 + peck), (0.25, 0.2, 0.45), (250, 150, 40))
    for sx in (-1, 1):
        part((sx * 0.3, 2.25 - peck, 1.3 + peck), (0.12, 0.14, 0.06), DARK)
        part((sx * 0.35, 0.45, 0.1), (0.14, 0.9, 0.14), (240, 130, 40))
        part((sx * 0.35, 0.05, 0.28), (0.3, 0.1, 0.5), (240, 130, 40))
    part((0, 1.55, 0.62), (1.4, 0.4, 0.4), (150, 120, 190))
    return out, o


def shadow_decal(origin, r, col):
    pts = []
    for i in range(10):
        a = i / 10 * math.tau
        pts.append((origin[0] + math.cos(a) * r, 0.03, origin[2] + math.sin(a) * r * 0.7))
    return pts, col


# ---------------------------------------------------------------------
# ROOMS
# ---------------------------------------------------------------------

def scene_bathroom():
    s = Scene("bathroom")
    s.bg = ((205, 222, 210), (160, 185, 170)); s.shadow = (125, 145, 140)
    s.floor((190, 200, 205), (172, 184, 190))
    s.back_wall((150, 205, 170)); s.wall_tiles((168, 218, 188), (152, 204, 172))
    s.side_walls((140, 195, 160))
    s.side_tiles(-1, (160, 212, 180), (146, 198, 166), z0=-14)
    s.side_tiles(1, (160, 212, 180), (146, 198, 166), z0=-14)
    for x in (-4.2, 0, 4.2):
        s.block((x, 3.6, 7.0), (3.9, 6.2, 0.15), (88, 170, 176))
        s.block((x + 1.3, 3.4, 6.9), (0.2, 0.5, 0.15), (60, 60, 70))
    for x in (-6.3, -2.1, 2.1, 6.3):
        s.block((x, 3.6, 8.2), (0.15, 6.2, 2.9), (78, 155, 162))
    s.block((6.6, 1.6, 3.0), (2.4, 3.2, 7.0), (232, 234, 240))
    for z in (1.2, 4.8):
        s.block((6.4, 3.4, z), (1.4, 0.25, 1.6), (70, 110, 160))
        s.block((7.4, 3.9, z), (0.3, 0.7, 0.3), (190, 195, 205))
    s.side_rect(1, 0.0, 6.5, 4.6, 9.0, (205, 235, 245))
    s.side_rect(1, 0.3, 6.2, 4.9, 8.7, (175, 215, 235))
    s.block((-6.4, 0.9, 2.2), (1.3, 1.8, 1.3), (110, 118, 130))
    s.block((-7.4, 5.0, 4.5), (0.5, 1.2, 1.3), (225, 228, 235))
    s.finalize(); return s


def scene_hallway():
    s = Scene("hallway")
    s.bg = ((230, 222, 196), (200, 190, 160)); s.shadow = (150, 140, 110)
    s.floor((218, 204, 172), (206, 192, 160))
    s.back_wall((228, 218, 192)); s.side_walls((236, 226, 200))
    rng = random.Random(4)
    for side in (-1, 1):
        for z in np.arange(-8.0, 9.0, 1.75):
            col = (45, 82, 205) if rng.random() < 0.5 else (38, 70, 188)
            x = side * 7.4
            s.block((x, 3.6, z + 0.87), (1.2, 7.0, 1.7), col)
            fx = side * 6.78
            s.block((fx, 6.1, z + 0.87), (0.06, 0.12, 0.8), (20, 30, 90))
            s.block((fx, 5.6, z + 0.87), (0.06, 0.12, 0.8), (20, 30, 90))
            s.block((fx, 3.4, z + 1.3), (0.08, 0.6, 0.14), (205, 210, 225))
    s.back_rect(-2.2, 2.2, 0, 6.8, (125, 85, 55))
    s.back_rect(-2.0, -0.1, 0.2, 6.5, (140, 98, 64))
    s.back_rect(0.1, 2.0, 0.2, 6.5, (140, 98, 64))
    s.back_rect(-1.2, 1.2, 7.4, 8.4, (40, 180, 95))
    s.back_rect(-6.2, -3.2, 3.2, 6.8, (200, 160, 100))
    for i, c in enumerate(((250, 240, 120), (240, 130, 150), (130, 200, 240), (250, 250, 250))):
        s.back_rect(-6.0 + (i % 2) * 1.4, -5.0 + (i % 2) * 1.4, 3.6 + (i // 2) * 1.5,
                    4.8 + (i // 2) * 1.5, c, z=8.95)
    s.finalize(); return s


def scene_office():
    s = Scene("office")
    s.bg = ((40, 60, 105), (60, 80, 120)); s.shadow = (70, 45, 35)
    s.floor((124, 84, 58), (112, 74, 50))
    s.back_wall((46, 72, 118)); s.side_walls((40, 63, 106))
    s.back_rect(-5.4, -1.4, 4.0, 8.6, (235, 235, 240))
    s.back_rect(-5.1, -1.7, 4.3, 8.3, (150, 205, 240))
    s.back_rect(-3.5, -3.3, 4.3, 8.3, (235, 235, 240))
    s.back_rect(-5.1, -1.7, 6.2, 6.4, (235, 235, 240))
    s.back_rect(2.2, 5.0, 4.6, 7.8, (95, 70, 45))
    s.back_rect(2.4, 4.8, 4.8, 7.6, (225, 190, 120))
    s.quad([(-4.5, 0.02, -2), (4.5, 0.02, -2), (4.5, 0.02, 6.4), (-4.5, 0.02, 6.4)], (170, 42, 52))
    s.block((0, 3.0, 7.4), (6.4, 0.35, 2.6), (140, 92, 55))
    s.block((0, 1.55, 6.3), (6.2, 2.9, 0.3), (112, 70, 42))
    for x in (-3.0, 3.0):
        s.block((x, 1.55, 7.4), (0.3, 2.9, 2.4), (112, 70, 42))
    s.block((-1.6, 4.0, 7.7), (1.8, 1.3, 0.15), (30, 32, 40))
    s.block((-1.6, 4.0, 7.62), (1.6, 1.1, 0.05), (90, 185, 240), flat=True)
    s.block((-1.6, 3.35, 7.7), (0.4, 0.4, 0.3), (60, 60, 70))
    s.block((1.4, 3.3, 7.2), (1.2, 0.25, 0.7), (55, 55, 62))
    rng = random.Random(11)
    for row in range(3):
        s.block((7.3, 1.2 + row * 2.2, 4.5), (1.0, 0.18, 6.0), (95, 64, 40))
        for k in range(10):
            col = rng.choice([(200, 60, 60), (70, 130, 210), (230, 190, 60), (70, 170, 110), (220, 220, 225)])
            s.block((7.35, 1.6 + row * 2.2, 1.9 + k * 0.55), (0.7, 0.9, 0.4), col)
    s.block((-6.5, 0.5, 6.8), (1.1, 1.0, 1.1), (150, 90, 60))
    s.block((-6.5, 1.9, 6.8), (1.8, 1.9, 1.8), (60, 150, 80))
    s.block((-6.8, 1.6, 3.4), (1.4, 3.2, 1.4), (150, 156, 168))
    s.sit_z = 8.4
    s.finalize(); return s


def scene_classroom():
    s = Scene("classroom")
    s.bg = ((235, 228, 180), (210, 200, 150)); s.shadow = (120, 90, 60)
    s.floor((190, 150, 100), (180, 140, 92))
    s.back_wall((238, 228, 172)); s.side_walls((228, 218, 162))
    s.back_rect(-5.8, 5.8, 3.0, 8.6, (112, 76, 46))
    s.back_rect(-5.5, 5.5, 3.3, 8.3, (44, 92, 76))
    for i, y in enumerate((7.3, 6.5, 5.7)):
        s.back_rect(-4.5, -1.5 - i, y, y + 0.12, (235, 235, 235), z=8.94)
    s.back_rect(1.0, 4.2, 6.0, 6.12, (235, 235, 235), z=8.94)
    s.block((0, 3.0, 8.75), (11.4, 0.2, 0.5), (112, 76, 46))
    for z0 in (1.0, 4.4):
        s.side_rect(1, z0, z0 + 2.8, 4.2, 8.6, (232, 232, 238))
        s.side_rect(1, z0 + 0.2, z0 + 2.6, 4.4, 8.4, (165, 215, 242))
    for x in (-5.4, 5.4):
        for z in (3.0, 6.2):
            s.block((x, 2.3, z), (2.4, 0.22, 1.6), (205, 165, 105))
            for dx in (-1, 1):
                for dz in (-1, 1):
                    s.block((x + dx * 1.0, 1.1, z + dz * 0.6), (0.14, 2.2, 0.14), (90, 90, 100))
    s.block((0, 2.8, 7.6), (4.4, 0.3, 1.8), (150, 100, 60))
    s.block((0, 1.4, 6.8), (4.2, 2.6, 0.2), (125, 82, 48))
    s.block((1.4, 3.15, 7.6), (0.5, 0.5, 0.5), (220, 40, 40))
    s.sit_z = 8.4
    s.finalize(); return s


def scene_kitchen():
    s = Scene("kitchen")
    s.bg = ((245, 228, 185), (225, 205, 160)); s.shadow = (130, 120, 105)
    s.floor((238, 238, 240), (205, 210, 216))
    s.back_wall((242, 222, 172)); s.side_walls((248, 230, 182))
    s.block((0.5, 1.6, 8.0), (13, 3.2, 2.0), (205, 160, 110))
    s.block((0.5, 3.3, 8.0), (13.2, 0.25, 2.2), (238, 238, 244))
    for x in np.arange(-5.5, 6.5, 2.4):
        s.block((x + 0.2, 1.7, 6.97), (2.2, 2.6, 0.06), (190, 144, 96))
        s.block((x + 0.9, 1.9, 6.93), (0.1, 0.5, 0.06), (230, 230, 235))
    s.back_rect(-2.2, 2.2, 5.0, 8.8, (235, 235, 240))
    s.back_rect(-1.9, 1.9, 5.3, 8.5, (150, 205, 242))
    s.back_rect(-0.05, 0.05, 5.3, 8.5, (235, 235, 240))
    for x in (-5.5, 5.5):
        s.block((x, 7.0, 8.4), (3.6, 2.4, 1.2), (205, 160, 110))
    s.block((-6.7, 3.5, 6.2), (2.4, 7.0, 2.6), (214, 218, 226))
    s.block((-5.45, 4.2, 4.85), (0.12, 2.2, 0.15), (120, 125, 135))
    s.block((-5.45, 2.0, 4.85), (0.12, 1.2, 0.15), (120, 125, 135))
    s.block((4.8, 1.8, 7.9), (3.0, 3.5, 1.9), (70, 74, 86))
    for dx in (-0.7, 0.7):
        for dz in (-0.4, 0.4):
            s.block((4.8 + dx, 3.6, 7.9 + dz), (0.8, 0.1, 0.8), (25, 25, 30))
    s.block((6.2, 2.6, 2.0), (3.0, 0.25, 3.0), (160, 110, 70))
    for dx in (-1.2, 1.2):
        for dz in (-1.2, 1.2):
            s.block((6.2 + dx, 1.3, 2.0 + dz), (0.2, 2.6, 0.2), (130, 88, 55))
    s.block((6.2, 2.85, 2.0), (1.1, 0.3, 1.1), (240, 220, 120))
    s.finalize(); return s


def scene_street():
    s = Scene("street")
    s.bg = ((120, 185, 240), (208, 232, 250)); s.shadow = (130, 130, 140)
    s.floor((205, 205, 210), (192, 192, 198), tile=4, x0=-28, x1=28, z0=-28, z1=5)
    s.quad([(-28, 0.01, 5), (28, 0.01, 5), (28, 0.01, 14), (-28, 0.01, 14)], (72, 74, 84))
    for x in np.arange(-28, 28, 5):
        s.quad([(x, 0.02, 9.8), (x + 2.5, 0.02, 9.8), (x + 2.5, 0.02, 10.2), (x, 0.02, 10.2)], (240, 240, 245))
    s.quad([(-28, 0.01, 14), (28, 0.01, 14), (28, 0.01, 17), (-28, 0.01, 17)], (198, 198, 204))
    rng = random.Random(21)
    x = -26
    while x < 26:
        w = rng.uniform(4.5, 7.0); h = rng.uniform(9, 20)
        col = rng.choice([(190, 120, 100), (120, 140, 190), (210, 185, 130), (150, 175, 150), (170, 130, 170)])
        s.block((x + w / 2, h / 2, 21), (w, h, 4), col)
        for wx in np.arange(x + 0.9, x + w - 0.6, 1.5):
            for wy in np.arange(2.6, h - 1, 2.6):
                wc = (255, 232, 150) if rng.random() < 0.4 else (110, 150, 195)
                s.quad([(wx, wy, 18.97), (wx + 0.9, wy, 18.97), (wx + 0.9, wy + 1.4, 18.97), (wx, wy + 1.4, 18.97)], wc)
        x += w + 0.4
    for tx in (-9, 8.5, -16, 17):
        s.block((tx, 1.6, 15.6), (0.7, 3.2, 0.7), (110, 75, 50))
        s.block((tx, 4.4, 15.6), (3.2, 2.8, 3.2), (70, 160, 85))
    for lx in (-6.5, 6.5):
        s.block((lx, 3.5, 14.8), (0.3, 7.0, 0.3), (60, 62, 72))
        s.block((lx, 7.2, 14.4), (0.3, 0.3, 1.4), (60, 62, 72))
        s.block((lx, 7.0, 13.8), (0.6, 0.3, 0.6), (255, 240, 170), flat=True)

    def car(t):
        cx = -22 + ((t * 7.0) % 44)
        z = 11.8
        col = (215, 50, 60)
        out = [box((cx, 1.0, z), (4.4, 1.2, 2.1), col),
               box((cx - 0.2, 2.1, z), (2.4, 1.0, 1.9), (170, 215, 240))]
        for dx in (-1.4, 1.4):
            for dz in (-1.0, 1.0):
                out.append(box((cx + dx, 0.45, z + dz), (0.9, 0.9, 0.35), (30, 30, 38)))
        return out
    s.dynamic = car
    s.finalize(); return s


def scene_park():
    s = Scene("park")
    s.bg = ((125, 195, 245), (210, 235, 250)); s.shadow = (70, 120, 70)
    s.floor((112, 188, 96), (100, 176, 88), tile=4, x0=-30, x1=30, z0=-28, z1=36)
    s.quad([(-30, 0.02, 5.5), (30, 0.02, 5.5), (30, 0.02, 8.5), (-30, 0.02, 8.5)], (205, 180, 125))
    rng = random.Random(8)
    for hx, hh in ((-14, 7), (0, 9), (15, 6.5)):
        s.block((hx, hh / 2, 34), (16, hh, 6), (88, 160, 92))
    s.block((-16, 18, 42), (5, 5, 1), (255, 235, 110), flat=True)
    for tx, tz in ((-9, 13), (7.5, 15), (-4, 20), (12, 22), (-14, 24)):
        s.block((tx, 2.2, tz), (1.0, 4.4, 1.0), (110, 75, 50))
        s.block((tx, 5.8, tz), (4.6, 3.2, 4.6), (62, 150, 80))
        s.block((tx, 8.0, tz), (3.2, 2.0, 3.2), (72, 165, 90))
    for x in np.arange(-14, 15, 2.0):
        s.block((x, 0.9, 12), (0.3, 1.8, 0.3), (225, 225, 230))
    s.block((0, 1.3, 12), (30, 0.2, 0.2), (225, 225, 230))
    s.block((0, 0.7, 12), (30, 0.2, 0.2), (225, 225, 230))
    s.block((-6.5, 1.3, 4.2), (3.4, 0.25, 1.1), (150, 100, 60))
    s.block((-6.5, 2.1, 4.75), (3.4, 1.0, 0.2), (150, 100, 60))
    for dx in (-1.4, 1.4):
        s.block((-6.5 + dx, 0.6, 4.2), (0.2, 1.2, 0.8), (70, 70, 80))
    for _ in range(26):
        fx, fz = rng.uniform(-14, 14), rng.uniform(9, 11)
        s.block((fx, 0.25, fz), (0.35, 0.5, 0.35),
                rng.choice([(250, 100, 130), (250, 220, 80), (255, 255, 255), (170, 120, 240)]))
    s.finalize(); return s


def scene_security_room():
    s = Scene("security_room")
    s.bg = ((28, 36, 56), (40, 50, 74)); s.shadow = (28, 32, 44)
    s.floor((52, 58, 74), (44, 49, 64))
    s.back_wall((32, 42, 64)); s.side_walls((26, 34, 54))
    rng = random.Random(3)
    for row, y in enumerate((7.3, 4.6)):
        for x in (-4.4, 0, 4.4):
            s.block((x, y, 8.6), (3.6, 2.5, 0.3), (18, 20, 28))
            s.block((x, y, 8.42), (3.3, 2.2, 0.05), (22, 82, 64), flat=True)
            for _ in range(2):
                s.block((x + rng.uniform(-1.1, 1.1), y + rng.uniform(-0.6, 0.6), 8.38),
                        (0.35, 0.4, 0.04), (255, 205, 70), flat=True)
    s.block((0, 2.3, 6.4), (10.5, 0.4, 2.5), (66, 72, 94))
    s.block((0, 1.1, 5.3), (10.5, 2.2, 0.3), (50, 55, 74))
    for i, c in enumerate(((235, 60, 60), (80, 220, 120), (250, 210, 70), (90, 160, 250), (235, 60, 60))):
        s.block((-3 + i * 1.5, 2.55, 6.0), (0.55, 0.15, 0.4), c, flat=True)
    s.block((-6.8, 1.2, 3), (1.6, 2.4, 1.6), (60, 66, 84))
    s.sit_z = 7.6
    s.finalize(); return s


def scene_store():
    s = Scene("store")
    s.bg = ((236, 240, 246), (220, 226, 236)); s.shadow = (150, 156, 170)
    s.floor((238, 238, 242), (216, 221, 230))
    s.back_wall((236, 241, 247)); s.side_walls((230, 236, 243))
    s.block((0, 3.2, 8.2), (15, 6.4, 1.6), (176, 214, 238))
    rng = random.Random(6)
    for x in np.arange(-6.4, 7, 2.1):
        s.back_rect(x - 0.9, x + 0.9, 0.6, 6.0, (205, 235, 250), z=7.38)
        for y in (1.4, 2.9, 4.4, 5.4):
            s.back_rect(x - 0.7 + rng.uniform(0, 0.2), x + 0.7, y, y + 0.7,
                        rng.choice([(230, 70, 70), (250, 200, 60), (70, 160, 230), (90, 200, 120)]), z=7.36)
    s.back_rect(-3, 3, 8.6, 10.2, (225, 40, 55), z=8.96)
    s.back_rect(-2.7, 2.7, 8.9, 9.9, (250, 250, 250), z=8.95)
    for side in (-1, 1):
        x = side * 6.9
        s.block((x, 2.8, 3.8), (1.4, 5.6, 9.0), (200, 202, 214))
        fx = x - side * 0.8
        for row in range(4):
            s.block((fx + side * 0.05, 0.9 + row * 1.3, 3.8), (0.15, 0.12, 8.6), (160, 162, 176))
            for k in range(9):
                s.block((fx - side * 0.15, 1.3 + row * 1.3, -0.2 + k * 0.95), (0.5, 0.7, 0.55),
                        rng.choice([(235, 70, 70), (250, 200, 60), (70, 160, 230), (90, 200, 120),
                                    (240, 130, 40), (170, 100, 220)]))
    s.block((-3.0, 1.5, 6.0), (3.0, 3.0, 2.2), (120, 130, 150))
    s.block((-3.0, 3.2, 6.0), (1.0, 0.8, 0.9), (50, 55, 66))
    s.finalize(); return s


SCENE_BUILDERS = {
    "bathroom": scene_bathroom, "hallway": scene_hallway, "office": scene_office,
    "classroom": scene_classroom, "kitchen": scene_kitchen, "street": scene_street,
    "park": scene_park, "security_room": scene_security_room, "store": scene_store,
}
_SCENE_CACHE = {}


def get_scene(name):
    if name not in SCENE_BUILDERS:
        name = "hallway"
    if name not in _SCENE_CACHE:
        sc = SCENE_BUILDERS[name]()
        if not hasattr(sc, "sit_z"):
            sc.sit_z = None
        _SCENE_CACHE[name] = sc
    return _SCENE_CACHE[name]


# ---------------------------------------------------------------------
# ENDLESS-RUNNER PANEL (the "gameplay" half of the split screen)
# Original blocky runner: dodges trains, jumps barriers, grabs coins.
# ---------------------------------------------------------------------

ROSTER["runner"] = dict(shirt=(40, 200, 220), pants=(60, 60, 80), skin=(240, 190, 150),
                        hair=(30, 30, 40), style="short", scale=1.0)
RUN_LANES = (-3.2, 0.0, 3.2)
RUN_SPEED = 26.0
_RUNNER = {}


def get_runner_scene():
    if "s" not in _RUNNER:
        sc = Scene("runner")
        sc.bg = ((255, 160, 95), (255, 224, 165)); sc.shadow = (70, 60, 80)
        sc.quad([(-90, 0, -30), (90, 0, -30), (90, 0, 170), (-90, 0, 170)], (88, 96, 132))
        sc.quad([(-5.4, 0.01, -30), (5.4, 0.01, -30), (5.4, 0.01, 170), (-5.4, 0.01, 170)], (128, 122, 138))
        for lx in RUN_LANES:
            for dx in (-0.8, 0.8):
                sc.quad([(lx + dx - 0.07, 0.02, -30), (lx + dx + 0.07, 0.02, -30),
                         (lx + dx + 0.07, 0.02, 170), (lx + dx - 0.07, 0.02, 170)], (195, 200, 210))
        sc.finalize()
        _RUNNER["s"] = sc
    return _RUNNER["s"]


def runner_lane_idx(seg):
    return random.Random(seg * 31 + 5).randrange(3)


def runner_lane_x(t):
    seg = int(t // 1.6)
    lane = runner_lane_idx(seg)
    prev = runner_lane_idx(seg - 1) if seg > 0 else 1
    k = ease(clamp((t - seg * 1.6) / 0.22, 0, 1))
    return RUN_LANES[prev] + (RUN_LANES[lane] - RUN_LANES[prev]) * k


def runner_obstacle(k):
    tk = 0.9 + k * 0.6
    seg = int(tk // 1.6)
    if tk - seg * 1.6 < 0.4:           # never place something mid lane-change
        return None
    rl = runner_lane_idx(seg)
    rng = random.Random(k * 7 + 1)
    kind = rng.choice(["train", "train", "barrier", "low", "coins", "coins", "coins"])
    if kind in ("train", "barrier"):
        lane = rng.choice([i for i in range(3) if i != rl])
    else:
        lane = rl
    return tk, kind, lane, rng.randrange(3)


def runner_jump(t):
    for k in range(max(0, int((t - 1.0) / 0.6) - 1), int((t + 1.0) / 0.6) + 2):
        ob = runner_obstacle(k)
        if ob and ob[1] == "low" and abs(t - ob[0]) < 0.4:
            return 2.2 * math.sin(math.pi * (t - ob[0] + 0.4) / 0.8)
    return 0.0


def runner_boxes(gt):
    S = RUN_SPEED
    out = []
    for i in range(45):                                   # track sleepers
        z = ((i * 3.0 - gt * S) % 135.0) - 8.0
        out.append(box((0, 0.06, z), (10.8, 0.12, 0.5), (72, 62, 68)))
    j0 = int(gt * S / 12.0)
    for j in range(j0, j0 + 14):                          # side buildings
        z = j * 12.0 - gt * S
        for side in (-1, 1):
            r = random.Random(j * 2 + (1 if side > 0 else 0))
            h, w = r.uniform(6, 18), r.uniform(5, 8)
            col = r.choice([(214, 120, 110), (120, 150, 215), (235, 190, 110), (150, 190, 160), (180, 130, 190)])
            out.append(box((side * (10 + w / 2), h / 2, z + 6), (w, h, 10), col))
            for wy in range(3, int(h), 3):
                for wz in (z + 3, z + 8):
                    out.append(box((side * 9.95, wy, wz), (0.15, 1.2, 1.6), (255, 228, 150), flat=True))
    kmin = max(0, int((gt - 1.6) / 0.6))
    kmax = int((gt - 0.9 + 125 / S) / 0.6) + 1
    for k in range(kmin, kmax + 1):
        ob = runner_obstacle(k)
        if not ob:
            continue
        tk, kind, lane, ci = ob
        z = (tk - gt) * S
        if z < -9 or z > 125:
            continue
        x = RUN_LANES[lane]
        if kind == "train":
            col = [(220, 60, 60), (60, 120, 220), (240, 170, 40)][ci]
            out += [box((x, 2.0, z + 4.5), (2.6, 3.4, 9), col),
                    box((x, 3.9, z + 4.5), (2.4, 0.3, 8.8), mulc(col, 0.7)),
                    box((x, 2.6, z - 0.02), (2.0, 1.1, 0.1), (200, 235, 250), flat=True),
                    box((x, 0.9, z - 0.02), (1.6, 0.35, 0.1), (255, 240, 160), flat=True)]
        elif kind == "barrier":
            for sx, c in ((-0.87, (30, 30, 36)), (0, (250, 205, 40)), (0.87, (30, 30, 36))):
                out.append(box((x + sx, 1.5, z), (0.87, 0.55, 0.45), c))
            for sx in (-1.1, 1.1):
                out.append(box((x + sx, 0.75, z), (0.25, 1.5, 0.3), (150, 150, 160)))
        elif kind == "low":
            out += [box((x, 0.5, z), (2.6, 1.0, 0.8), (240, 140, 45)),
                    box((x, 0.5, z - 0.41), (2.6, 0.3, 0.05), (255, 255, 255), flat=True)]
        else:
            for c in range(5):
                cy = 1.5 + 0.9 * math.sin(math.pi * c / 4)
                out.append(box((x, cy, z + c * 1.8), (0.85, 0.85, 0.14), (255, 212, 40),
                               R=rot_y(gt * 8 + c), flat=True))
    return out


def compose_runner(gt):
    sc = get_runner_scene()
    rx = runner_lane_x(gt)
    bx, origin = build_character("runner", rx, 0.0, 0.0, "sprint", "happy", gt, runner_jump(gt))
    dyn = {"boxes": runner_boxes(gt) + bx,
           "decals": [shadow_decal(origin, 1.4, sc.shadow)]}
    cam = Camera((rx * 0.4, 7.2, -14.0), (rx * 0.25, 1.6, 12.0), fov=62)
    return sc, cam, dyn


# ---------------------------------------------------------------------
# SHOT COMPOSITION (cast placement + camera)
# ---------------------------------------------------------------------

SLOTS = {1: [0.0], 2: [-2.3, 2.3], 3: [-3.5, 0.0, 3.5]}
SHOT_DIST = {"close": 21.0, "medium": 26.0, "wide": 31.0}


def compose_frame(scene, beat, bi, t, u, aspect=0.5625, dscale=1.0):
    """Boxes + camera for one frame of a beat. t = secs into beat, u = 0..1."""
    cast = beat["cast"]
    people = [c for c in cast if c["who"] != "pigeon"][:3]
    has_pigeon = any(c["who"] == "pigeon" for c in cast)
    xs = SLOTS.get(len(people), [0.0])
    az = [0.0, 0.32, -0.32, 0.18, -0.2, 0.28][bi % 6]
    boxes, decals = [], []
    used_x = []

    new_scene = beat.get("_prev_scene") != beat["scene"]
    for ci, (c, x) in enumerate(zip(people, xs)):
        pose, emo = c["pose"], c["emotion"]
        z = 3.2
        if pose == "sit":
            z = scene.sit_z if scene.sit_z else 3.2
            if scene.sit_z is None:
                z = 3.4
        yaw = math.pi + az * 0.7 + (0.22 if x < 0 else -0.22 if x > 0 else 0.0)
        # look toward the nearest other person (head only)
        look = 0.0
        others = [xx for xx in xs[:len(people)] if xx != x]
        if others and pose not in ("walk", "run", "cry", "panic"):
            tgt = min(others, key=lambda o: abs(o - x))
            look = 0.55 if tgt > x else -0.55
            look *= 0.6 + 0.4 * math.sin(t * 0.9 + ci)
        enter = None
        if (new_scene or c["who"] not in beat.get("_prev_who", [c["who"]])) \
                and pose in ("idle", "point", "shrug") and t < 1.1:
            side = -1 if (x < 0 or (x == 0 and bi % 2 == 0)) else 1
            enter = (side * 7.5, t / 0.9)
        bx, origin = build_character(c["who"], x, z, yaw, pose, emo, t, u, look, enter)
        boxes += bx
        decals.append(shadow_decal(origin, 1.5 * ROSTER[c["who"]]["scale"], scene.shadow))
        used_x.append(x)
        if pose == "sit":
            sx, sz = origin[0], origin[2]
            sc_ = ROSTER[c["who"]]["scale"]
            boxes.append(box((sx, 0.8 * sc_, sz - 0.1), (2.2, 0.3, 1.9), (70, 60, 90)))
            for dx in (-0.9, 0.9):
                for dz in (-0.8, 0.8):
                    boxes.append(box((sx + dx, 0.35, sz - 0.1 + dz), (0.18, 0.7, 0.18), (50, 50, 60)))
            boxes.append(box((sx, 2.0, sz - 1.0), (2.2, 2.4, 0.25), (70, 60, 90)))

    if has_pigeon:
        px = 4.6 if all(abs(x - 4.6) > 2 for x in used_x) else -4.6
        bx, origin = build_pigeon(px, 1.2, t, math.pi + az * 0.5 - (0.3 if px > 0 else -0.3))
        boxes += bx
        decals.append(shadow_decal(origin, 1.1, scene.shadow))
        used_x.append(px)

    if scene.dynamic:
        boxes += scene.dynamic(beat["_t0"] + t)

    maxx = max([abs(x) for x in used_x] + [0.0])
    d = max(SHOT_DIST.get(beat["shot"], 26.0) * dscale, (maxx + 2.6) / (aspect * 0.414))
    style = bi % 4
    if style == 0:
        d *= 1.0 - 0.14 * ease(u)                  # push in
    elif style == 1:
        d *= 0.9 + 0.1 * ease(u)                   # pull out
    elif style == 2:
        d *= 0.97
    else:
        d *= 1.0 - 0.07 * ease(u)
    tx = (sum(used_x) / len(used_x)) * 0.5 if used_x else 0.0
    target = np.array([tx, 3.6 + (0.5 if style == 2 else 0.0), 3.4])
    drift = (u - 0.5) * (0.30 if style in (2, 3) else 0.10) * (1 if bi % 2 == 0 else -1)
    ang = az + drift + math.sin(bi * 1.3 + t * 0.35) * 0.04
    lift = (0.21, 0.12, 0.30, 0.18)[style]         # camera height: low / high angles
    pos = target + np.array([math.sin(ang) * d, d * lift, -math.cos(ang) * d])
    if beat["fx"] == "shake" and t < 0.55:
        k = (1 - t / 0.55) * 0.35
        pos = pos + np.array([math.sin(t * 90) * k, math.cos(t * 77) * k, 0])
    return Camera(pos, target), {"boxes": boxes, "decals": decals}


# ---------------------------------------------------------------------
# STORY (validation + generation)
# ---------------------------------------------------------------------

def _b(text, scene, shot, cast, chip=None, trouble=None, hl=(), fx="none",
       sign=0, insert=None):
    return {
        "sign": sign,
        "insert": ({"kind": insert[0], "title": insert[1], "lines": list(insert[2:])}
                   if insert else None),
        "text": text, "scene": scene, "shot": shot,
        "cast": [{"who": w, "pose": p, "emotion": e} for w, p, e in cast],
        "chip": {"text": chip[0], "color": chip[1]} if chip else None,
        "trouble": trouble, "hl": list(hl), "fx": fx,
    }


DEMO_STORY = {
    "title": "When Is Living With Your Best Friend Too Much?",
    "description": "Five signs you and your roommate are basically the same person.",
    "summary": "Five signs a roommate and a best friend have merged into one person.",
    "topic": "roommate twins",
    "tags": ["shorts", "signs", "relatable", "funny"],
    "beats": [
        _b("When is living with your best friend too much? Chat, here are five signs.",
           "kitchen", "medium", [("hero", "shrug", "shock"), ("friend", "idle", "smug")],
           hl=["too much", "five"], fx="shake"),
        _b("Sign one: you both wake up at the exact same time. Every single day.",
           "kitchen", "medium", [("hero", "idle", "sad"), ("friend", "idle", "sad")],
           hl=["Sign one", "same time"], sign=1),
        _b("Same alarm, same snooze, same sad face. Lowkey scary.",
           "kitchen", "close", [("hero", "idle", "sad")], hl=["snooze", "scary"],
           insert=("alarm", "7:00", "BOTH PHONES RINGING")),
        _b("Sign two: you share one brain. You say the same thing, fr fr.",
           "hallway", "medium", [("hero", "point", "shock"), ("friend", "point", "shock")],
           hl=["Sign two", "one brain"], sign=2),
        _b("You text each other the same message at the same time.",
           "hallway", "close", [("hero", "idle", "shock")], hl=["same message"],
           insert=("phone", "Derek", "did u see that", "omg yes!!", "wait we said it together")),
        _b("Sign three: you split the groceries without ever talking about it.",
           "store", "wide", [("hero", "idle", "smug"), ("friend", "idle", "smug")],
           hl=["Sign three", "groceries"], sign=3),
        _b("One list, two handwritings. No cap.",
           "store", "close", [("friend", "idle", "happy")], hl=["No cap"],
           insert=("note", "WE NEED", "milk", "chips", "more chips")),
        _b("Sign four: your friends say your names as one word.",
           "park", "wide", [("hero", "celebrate", "happy"), ("friend", "celebrate", "happy"),
                            ("kid", "shrug", "neutral")],
           hl=["Sign four", "one word"], sign=4),
        _b("Sign five: it's Sunday and nobody planned anything. Both cooked.",
           "kitchen", "medium", [("hero", "cry", "sad"), ("friend", "cry", "sad")],
           hl=["Sign five", "cooked"], sign=5),
        _b("Zero plans, full aura loss. Only in Ohio.",
           "kitchen", "close", [("hero", "idle", "sad")], hl=["aura", "Ohio"],
           insert=("calendar", "SUNDAY", "12", "NO PLANS")),
        _b("Hit three of these and you two are basically one NPC. Share this with your roommate.",
           "park", "medium", [("hero", "celebrate", "happy"), ("friend", "celebrate", "happy")],
           hl=["three", "NPC", "Share"]),
    ],
}


def clean_beat(beat):
    if not isinstance(beat, dict):
        raise ValueError("A story beat was not an object.")

    text = re.sub(r"\s+", " ", str(beat.get("text", ""))).strip()
    if len(text.split()) < 3:
        raise ValueError("A story beat has no narration.")

    scene = beat.get("scene", "hallway")
    scene = scene if scene in SCENES else "hallway"
    shot = beat.get("shot", "medium")
    shot = shot if shot in SHOTS else "medium"
    fx = "shake" if beat.get("fx") in ("shake", "shock", "explosion") else "none"

    cast = []
    for item in (beat.get("cast") or [])[:4]:
        if not isinstance(item, dict):
            continue
        who = item.get("who", "hero")
        who = who if who in WHO else "hero"
        if any(c["who"] == who for c in cast):
            continue
        pose = item.get("pose", "idle")
        pose = pose if pose in POSES else "idle"
        emo = item.get("emotion", "neutral")
        emo = emo if emo in EMOTIONS else "neutral"
        cast.append({"who": who, "pose": pose, "emotion": emo})
    if not [c for c in cast if c["who"] != "pigeon"]:
        cast.insert(0, {"who": "hero", "pose": "idle", "emotion": "neutral"})
    # at most 3 people (the pigeon is extra)
    people = [c for c in cast if c["who"] != "pigeon"][:3]
    cast = people + [c for c in cast if c["who"] == "pigeon"][:1]
    if scene not in ("office", "classroom", "security_room"):
        for c in cast:
            if c["pose"] == "sit":
                c["pose"] = "idle"

    chip = None
    raw = beat.get("chip")
    if isinstance(raw, dict) and str(raw.get("text", "")).strip():
        color = raw.get("color", "purple")
        chip = {
            "text": re.sub(r"\s+", " ", str(raw["text"])).strip().upper()[:30],
            "color": color if color in CHIP_COLORS else "purple",
        }

    trouble = beat.get("trouble")
    trouble = trouble if trouble in TROUBLE else None

    hl = beat.get("hl") or []
    hl = [str(x) for x in hl][:6] if isinstance(hl, list) else []

    try:
        sign = max(0, min(9, int(beat.get("sign") or 0)))
    except (TypeError, ValueError):
        sign = 0
    insert = None
    raw = beat.get("insert")
    if isinstance(raw, dict) and raw.get("kind") in INSERT_KINDS:
        lines = raw.get("lines") or []
        lines = [re.sub(r"\s+", " ", str(x)).strip()[:34] for x in lines
                 if str(x).strip()][:4] if isinstance(lines, list) else []
        insert = {"kind": raw["kind"],
                  "title": re.sub(r"\s+", " ", str(raw.get("title", ""))).strip()[:28],
                  "lines": lines}

    return {
        "text": text, "scene": scene, "shot": shot, "cast": cast,
        "chip": chip, "trouble": trouble, "hl": hl, "fx": fx,
        "sign": sign, "insert": insert,
    }


def write_episode(episode_number, state):
    if CFG["demo"]:
        data = json.loads(json.dumps(DEMO_STORY))
        data["beats"] = [clean_beat(b) for b in data["beats"]]
        return data

    trends = trending_topics()
    recent = state.get("topics", [])[-40:]
    twist = random.choice(trends) if trends else "an everyday problem"
    area = random.choice([
        "school", "a job", "driving and traffic", "shopping", "social media",
        "neighbors", "airports and travel", "restaurants", "sports", "online gaming",
        "borrowing things", "public transport", "a group chat", "a family trip",
    ])

    signs_prompt = f"""
You write viral 35-50 second vertical Shorts in the "SIGNS" listicle style: a
relatable hook question, then a numbered checklist of signs, each shown with a
cartoon scene or a close-up of a phone text, sticky note, alarm, laptop or
calendar. It is fast, funny, a little chaotic, and speaks straight to "you".

Topic area for this episode: {area}
Loose trend inspiration (optional): {twist}
Do NOT repeat these topics: {json.dumps(recent[-30:])}

STRUCTURE (11 to 15 beats)
1. HOOK beat (sign 0): a question the viewer wants answered, like "When is
   living with your best friend too much?" or "Signs your group chat is
   running your life." End it with "here are five signs".
2. For each sign number k from 1 to N (N is 4, 5 or 6) write a beat that STARTS
   with "Sign one:", "Sign two:" and so on, and set "sign": k on that beat.
   Directly after it, add a short follow-up beat (sign 0) that makes the sign
   funnier. Give at least 3 of those follow-ups an "insert" close-up.
3. The LAST sign is the twist or funniest one.
4. FINAL beat (sign 0): a punchline plus a share line ("Share this with the
   friend who...").

VOICE
- Second person: "you", "your". Relatable everyday situations the viewer has
  lived. Light modern slang (cooked, no cap, fr, aura, NPC, lowkey, locked in,
  only in Ohio, sigma, mid), one or two per beat, never forced.
- Spoken lines are 6 to 14 words. Total spoken words 90 to 160.
- Family friendly, no swearing, no real people, no brands, no advice.
- "hl" lists 1-3 words from the beat that pop in the captions.

INSERT CLOSE-UPS (the "insert" field, only on follow-up beats)
- {{"kind":"phone","title":"Derek","lines":["did u see that","omg yes","same brain"]}}
  = a text chat. Lines alternate: incoming, outgoing, incoming...
- {{"kind":"note","title":"WE NEED","lines":["milk","chips"]}}   = sticky note
- {{"kind":"alarm","title":"7:00","lines":["BOTH PHONES RINGING"]}}
- {{"kind":"laptop","title":"Essay","lines":["Chapter one is","pure panic"]}}
- {{"kind":"calendar","title":"FRIDAY","lines":["13","NO PLANS"]}}
Keep every title and line short (under 30 characters).

Return ONLY valid JSON in exactly this shape:
{{
 "title": "WHEN IS LIVING WITH YOUR BEST FRIEND TOO MUCH?",
 "description": "one sentence",
 "summary": "one sentence plot summary",
 "topic": "short topic label",
 "tags": ["shorts","relatable","funny"],
 "beats": [
  {{"text": "Sign one: you both wake up at the same time.",
    "scene": "kitchen", "shot": "medium",
    "cast": [{{"who":"hero","pose":"idle","emotion":"sad"}},
             {{"who":"friend","pose":"idle","emotion":"sad"}}],
    "hl": ["Sign one","same time"], "fx": "none", "sign": 1, "insert": null}}
 ]
}}
Use only these values:
scene: {SCENES}
shot: {SHOTS}
who: {WHO}  (hero = "you")
pose: {POSES}
emotion: {EMOTIONS}
fx: none or shake (only on the hook beat)
Use "sit" only in office, classroom or security_room. Vary scenes and who is on
screen. Use at most 3 people per beat.
"""

    prompt = f"""
You write viral 35-50 second vertical YouTube Shorts in the modern "brainrot
storytime" style: a hyped, chaotic, chronically-online narrator tells a funny
story about someone's dumb decision and what could realistically happen next,
over a split screen of endless-runner gameplay.

STRUCTURE (AT LEAST 12 beats, up to 16; short lines, one idea each)
1. HOOK beat: states the dumb move and that the hero is cooked / got caught.
2. Beats that show how it started, then how it escalated.
3. Several beats that explain the real-world consequences ("In many places...",
   "Possible results are...", "It can count as..."). Each consequence gets a
   "chip" (a short tag, 2-4 words, UPPERCASE) that stacks on screen.
4. A TWIST beat ("But here's the twist...") showing what he should have done.
5. A funny last beat that ends with a question for the comments ("...comment
   below, chat").
Topic area for this episode: {area}
Loose trend inspiration (optional): {twist}
Do NOT repeat these topics: {json.dumps(recent[-30:])}

VOICE AND SLANG (this is the whole point: it must sound like brainrot)
- The narrator is a hyped, chronically online Gen Z voice. Every line has 2 or
  3 slang terms or meme phrases, but it must still be easy to follow out loud.
- Address the audience as "chat" or "bro" and react live ("chat, he did NOT").
- Slang and meme phrases to mix freely: cooked, so cooked, locked in, lock in,
  aura, "-5000 aura", "plus aura", no cap, fr fr, lowkey, highkey, bro, bruh,
  chat is this real, sigma, mid, NPC, slop, brain AFK, brainrot, "only in Ohio",
  six seven, huss, skibidi, rizz, mewing, fanum tax, sus, sheesh, delulu,
  "it's giving", "bro thought he was him", "bro really said", "not the ...",
  "ate and left no crumbs", "huge L", "massive W", "side quest", "main
  character energy", "I'm not even lying", "the audacity", "ratio".
- Open with a hook like "Bro really thought he could..." or "Chat, this man
  is so cooked." Narrate dumb decisions with dramatic, deadpan confidence.
- Chips are a mix of rule tags with a trailing * ("FAKING SICK = LYING*") and
  brainrot tags ("-5000 AURA", "+500 AURA", "COOKED", "NPC MOVE", "ONLY IN OHIO",
  "HUGE L", "BRAIN AFK").
- "hl" should pick the slang words so they pop in the captions.
- Never use sexual, hateful or insulting slang. No swearing.

RULES
- The hero has a first name. Spoken lines are 7 to 13 words, punchy, funny.
- Total spoken words: 85 to 140.
- Be accurate and hedged about real-world rules: say "can", "may", "often",
  "in many places". Never give medical or legal advice and never claim a
  specific law or penalty.
- Family friendly. No real people, brands, or copyrighted characters.
- Vary scenes, shots, poses and who is on screen. Use "sit" only in office,
  classroom or security_room. A silent pigeon may appear in 2-3 beats.
- "hl" lists 1-3 important or funny words from that beat's text (they pop up
  big and yellow in the captions).
- "trouble" is big, some, or none (shown as COOKED, MID, LOCKED IN). Start
  with big, move to some around the twist, end with none. Only set it when it
  changes.

Return ONLY valid JSON in exactly this shape:
{{
 "title": "SHORT TITLE LIKE THE FAKE SICK DAY",
 "description": "one sentence",
 "summary": "one sentence plot summary",
 "topic": "short topic label",
 "tags": ["shorts","storytime","funny"],
 "beats": [
  {{
   "text": "Jake called in sick, then posted his whole day at the park.",
   "scene": "office",
   "shot": "medium",
   "cast": [{{"who":"boss","pose":"point","emotion":"angry"}},
            {{"who":"hero","pose":"shock","emotion":"scared"}}],
   "chip": {{"text":"JAKE GOT CAUGHT","color":"red"}},
   "trouble": "big",
   "hl": ["sick","park"],
   "fx": "shake"
  }}
 ]
}}
Use only these values:
scene: {SCENES}
shot: {SHOTS}
who: {WHO}  (hero = the main character)
pose: {POSES}
emotion: {EMOTIONS}
chip color: {sorted(CHIP_COLORS)}
fx: none or shake
"""

    use_signs = CFG["format"] != "story"
    if use_signs:
        prompt = signs_prompt
    last_error = None
    for attempt in range(4):
        try:
            data = llm_json(prompt)
            beats = [clean_beat(x) for x in data.get("beats", [])]
            words = sum(len(x["text"].split()) for x in beats)
            chips = sum(1 for x in beats if x["chip"])

            if not 8 <= len(beats) <= 20 or not 60 <= words <= 190:
                raise ValueError(f"Story length outside limits: {len(beats)} beats, {words} words")
            if use_signs:
                if max([x["sign"] for x in beats] + [0]) < 3:
                    raise ValueError("Signs video needs at least 3 numbered signs.")
            elif chips < 2:
                raise ValueError("Story needs at least 2 consequence chips.")

            data["beats"] = beats
            data["title"] = str(data.get("title", f"Dumb Moves #{episode_number}"))[:60]
            data["topic"] = str(data.get("topic", area))[:100]
            data["summary"] = str(data.get("summary", ""))[:300]
            data["description"] = str(data.get("description", ""))[:500]
            data["tags"] = [str(t)[:30] for t in data.get("tags", [])[:12]]
            return data

        except Exception as exc:
            last_error = exc
            print(f"Story attempt {attempt + 1} failed:", exc)
            if "429" in str(exc):
                break  # quota is gone; more retries only waste requests
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Could not create a valid story: {last_error}")


# ---------------------------------------------------------------------
# NARRATION (natural neural voice, Piper as fallback)
# ---------------------------------------------------------------------

TRIM = (
    "silenceremove=start_periods=1:start_silence=0.02:start_threshold=-48dB,"
    "areverse,"
    "silenceremove=start_periods=1:start_silence=0.02:start_threshold=-48dB,"
    "areverse,"
    "highpass=f=70,apad=pad_dur=0.2"
)


def synthesize_edge(text, out_path):
    """Microsoft neural voice through the free edge-tts package."""
    last = ""
    for attempt in range(3):
        out_path.unlink(missing_ok=True)
        result = subprocess.run(
            [sys.executable, "-m", "edge_tts",
             f"--voice={CFG['voice']}", f"--rate={CFG['voice_rate']}",
             f"--pitch={CFG['voice_pitch']}", f"--text={text}",
             f"--write-media={out_path}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90,
        )
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 800:
            return
        last = (result.stderr or result.stdout)[-600:]
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("edge-tts failed: " + last)


SAMPLE_VOICES = [
    ("en-US-AndrewNeural", "+10%", "-2Hz"), ("en-US-BrianNeural", "+10%", "-2Hz"),
    ("en-US-GuyNeural", "+8%", "-2Hz"), ("en-US-ChristopherNeural", "+10%", "+0Hz"),
    ("en-US-EricNeural", "+10%", "-2Hz"), ("en-US-RogerNeural", "+10%", "+0Hz"),
    ("en-US-SteffanNeural", "+10%", "-2Hz"), ("en-GB-RyanNeural", "+10%", "+0Hz"),
    ("en-AU-WilliamNeural", "+10%", "+0Hz"), ("en-US-AndrewMultilingualNeural", "+10%", "-2Hz"),
    ("en-US-BrianMultilingualNeural", "+10%", "-2Hz"),
]


def make_voice_samples():
    """VOICE=samples -> one short clip per voice so you can listen and pick."""
    d = OUT_DAY / "voice_samples"
    d.mkdir(parents=True, exist_ok=True)
    line = ("Bro walked into the bathroom, no cap, and the cop was already there. "
            "He is so cooked. Only in Ohio.")
    old = (CFG["voice"], CFG["voice_rate"], CFG["voice_pitch"])
    for name, rate, pitch in SAMPLE_VOICES:
        CFG["voice"], CFG["voice_rate"], CFG["voice_pitch"] = name, rate, pitch
        try:
            synthesize_edge(line, d / f"{name}.mp3")
            print("sample ok:", name)
        except Exception as exc:
            print("sample failed:", name, exc)
    CFG["voice"], CFG["voice_rate"], CFG["voice_pitch"] = old


def synthesize_piper(text, out_path):
    binary = CFG["piper_bin"]
    if not shutil.which(binary) and not Path(binary).exists():
        raise RuntimeError(f"Piper executable not found: {binary}")
    model = Path(CFG["piper_model"])
    if not model.exists():
        raise RuntimeError(f"Piper model not found: {model}")
    last = ""
    for extra in (["--length_scale", "0.94"], ["--length-scale", "0.94"], []):
        out_path.unlink(missing_ok=True)
        result = subprocess.run(
            [binary, "--model", str(model), *extra, "--output_file", str(out_path)],
            input=text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            return
        last = result.stderr[-800:]
    raise RuntimeError("Piper failed: " + last)


_TTS_STATE = {"edge_ok": True, "announced": False}


def synthesize(text, raw_path):
    """Write speech for one line to raw_path. Neural voice first."""
    if _TTS_STATE["edge_ok"]:
        try:
            synthesize_edge(text, raw_path)
            if not _TTS_STATE["announced"]:
                print("Narration voice:", CFG["voice"])
                _TTS_STATE["announced"] = True
            return
        except Exception as exc:
            print("Neural voice unavailable, falling back to Piper:", str(exc)[:300])
            _TTS_STATE["edge_ok"] = False
    synthesize_piper(text, raw_path)


def voice_beats(beats, folder):
    clips, durations = [], []
    for index, beat in enumerate(beats):
        raw = folder / f"voice_raw_{index:02d}.mp3"
        synthesize(beat["text"], raw)
        clip = folder / f"voice_{index:02d}.wav"
        run(["ffmpeg", "-y", "-i", str(raw), "-af", TRIM,
             "-ar", "24000", "-ac", "1", str(clip)])
        raw.unlink(missing_ok=True)
        clips.append(clip)
        durations.append(probe_duration(clip))

    concat = folder / "voice_list.txt"
    concat.write_text("".join(f"file '{p.name}'\n" for p in clips), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat.name,
         "-c:a", "pcm_s16le", "voice.wav"], cwd=folder)
    return durations


# ---------------------------------------------------------------------
# TIMELINE (beat times, word times, caption chunks, chips, sound effects)
# ---------------------------------------------------------------------

_FONTS = {}


def font(size):
    if size in _FONTS:
        return _FONTS[size]
    candidates = [
        ROOT / "fonts" / "Poppins-Bold.ttf",
        "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    f = None
    for c in candidates:
        if Path(c).exists():
            f = ImageFont.truetype(str(c), size)
            break
    if f is None:
        try:
            f = ImageFont.load_default(size)
        except TypeError:
            f = ImageFont.load_default()
    _FONTS[size] = f
    return f


def wkey(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def wrap_words(words, fnt, maxw):
    lines, cur = [], []
    space = fnt.getlength(" ") + 16
    for w in words:
        trial = cur + [w]
        width = sum(fnt.getlength(x) for x in trial) + space * (len(trial) - 1)
        if cur and width > maxw:
            lines.append(cur)
            cur = [w]
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def chunk_words(words, fnt, maxw=780, max_words=7):
    chunks, cur = [], []
    for w in words:
        trial = cur + [w]
        if cur and (len(trial) > max_words or len(wrap_words(trial, fnt, maxw)) > 2):
            chunks.append(cur)
            cur = [w]
        else:
            cur = trial
        if re.search(r"[.!?,;:]$", w) and len(cur) >= 3:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def build_timeline(beats, durations):
    """Fill beats with start times / durations and build HUD + sound timing."""
    br = CFG["brainrot"]
    t = 0.0
    trouble = "big"
    chips, sfx, captions, punches = [], [], [], []
    fnt = font(72)
    n_signs = max([b.get("sign", 0) for b in beats] + [0])
    signs_mode = n_signs >= 3
    sign_t, cur_sign = [], 0

    for i, (beat, dur) in enumerate(zip(beats, durations)):
        beat["_t0"] = t
        pv = beats[i - 1] if i else None
        beat["_prev_scene"] = pv["scene"] if pv else None
        beat["_prev_who"] = [c["who"] for c in pv["cast"]] if pv else []
        beat["_dur"] = max(1.3 if br else 1.7, dur) + (0.45 if i == len(beats) - 1 else 0.0)
        speech = max(0.4, dur - 0.2)
        if br and i == 0:
            beat["fx"] = "shake"               # chaotic hook frame

        if signs_mode:
            if beat.get("sign", 0) > cur_sign:
                cur_sign = beat["sign"]
                sign_t.append((t, cur_sign))
                sfx.append((t + 0.05, "ding"))
            beat["_sign"] = cur_sign
            if beat.get("insert"):
                sfx.append((t + 0.02, "whoosh"))
                sfx.append((t + 0.3, "pop"))
            beat["chip"] = None
            beat["trouble"] = None
        if beat["trouble"] and beat["trouble"] != trouble:
            if beat["trouble"] == "some":
                sfx.append((t + 0.05, "horn"))
            elif beat["trouble"] == "none":
                sfx.append((t + 0.05, "ding"))
        if beat["trouble"]:
            trouble = beat["trouble"]
        beat["_trouble"] = trouble

        if br and i > 0:
            sfx.append((max(0.0, t - 0.06), "whoosh"))
        if beat["chip"]:
            ct = t + min(0.5, speech * 0.25)
            chips.append({"text": beat["chip"]["text"],
                          "color": CHIP_COLORS[beat["chip"]["color"]], "t0": ct})
            sfx.append((ct, "pop"))
        if beat["fx"] == "shake" or any(c["pose"] in ("shock", "fall") for c in beat["cast"]):
            sfx.append((t + 0.02, "boom"))

        words = beat["text"].split()
        weights = [max(1, len(re.sub(r"\W", "", w))) + 1 for w in words]
        total = sum(weights)
        hl = set()
        for phrase in beat["hl"]:
            hl.update(wkey(x) for x in phrase.split())
        hl.discard("")

        cursor = t
        timed = []
        for w, wt in zip(words, weights):
            span = speech * wt / total
            timed.append((cursor, cursor + span, w))
            cursor += span

        idx = 0
        for chunk in chunk_words(words, fnt):
            entries = [(w, wkey(w) in hl, timed[idx + j][0]) for j, w in enumerate(chunk)]
            idx += len(chunk)
            captions.append([entries[0][2], None, entries])
        n_punch = 0
        for (w, is_hl, ws) in [(timed[k][2], wkey(timed[k][2]) in hl, timed[k][0]) for k in range(len(timed))]:
            if is_hl and n_punch < 2:
                punches.append(ws)
                n_punch += 1
        t += beat["_dur"]

    for k, cap in enumerate(captions):
        cap[1] = captions[k + 1][0] if k + 1 < len(captions) else t
    return {"total": t, "chips": chips, "sfx": sfx, "captions": captions, "punches": punches,
            "signs": n_signs if signs_mode else 0, "sign_t": sign_t}


# ---------------------------------------------------------------------
# HUD (title, trouble bar, stacking chips, captions)
# ---------------------------------------------------------------------

def stroke_text(draw, xy, text, fnt, fill, stroke=6, anchor="la"):
    draw.text(xy, text, font=fnt, fill=fill, stroke_width=stroke,
              stroke_fill=(16, 16, 28), anchor=anchor)


def build_title_blocks(title, episode):
    blocks = {}
    title = title.upper()
    labels = TROUBLE_BR
    for key, (label, colour) in labels.items():
        img = Image.new("RGBA", (TW, 330), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        stroke_text(d, (64, 6), f"{SERIES_NAME}  •  EP {episode}", font(30),
                    (255, 225, 120), 5)
        size = 78
        while size > 46 and font(size).getlength(title) > TW - 128:
            size -= 4
        lines = [title]
        if font(size).getlength(title) > TW - 128:
            words = title.split()
            half = max(1, len(words) // 2)
            lines = [" ".join(words[:half]), " ".join(words[half:])]
            size = 62
        y = 52
        for line in lines:
            stroke_text(d, (64, y), line, font(size), (255, 255, 255), 5)
            y += int(size * 1.2)
        by = y + 10
        d.rounded_rectangle((64, by, 64 + 600, by + 54), radius=27, fill=colour,
                            outline=(255, 255, 255), width=4)
        d.text((64 + 300, by + 28), label, font=font(30), fill=(255, 255, 255), anchor="mm")
        blocks[key] = (img, by + 54)
    return blocks


def build_chip(text, colour):
    fnt = font(40)
    w = int(fnt.getlength(text)) + 90
    img = Image.new("RGBA", (w + 16, 100), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((12, 14, w + 8, 82), radius=34, fill=(0, 0, 0, 110))
    d.rounded_rectangle((8, 8, w + 4, 74), radius=34, fill=colour,
                        outline=(255, 255, 255), width=4)
    d.text((8 + (w - 4) // 2 + 2, 42), text, font=fnt, fill=(255, 255, 255), anchor="mm")
    return img


def build_caption(words, upto):
    """Kinetic caption: words = [(word, highlight, start)]. Only words up to
    index `upto` are drawn, but the layout is fixed so nothing jumps around.
    Highlighted words are bigger and yellow."""
    base, big = font(76), font(92)
    maxw = 900
    space = base.getlength(" ") + 18

    def wf(h):
        return big if h else base

    lines, cur, curw = [], [], 0.0
    for i, (w, h, _) in enumerate(words):
        ww = wf(h).getlength(w)
        add = ww + (space if cur else 0)
        if cur and curw + add > maxw:
            lines.append(cur)
            cur, curw, add = [], 0.0, ww
        cur.append((i, w, h, ww))
        curw += add
    lines.append(cur)

    line_h = 120
    img = Image.new("RGBA", (TW, line_h * len(lines) + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for li, line in enumerate(lines):
        total = sum(x[3] for x in line) + space * (len(line) - 1)
        x = (TW - total) / 2
        y = li * line_h + 100
        for i, w, h, ww in line:
            if i <= upto:
                col = (255, 226, 60) if h else (255, 255, 255)
                d.text((x, y), w, font=wf(h), fill=col, stroke_width=10,
                       stroke_fill=(16, 16, 28), anchor="ls")
            x += ww + space
    return img


def build_footer():
    img = Image.new("RGBA", (TW, 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((TW / 2, 30), "Fictional comedy for fun. Not advice.",
           font=font(26), fill=(235, 235, 240), anchor="mm",
           stroke_width=4, stroke_fill=(16, 16, 28))
    return img


def build_sign_bars(title, n):
    """Header images: k=0 shows the hook question, k>=1 shows 'SIGN k OF n'."""
    bars = []
    for k in range(n + 1):
        img = Image.new("RGBA", (TW, 330), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if k == 0:
            fs = 64
            lines = _wrap_text(title, font(fs), 800)
            while len(lines) > 3 and fs > 44:
                fs -= 6
                lines = _wrap_text(title, font(fs), 800)
            lh = int(fs * 1.22)
            h = lh * len(lines) + 44
            d.rounded_rectangle((70, 40, TW - 70, 40 + h), radius=36,
                                fill=(46, 184, 96), outline=(255, 255, 255), width=6)
            for i, line in enumerate(lines):
                stroke_text(d, (TW // 2, 40 + 22 + lh * i + lh // 2), line, font(fs),
                            (255, 255, 255), 5, "mm")
        bw, gap = min(96, (TW - 160) // n - 12), 12
        total = n * bw + (n - 1) * gap
        x0 = (TW - total) // 2
        by = 8 if k == 0 else 12
        if k == 0:
            by = 40 + 0
        for i in range(n):
            x = x0 + i * (bw + gap)
            if k == 0:
                continue
            done = i < k
            d.rounded_rectangle((x, by, x + bw, by + 38), radius=10,
                                fill=(46, 184, 96) if done else (255, 255, 255),
                                outline=(16, 16, 28), width=4)
            if done:
                cx, cy = x + bw // 2, by + 19
                d.line([(cx - 14, cy), (cx - 4, cy + 10), (cx + 16, cy - 10)],
                       fill=(255, 255, 255), width=7, joint="curve")
        if k > 0:
            label = f"SIGN {k} OF {n}"
            fnt = font(58)
            w = int(fnt.getlength(label)) + 90
            d.rounded_rectangle(((TW - w) // 2, by + 62, (TW + w) // 2, by + 62 + 88),
                                radius=30, fill=(252, 250, 236), outline=(16, 16, 28), width=5)
            d.text((TW // 2, by + 62 + 46), label, font=fnt, fill=(24, 32, 44), anchor="mm")
        bars.append(img)
    return bars


def _wrap_text(text, fnt, maxw):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if cur and fnt.getlength(trial) > maxw:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


INSERT_BG = [(190, 226, 156), (168, 204, 236), (250, 214, 150), (240, 186, 202),
             (204, 192, 236), (255, 226, 140)]
_STRIPES = {}


def _insert_bg(idx):
    col = INSERT_BG[idx % len(INSERT_BG)]
    if col not in _STRIPES:
        img = Image.new("RGB", (TW, TH), col)
        d = ImageDraw.Draw(img)
        light = tuple(min(255, c + 16) for c in col)
        for x in range(-TH, TW, 220):
            d.polygon([(x, TH), (x + 90, TH), (x + 90 + TH, 0), (x + TH, 0)], fill=light)
        _STRIPES[col] = img
    return _STRIPES[col].copy()


def _card_phone(ins, t, dur):
    cv = Image.new("RGBA", (900, 660), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)
    d.rounded_rectangle((250, 0, 650, 660), radius=56, fill=(28, 30, 46))
    d.rounded_rectangle((264, 14, 636, 646), radius=44, fill=(250, 250, 252))
    d.rounded_rectangle((264, 14, 636, 118), radius=44, fill=(236, 238, 246))
    d.rectangle((264, 70, 636, 118), fill=(236, 238, 246))
    d.ellipse((284, 34, 344, 94), fill=(222, 70, 80))
    name = ins["title"] or "Chat"
    d.text((314, 64), name[:1].upper(), font=font(34), fill=(255, 255, 255), anchor="mm")
    d.text((362, 64), name, font=font(34), fill=(30, 34, 50), anchor="lm")
    lines = ins["lines"] or ["..."]
    n = len(lines)
    step = min(0.8, max(0.3, dur * 0.6 / n))
    y = 138
    fnt = font(28)
    for i, line in enumerate(lines):
        r = 0.3 + i * step
        wrapped = _wrap_text(line, fnt, 270)
        bh = 24 + 36 * len(wrapped)
        mine = i % 2 == 1
        if t < r:
            if t > r - 0.4 and not mine:
                d.rounded_rectangle((284, y, 384, y + 50), radius=25, fill=(230, 232, 238))
                for j in range(3):
                    dy = abs(math.sin(t * 9 + j)) * 6
                    d.ellipse((304 + j * 28, y + 21 - dy, 316 + j * 28, y + 33 - dy),
                              fill=(130, 134, 150))
            break
        bw = int(max(fnt.getlength(x) for x in wrapped)) + 36
        if mine:
            x0 = 616 - bw
            d.rounded_rectangle((x0, y, 616, y + bh), radius=26, fill=(40, 105, 225))
            col = (255, 255, 255)
        else:
            x0 = 284
            d.rounded_rectangle((x0, y, x0 + bw, y + bh), radius=26, fill=(230, 232, 238))
            col = (30, 34, 50)
        for j, wl in enumerate(wrapped):
            d.text((x0 + 18, y + 12 + 36 * j), wl, font=fnt, fill=col)
        y += bh + 18
    return cv


def _card_note(ins, t, dur):
    cv = Image.new("RGBA", (900, 660), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)
    x0, y0, sz = 170, 40, 560
    d.rectangle((x0 + 12, y0 + 16, x0 + sz + 12, y0 + sz + 16), fill=(0, 0, 0, 60))
    d.rectangle((x0, y0, x0 + sz, y0 + sz), fill=(255, 226, 92))
    d.rectangle((x0 + 190, y0 - 22, x0 + 370, y0 + 34), fill=(255, 255, 255, 170))
    y = y0 + 70
    for ln in _wrap_text(ins["title"], font(66), sz - 80)[:2]:
        d.text((x0 + 40, y), ln, font=font(66), fill=(40, 36, 50))
        y += 82
    y += 10
    k = max(1, len(ins["lines"]))
    for i, line in enumerate(ins["lines"]):
        if t < 0.3 + i * min(0.6, dur * 0.5 / k):
            break
        d.text((x0 + 40, y), "- " + line, font=font(46), fill=(60, 52, 70))
        y += 64
    return cv.rotate(-4, resample=Image.BICUBIC)


def _card_alarm(ins, t, dur):
    cv = Image.new("RGBA", (900, 660), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)
    d.rounded_rectangle((240, 0, 660, 660), radius=60, fill=(22, 24, 44))
    d.text((450, 250), ins["title"] or "7:00", font=font(140), fill=(255, 255, 255), anchor="mm")
    cx, cy = 450, 440
    d.ellipse((cx - 56, cy - 62, cx + 56, cy + 40), fill=(255, 205, 50))
    d.rectangle((cx - 66, cy + 26, cx + 66, cy + 46), fill=(255, 205, 50))
    d.ellipse((cx - 14, cy + 44, cx + 14, cy + 72), fill=(255, 205, 50))
    a = int(120 + 120 * abs(math.sin(t * 14)))
    for sx in (-1, 1):
        for r in (84, 112):
            bx = cx + sx * r
            d.arc((bx - 26, cy - 60, bx + 26, cy + 10), 270 if sx > 0 else 90,
                  90 if sx > 0 else 270, fill=(255, 205, 50, a), width=8)
    if ins["lines"]:
        d.rounded_rectangle((270, 560, 630, 625), radius=30, fill=(122, 74, 210))
        d.text((450, 593), ins["lines"][0][:24], font=font(26), fill=(255, 255, 255), anchor="mm")
    return cv.rotate(math.sin(t * 30) * 3.0, resample=Image.BICUBIC)


def _card_laptop(ins, t, dur):
    cv = Image.new("RGBA", (900, 660), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)
    d.rounded_rectangle((60, 20, 840, 540), radius=32, fill=(40, 44, 60))
    d.rounded_rectangle((84, 44, 816, 516), radius=14, fill=(250, 250, 252))
    d.rectangle((84, 44, 816, 100), fill=(226, 230, 242))
    d.text((112, 72), ins["title"] or "Document", font=font(32), fill=(30, 34, 50), anchor="lm")
    for i, c in enumerate(((235, 90, 90), (240, 190, 60), (80, 190, 110))):
        d.ellipse((730 + i * 26, 62, 748 + i * 26, 80), fill=c)
    d.rounded_rectangle((30, 548, 870, 590), radius=20, fill=(176, 182, 200))
    text = " ".join(ins["lines"]) or "..."
    shown = int(len(text) * clamp(t / max(0.8, dur * 0.7), 0, 1))
    y, fnt, left = 130, font(40), text[:shown]
    for ln in _wrap_text(left + ("|" if int(t * 3) % 2 == 0 else " "), fnt, 680)[:7]:
        d.text((114, y), ln, font=fnt, fill=(30, 34, 50))
        y += 56
    return cv


def _card_calendar(ins, t, dur):
    cv = Image.new("RGBA", (900, 660), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)
    d.rounded_rectangle((170, 0, 730, 640), radius=44, fill=(255, 255, 255))
    d.rounded_rectangle((170, 0, 730, 150), radius=44, fill=(226, 52, 62))
    d.rectangle((170, 90, 730, 150), fill=(226, 52, 62))
    d.text((450, 78), ins["title"][:12] or "FRIDAY", font=font(70), fill=(255, 255, 255), anchor="mm")
    day = ins["lines"][0] if ins["lines"] else ""
    d.text((450, 340), day[:3], font=font(240), fill=(30, 34, 50), anchor="mm")
    if len(ins["lines"]) > 1 and t > 0.5:
        k = ease_out_back(min(1.0, (t - 0.5) / 0.3))
        txt = ins["lines"][1][:16]
        w = int(font(46).getlength(txt)) + 60
        lay = Image.new("RGBA", (w + 20, 110), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        ld.rounded_rectangle((8, 8, w + 12, 100), radius=22, outline=(226, 52, 62), width=8,
                             fill=(255, 255, 255, 230))
        ld.text(((w + 20) // 2, 54), txt, font=font(46), fill=(226, 52, 62), anchor="mm")
        lay = lay.rotate(-6, expand=True, resample=Image.BICUBIC)
        kk = max(0.05, k)
        lay = lay.resize((max(1, int(lay.width * kk)), max(1, int(lay.height * kk))))
        cv.paste(lay, (450 - lay.width // 2, 520 - lay.height // 2), lay)
    return cv


_CARDS = {"phone": _card_phone, "note": _card_note, "alarm": _card_alarm,
          "laptop": _card_laptop, "calendar": _card_calendar}


def insert_frame(ins, t, dur, idx):
    """Full-frame close-up (phone chat, sticky note, alarm...) on a pastel bg."""
    img = _insert_bg(idx)
    cv = _CARDS[ins["kind"]](ins, t, dur)
    k = max(0.05, ease_out_back(min(1.0, t / 0.35)))
    sc = (1.4 if ins['kind'] == 'phone' else 1.15) * k
    cv = cv.resize((max(1, int(cv.width * sc)), max(1, int(cv.height * sc))), Image.BILINEAR)
    cx, cy = TW // 2, 800 + int(math.sin(t * 2.2) * 9)
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse((cx - int(cv.width * 0.32), cy + int(cv.height * 0.42),
                                cx + int(cv.width * 0.32), cy + int(cv.height * 0.5)),
                               fill=(0, 0, 0, 55))
    img.paste(sh, (0, 0), sh)
    img.paste(cv, (cx - cv.width // 2, cy - cv.height // 2), cv)
    return img


class Hud:
    def __init__(self, title, episode, timeline):
        self.br = CFG["brainrot"]
        self.n_signs = timeline.get("signs", 0)
        self.sign_t = timeline.get("sign_t", [])
        self.sign_bars = build_sign_bars(title, self.n_signs) if self.n_signs else None
        self.blocks = build_title_blocks(title, episode)
        self.chips = timeline["chips"]
        self.chip_imgs = [build_chip(c["text"], c["color"]) for c in self.chips]
        self.caps = timeline["captions"]
        self.footer = build_footer()
        self._cache = {}

    def caption_img(self, k, upto):
        key = (k, upto)
        if key not in self._cache:
            if len(self._cache) > 24:
                self._cache.clear()
            self._cache[key] = build_caption(self.caps[k][2], upto)
        return self._cache[key]

    def draw(self, img, gt, trouble):
        if self.sign_bars:
            k, t0 = 0, 0.0
            for st, kk in self.sign_t:
                if st <= gt:
                    k, t0 = kk, st
            bar = self.sign_bars[min(k, self.n_signs)]
            age = gt - t0
            if k > 0 and age < 0.3:
                sc = max(0.05, ease_out_back(age / 0.3))
                bar = bar.resize((max(1, int(bar.width * sc)), max(1, int(bar.height * sc))),
                                 Image.BILINEAR)
            img.paste(bar, ((TW - bar.width) // 2, 150 + (330 - bar.height) // 2), bar)
            bottom = 330
            self._draw_caption(img, gt)
            img.paste(self.footer, (0, 1760), self.footer)
            return
        block, bottom = self.blocks[trouble]
        img.paste(block, (0, 150), block)

        max_chips, step = (4, 76) if self.br else (5, 86)
        shown = [i for i, c in enumerate(self.chips) if c["t0"] <= gt][-max_chips:]
        y = 150 + bottom + (14 if self.br else 34)
        for i in shown:
            chip = self.chip_imgs[i]
            age = gt - self.chips[i]["t0"]
            cy = y
            if age < 0.32:
                k = max(0.05, ease_out_back(age / 0.32))
                chip = chip.resize((max(1, int(chip.width * k)), max(1, int(chip.height * k))),
                                   Image.BILINEAR)
                cy = y + (100 - chip.height) // 2
            img.paste(chip, ((TW - chip.width) // 2, cy), chip)
            y += step

        self._draw_caption(img, gt)

        fy = 1800 if self.br else 1480
        img.paste(self.footer, (0, fy), self.footer)

    def _draw_caption(self, img, gt):
        for k, (s, e, words) in enumerate(self.caps):
            if s <= gt < e:
                upto = 0
                for i, wd in enumerate(words):
                    if wd[2] <= gt:
                        upto = i
                if not self.br:
                    upto = len(words) - 1          # show the whole phrase at once
                cap = self.caption_img(k, upto)
                h0 = cap.height
                age = gt - words[upto][2]
                if self.br and age < 0.12:                      # pop on every new word
                    sc = 1.0 + 0.14 * (1 - age / 0.12)
                    cap = cap.resize((int(cap.widt