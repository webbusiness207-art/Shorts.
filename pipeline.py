#!/usr/bin/env python3
"""Shorts Factory v3 — animated comedy Shorts."""

import datetime as dt
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
TODAY = dt.date.today().isoformat()
OUT_DAY = ROOT / "output" / TODAY
STATE_FILE = ROOT / "used_topics.json"
SERIES_NAME = os.getenv("SERIES_NAME", "KEVIN'S WORST IDEAS")

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

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}

CFG = {
    "videos_per_run": max(1, min(5, env_int("VIDEOS_PER_RUN", 3))),
    "visuals": os.getenv("VISUALS", "toon").lower().strip(),
    "voice": os.getenv("VOICE", "en_US-lessac-medium"),
    "piper_bin": os.getenv("PIPER_BIN", "piper"),
    "piper_model": os.getenv(
        "PIPER_MODEL",
        str(ROOT / "models" / "en_US-lessac-medium.onnx")
    ),
    "gemini_model": os.getenv("GEMINI_MODEL", "").strip(),
    "ollama_url": os.getenv(
        "OLLAMA_URL", "http://localhost:11434/api/generate"
    ),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    "music_volume": max(0, min(1, env_float("MUSIC_VOLUME", 0.075))),
    "fps": 24,
    "width": 720,
    "height": 1280,
}

TW, TH, FPS = CFG["width"], CFG["height"], CFG["fps"]
INK = (27, 31, 43)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
TRENDS_URL = "https://trends.google.com/trending/rss?geo=US"


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def run(command, cwd=None, input_text=None, check=True):
    result = subprocess.run(
        [str(x) for x in command],
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=input_text is not None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        error = result.stderr
        if isinstance(error, bytes):
            error = error.decode("utf-8", "replace")
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
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("topics", [])
                data.setdefault("episodes", [])
                return data
        except Exception as exc:
            print("Could not read topic history:", exc)
    return {"topics": [], "episodes": []}


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
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


def discover_gemini_models(api_key):
    try:
        response = requests.get(
            f"{GEMINI_BASE}/models",
            params={"key": api_key},
            timeout=20,
        )
        response.raise_for_status()
        models = []
        for item in response.json().get("models", []):
            name = item.get("name", "")
            methods = item.get("supportedGenerationMethods", [])
            if (
                "generateContent" in methods
                and "flash" in name.lower()
                and not any(x in name.lower() for x in (
                    "image", "audio", "tts", "live", "embedding"
                ))
            ):
                models.append(name.split("/")[-1])
        models.sort(key=lambda n: ("latest" not in n.lower(), n))
        return models
    except Exception as exc:
        print("Gemini model discovery failed:", exc)
        return []


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


def gemini_generate(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    models = discover_gemini_models(api_key)
    candidates = (
        ([CFG["gemini_model"]] if CFG["gemini_model"] else [])
        + models
        + ["gemini-2.5-flash", "gemini-2.0-flash"]
    )

    last_error = None

    for model in dict.fromkeys(candidates):
        for attempt in range(3):
            response = None
            try:
                response = requests.post(
                    f"{GEMINI_BASE}/models/{model}:generateContent",
                    params={"key": api_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.9,
                            "responseMimeType": "application/json",
                        },
                    },
                    timeout=75,
                )

                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"{model}: HTTP {response.status_code}"
                    if attempt < 2:
                        retry = response.headers.get("Retry-After")
                        try:
                            delay = clamp(float(retry), 1, 20)
                        except (TypeError, ValueError):
                            delay = min(2 ** attempt + random.random(), 15)
                        print(f"Gemini retrying in {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    break

                response.raise_for_status()
                payload = response.json()
                parts = payload.get("candidates", [{}])[0].get(
                    "content", {}
                ).get("parts", [])
                text = "\n".join(
                    p.get("text", "") for p in parts if p.get("text")
                )
                if not text:
                    raise RuntimeError("Gemini returned no text.")
                return extract_json(text)

            except Exception as exc:
                last_error = str(exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    break

    raise RuntimeError(f"Gemini failed: {last_error}")


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
        print("Gemini unavailable; trying Ollama:", gemini_error)
        try:
            return ollama_generate(prompt)
        except Exception as ollama_error:
            raise RuntimeError(
                f"Story generation failed. Gemini: {gemini_error}; "
                f"Ollama: {ollama_error}"
            )


SCENES = {
    "classroom", "bedroom", "kitchen", "office", "lab",
    "stage", "street", "city_night", "park", "space",
}
EMOTIONS = {"neutral", "happy", "angry", "shock", "scared"}
ACTIONS = {"idle", "jump", "shake", "slide", "panic", "point", "run"}
OBJECTS = {
    "none", "phone", "box", "rocket", "car", "button", "pancake",
    "cake", "money", "computer", "machine", "trophy", "chair",
    "table", "sign", "ball", "planet", "giant_coin", "pizza",
    "suitcase", "tv", "present",
}
OBJECT_ACTIONS = {
    "idle", "bounce", "spin", "fall", "grow", "shake",
    "fly", "explode", "slide", "drop",
}
EFFECTS = {"none", "explosion", "fire", "sparkle", "shock"}


def clean_beat(beat):
    if not isinstance(beat, dict):
        raise ValueError("A story beat was not an object.")

    text = re.sub(r"\s+", " ", str(beat.get("text", ""))).strip()
    if not text:
        raise ValueError("A story beat has no narration.")

    scene = beat.get("scene", "classroom")
    if scene not in SCENES:
        scene = "classroom"

    fx = beat.get("fx", "none")
    if fx not in EFFECTS:
        fx = "none"

    chars = []
    for item in beat.get("chars", [])[:3]:
        if not isinstance(item, dict):
            continue
        who = item.get("who", "kevin")
        if who not in {"kevin", "dave", "boss", "cop", "kid", "pigeon"}:
            who = "kevin"
        emotion = item.get("emotion", "neutral")
        action = item.get("action", "idle")
        chars.append({
            "who": who,
            "emotion": emotion if emotion in EMOTIONS else "neutral",
            "action": action if action in ACTIONS else "idle",
        })

    if not chars:
        chars = [{"who": "kevin", "emotion": "neutral", "action": "idle"}]

    objects = []
    for item in beat.get("objects", [])[:2]:
        if not isinstance(item, dict):
            continue
        kind = item.get("type", "none")
        action = item.get("action", "idle")
        objects.append({
            "type": kind if kind in OBJECTS else "none",
            "action": action if action in OBJECT_ACTIONS else "idle",
            "size": clamp(float(item.get("size", 1)), 0.4, 2),
            "position": (
                item.get("position")
                if item.get("position") in {"left", "center", "right"}
                else "center"
            ),
        })

    if not objects:
        objects = [{
            "type": "none", "action": "idle",
            "size": 1, "position": "center",
        }]

    return {
        "label": str(beat.get("label", "STORY"))[:24],
        "text": text,
        "scene": scene,
        "chars": chars,
        "objects": objects,
        "fx": fx,
        "visual": str(beat.get("visual", ""))[:80],
    }


def write_episode(episode_number, state):
    trends = trending_topics()
    recent = state.get("topics", [])[-30:]
    topic_hint = random.choice(trends) if trends else "an everyday problem"

    prompt = f"""
Create one original, funny, fast-paced vertical YouTube Shorts episode
for the fictional comedy series "{SERIES_NAME}".

Kevin is wildly overconfident and misunderstands simple things.
Dave is the practical friend who warns him.
A pigeon silently witnesses the chaos.

Current inspiration topic, if useful: {topic_hint}
Avoid repeating these topics: {json.dumps(recent[-20:])}

Return ONLY valid JSON in this exact shape:
{{
 "title":"short clickable title",
 "description":"one sentence",
 "summary":"one sentence plot summary",
 "topic":"short topic label",
 "tags":["tag1","tag2","tag3"],
 "beats":[
  {{
   "label":"HOOK",
   "text":"One spoken sentence, ideally 7-14 words.",
   "scene":"classroom",
   "visual":"short visible action",
   "chars":[{{"who":"kevin","emotion":"shock","action":"panic"}}],
   "objects":[{{"type":"phone","action":"shake","size":1.0,"position":"right"}}],
   "fx":"shock"
  }}
 ]
}}

Make 10 to 13 beats and about 100 to 145 spoken words.
Beat one must be an immediate curiosity hook.
Escalate every beat. Include Kevin, Dave, and the pigeon.
Finish with a funny cliffhanger or comment question.
Vary scenes, character reactions, actions, and objects.
Keep it family-friendly and original.
Use only these values:
Scenes: {sorted(SCENES)}
Characters: kevin, dave, boss, cop, kid, pigeon
Emotions: {sorted(EMOTIONS)}
Character actions: {sorted(ACTIONS)}
Objects: {sorted(OBJECTS)}
Object actions: {sorted(OBJECT_ACTIONS)}
Effects: {sorted(EFFECTS)}
Do not use copyrighted characters, brands, lyrics, or real people's names.
"""

    last_error = None

    for attempt in range(4):
        try:
            data = llm_json(prompt)
            beats = [clean_beat(x) for x in data.get("beats", [])]
            words = sum(len(x["text"].split()) for x in beats)

            if not 9 <= len(beats) <= 14 or not 85 <= words <= 165:
                raise ValueError(
                    f"Story length outside limits: {len(beats)} beats, {words} words"
                )

            data["beats"] = beats
            data["title"] = str(
                data.get("title", f"Kevin's Worst Idea #{episode_number}")
            )[:90]
            data["topic"] = str(data.get("topic", topic_hint))[:100]
            data["summary"] = str(data.get("summary", ""))[:300]
            data["description"] = str(data.get("description", ""))[:500]
            data["tags"] = [
                str(t)[:30] for t in data.get("tags", [])[:12]
            ]

            state["topics"].append(data["topic"])
            return data

        except Exception as exc:
            last_error = exc
            print(f"Story attempt {attempt + 1} failed:", exc)
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Could not create a valid story: {last_error}")


# ---------------------------------------------------------------------
# NARRATION
# ---------------------------------------------------------------------

def init_tts():
    print("Using Piper for narration.")


def synthesize_piper(text, output_path):
    binary = CFG["piper_bin"]
    if not shutil.which(binary) and not Path(binary).exists():
        raise RuntimeError(f"Piper executable not found: {binary}")

    model = Path(CFG["piper_model"])
    if not model.exists():
        raise RuntimeError(f"Piper model not found: {model}")

    result = subprocess.run(
        [
            binary, "--model", str(model),
            "--length_scale", "0.94",
            "--output_file", str(output_path),
        ],
        input=text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode:
        raise RuntimeError("Piper failed: " + result.stderr[-2000:])

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("Piper did not produce a usable audio file.")


def voice_beats(beats, folder):
    raw_files = []
    durations = []

    for index, beat in enumerate(beats):
        raw = folder / f"voice_raw_{index:02d}.wav"
        synthesize_piper(beat["text"], raw)

        processed = folder / f"voice_{index:02d}.wav"
        run([
            "ffmpeg", "-y", "-i", str(raw),
            "-af",
            "highpass=f=75,lowpass=f=11000,"
            "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=100,"
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "24000", "-ac", "1", str(processed),
        ])

        raw.unlink(missing_ok=True)
        raw_files.append(processed)
        durations.append(probe_duration(processed))

    concat_list = folder / "voice_list.txt"
    concat_list.write_text(
        "".join(f"file '{p.name}'\n" for p in raw_files),
        encoding="utf-8",
    )

    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list.name, "-c:a", "pcm_s16le", "voice.wav",
    ], cwd=folder)

    return durations


# ---------------------------------------------------------------------
# CAPTIONS
# ---------------------------------------------------------------------

def ass_time(seconds):
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def fallback_word_times(beats, durations):
    result = []
    current = 0

    for beat, duration in zip(beats, durations):
        words = beat["text"].split()
        weights = [max(1, len(re.sub(r"\W", "", w))) for w in words]
        total = sum(weights) or 1
        cursor = current

        for word, weight in zip(words, weights):
            span = duration * weight / total
            result.append((cursor, cursor + span, word))
            cursor += span

        current += duration

    return result


def transcribe_words(audio_path, beats, durations):
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            os.getenv("WHISPER_MODEL", "small"),
            device="cpu",
            compute_type="int8",
        )

        segments, _ = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
        )

        words = []
        for segment in segments:
            if segment.words:
                for item in segment.words:
                    word = item.word.strip()
                    if word:
                        words.append((
                            float(item.start),
                            float(item.end),
                            word,
                        ))

        if words:
            return words

    except Exception as exc:
        print("Whisper unavailable; using estimated word timing:", exc)

    return fallback_word_times(beats, durations)


def write_ass(words, beats, durations, episode_number, path):
    header = """[Script Info]
Title: Shorts Factory Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,76,&H00FFFFFF,&H0000FFFF,&H00151520,&H80000000,1,0,0,0,100,100,0,0,1,5,2,2,90,90,300,1
Style: Top,DejaVu Sans,40,&H00FFFFFF,&H0000FFFF,&H00151520,&H80000000,1,0,0,0,100,100,0,0,1,3,1,8,70,70,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    total = sum(durations)

    lines.append(
        f"Dialogue: 0,0:00:00.00,{ass_time(min(2, total))},"
        f"Top,,0,0,0,,{SERIES_NAME.replace(',', '')}  •  EPISODE {episode_number}\n"
    )

    for start, end, word in words:
        clean = word.replace("{", "").replace("}", "").replace("\\", "")
        if not clean:
            continue

        lines.append(
            f"Dialogue: 1,{ass_time(start)},"
            f"{ass_time(max(start + 0.08, end))},"
            f"Default,,0,0,0,,{clean}\n"
        )

    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# CARTOON VISUALS
# ---------------------------------------------------------------------

def mix(a, b, t):
    t = clamp(t, 0, 1)
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def ease(t):
    t = clamp(t, 0, 1)
    return t * t * (3 - 2 * t)


def make_bg(scene):
    palettes = {
        "classroom": ((100, 185, 225), (238, 220, 170)),
        "bedroom": ((75, 105, 165), (210, 175, 155)),
        "kitchen": ((110, 185, 175), (245, 225, 190)),
        "office": ((90, 125, 165), (205, 215, 220)),
        "lab": ((80, 160, 190), (205, 235, 225)),
        "stage": ((70, 55, 110), (230, 120, 95)),
        "street": ((100, 170, 205), (220, 190, 140)),
        "city_night": ((25, 35, 75), (90, 80, 130)),
        "park": ((90, 175, 125), (215, 220, 160)),
        "space": ((20, 25, 65), (70, 75, 135)),
    }

    top, bottom = palettes.get(scene, palettes["classroom"])
    img = Image.new("RGB", (TW, TH))
    pixels = img.load()

    for y in range(TH):
        color = mix(top, bottom, y / TH)
        for x in range(TW):
            pixels[x, y] = color

    draw = ImageDraw.Draw(img)

    if scene == "space":
        rng = random.Random(12)
        for _ in range(90):
            x, y = rng.randrange(TW), rng.randrange(TH)
            r = rng.choice((1, 2, 3))
            draw.ellipse((x-r, y-r, x+r, y+r), fill="white")
        draw.ellipse((480, 150, 700, 370), fill=(100, 115, 205))
    else:
        draw.rectangle((0, 950, TW, TH), fill=(75, 100, 95))

        if scene in {"street", "city_night"}:
            rng = random.Random(scene)
            for x in range(0, TW, 90):
                height = rng.randint(250, 570)
                draw.rectangle(
                    (x, 950-height, x+75, 950),
                    fill=(55, 65, 90),
                )
                for wy in range(1000-height, 900, 65):
                    for wx in range(x+15, x+65, 28):
                        draw.rectangle(
                            (wx, wy, wx+10, wy+18),
                            fill=(255, 220, 120),
                        )
        else:
            draw.rectangle(
                (40, 180, TW-40, 850),
                fill=(235, 230, 205),
                outline=INK,
                width=8,
            )
            draw.rectangle(
                (75, 220, TW-75, 800),
                fill=(180, 220, 230),
            )
            draw.rectangle((0, 900, TW, 930), fill=(145, 105, 75))

    return img


ROSTER = {
    "kevin": {
        "shirt": (235, 75, 65),
        "pants": (45, 65, 105),
        "hair": (65, 40, 25),
    },
    "dave": {
        "shirt": (55, 125, 220),
        "pants": (45, 55, 75),
        "hair": (55, 45, 35),
        "glasses": True,
    },
    "boss": {
        "shirt": (65, 65, 80),
        "pants": (40, 45, 60),
        "hair": (45, 35, 30),
        "hat": (50, 50, 60),
    },
    "cop": {
        "shirt": (40, 80, 155),
        "pants": (35, 45, 85),
        "hair": (50, 40, 30),
        "hat": (35, 55, 105),
    },
    "kid": {
        "shirt": (240, 180, 55),
        "pants": (60, 105, 75),
        "hair": (80, 45, 25),
    },
}


def draw_pigeon(draw, x, y, scale=1, beat_t=0):
    bob = math.sin(beat_t * 6) * 5 * scale
    y += bob

    draw.ellipse(
        (x-48*scale, y-45*scale, x+48*scale, y+38*scale),
        fill=(115, 125, 140), outline=INK, width=4,
    )
    draw.ellipse(
        (x+10*scale, y-78*scale, x+58*scale, y-25*scale),
        fill=(125, 135, 150), outline=INK, width=4,
    )
    draw.polygon(
        [
            (x+45*scale, y-55*scale),
            (x+80*scale, y-43*scale),
            (x+45*scale, y-35*scale),
        ],
        fill=(240, 155, 45),
        outline=INK,
    )
    draw.ellipse(
        (x+35*scale, y-62*scale, x+43*scale, y-54*scale),
        fill=INK,
    )
    draw.line(
        (x-12*scale, y+30*scale, x-15*scale, y+58*scale),
        fill=(220, 145, 45), width=4,
    )
    draw.line(
        (x+18*scale, y+30*scale, x+20*scale, y+58*scale),
        fill=(220, 145, 45), width=4,
    )


def draw_character(draw, who, x, ground_y, scale, emotion, action, global_t, beat_t):
    if who == "pigeon":
        draw_pigeon(draw, x, ground_y-20*scale, scale*.8, beat_t)
        return

    spec = ROSTER.get(who, ROSTER["kevin"])
    bob = 0

    if action in {"jump", "panic"}:
        bob = -abs(math.sin(beat_t * 8)) * 45 * scale
    elif action == "shake":
        x += math.sin(beat_t * 30) * 12 * scale
    elif action == "slide":
        x += math.sin(beat_t * 3) * 45 * scale

    y = ground_y + bob
    head_size = 100 * scale
    body_top = y - 250 * scale
    head_top = body_top - head_size + 10 * scale

    draw.line(
        (x-25*scale, y-75*scale, x-42*scale, y),
        fill=spec["pants"], width=max(1, int(24*scale)),
    )
    draw.line(
        (x+25*scale, y-75*scale, x+42*scale, y),
        fill=spec["pants"], width=max(1, int(24*scale)),
    )

    draw.rounded_rectangle(
        (x-58*scale, body_top, x+58*scale, y-65*scale),
        radius=max(1, int(25*scale)),
        fill=spec["shirt"], outline=INK, width=4,
    )

    arm_raise = -45 if action in {"point", "panic"} else 0

    draw.line(
        (
            x-50*scale, body_top+25*scale,
            x-95*scale, body_top+100*scale+arm_raise*scale,
        ),
        fill=spec["shirt"], width=max(1, int(22*scale)),
    )
    draw.line(
        (
            x+50*scale, body_top+25*scale,
            x+95*scale, body_top+100*scale-arm_raise*scale,
        ),
        fill=spec["shirt"], width=max(1, int(22*scale)),
    )

    draw.ellipse(
        (
            x-head_size/2, head_top,
            x+head_size/2, head_top+head_size,
        ),
        fill=(245, 190, 145), outline=INK, width=4,
    )

    draw.pieslice(
        (
            x-head_size/2-3, head_top-12*scale,
            x+head_size/2+3, head_top+48*scale,
        ),
        180, 360,
        fill=spec["hair"], outline=INK,
    )

    face_y = head_top + head_size*.56
    eye_radius = (
        17 if emotion in {"shock", "scared"} else 13
    ) * scale

    for side in (-1, 1):
        eye_x = x + side*24*scale

        draw.ellipse(
            (
                eye_x-eye_radius,
                face_y-8*scale-eye_radius,
                eye_x+eye_radius,
                face_y-8*scale+eye_radius,
            ),
            fill="white", outline=INK, width=3,
        )

        pupil = eye_radius*.4
        draw.ellipse(
            (
                eye_x-pupil,
                face_y-8*scale-pupil,
                eye_x+pupil,
                face_y-8*scale+pupil,
            ),
            fill=INK,
        )

        if spec.get("glasses"):
            draw.ellipse(
                (
                    eye_x-eye_radius-5*scale,
                    face_y-8*scale-eye_radius-5*scale,
                    eye_x+eye_radius+5*scale,
                    face_y-8*scale+eye_radius+5*scale,
                ),
                outline=INK, width=3,
            )

    mouth_y = face_y + 30*scale

    if emotion == "happy":
        draw.arc(
            (
                x-24*scale, mouth_y-24*scale,
                x+24*scale, mouth_y+14*scale,
            ),
            15, 165, fill=INK, width=5,
        )
    elif emotion in {"shock", "scared"}:
        draw.ellipse(
            (
                x-14*scale, mouth_y-5*scale,
                x+14*scale, mouth_y+28*scale,
            ),
            fill=(70, 12, 24), outline=INK, width=3,
        )
    elif emotion == "angry":
        draw.arc(
            (
                x-20*scale, mouth_y-2*scale,
                x+20*scale, mouth_y+30*scale,
            ),
            200, 340, fill=INK, width=5,
        )
    else:
        draw.line(
            (x-14*scale, mouth_y, x+14*scale, mouth_y),
            fill=INK, width=5,
        )

    if spec.get("hat"):
        draw.rectangle(
            (
                x-head_size/2-8*scale,
                head_top-16*scale,
                x+head_size/2+8*scale,
                head_top+24*scale,
            ),
            fill=spec["hat"], outline=INK, width=4,
        )


def object_position(position):
    return {
        "left": TW*.23,
        "center": TW*.5,
        "right": TW*.77,
    }.get(position, TW*.5)


def draw_object(draw, obj, beat_t, index):
    kind = obj["type"]
    if kind == "none":
        return

    unit = obj["size"]
    x, y = object_position(obj["position"]), 700
    action = obj["action"]

    if action == "bounce":
        y -= abs(math.sin(beat_t*7+index))*90
    elif action in {"fall", "drop"}:
        y = 250 + ease(beat_t/.7)*450
    elif action == "grow":
        unit *= .6 + ease(beat_t/.8)*.7
    elif action == "fly":
        x += math.sin(beat_t*3+index)*110
        y -= ease(beat_t/.7)*250
    elif action == "shake":
        x += math.sin(beat_t*35)*15

    w, h = 85*unit, 85*unit
    outline = INK

    if kind == "phone":
        draw.rounded_rectangle(
            (x-w*.45, y-h, x+w*.45, y+h),
            radius=16, fill=(35, 40, 55), outline=outline, width=5,
        )
        draw.rectangle(
            (x-w*.34, y-h*.78, x+w*.34, y+h*.65),
            fill=(80, 190, 240),
        )

    elif kind in {"box", "present", "suitcase"}:
        color = (190, 105, 45) if kind == "box" else (180, 55, 90)
        draw.rectangle(
            (x-w, y-h*.6, x+w, y+h*.6),
            fill=color, outline=outline, width=5,
        )
        draw.rectangle(
            (x-w*.12, y-h*.6, x+w*.12, y+h*.6),
            fill=(255, 215, 70),
        )

    elif kind in {"rocket", "machine"}:
        draw.rounded_rectangle(
            (x-w*.42, y-h*1.5, x+w*.42, y+h*.8),
            radius=25, fill=(205, 210, 220),
            outline=outline, width=5,
        )
        draw.ellipse(
            (x-w*.22, y-h*.8, x+w*.22, y-h*.35),
            fill=(70, 180, 240), outline=outline, width=3,
        )
        if action in {"fly", "explode"}:
            draw.polygon(
                (
                    (x-w*.25, y+h*.7),
                    (x, y+h*1.8+abs(math.sin(beat_t*15))*45),
                    (x+w*.25, y+h*.7),
                ),
                fill=(255, 120, 20),
            )

    elif kind in {"pancake", "pizza", "cake"}:
        color = {
            "pancake": (225, 165, 75),
            "pizza": (235, 180, 75),
            "cake": (245, 180, 195),
        }[kind]
        draw.ellipse(
            (x-w, y-h*.45, x+w, y+h*.45),
            fill=color, outline=outline, width=5,
        )
        if kind == "pizza":
            for dx, dy in ((-25, -8), (25, 5), (0, 18)):
                draw.ellipse(
                    (x+dx-9, y+dy-9, x+dx+9, y+dy+9),
                    fill=(190, 50, 45),
                )

    elif kind in {
        "money", "giant_coin", "button", "ball", "trophy", "planet"
    }:
        color = {
            "money": (80, 190, 100),
            "giant_coin": (255, 205, 40),
            "button": (230, 45, 50),
            "ball": (60, 140, 235),
            "trophy": (255, 205, 40),
            "planet": (105, 110, 220),
        }[kind]
        draw.ellipse(
            (x-w, y-h, x+w, y+h),
            fill=color, outline=outline, width=5,
        )
        label = "$" if kind == "money" else (
            "DO NOT" if kind == "button" else "★"
        )
        draw.text(
            (x, y), label, fill="white",
            anchor="mm", stroke_width=2, stroke_fill=INK,
        )

    elif kind in {"computer", "tv"}:
        draw.rectangle(
            (x-w, y-h, x+w, y+h*.65),
            fill=(35, 40, 55), outline=outline, width=5,
        )
        draw.rectangle(
            (x-w*.8, y-h*.8, x+w*.8, y+h*.4),
            fill=(80, 185, 235),
        )
        draw.rectangle(
            (x-8, y+h*.65, x+8, y+h),
            fill=(70, 70, 80),
        )

    elif kind in {"car", "chair"}:
        draw.rounded_rectangle(
            (x-w*1.3, y-h*.6, x+w*1.3, y+h*.35),
            radius=20, fill=(210, 45, 55),
            outline=outline, width=5,
        )
        for wx in (x-w*.7, x+w*.7):
            draw.ellipse(
                (wx-20, y+h*.1, wx+20, y+h*.7),
                fill=(35, 35, 45),
            )

    elif kind == "table":
        draw.rectangle(
            (x-w*1.3, y-h*.2, x+w*1.3, y+h*.1),
            fill=(150, 95, 55), outline=outline, width=4,
        )
        draw.rectangle(
            (x-w, y+h*.1, x-w*.8, y+h),
            fill=(150, 95, 55),
        )
        draw.rectangle(
            (x+w*.8, y+h*.1, x+w, y+h),
            fill=(150, 95, 55),
        )

    elif kind == "sign":
        draw.rectangle(
            (x-w, y-h, x+w, y+h*.5),
            fill=(255, 220, 65), outline=outline, width=5,
        )
        draw.text((x, y-h*.2), "WARNING", fill=INK, anchor="mm")


def draw_effect(draw, effect, beat_t, frame):
    cx, cy = TW/2, 520

    if effect == "explosion":
        radius = 35 + ease(beat_t/.35)*220
        points = []
        for i in range(24):
            r = radius * (1 if i % 2 == 0 else .55)
            angle = i/24*math.tau
            points.append((
                cx+math.cos(angle)*r,
                cy+math.sin(angle)*r,
            ))
        draw.polygon(
            points,
            fill=(255, 130, 20),
            outline=(220, 40, 40),
        )

    elif effect == "fire":
        for i in range(6):
            x = 45 + i*120
            height = 140 + 55*math.sin(beat_t*10+i)
            draw.polygon(
                ((x-35, 1050), (x, 1050-height), (x+35, 1050)),
                fill=(255, 100, 15),
            )

    elif effect == "shock" and beat_t < .12:
        draw.rectangle((0, 0, TW, TH), fill="white")

    elif effect == "sparkle":
        rng = random.Random(7)
        for i in range(12):
            x = rng.randint(35, TW-35)
            y = rng.randint(150, 900)
            r = 8 + 10*abs(math.sin(beat_t*7+i))
            draw.line((x-r, y, x+r, y), fill=(255, 245, 140), width=4)
            draw.line((x, y-r, x, y+r), fill=(255, 245, 140), width=4)


def draw_visual_description(draw, text, font):
    clean = re.sub(r"[^A-Za-z0-9 !?'$-]", "", text).strip()
    if not clean:
        return

    clean = clean[:45] + ("..." if len(clean) > 45 else "")
    draw.rounded_rectangle(
        (28, 540, TW-28, 600),
        radius=14, fill=(25, 25, 40),
        outline="white", width=2,
    )
    draw.text(
        (TW/2, 570), clean,
        font=font, fill="white", anchor="mm",
    )


# ---------------------------------------------------------------------
# BLENDER RENDERER
# ---------------------------------------------------------------------

def blender_background(beats, durations, folder):
    blender = shutil.which("blender")
    if not blender:
        raise RuntimeError("Blender is not installed.")

    scene_script = Path(__file__).with_name("blender_scene.py")
    if not scene_script.exists():
        raise RuntimeError("blender_scene.py is missing.")

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    payload = folder / "blender_beats.json"
    output = folder / "bg.mp4"

    payload.write_text(
        json.dumps({
            "beats": beats,
            "durations": durations,
            "width": 1080,
            "height": 1920,
            "fps": CFG["fps"],
        }),
        encoding="utf-8",
    )

    command = [
        blender, "-b", "-t", "2",
        "--python-exit-code", "1",
        "--python", str(scene_script),
        "--", str(payload), str(output),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=5400,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Blender rendering failed:\n"
            + result.stdout[-4000:]
            + result.stderr[-4000:]
        )

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Blender did not create the background video.")

    return output, []


# ---------------------------------------------------------------------
# TOON RENDERER
# ---------------------------------------------------------------------

def toon_background(beats, durations, folder):
    bgs = {}

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = (
        ImageFont.truetype(font_path, 20)
        if Path(font_path).exists()
        else ImageFont.load_default()
    )

    output = folder / "bg.mp4"

    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{TW}x{TH}",
        "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(output),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    frame_number = 0

    try:
        for index, (beat, duration) in enumerate(zip(beats, durations)):
            scene = beat["scene"]

            if scene not in bgs:
                bgs[scene] = make_bg(scene)

            chars = beat["chars"]
            count = len(chars)

            sizes = {1: 1.45, 2: 1.12, 3: .92}
            positions = {
                1: [.5],
                2: [.30, .70],
                3: [.20, .50, .80],
            }

            start_frame = frame_number
            end_frame = start_frame + max(1, round(duration*FPS))

            for frame in range(start_frame, end_frame):
                beat_t = (frame-start_frame)/FPS
                img = bgs[scene].copy()
                draw = ImageDraw.Draw(img)

                draw_effect(draw, beat["fx"], beat_t, frame)

                for oi, obj in enumerate(beat["objects"]):
                    draw_object(draw, obj, beat_t, oi)

                for char, pos in zip(chars, positions[count]):
                    draw_character(
                        draw,
                        char["who"],
                        TW*pos,
                        1030,
                        sizes[count],
                        char["emotion"],
                        char["action"],
                        frame/FPS,
                        beat_t,
                    )

                if beat_t < 1.2 and beat.get("visual"):
                    draw_visual_description(draw, beat["visual"], font)

                progress = (
                    index + beat_t/max(duration, .1)
                ) / len(beats)

                draw.rounded_rectangle(
                    (90, 410, 630, 445),
                    radius=15,
                    fill=(25, 25, 40),
                    outline="white",
                    width=2,
                )

                draw.rounded_rectangle(
                    (
                        94, 414,
                        94+max(5, int(532*progress)),
                        441,
                    ),
                    radius=12,
                    fill=mix(
                        (65, 220, 100),
                        (245, 55, 55),
                        progress,
                    ),
                )

                draw.text(
                    (TW/2, 385),
                    "CHAOS LEVEL",
                    font=font,
                    fill="white",
                    anchor="mm",
                    stroke_width=2,
                    stroke_fill=INK,
                )

                zoom = (
                    1.0
                    + .075*(1-ease(beat_t/.25))
                    + .012*math.sin(beat_t*2)
                )

                shake = (
                    8*max(0, 1-beat_t/.5)
                    if beat["fx"] in {"explosion", "shock"}
                    else 0
                )

                crop_w, crop_h = TW/zoom, TH/zoom

                left = clamp(
                    TW/2 + math.sin(frame*2.7)*shake - crop_w/2,
                    0, TW-crop_w,
                )
                top = clamp(
                    TH/2 + math.cos(frame*3.1)*shake - crop_h/2,
                    0, TH-crop_h,
                )

                right = clamp(left+crop_w, 1, TW)
                bottom = clamp(top+crop_h, 1, TH)

                crop = img.crop((
                    int(left),
                    int(top),
                    max(int(left)+1, int(math.ceil(right))),
                    max(int(top)+1, int(math.ceil(bottom))),
                ))

                crop = crop.resize(
                    (TW, TH),
                    Image.Resampling.BILINEAR,
                )

                process.stdin.write(crop.tobytes())

            frame_number = end_frame

    except Exception:
        if process.stdin:
            process.stdin.close()
        process.kill()
        process.wait()
        raise

    else:
        if process.stdin:
            process.stdin.close()

    stderr = (
        process.stderr.read().decode("utf-8", "replace")
        if process.stderr else ""
    )

    return_code = process.wait()

    if return_code:
        raise RuntimeError(
            "Cartoon video render failed: " + stderr[-3000:]
        )

    return output, []


# ---------------------------------------------------------------------
# STOCK FOOTAGE
# ---------------------------------------------------------------------

def pixabay_candidates(query):
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        return []

    response = requests.get(
        "https://pixabay.com/api/videos/",
        params={
            "key": key,
            "q": query[:100],
            "per_page": 15,
            "safesearch": "true",
            "video_type": "film",
        },
        timeout=30,
    )
    response.raise_for_status()

    results = []

    for hit in response.json().get("hits", []):
        videos = hit.get("videos", {})
        chosen = next(
            (
                videos[k]
                for k in ("medium", "small", "large")
                if videos.get(k, {}).get("url")
            ),
            None,
        )
        if chosen:
            results.append((
                chosen["url"],
                hit.get("pageURL", ""),
            ))

    return results


def pexels_candidates(query):
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return []

    response = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": key},
        params={
            "query": query,
            "orientation": "portrait",
            "per_page": 12,
        },
        timeout=30,
    )
    response.raise_for_status()

    results = []

    for video in response.json().get("videos", []):
        files = [
            f for f in video.get("video_files", [])
            if f.get("file_type") == "video/mp4" and f.get("link")
        ]
        if files:
            chosen = min(
                files,
                key=lambda f: abs((f.get("width") or 720)-720),
            )
            results.append((
                chosen["link"],
                video.get("url", ""),
            ))

    return results


def download(url, path):
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with path.open("wb") as output:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    output.write(chunk)


def make_seg(raw, duration, output):
    raw_duration = probe_duration(raw)
    start = random.uniform(
        0,
        max(0, raw_duration-duration-.2),
    )

    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,eq=saturation=1.15:contrast=1.05,"
        "setsar=1,fps=24"
    )

    run([
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-i", str(raw),
        "-an",
        "-t", f"{duration:.2f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "21",
        "-pix_fmt", "yuv420p",
        str(output),
    ])


def build_background(beats, durations, folder):
    local = pick_files(ROOT / "footage", VIDEO_EXT)

    if os.getenv("PIXABAY_API_KEY"):
        api_search = pixabay_candidates
    else:
        api_search = pexels_candidates

    api_enabled = bool(
        os.getenv("PIXABAY_API_KEY")
        or os.getenv("PEXELS_API_KEY")
    )

    pool, sources, segments, used = [], [], [], set()

    for index, (beat, duration) in enumerate(zip(beats, durations)):
        raw = None

        if api_enabled:
            try:
                candidates = [
                    x for x in api_search(
                        beat["visual"] or beat["scene"]
                    )
                    if x[0] not in used
                ]

                if candidates:
                    url, page = random.choice(candidates[:6])
                    raw = folder / f"raw_{index}.mp4"
                    download(url, raw)
                    used.add(url)
                    pool.append(raw)

                    if page:
                        sources.append(page)

            except Exception as exc:
                print("Stock footage lookup failed:", exc)

        if raw is None and pool:
            raw = random.choice(pool)

        if raw is None and local:
            raw = random.choice(local)

        if raw is None:
            raise RuntimeError(
                "Stock visuals selected, but no footage was available."
            )

        segment = folder / f"segment_{index}.mp4"
        make_seg(raw, duration, segment)
        segments.append(segment)

    concat = folder / "segments.txt"

    concat.write_text(
        "".join(f"file '{p.name}'\n" for p in segments),
        encoding="utf-8",
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat.name,
        "-c", "copy",
        "bg.mp4",
    ], cwd=folder)

    return folder / "bg.mp4", sources


# ---------------------------------------------------------------------
# FINAL VIDEO COMPOSITION
# ---------------------------------------------------------------------

def render(background, voice, workdir, output, duration):
    music_files = pick_files(ROOT / "music", AUDIO_EXT)
    music = random.choice(music_files) if music_files else None

    command = [
        "ffmpeg", "-y",
        "-i", str(background),
        "-i", str(voice),
    ]

    # Scale first, then burn captions at the final video resolution.
    video_filter = (
        "[0:v]scale=1080:1920:flags=lanczos,"
        "ass=captions.ass[v];"
    )

    if music:
        command += [
            "-stream_loop", "-1",
            "-i", str(music),
        ]

        audio_filter = (
            f"[2:a]volume={CFG['music_volume']}[music];"
            "[1:a][music]amix=inputs=2:duration=first:"
            "dropout_transition=0:normalize=0[a]"
        )
    else:
        audio_filter = "[1:a]anull[a]"

    command += [
        "-filter_complex",
        video_filter + audio_filter,
        "-map", "[v]",
        "-map", "[a]",
        "-t", f"{duration+.3:.2f}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ]

    run(command, cwd=workdir)

    return music.name if music else None


# ---------------------------------------------------------------------
# PRODUCE ONE EPISODE
# ---------------------------------------------------------------------

def make_video(index):
    state = load_state()
    episode_number = len(state["episodes"]) + 1

    data = write_episode(episode_number, state)

    folder = OUT_DAY / (
        f"{index:02d}_ep{episode_number}_{safe_name(data['title'])}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    beats = data["beats"]

    (folder / "script.txt").write_text(
        "\n".join(
            f"[{b['label']}] {b['text']}"
            for b in beats
        ),
        encoding="utf-8",
    )

    print("Generating narration...")
    durations = voice_beats(beats, folder)
    total_duration = sum(durations)

    print("Generating captions...")
    words = transcribe_words(
        folder / "voice.wav",
        beats,
        durations,
    )

    write_ass(
        words,
        beats,
        durations,
        episode_number,
        folder / "captions.ass",
    )

    print("Rendering animated scenes...")

    if CFG["visuals"] == "blender":
        background, sources = blender_background(
            beats, durations, folder
        )

    elif CFG["visuals"] == "toon":
        background, sources = toon_background(
            beats, durations, folder
        )

    elif CFG["visuals"] == "stock":
        background, sources = build_background(
            beats, durations, folder
        )

    else:
        raise ValueError(
            "VISUALS must be 'blender', 'toon', or 'stock'."
        )

    print("Compositing final Short...")

    music = render(
        background,
        folder / "voice.wav",
        folder,
        folder / "video.mp4",
        total_duration,
    )

    for pattern in (
        "voice_raw_*.wav",
        "voice_*.wav",
        "raw_*.mp4",
        "segment_*.mp4",
        "segments.txt",
        "voice_list.txt",
        "bg.mp4",
    ):
        for path in folder.glob(pattern):
            path.unlink(missing_ok=True)

    metadata = {
        "episode": episode_number,
        "series": SERIES_NAME,
        "title": data["title"],
        "description": (
            data.get("description", "")
            + "\n\nFictional comedy series. #Shorts"
        ),
        "summary": data.get("summary", ""),
        "tags": data.get("tags", []),
        "duration_sec": round(total_duration, 1),
        "voice_engine": "Piper",
        "voice_model": CFG["voice"],
        "visual_style": CFG["visuals"],
        "footage_sources": sources,
        "music_used": music,
        "review_checklist": [
            "Watch the complete video before uploading.",
            "Check narration and caption timing.",
            "Confirm any music or stock-footage license.",
            "Apply synthetic-content disclosure if applicable.",
        ],
    }

    (folder / "meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    state["episodes"].append({
        "ep": episode_number,
        "title": data["title"],
        "summary": data.get("summary", ""),
    })

    save_state(state)

    return str(folder), data["title"]


# ---------------------------------------------------------------------
# NOTIFICATIONS
# ---------------------------------------------------------------------

def notify(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat:
        return

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat,
                "text": message[:4000],
            },
            timeout=15,
        )
    except Exception as exc:
        print("Telegram message failed:", exc)


def notify_video(path, caption):
    token = os.getenv("TELEGRAM_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat:
        return

    try:
        with open(path, "rb") as video:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendVideo",
                data={
                    "chat_id": chat,
                    "caption": caption[:900],
                },
                files={"video": video},
                timeout=180,
            )

        response.raise_for_status()

    except Exception as exc:
        print("Telegram video notification failed:", exc)


# ---------------------------------------------------------------------
# PREFLIGHT CHECKS
# ---------------------------------------------------------------------

def preflight():
    errors = []

    if CFG["visuals"] not in {"blender", "toon", "stock"}:
        errors.append(
            "VISUALS must be 'blender', 'toon', or 'stock'."
        )

    if CFG["visuals"] == "blender":
        if not shutil.which("blender"):
            errors.append("Blender executable not found.")

        if not Path(__file__).with_name("blender_scene.py").exists():
            errors.append("blender_scene.py not found.")

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            errors.append(f"{tool} is missing from PATH.")

    piper = CFG["piper_bin"]

    if not (shutil.which(piper) or Path(piper).exists()):
        errors.append(f"Piper executable not found: {piper}")

    if not Path(CFG["piper_model"]).exists():
        errors.append(
            f"Piper model is missing: {CFG['piper_model']}"
        )

    if CFG["visuals"] == "stock" and not (
        os.getenv("PIXABAY_API_KEY")
        or os.getenv("PEXELS_API_KEY")
        or pick_files(ROOT / "footage", VIDEO_EXT)
    ):
        errors.append(
            "Stock mode requires an API key or footage in ./footage."
        )

    if not os.getenv("GEMINI_API_KEY"):
        try:
            requests.get(
                CFG["ollama_url"].rsplit("/api", 1)[0],
                timeout=4,
            )
        except Exception:
            errors.append(
                "Set GEMINI_API_KEY or run a reachable Ollama server."
            )

    return errors


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    OUT_DAY.mkdir(parents=True, exist_ok=True)

    print("Starting Shorts Factory v3...")
    print("Visual engine:", CFG["visuals"])
    print("Videos requested:", CFG["videos_per_run"])

    init_tts()

    errors = preflight()
    completed = []

    if not errors:
        existing = [
            p for p in OUT_DAY.iterdir()
            if p.is_dir()
        ]

        start_index = len(existing) + 1

        for offset in range(CFG["videos_per_run"]):
            try:
                folder, title = make_video(start_index + offset)

                completed.append((title, folder))

                notify_video(
                    Path(folder) / "video.mp4",
                    title,
                )

                print(f"Completed: {title}")

            except Exception as exc:
                message = f"Video {offset+1}: {exc}"
                errors.append(message)
                print(message)

                with (OUT_DAY / "errors.log").open(
                    "a", encoding="utf-8"
                ) as log:
                    log.write(traceback.format_exc() + "\n")

    else:
        for error in errors:
            print("PREFLIGHT:", error)

    report = [
        f"# Daily report — {TODAY}",
        f"Completed: {len(completed)}",
        f"Errors: {len(errors)}",
        f"Voice: Piper / {CFG['voice']}",
        f"Visuals: {CFG['visuals']}",
        "",
    ]

    report.extend(
        f"- COMPLETE: {title} -> {folder}"
        for title, folder in completed
    )

    report.extend(
        f"- ERROR: {error}"
        for error in errors
    )

    report.append(
        "\nReview each video's video.mp4 and meta.json before uploading."
    )

    report_text = "\n".join(report)

    with (OUT_DAY / "report.md").open(
        "a", encoding="utf-8"
    ) as output:
        output.write(report_text + "\n\n")

    print(report_text)
    notify(report_text)


if __name__ == "__main__":
    main()
