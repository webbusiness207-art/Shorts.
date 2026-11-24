
#!/usr/bin/env python3
"""Shorts Factory v4 — animated comedy YouTube Shorts."""

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

import requests
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

SERIES_NAME = os.getenv("SERIES_NAME") or "KEVIN'S WORST IDEAS"

DEFAULT_PREMISE = (
    "Kevin is an overconfident, clueless guy who tries one absurd "
    "experiment per episode. Every beat escalates. Dave warns Kevin "
    "and gets ignored. A pigeon mysteriously watches every disaster. "
    "End on a funny cliffhanger."
)

CFG = {
    "videos_per_run": int(os.getenv("VIDEOS_PER_RUN") or 3),
    "premise": os.getenv("NICHE") or DEFAULT_PREMISE,
    "voice": os.getenv("VOICE") or "am_fenrir",
    "speed": float(os.getenv("VOICE_SPEED") or 1.08),
    "engine": "kokoro",
    "visuals": os.getenv("VISUALS") or "toon",
    "ollama_url": os.getenv(
        "OLLAMA_URL",
        "http://localhost:11434/api/generate"
    ),
    "ollama_model": "llama3.1:8b",
    "piper_bin": os.getenv("PIPER_BIN") or "piper",
    "piper_model": str(
        ROOT / "models" / "en_US-lessac-medium.onnx"
    ),
    "font": os.getenv("CAPTION_FONT") or "DejaVu Sans",
    "whisper_model": "small",
    "music_volume": 0.075,
    "words_min": 90,
    "words_max": 145,
}

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg"}

TODAY = dt.date.today().isoformat()
OUT_DAY = ROOT / "output" / TODAY
STATE_FILE = ROOT / "used_topics.json"

TW, TH = 720, 1280
FPS = 30
INK = (22, 22, 34)

# ============================================================
# UTILITIES
# ============================================================

def run(cmd, **kwargs):
    return subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        **kwargs
    )


def probe_duration(path):
    result = run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(path)
    ])
    return float(result.stdout.strip())


def pick_files(folder, extensions):
    if not folder.exists():
        return []

    return [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()

    return json.loads(text)


# ============================================================
# GEMINI AI
# ============================================================

_GEMINI_OK = None


def gemini_models(key):
    response = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": key},
        params={"pageSize": 200},
        timeout=30
    )

    response.raise_for_status()

    names = []

    for model in response.json().get("models", []):
        name = model["name"].split("/")[-1]
        supported = model.get("supportedGenerationMethods", [])

        if (
            "generateContent" in supported
            and "flash" in name.lower()
            and not re.search(
                r"image|tts|live|audio|embed|robotics",
                name,
                re.I
            )
        ):
            names.append(name)

    def rank(name):
        version = re.search(
            r"gemini-(\d+(?:\.\d+)?)",
            name
        )

        number = float(version.group(1)) if version else 0

        return (
            "preview" in name or "exp" in name,
            "lite" not in name,
            -number
        )

    return sorted(names, key=rank)


def llm_json(prompt):
    global _GEMINI_OK

    key = os.getenv("GEMINI_API_KEY")

    if not key:
        response = requests.post(
            CFG["ollama_url"],
            json={
                "model": CFG["ollama_model"],
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=900
        )

        response.raise_for_status()

        return safe_json(response.json()["response"])

    candidates = []

    if _GEMINI_OK:
        candidates.append(_GEMINI_OK)

    if os.getenv("GEMINI_MODEL"):
        candidates.append(os.getenv("GEMINI_MODEL"))

    try:
        candidates.extend(gemini_models(key))
    except Exception as exc:
        print("Gemini model discovery failed:", exc)

    candidates.extend([
        "gemini-2.5-flash",
        "gemini-flash-latest"
    ])

    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        raise RuntimeError("No Gemini models were found.")

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.85
        }
    }

    last_error = "No models worked."

    for model in candidates:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )

        for attempt in range(4):
            try:
                response = requests.post(
                    url,
                    headers={"x-goog-api-key": key},
                    json=body,
                    timeout=90
                )

                if response.status_code in (
                    429, 500, 502, 503, 504
                ):
                    last_error = (
                        f"{model}: HTTP {response.status_code}"
                    )

                    retry_after = response.headers.get("Retry-After")

                    try:
                        wait = float(retry_after)
                    except (TypeError, ValueError):
                        wait = min(8 * (2 ** attempt), 45)

                    print(
                        f"Gemini temporary error: {last_error}. "
                        f"Waiting {wait:.0f} seconds."
                    )

                    if attempt < 3:
                        time.sleep(
                            wait + random.uniform(0, 2)
                        )
                        continue

                    break

                if response.status_code in (400, 403, 404):
                    last_error = (
                        f"{model}: HTTP {response.status_code}"
                    )
                    print("Trying another model:", last_error)
                    break

                response.raise_for_status()

                result = response.json()

                content = (
                    result["candidates"][0]
                    ["content"]["parts"][0]["text"]
                )

                data = safe_json(content)

                _GEMINI_OK = model

                print("Using Gemini model:", model)

                return data

            except requests.exceptions.RequestException as exc:
                last_error = f"{model}: {exc}"

                print("Gemini request failed:", last_error)

                if attempt < 3:
                    time.sleep(min(8 * (2 ** attempt), 45))

            except (
                KeyError,
                IndexError,
                json.JSONDecodeError,
                TypeError
            ) as exc:
                last_error = f"{model}: Invalid response: {exc}"
                break

    raise RuntimeError(
        "No Gemini model worked: " + last_error
    )


# ============================================================
# SERIES MEMORY
# ============================================================

def load_state():
    if STATE_FILE.exists():
        try:
            data = json.loads(
                STATE_FILE.read_text(encoding="utf-8")
            )

            if (
                isinstance(data, dict)
                and isinstance(data.get("episodes"), list)
            ):
                return data

        except Exception:
            pass

    return {"episodes": []}


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )


def trending_hints():
    try:
        response = requests.get(
            "https://trends.google.com/trending/rss?geo=US",
            timeout=20
        )

        response.raise_for_status()

        root = ET.fromstring(response.content)

        return [
            item.findtext("title")
            for item in root.iter("item")
            if item.findtext("title")
        ][:10]

    except Exception as exc:
        print("Trends unavailable:", exc)
        return []


# ============================================================
# STORY AND VISUAL SCHEMA
# ============================================================

SCENE_NAMES = [
    "classroom", "bedroom", "kitchen", "office", "lab",
    "stage", "street", "city_night", "park", "space"
]

EMOTIONS = [
    "neutral", "happy", "angry", "shock", "scared"
]

ACTIONS = [
    "idle", "jump", "shake", "slide", "panic", "point", "run"
]

OBJECT_NAMES = [
    "none", "phone", "box", "rocket", "car", "button",
    "pancake", "cake", "money", "computer", "machine",
    "trophy", "chair", "table", "sign", "ball", "planet",
    "giant_coin", "pizza", "suitcase", "tv", "present"
]

OBJECT_ACTIONS = [
    "idle", "bounce", "spin", "fall", "grow",
    "shake", "fly", "explode", "slide", "drop"
]

FX_NAMES = [
    "none", "explosion", "fire", "sparkle", "shock"
]


ROSTER = {
    "kevin": {
        "skin": (255, 205, 160),
        "shirt": (255, 135, 35),
        "pants": (48, 57, 130),
        "hair": (85, 48, 28)
    },
    "dave": {
        "skin": (225, 170, 130),
        "shirt": (55, 135, 225),
        "pants": (55, 55, 70),
        "hair": (25, 25, 30),
        "glasses": True
    },
    "cop": {
        "skin": (240, 190, 150),
        "shirt": (35, 55, 120),
        "pants": (25, 35, 80),
        "hair": (40, 30, 20),
        "hat": (25, 40, 100)
    },
    "boss": {
        "skin": (210, 160, 120),
        "shirt": (95, 95, 105),
        "pants": (60, 60, 70),
        "hair": (170, 170, 175),
        "tie": (210, 40, 40)
    },
    "kid": {
        "skin": (250, 215, 180),
        "shirt": (170, 90, 220),
        "pants": (40, 40, 60),
        "hair": (240, 200, 70)
    },
    "pigeon": {}
}


SCENES = {
    "classroom": ((238, 216, 150), (150, 105, 70)),
    "bedroom": ((176, 198, 240), (120, 90, 70)),
    "kitchen": ((252, 238, 200), (196, 200, 212)),
    "office": ((186, 212, 200), (90, 100, 122)),
    "lab": ((58, 70, 96), (36, 42, 60)),
    "stage": ((88, 28, 112), (62, 20, 72)),
    "street": ((110, 196, 255), (70, 70, 82)),
    "city_night": ((18, 18, 60), (35, 35, 65)),
    "park": ((130, 210, 255), (96, 196, 96)),
    "space": ((4, 4, 24), (25, 10, 55))
}


def clean_beat(beat):
    if not isinstance(beat, dict):
        raise ValueError("A story beat was not a JSON object.")

    beat["text"] = str(beat.get("text") or "").strip()
    beat["label"] = str(
        beat.get("label") or ""
    ).upper()[:28]

    if beat.get("scene") not in SCENE_NAMES:
        beat["scene"] = "street"

    if beat.get("fx") not in FX_NAMES:
        beat["fx"] = "none"

    chars = []

    for char in (beat.get("chars") or [])[:3]:
        if not isinstance(char, dict):
            continue

        who = char.get("who")
        if who not in ROSTER:
            who = "kevin"

        emotion = char.get("emotion")
        if emotion not in EMOTIONS:
            emotion = "neutral"

        action = char.get("action")
        if action not in ACTIONS:
            action = "idle"

        chars.append({
            "who": who,
            "emotion": emotion,
            "action": action
        })

    beat["chars"] = chars or [{
        "who": "kevin",
        "emotion": "neutral",
        "action": "idle"
    }]

    objects = []

    for obj in (beat.get("objects") or [])[:3]:
        if not isinstance(obj, dict):
            continue

        kind = obj.get("type")
        if kind not in OBJECT_NAMES:
            kind = "none"

        action = obj.get("action")
        if action not in OBJECT_ACTIONS:
            action = "idle"

        try:
            size = float(obj.get("size", 1.0))
        except (TypeError, ValueError):
            size = 1.0

        position = obj.get("position")

        if position not in ("left", "center", "right"):
            position = "center"

        objects.append({
            "type": kind,
            "action": action,
            "size": clamp(size, 0.4, 2.8),
            "position": position
        })

    beat["objects"] = objects

    visual = beat.get("visual")

    if not isinstance(visual, str) or not visual.strip():
        visual = beat["scene"].replace("_", " ")

    beat["visual"] = visual[:180]


def write_episode(number, state):
    history = "\n".join(
        f"EP {episode.get('ep')}: "
        f"{episode.get('title')} - "
        f"{episode.get('summary', '')}"
        for episode in state["episodes"][-7:]
    ) or "No previous episodes."

    trends = trending_hints()

    prompt = f"""
You are the head writer of a funny animated YouTube Shorts series.

SERIES: {SERIES_NAME}
PREMISE: {CFG['premise']}

PREVIOUS EPISODES:
{history}

OPTIONAL TREND IDEAS:
{trends}

Create episode {number}.

STORY RULES:
- Write 10 to 13 beats.
- Each beat is one spoken sentence, ideally 7 to 14 words.
- Total story should contain 90 to 145 spoken words.
- The opening must immediately create curiosity.
- Every beat must add a new event, twist, reaction, or escalation.
- No slow introductions or filler.
- Make the story understandable without context.
- Kevin is the main character.
- Dave warns Kevin and gets ignored.
- The pigeon appears in at least two beats.
- End with a funny cliffhanger about Kevin's next idea.
- Ask viewers what Kevin should try next.
- PG comedy only.
- Do not give real dangerous instructions.
- Do not copy existing videos or scripts.
- Avoid repeating previous episode ideas.

VISUAL RULES:
The visuals must literally show what the narrator describes.
Use concrete visible objects and actions.
Do not use vague descriptions like "Kevin is shocked."

For every beat return:
1. text: exact spoken sentence.
2. scene: one of {SCENE_NAMES}.
3. visual: concrete description of what viewers see.
4. chars: characters on screen.
5. objects: actual objects visible.
6. fx: visual effect.
7. label: short on-screen text.

CHARACTER OPTIONS:
kevin, dave, cop, boss, kid, pigeon

EMOTIONS:
{EMOTIONS}

CHARACTER ACTIONS:
{ACTIONS}

OBJECT TYPES:
{OBJECT_NAMES}

OBJECT ACTIONS:
{OBJECT_ACTIONS}

OBJECT POSITIONS:
left, center, right

EFFECTS:
{FX_NAMES}

Examples:
Kevin launches a giant pancake:
objects=[{{"type":"pancake","action":"fly","size":2.0,"position":"center"}}]

A phone shakes:
objects=[{{"type":"phone","action":"shake","size":1.0,"position":"center"}}]

A giant rocket appears:
objects=[{{"type":"rocket","action":"grow","size":2.0,"position":"right"}}]

Kevin gets rich:
objects=[{{"type":"money","action":"fall","size":1.5,"position":"center"}}]

Return valid JSON only:
{{
"title": "title under 60 characters",
"description": "short description",
"tags": ["tag1", "tag2"],
"summary": "one sentence summary",
"beats": [
{{
"text": "spoken line",
"scene": "kitchen",
"visual": "Kevin stands beside a giant pancake cannon",
"label": "PANCAKE MODE",
"fx": "sparkle",
"chars": [
{{"who":"kevin","emotion":"shock","action":"jump"}},
{{"who":"dave","emotion":"scared","action":"panic"}}
],
"objects": [
{{"type":"pancake","action":"fly","size":2.0,"position":"center"}}
]
}}
]
}}
"""

    for attempt in range(4):
        data = llm_json(prompt)
        beats = data.get("beats") or []

        if not isinstance(beats, list):
            continue

        words = sum(
            len(str(beat.get("text", "")).split())
            for beat in beats
            if isinstance(beat, dict)
        )

        valid = (
            9 <= len(beats) <= 14
            and CFG["words_min"] <= words <= CFG["words_max"] + 20
            and all(
                isinstance(beat, dict) and beat.get("text")
                for beat in beats
            )
        )

        if valid:
            for beat in beats:
                clean_beat(beat)

            return data

        print("Story failed validation. Regenerating...")

    raise RuntimeError(
        "AI could not create a valid episode after four attempts."
    )


# ============================================================
# VOICE GENERATION
# ============================================================

_KP = None


def init_tts():
    global _KP

    try:
        from kokoro import KPipeline

        _KP = KPipeline(lang_code="a")
        CFG["engine"] = "kokoro"

        print("Kokoro voice engine ready:", CFG["voice"])
        return

    except Exception as exc:
        print("Kokoro unavailable:", exc)

    CFG["engine"] = "piper"

    if os.getenv("GITHUB_ACTIONS"):
        try:
            subprocess.run(
                [
                    "sudo", "apt-get", "install",
                    "-y", "-qq", "espeak-ng"
                ],
                check=False
            )

            subprocess.run(
                [
                    sys.executable, "-m", "pip",
                    "install", "-q", "piper-tts"
                ],
                check=False
            )

        except Exception as exc:
            print("Piper setup issue:", exc)


def tts(text, wav):
    if CFG["engine"] == "kokoro":
        import numpy as np
        import soundfile as sf

        chunks = []

        for _, _, audio in _KP(
            text,
            voice=CFG["voice"],
            speed=CFG["speed"]
        ):
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            elif hasattr(audio, "numpy"):
                audio = audio.numpy()

            chunks.append(
                np.asarray(audio, dtype=np.float32)
            )

        if not chunks:
            raise RuntimeError("Kokoro returned no audio.")

        sf.write(
            str(wav),
            np.concatenate(chunks),
            24000
        )

    else:
        run(
            [
                CFG["piper_bin"],
                "--model", CFG["piper_model"],
                "--length_scale", "0.90",
                "--output_file", str(wav)
            ],
            input=text
        )


def voice_beats(beats, folder):
    durations = []
    files = []

    for index, beat in enumerate(beats):
        raw = folder / f"voice_raw_{index}.wav"
        final = folder / f"voice_{index}.wav"

        tts(beat["text"], raw)

        run([
            "ffmpeg", "-y",
            "-i", str(raw),
            "-af",
            "highpass=f=75,"
            "lowpass=f=11500,"
            "equalizer=f=250:t=q:w=1:g=-1.5,"
            "equalizer=f=3200:t=q:w=1:g=1.5,"
            "acompressor=threshold=0.12:ratio=2.5:attack=15:release=120,"
            "loudnorm=I=-16:TP=-1.5:LRA=8,"
            "apad=pad_dur=0.12",
            "-ar", "24000",
            "-ac", "1",
            str(final)
        ])

        raw.unlink(missing_ok=True)

        durations.append(probe_duration(final))
        files.append(final)

    concat_file = folder / "voice_list.txt"

    concat_file.write_text(
        "".join(
            f"file '{file.name}'\n"
            for file in files
        ),
        encoding="utf-8"
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-i", concat_file.name,
        "-c", "copy",
        "voice.wav"
    ], cwd=folder)

    return durations


# ============================================================
# WORD CAPTIONS
# ============================================================

def transcribe_words(wav):
    from faster_whisper import WhisperModel

    model = WhisperModel(
        CFG["whisper_model"],
        device="cpu",
        compute_type="int8"
    )

    segments, _ = model.transcribe(
        str(wav),
        word_timestamps=True,
        vad_filter=True
    )

    return [
        (word.word.strip(), word.start, word.end)
        for segment in segments
        for word in segment.words
        if word.word.strip()
    ]


def ass_time(seconds):
    return (
        f"{int(seconds // 3600)}:"
        f"{int(seconds % 3600 // 60):02d}:"
        f"{seconds % 60:05.2f}"
    )


def write_ass(words, beats, durations, episode, path):
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,"
        "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Word,@FONT,112,&H0000E5FF,&H000000FF,&H00000000,"
        "&H80000000,-1,0,0,0,100,100,0,0,1,9,3,5,60,60,60,1\n"
        "Style: Head,@FONT,44,&H00FFFFFF,&H000000FF,&H00000000,"
        "&H80000000,-1,0,0,0,100,100,2,0,1,5,0,8,60,60,60,1\n"
        "Style: Label,@FONT,68,&H00FFFFFF,&H000000FF,&H003030E0,"
        "&H00000000,-1,0,0,0,100,100,0,0,3,12,0,8,60,60,60,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,"
        "MarginV,Effect,Text\n"
    ).replace("@FONT", CFG["font"])

    total = sum(durations)

    events = [
        f"Dialogue: 0,{ass_time(0)},{ass_time(total)},Head,,0,0,0,,"
        f"{{\\an8\\pos(540,150)}}"
        f"{SERIES_NAME.upper()}  |  EP {episode}"
    ]

    elapsed = 0.0

    for beat, duration in zip(beats, durations):
        label = re.sub(
            r"[{}\\]",
            "",
            beat.get("label", "")
        )

        if label:
            events.append(
                f"Dialogue: 0,{ass_time(elapsed)},"
                f"{ass_time(elapsed + duration)},Label,,0,0,0,,"
                f"{{\\an8\\pos(540,270)}}{label}"
            )

        elapsed += duration

    pop = (
        "\\fscx85\\fscy85"
        "\\t(0,90,\\fscx108\\fscy108)"
        "\\t(90,150,\\fscx100\\fscy100)"
    )

    for index, (word, start, end) in enumerate(words):
        next_start = (
            words[index + 1][1]
            if index + 1 < len(words)
            else None
        )

        if next_start is not None and next_start - end < 0.35:
            finish = next_start
        else:
            finish = end + 0.06

        finish = max(finish, start + 0.10)

        clean = re.sub(
            r"[{}\\]",
            "",
            word
        ).upper()

        events.append(
            f"Dialogue: 1,{ass_time(start)},{ass_time(finish)},"
            f"Word,,0,0,0,,"
            f"{{\\an5\\pos(540,1120){pop}}}{clean}"
        )

    path.write_text(
        header + "\n".join(events) + "\n",
        encoding="utf-8"
    )


# ============================================================
# CARTOON BACKGROUNDS
# ============================================================

def lerp(a, b, t):
    return a + (b - a) * t


def mix(a, b, t):
    return tuple(
        int(lerp(x, y, t))
        for x, y in zip(a, b)
    )


def ease(value):
    value = clamp(value, 0, 1)
    return 1 - (1 - value) ** 3


def make_bg(scene):
    top, bottom = SCENES.get(
        scene,
        SCENES["street"]
    )

    img = Image.new("RGB", (TW, TH))
    draw = ImageDraw.Draw(img)

    for y in range(TH):
        color = mix(top, bottom, y / TH)
        draw.line(
            [(0, y), (TW, y)],
            fill=color
        )

    rnd = random.Random(
        sum(map(ord, scene))
    )

    if scene in (
        "classroom", "bedroom", "kitchen",
        "office", "lab", "stage"
    ):
        draw.rectangle(
            [0, 870, TW, TH],
            fill=bottom,
            outline=INK,
            width=4
        )

        if scene == "classroom":
            draw.rectangle(
                [70, 170, 650, 470],
                fill=(40, 92, 72),
                outline=(130, 85, 40),
                width=14
            )

            draw.text(
                (360, 320),
                "???",
                fill=(255, 255, 255),
                anchor="mm"
            )

        elif scene == "bedroom":
            draw.rectangle(
                [20, 700, 330, 880],
                fill=(235, 235, 252),
                outline=INK,
                width=5
            )

            draw.rectangle(
                [470, 150, 690, 420],
                fill=(150, 212, 255),
                outline=INK,
                width=7
            )

        elif scene == "kitchen":
            draw.rectangle(
                [0, 700, TW, 780],
                fill=(250, 250, 250),
                outline=INK,
                width=4
            )

            draw.rectangle(
                [0, 780, TW, 870],
                fill=(140, 100, 70),
                outline=INK,
                width=4
            )

        elif scene == "office":
            draw.rectangle(
                [60, 800, 660, 870],
                fill=(150, 105, 70),
                outline=INK,
                width=5
            )

            draw.rectangle(
                [250, 640, 470, 790],
                fill=(50, 55, 70),
                outline=INK,
                width=6
            )

        elif scene == "lab":
            draw.rectangle(
                [40, 770, 680, 860],
                fill=(120, 130, 150),
                outline=INK,
                width=5
            )

            for x, color in (
                (120, (90, 255, 140)),
                (330, (255, 90, 200)),
                (540, (90, 200, 255))
            ):
                draw.rectangle(
                    [x, 660, x + 70, 770],
                    fill=color,
                    outline=INK,
                    width=4
                )

        else:
            draw.polygon(
                [(0, 0), (180, 0), (80, 870), (0, 870)],
                fill=(190, 30, 50)
            )

            draw.polygon(
                [
                    (TW, 0),
                    (TW - 180, 0),
                    (TW - 80, 870),
                    (TW, 870)
                ],
                fill=(190, 30, 50)
            )

    elif scene in ("street", "city_night"):
        night = scene == "city_night"
        x = -10

        while x < TW:
            width = rnd.randint(90, 170)
            height = rnd.randint(320, 700)

            if night:
                color = mix(
                    (35, 35, 75),
                    (95, 70, 125),
                    rnd.random()
                )
            else:
                color = mix(
                    (240, 120, 120),
                    (255, 215, 120),
                    rnd.random()
                )

            draw.rectangle(
                [x, 860 - height, x + width, 860],
                fill=color,
                outline=INK,
                width=4
            )

            for wy in range(860 - height + 35, 820, 60):
                for wx in range(x + 15, x + width - 25, 40):
                    window = (
                        (255, 230, 120)
                        if night and rnd.random() < 0.5
                        else (190, 230, 255)
                    )

                    draw.rectangle(
                        [wx, wy, wx + 20, wy + 30],
                        fill=window
                    )

            x += width + 5

        draw.rectangle(
            [0, 860, TW, TH],
            fill=(65, 65, 78)
        )

        for x in range(10, TW, 150):
            draw.rectangle(
                [x, 1120, x + 80, 1138],
                fill=(245, 245, 245)
            )

    elif scene == "park":
        draw.ellipse(
            [-200, 640, 520, 1000],
            fill=(110, 200, 110),
            outline=INK,
            width=4
        )

        draw.ellipse(
            [260, 600, 940, 1000],
            fill=(90, 184, 100),
            outline=INK,
            width=4
        )

        for x, height in (
            (90, 230),
            (590, 280),
            (430, 190)
        ):
            draw.rectangle(
                [x - 18, 860 - height, x + 18, 870],
                fill=(120, 80, 50)
            )

            draw.ellipse(
                [
                    x - 90,
                    860 - height - 130,
                    x + 90,
                    860 - height + 50
                ],
                fill=(60, 160, 70),
                outline=INK,
                width=4
            )

    else:
        for _ in range(150):
            x = rnd.randint(0, TW)
            y = rnd.randint(0, TH)
            radius = rnd.choice((1, 2, 3))

            draw.ellipse(
                [
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius
                ],
                fill=(255, 255, 255)
            )

        draw.ellipse(
            [360, 180, 680, 500],
            fill=(232, 124, 72),
            outline=INK,
            width=5
        )

    return img


# ============================================================
# CHARACTERS
# ============================================================

def rotate_points(pivot, points, angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)

    return [
        (
            pivot[0] + x * cosine - y * sine,
            pivot[1] + x * sine + y * cosine
        )
        for x, y in points
    ]


def draw_pigeon(draw, cx, base, unit, t, action):
    bob = math.sin(t * 8) * 5 * unit

    if action in ("shake", "panic"):
        cx += math.sin(t * 35) * 8 * unit

    draw.line(
        [
            (cx - 12 * unit, base - 30 * unit + bob),
            (cx - 12 * unit, base)
        ],
        fill=(255, 150, 40),
        width=max(2, int(6 * unit))
    )

    draw.line(
        [
            (cx + 16 * unit, base - 30 * unit + bob),
            (cx + 16 * unit, base)
        ],
        fill=(255, 150, 40),
        width=max(2, int(6 * unit))
    )

    draw.ellipse(
        [
            cx - 72 * unit,
            base - 118 * unit + bob,
            cx + 72 * unit,
            base - 22 * unit + bob
        ],
        fill=(162, 162, 178),
        outline=INK,
        width=4
    )

    draw.ellipse(
        [
            cx + 30 * unit,
            base - 156 * unit + bob,
            cx + 94 * unit,
            base - 92 * unit + bob
        ],
        fill=(142, 142, 162),
        outline=INK,
        width=4
    )

    draw.polygon(
        [
            (cx + 82 * unit, base - 136 * unit + bob),
            (cx + 120 * unit, base - 124 * unit + bob),
            (cx + 82 * unit, base - 114 * unit + bob)
        ],
        fill=(255, 172, 40),
        outline=INK
    )

    draw.ellipse(
        [
            cx + 65 * unit,
            base - 145 * unit + bob,
            cx + 83 * unit,
            base - 127 * unit + bob
        ],
        fill=(255, 255, 255),
        outline=INK
    )

    draw.ellipse(
        [
            cx + 72 * unit,
            base - 139 * unit + bob,
            cx + 79 * unit,
            base - 132 * unit + bob
        ],
        fill=INK
    )


def draw_character(
    draw, who, cx, base, unit,
    emotion, action, t, beat_t
):
    if who == "pigeon":
        draw_pigeon(
            draw,
            cx,
            base,
            unit * 0.9,
            t,
            action
        )
        return

    spec = ROSTER.get(who, ROSTER["kevin"])

    dx = 0
    dy = 0

    if action == "jump":
        dy = -abs(math.sin(t * 5)) * 125
    elif action == "shake":
        dx = math.sin(t * 32) * 10
    elif action == "panic":
        dx = math.sin(t * 30) * 12
        dy = -abs(math.sin(t * 14)) * 30
    elif action == "run":
        dx = math.sin(t * 8) * 18
        dy = abs(math.sin(t * 15)) * -18
    else:
        dy = math.sin(t * 6) * 4

    x = cx + dx
    y = base + dy

    leg_height = 88 * unit
    torso_height = 138 * unit
    torso_width = 124 * unit
    head_size = 112 * unit

    draw.ellipse(
        [
            x - 72 * unit,
            base - 10 * unit,
            x + 72 * unit,
            base + 16 * unit
        ],
        fill=(35, 35, 45)
    )

    for side in (-1, 1):
        leg_x = x + side * 24 * unit

        draw.rectangle(
            [
                leg_x - 19 * unit,
                y - leg_height,
                leg_x + 19 * unit,
                y - 8 * unit
            ],
            fill=spec["pants"],
            outline=INK,
            width=4
        )

        draw.rounded_rectangle(
            [
                leg_x - 24 * unit,
                y - 14 * unit,
                leg_x + 24 * unit,
                y + 2 * unit
            ],
            radius=max(3, int(6 * unit)),
            fill=(245, 245, 245),
            outline=INK,
            width=3
        )

    top = y - leg_height - torso_height

    draw.rounded_rectangle(
        [
            x - torso_width / 2,
            top,
            x + torso_width / 2,
            y - leg_height + 6 * unit
        ],
        radius=max(5, int(14 * unit)),
        fill=spec["shirt"],
        outline=INK,
        width=4
    )

    raised = (
        emotion in ("shock", "scared")
        or action in ("panic", "jump")
    )

    for side in (-1, 1):
        pivot = (
            x + side * (torso_width / 2),
            top + 22 * unit
        )

        angle = (
            2.5 * side
            if raised
            else 0.25 * side + math.sin(t * 6) * 0.12
        )

        points = rotate_points(
            pivot,
            [
                (-18 * unit, 0),
                (18 * unit, 0),
                (18 * unit, 120 * unit),
                (-18 * unit, 120 * unit)
            ],
            angle
        )

        draw.polygon(
            points,
            fill=spec["shirt"],
            outline=INK
        )

    head_top = top - head_size

    draw.rounded_rectangle(
        [
            x - head_size / 2,
            head_top,
            x + head_size / 2,
            head_top + head_size
        ],
        radius=max(5, int(22 * unit)),
        fill=spec["skin"],
        outline=INK,
        width=4
    )

    draw.rounded_rectangle(
        [
            x - head_size / 2,
            head_top,
            x + head_size / 2,
            head_top + 34 * unit
        ],
        radius=max(5, int(18 * unit)),
        fill=spec["hair"],
        outline=INK,
        width=3
    )

    if spec.get("hat"):
        draw.rectangle(
            [
                x - head_size / 2 - 8 * unit,
                head_top - 16 * unit,
                x + head_size / 2 + 8 * unit,
                head_top + 24 * unit
            ],
            fill=spec["hat"],
            outline=INK,
            width=4
        )

    face_y = head_top + head_size * 0.56

    eye_radius = (
        17 if emotion in ("shock", "scared") else 13
    ) * unit

    for side in (-1, 1):
        eye_x = x + side * 24 * unit

        draw.ellipse(
            [
                eye_x - eye_radius,
                face_y - 8 * unit - eye_radius,
                eye_x + eye_radius,
                face_y - 8 * unit + eye_radius
            ],
            fill=(255, 255, 255),
            outline=INK,
            width=3
        )

        pupil = eye_radius * 0.4

        draw.ellipse(
            [
                eye_x - pupil,
                face_y - 8 * unit - pupil,
                eye_x + pupil,
                face_y - 8 * unit + pupil
            ],
            fill=INK
        )

        if spec.get("glasses"):
            draw.ellipse(
                [
                    eye_x - eye_radius - 5 * unit,
                    face_y - 8 * unit - eye_radius - 5 * unit,
                    eye_x + eye_radius + 5 * unit,
                    face_y - 8 * unit + eye_radius + 5 * unit
                ],
                outline=INK,
                width=3
            )

    mouth_y = face_y + 30 * unit

    if emotion == "happy":
        draw.arc(
            [
                x - 24 * unit,
                mouth_y - 24 * unit,
                x + 24 * unit,
                mouth_y + 14 * unit
            ],
            15, 165,
            fill=INK,
            width=5
        )

    elif emotion in ("shock", "scared"):
        draw.ellipse(
            [
                x - 14 * unit,
                mouth_y - 5 * unit,
                x + 14 * unit,
                mouth_y + 28 * unit
            ],
            fill=(70, 12, 24),
            outline=INK,
            width=3
        )

    elif emotion == "angry":
        draw.arc(
            [
                x - 20 * unit,
                mouth_y - 2 * unit,
                x + 20 * unit,
                mouth_y + 30 * unit
            ],
            200, 340,
            fill=INK,
            width=5
        )

    else:
        draw.line(
            [
                (x - 14 * unit, mouth_y),
                (x + 14 * unit, mouth_y)
            ],
            fill=INK,
            width=5
        )


# ============================================================
# OBJECTS AND STORY ACTIONS
# ============================================================

def object_position(position):
    return {
        "left": TW * 0.24,
        "center": TW * 0.50,
        "right": TW * 0.76
    }.get(position, TW * 0.5)


def draw_object(draw, obj, beat_t, index):
    kind = obj["type"]

    if kind == "none":
        return

    unit = obj["size"]
    x = object_position(obj["position"])
    y = 680

    action = obj["action"]

    if action == "bounce":
        y -= abs(
            math.sin(beat_t * 7 + index)
        ) * 100

    elif action in ("fall", "drop"):
        y = 220 + ease(beat_t / 0.7) * 460

    elif action == "grow":
        unit *= 0.6 + ease(beat_t / 0.8) * 0.8

    elif action == "fly":
        x += math.sin(beat_t * 3 + index) * 120
        y -= ease(beat_t / 0.7) * 280

    elif action == "spin":
        x += math.sin(beat_t * 5) * 40
        y += math.cos(beat_t * 5) * 30

    elif action == "shake":
        x += math.sin(beat_t * 35) * 18

    w = 100 * unit
    h = 100 * unit

    if kind == "phone":
        draw.rounded_rectangle(
            [
                x - w * 0.45, y - h,
                x + w * 0.45, y + h
            ],
            radius=20,
            fill=(35, 40, 55),
            outline=INK,
            width=6
        )

        draw.rectangle(
            [
                x - w * 0.34, y - h * 0.78,
                x + w * 0.34, y + h * 0.65
            ],
            fill=(80, 190, 240)
        )

        draw.ellipse(
            [
                x - 7, y + h * 0.76,
                x + 7, y + h * 0.9
            ],
            fill=(220, 220, 220)
        )

    elif kind in ("box", "present", "suitcase"):
        color = (
            (190, 105, 45)
            if kind == "box"
            else (180, 55, 90)
        )

        draw.rectangle(
            [
                x - w, y - h * 0.6,
                x + w, y + h * 0.6
            ],
            fill=color,
            outline=INK,
            width=6
        )

        draw.rectangle(
            [
                x - w * 0.12, y - h * 0.6,
                x + w * 0.12, y + h * 0.6
            ],
            fill=(255, 215, 70),
            outline=INK,
            width=3
        )

    elif kind in ("rocket", "machine"):
        draw.rounded_rectangle(
            [
                x - w * 0.42, y - h * 1.5,
                x + w * 0.42, y + h * 0.8
            ],
            radius=35,
            fill=(205, 210, 220),
            outline=INK,
            width=6
        )

        draw.polygon(
            [
                (x - w * 0.42, y - h * 0.9),
                (x - w, y + h * 0.4),
                (x - w * 0.42, y + h * 0.2)
            ],
            fill=(230, 55, 55),
            outline=INK
        )

        draw.polygon(
            [
                (x + w * 0.42, y - h * 0.9),
                (x + w, y + h * 0.4),
                (x + w * 0.42, y + h * 0.2)
            ],
            fill=(230, 55, 55),
            outline=INK
        )

        draw.ellipse(
            [
                x - w * 0.22, y - h * 0.8,
                x + w * 0.22, y - h * 0.35
            ],
            fill=(70, 180, 240),
            outline=INK,
            width=4
        )

        if action in ("fly", "explode"):
            draw.polygon(
                [
                    (x - w * 0.25, y + h * 0.7),
                    (
                        x,
                        y + h * 1.8
                        + abs(math.sin(beat_t * 15)) * 60
                    ),
                    (x + w * 0.25, y + h * 0.7)
                ],
                fill=(255, 120, 20)
            )

    elif kind in ("pancake", "pizza", "cake"):
        color = {
            "pancake": (225, 165, 75),
            "pizza": (235, 180, 75),
            "cake": (245, 180, 195)
        }[kind]

        draw.ellipse(
            [
                x - w, y - h * 0.45,
                x + w, y + h * 0.45
            ],
            fill=color,
            outline=INK,
            width=6
        )

        if kind == "pizza":
            for dx, dy in (
                (-30, -10),
                (30, 5),
                (0, 15)
            ):
                draw.ellipse(
                    [
                        x + dx - 10,
                        y + dy - 10,
                        x + dx + 10,
                        y + dy + 10
                    ],
                    fill=(190, 50, 45)
                )

        elif kind == "cake":
            draw.rectangle(
                [
                    x - w * 0.8, y - 5,
                    x + w * 0.8, y + h * 0.4
                ],
                fill=(245, 220, 220),
                outline=INK,
                width=4
            )

    elif kind in ("money", "giant_coin"):
        color = (
            (80, 190, 100)
            if kind == "money"
            else (255, 205, 40)
        )

        draw.ellipse(
            [
                x - w, y - h,
                x + w, y + h
            ],
            fill=color,
            outline=INK,
            width=6
        )

        draw.text(
            (x, y),
            "$" if kind == "money" else "★",
            fill=(255, 255, 255),
            anchor="mm",
            stroke_width=3,
            stroke_fill=INK
        )

    elif kind in ("car", "chair"):
        draw.rounded_rectangle(
            [
                x - w * 1.3, y - h * 0.6,
                x + w * 1.3, y + h * 0.35
            ],
            radius=25,
            fill=(210, 45, 55),
            outline=INK,
            width=6
        )

        draw.ellipse(
            [
                x - w, y + h * 0.1,
                x - w * 0.45, y + h * 0.7
            ],
            fill=(35, 35, 45),
            outline=INK,
            width=4
        )

        draw.ellipse(
            [
                x + w * 0.45, y + h * 0.1,
                x + w, y + h * 0.7
            ],
            fill=(35, 35, 45),
            outline=INK,
            width=4
        )

    elif kind in ("button", "ball"):
        color = (
            (230, 45, 50)
            if kind == "button"
            else (60, 140, 235)
        )

        draw.ellipse(
            [
                x - w, y - w,
                x + w, y + w
            ],
            fill=color,
            outline=INK,
            width=7
        )

        if kind == "button":
            draw.text(
                (x, y),
                "DO NOT",
                fill=(255, 255, 255),
                anchor="mm"
            )

    elif kind in ("computer", "tv"):
        draw.rectangle(
            [
                x - w, y - h,
                x + w, y + h * 0.65
            ],
            fill=(35, 40, 55),
            outline=INK,
            width=7
        )

        draw.rectangle(
            [
                x - w * 0.8, y - h * 0.8,
                x + w * 0.8, y + h * 0.4
            ],
            fill=(80, 185, 235)
        )

        draw.rectangle(
            [
                x - 10, y + h * 0.65,
                x + 10, y + h
            ],
            fill=(70, 70, 80)
        )

    elif kind in ("trophy", "planet"):
        color = (
            (255, 205, 40)
            if kind == "trophy"
            else (105, 110, 220)
        )

        draw.ellipse(
            [
                x - w, y - h,
                x + w, y + h
            ],
            fill=color,
            outline=INK,
            width=6
        )

        if kind == "trophy":
            draw.rectangle(
                [
                    x - w * 0.2, y + h * 0.6,
                    x + w * 0.2, y + h * 1.1
                ],
                fill=color,
                outline=INK,
                width=4
            )

            draw.rectangle(
                [
                    x - w * 0.6, y + h * 1.05,
                    x + w * 0.6, y + h * 1.3
                ],
                fill=color,
                outline=INK,
                width=4
            )

    elif kind == "table":
        draw.rectangle(
            [
                x - w * 1.3, y - h * 0.2,
                x + w * 1.3, y + h * 0.1
            ],
            fill=(150, 95, 55),
            outline=INK,
            width=5
        )

        draw.rectangle(
            [
                x - w, y + h * 0.1,
                x - w * 0.8, y + h
            ],
            fill=(150, 95, 55),
            outline=INK,
            width=4
        )

        draw.rectangle(
            [
                x + w * 0.8, y + h * 0.1,
                x + w, y + h
            ],
            fill=(150, 95, 55),
            outline=INK,
            width=4
        )

    elif kind == "sign":
        draw.rectangle(
            [
                x - w, y - h,
                x + w, y + h * 0.5
            ],
            fill=(255, 220, 65),
            outline=INK,
            width=6
        )

        draw.text(
            (x, y - h * 0.2),
            "WARNING",
            fill=INK,
            anchor="mm"
        )


# ============================================================
# EFFECTS AND VISUAL DESCRIPTIONS
# ============================================================

def draw_effect(draw, effect, beat_t, frame):
    cx = TW / 2
    cy = 500

    if effect == "explosion":
        radius = (
            40
            + ease(beat_t / 0.35) * 280
            + math.sin(beat_t * 25) * 10
        )

        for scale, color in (
            (1.0, (220, 40, 40)),
            (0.78, (255, 130, 20)),
            (0.53, (255, 220, 55)),
            (0.27, (255, 255, 220))
        ):
            points = []

            for i in range(24):
                r = radius * scale * (
                    1 if i % 2 == 0 else 0.55
                )

                angle = i / 24 * math.tau

                points.append((
                    cx + math.cos(angle) * r,
                    cy + math.sin(angle) * r
                ))

            draw.polygon(points, fill=color)

    elif effect == "fire":
        for i in range(7):
            x = 40 + i * 100
            height = 170 + 70 * math.sin(
                beat_t * 10 + i
            )

            draw.polygon(
                [
                    (x - 45, 1000),
                    (x, 1000 - height),
                    (x + 45, 1000)
                ],
                fill=(255, 100, 15)
            )

            draw.polygon(
                [
                    (x - 20, 1000),
                    (x, 1000 - height * 0.6),
                    (x + 20, 1000)
                ],
                fill=(255, 220, 65)
            )

    elif effect == "shock":
        if beat_t < 0.35:
            draw.rectangle(
                [0, 0, TW, TH],
                fill=(255, 255, 255)
            )

    elif effect == "sparkle":
        rnd = random.Random(7)

        for i in range(10):
            x = rnd.randint(40, TW - 40)
            y = rnd.randint(150, 900)

            radius = 12 + 12 * abs(
                math.sin(beat_t * 7 + i)
            )

            draw.line(
                [
                    (x - radius, y),
                    (x + radius, y)
                ],
                fill=(255, 245, 140),
                width=5
            )

            draw.line(
                [
                    (x, y - radius),
                    (x, y + radius)
                ],
                fill=(255, 245, 140),
                width=5
            )


def draw_visual_description(draw, text, font):
    if not text:
        return

    clean = re.sub(
        r"[^A-Za-z0-9 !?'$-]",
        "",
        text
    ).strip()

    if not clean:
        return

    if len(clean) > 48:
        clean = clean[:45] + "..."

    draw.rounded_rectangle(
        [35, 520, TW - 35, 580],
        radius=16,
        fill=(25, 25, 40),
        outline=(255, 255, 255),
        width=2
    )

    draw.text(
        (TW / 2, 550),
        clean,
        font=font,
        fill=(255, 255, 255),
        anchor="mm"
    )


# ============================================================
# CARTOON VIDEO RENDERER
# ============================================================

def toon_background(beats, durations, folder):
    bgs = {}

    font_path = (
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf"
    )

    if Path(font_path).exists():
        font = ImageFont.truetype(font_path, 22)
    else:
        font = ImageFont.load_default()

    output_path = folder / "bg.mp4"

    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{TW}x{TH}",
        "-r", str(FPS),
        "-i", "-",
        "-vf", "scale=1080:1920:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    frame_number = 0

    try:
        for index, (beat, duration) in enumerate(
            zip(beats, durations)
        ):
            scene = beat["scene"]

            if scene not in bgs:
                bgs[scene] = make_bg(scene)

            characters = beat["chars"]
            count = len(characters)

            sizes = {
                1: 1.55,
                2: 1.25,
                3: 1.00
            }

            positions = {
                1: [0.50],
                2: [0.30, 0.70],
                3: [0.20, 0.50, 0.80]
            }

            start_frame = frame_number

            end_frame = (
                start_frame
                + round(duration * FPS)
            )

            for frame in range(start_frame, end_frame):
                beat_t = (frame - start_frame) / FPS
                global_t = frame / FPS

                img = bgs[scene].copy()
                draw = ImageDraw.Draw(img)

                draw_effect(
                    draw,
                    beat["fx"],
                    beat_t,
                    frame
                )

                for obj_index, obj in enumerate(
                    beat["objects"]
                ):
                    draw_object(
                        draw,
                        obj,
                        beat_t,
                        obj_index
                    )

                for char, position in zip(
                    characters,
                    positions[count]
                ):
                    draw_character(
                        draw,
                        char["who"],
                        TW * position,
                        1020,
                        sizes[count],
                        char["emotion"],
                        char["action"],
                        global_t,
                        beat_t
                    )

                if beat["fx"] == "sparkle":
                    draw_effect(
                        draw,
                        "sparkle",
                        beat_t,
                        frame
                    )

                if beat_t < 1.2 and beat.get("visual"):
                    draw_visual_description(
                        draw,
                        beat["visual"],
                        font
                    )

                progress = (
                    index + beat_t / max(duration, 0.1)
                ) / len(beats)

                draw.rounded_rectangle(
                    [95, 420, 625, 454],
                    radius=16,
                    fill=(25, 25, 40),
                    outline=(255, 255, 255),
                    width=2
                )

                progress_right = clamp(
                    99 + max(5, int(522 * progress)),
                    104,
                    621
                )

                draw.rounded_rectangle(
                    [
                        99, 424,
                        progress_right, 450
                    ],
                    radius=12,
                    fill=mix(
                        (65, 220, 100),
                        (245, 55, 55),
                        progress
                    )
                )

                draw.text(
                    (TW / 2, 395),
                    "CHAOS LEVEL",
                    font=font,
                    fill=(255, 255, 255),
                    anchor="mm",
                    stroke_width=2,
                    stroke_fill=INK
                )

                zoom = (
                    1.0
                    + 0.10 * (1 - ease(beat_t / 0.25))
                    + 0.025 * math.sin(beat_t * 2)
                )

                zoom = max(1.0, zoom)

                shake = (
                    12 * max(0, 1 - beat_t / 0.5)
                    if beat["fx"] in ("explosion", "shock")
                    else 0
                )

                shake_x = math.sin(frame * 2.7) * shake
                shake_y = math.cos(frame * 3.1) * shake

                # Safe crop: never allow a crop box outside the image.
                crop_w = max(1, min(TW, TW / zoom))
                crop_h = max(1, min(TH, TH / zoom))

                left = clamp(
                    TW / 2 + shake_x - crop_w / 2,
                    0,
                    TW - crop_w
                )

                top = clamp(
                    TH / 2 + shake_y - crop_h / 2,
                    0,
                    TH - crop_h
                )

                right = min(
                    TW,
                    max(left + 1, left + crop_w)
                )

                bottom = min(
                    TH,
                    max(top + 1, top + crop_h)
                )

                img = img.crop((
                    int(left),
                    int(top),
                    int(right),
                    int(bottom)
                ))

                img = img.resize(
                    (TW, TH),
                    Image.Resampling.BILINEAR
                )

                process.stdin.write(img.tobytes())

            frame_number = end_frame

    except Exception:
        if process.stdin:
            process.stdin.close()

        process.wait()
        raise

    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()

    if process.wait() != 0:
        raise RuntimeError(
            "Cartoon video rendering failed."
        )

    return output_path, []


# ============================================================
# OPTIONAL STOCK FOOTAGE
# ============================================================

def pixabay_candidates(query):
    key = os.getenv("PIXABAY_API_KEY")

    if not key:
        return []

    response = requests.get(
        "https://pixabay.com/api/videos/",
        params={
            "key": key,
            "q": query[:100],
            "per_page": 20,
            "safesearch": "true",
            "video_type": "film"
        },
        timeout=30
    )

    response.raise_for_status()

    results = []

    for hit in response.json().get("hits", []):
        videos = hit.get("videos", {})

        chosen = next(
            (
                videos[name]
                for name in ("medium", "small", "large")
                if videos.get(name, {}).get("url")
            ),
            None
        )

        if chosen:
            results.append((
                chosen["url"],
                hit.get("pageURL")
            ))

    return results


def download(url, path):
    with requests.get(
        url,
        stream=True,
        timeout=90
    ) as response:
        response.raise_for_status()

        with path.open("wb") as output:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    output.write(chunk)


def make_seg(raw, duration, output):
    raw_duration = probe_duration(raw)

    start = random.uniform(
        0,
        max(0.0, raw_duration - duration - 0.2)
    )

    vf = (
        "scale=1188:2112:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "eq=saturation=1.2:contrast=1.06,"
        "setsar=1,fps=30"
    )

    run([
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-stream_loop", "-1",
        "-i", str(raw),
        "-an",
        "-t", f"{duration:.2f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "21",
        "-pix_fmt", "yuv420p",
        str(output)
    ])


def build_background(beats, durations, folder):
    use_api = bool(os.getenv("PIXABAY_API_KEY"))

    local = pick_files(
        ROOT / "footage",
        VIDEO_EXT
    )

    pool = []
    used = set()
    sources = []
    segments = []

    for index, (beat, duration) in enumerate(
        zip(beats, durations)
    ):
        raw = None

        if use_api:
            try:
                candidates = [
                    item for item in pixabay_candidates(
                        beat["visual"]
                    )
                    if item[0] not in used
                ]

                if not candidates:
                    candidates = [
                        item for item in pixabay_candidates(
                            beat["scene"].replace("_", " ")
                        )
                        if item[0] not in used
                    ]

                if candidates:
                    url, page = random.choice(candidates[:8])

                    raw = folder / f"raw_{index}.mp4"

                    download(url, raw)

                    used.add(url)
                    pool.append(raw)
                    sources.append(page)

            except Exception as exc:
                print("Footage lookup failed:", exc)

        if raw is None and pool:
            raw = random.choice(pool)

        if raw is None and local:
            raw = random.choice(local)

        if raw is None:
            raise RuntimeError(
                f"No footage found for beat {index + 1}."
            )

        segment = folder / f"segment_{index}.mp4"

        make_seg(raw, duration, segment)

        segments.append(segment)

    concat_file = folder / "segments.txt"

    concat_file.write_text(
        "".join(
            f"file '{segment.name}'\n"
            for segment in segments
        ),
        encoding="utf-8"
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-i", concat_file.name,
        "-c", "copy",
        "bg.mp4"
    ], cwd=folder)

    return folder / "bg.mp4", sources


# ============================================================
# FINAL VIDEO
# ============================================================

def render(background, voice, workdir, output, duration):
    music_files = pick_files(
        ROOT / "music",
        AUDIO_EXT
    )

    music = (
        random.choice(music_files)
        if music_files
        else None
    )

    command = [
        "ffmpeg", "-y",
        "-i", str(background),
        "-i", str(voice)
    ]

    video_filter = (
        "[0:v]ass=captions.ass,"
        "scale=1080:1920:flags=lanczos[v];"
    )

    if music:
        command += [
            "-stream_loop", "-1",
            "-i", str(music)
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
        "-t", f"{duration + 0.25:.2f}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output)
    ]

    run(command, cwd=workdir)

    return music.name if music else None


# ============================================================
# EPISODE PRODUCTION
# ============================================================

def make_video(index):
    state = load_state()

    episode_number = len(state["episodes"]) + 1

    data = write_episode(
        episode_number,
        state
    )

    beats = data["beats"]

    folder = (
        OUT_DAY
        / f"{index:02d}_ep{episode_number}"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    (folder / "script.txt").write_text(
        "\n".join(
            f"[{beat['label']}] {beat['text']}"
            for beat in beats
        ),
        encoding="utf-8"
    )

    print("Generating narration...")

    durations = voice_beats(
        beats,
        folder
    )

    total_duration = sum(durations)

    print("Generating word captions...")

    words = transcribe_words(
        folder / "voice.wav"
    )

    write_ass(
        words,
        beats,
        durations,
        episode_number,
        folder / "captions.ass"
    )

    print("Animating scenes...")

    if CFG["visuals"] == "toon":
        background, sources = toon_background(
            beats,
            durations,
            folder
        )

    else:
        background, sources = build_background(
            beats,
            durations,
            folder
        )

    print("Rendering final video...")

    music = render(
        background,
        folder / "voice.wav",
        folder,
        folder / "video.mp4",
        total_duration
    )

    for pattern in (
        "voice_raw_*.wav",
        "voice_*.wav",
        "raw_*.mp4",
        "segment_*.mp4",
        "segments.txt",
        "voice_list.txt",
        "bg.mp4"
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
        "tags": data.get("tags", []),
        "duration_sec": round(total_duration, 1),
        "voice_engine": CFG["engine"],
        "voice": CFG["voice"],
        "visual_style": CFG["visuals"],
        "footage_sources": sources,
        "music_used": music,
        "review_checklist": [
            "Review the complete video before uploading",
            "Check captions and narration timing",
            "Confirm music and footage usage rights",
            "Use synthetic-content disclosure if applicable"
        ]
    }

    (folder / "meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    state["episodes"].append({
        "ep": episode_number,
        "title": data["title"],
        "summary": data.get("summary", "")
    })

    save_state(state)

    return str(folder), data["title"]


# ============================================================
# NOTIFICATIONS
# ============================================================

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
                "text": message[:4000]
            },
            timeout=15
        )

    except Exception:
        pass


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
                    "caption": caption[:900]
                },
                files={
                    "video": video
                },
                timeout=180
            )

            response.raise_for_status()

    except Exception as exc:
        print("Telegram video notification failed:", exc)


# ============================================================
# PREFLIGHT CHECKS
# ============================================================

def preflight():
    errors = []

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            errors.append(
                f"{tool} is missing from PATH."
            )

    if CFG["engine"] == "piper":
        if not (
            shutil.which(CFG["piper_bin"])
            or Path(CFG["piper_bin"]).exists()
        ):
            errors.append(
                "Piper is not installed or not found."
            )

        elif not Path(CFG["piper_model"]).exists():
            errors.append(
                "Piper model is missing: "
                + CFG["piper_model"]
            )

    if CFG["visuals"] == "stock":
        if (
            not os.getenv("PIXABAY_API_KEY")
            and not pick_files(ROOT / "footage", VIDEO_EXT)
        ):
            errors.append(
                "Stock mode needs a Pixabay API key "
                "or licensed footage in the footage folder."
            )

    if not os.getenv("GEMINI_API_KEY"):
        try:
            requests.get(
                CFG["ollama_url"].rsplit("/api", 1)[0],
                timeout=5
            )

        except Exception:
            errors.append(
                "No Gemini API key and Ollama is not running."
            )

    return errors


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DAY.mkdir(
        parents=True,
        exist_ok=True
    )

    completed = []
    errors = []

    print("Starting Shorts Factory v4...")
    print("Visual engine:", CFG["visuals"])

    init_tts()

    errors.extend(preflight())

    if not errors:
        existing = [
            path for path in OUT_DAY.iterdir()
            if path.is_dir()
        ]

        start_index = len(existing) + 1

        for offset in range(CFG["videos_per_run"]):
            try:
                path, title = make_video(
                    start_index + offset
                )

                completed.append((title, path))

                notify_video(
                    Path(path) / "video.mp4",
                    title
                )

            except Exception as exc:
                error_message = (
                    f"Video {offset + 1}: {exc}"
                )

                errors.append(error_message)

                with (
                    OUT_DAY / "errors.log"
                ).open("a", encoding="utf-8") as log:
                    log.write(
                        traceback.format_exc() + "\n"
                    )

                print(error_message)

    report = [
        f"# Daily report — {TODAY}",
        f"Completed: {len(completed)}",
        f"Errors: {len(errors)}",
        f"Voice: {CFG['engine']} / {CFG['voice']}",
        f"Visuals: {CFG['visuals']}",
        ""
    ]

    report.extend(
        f"- COMPLETE: {title} -> {path}"
        for title, path in completed
    )

    report.extend(
        f"- ERROR: {error}"
        for error in errors
    )

    report.append(
        "\nReview each video's video.mp4 and meta.json before uploading."
    )

    report_text = "\n".join(report)

    with (
        OUT_DAY / "report.md"
    ).open("a", encoding="utf-8") as output:
        output.write(report_text + "\n\n")

    print(report_text)

    notify(report_text)


if __name__ == "__main__":
    main()
