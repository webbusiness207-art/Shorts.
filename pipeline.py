#!/usr/bin/env python3
"""Shorts Factory v4 - "Huss Huss Huss" format Shorts.

A deadpan blocky avatar explains that everybody in town goes out partying
except him, because he is secretly grinding one skill. Between lines he does
the skill and breathes "Huss, huss, huss."
"""

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

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
TODAY = dt.date.today().isoformat()
OUT_DAY = ROOT / "output" / TODAY
STATE_FILE = ROOT / "used_topics.json"
SERIES_NAME = os.getenv("SERIES_NAME", "IN THIS TOWN")


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
        "PIPER_MODEL", str(ROOT / "models" / "en_US-lessac-medium.onnx")
    ),
    "gemini_model": os.getenv("GEMINI_MODEL", "").strip(),
    "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    "music_volume": max(0, min(1, env_float("MUSIC_VOLUME", 0.075))),
    "fps": 24,
    "blender_fps": env_int("BLENDER_FPS", 18),
    "width": 720,
    "height": 1280,
    "use_whisper": os.getenv("USE_WHISPER", "0") == "1",
}

TW, TH, FPS = CFG["width"], CFG["height"], CFG["fps"]
INK = (27, 31, 43)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
TRENDS_URL = "https://trends.google.com/trending/rss?geo=US"

# Vocabulary shared by the story prompt, the validator and both renderers.
SCENES = {
    "classroom", "bedroom", "kitchen", "office", "lab",
    "stage", "street", "city_night", "park", "space",
}
EMOTIONS = {"neutral", "happy", "shock", "scared", "angry", "sad"}
ACTIONS = {"idle", "jump", "panic", "point", "shake", "slide", "huss"}
EFFECTS = {"none", "explosion", "fire", "shock", "sparkle"}
ACTIVITIES = {
    "basketball", "jump_rope", "weights", "boxing", "running", "typing",
}
PERIOD = 0.42  # seconds per "huss" repetition


# ---------------------------------------------------------------------
# UTILITIES
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
                    raise RuntimeError(
                        "Gemini keeps returning HTTP 429. Check your Google "
                        "AI Studio quota and rate limits. " + last_error
                    )

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


def clean_beat(beat, default_activity):
    if not isinstance(beat, dict):
        raise ValueError("A story beat was not an object.")

    kind = "huss" if str(beat.get("kind", "line")).lower() == "huss" else "line"
    text = re.sub(r"\s+", " ", str(beat.get("text", ""))).strip()
    if kind == "huss" and not text:
        text = "Huss, huss, huss."
    if not text:
        raise ValueError("A story beat has no narration.")

    scene = beat.get("scene", "city_night")
    if scene not in SCENES:
        scene = "city_night"

    activity = beat.get("activity", default_activity)
    if activity not in ACTIVITIES:
        activity = default_activity

    emotion = beat.get("emotion", "neutral")
    if emotion not in EMOTIONS:
        emotion = "neutral"

    fx = beat.get("fx", "none")
    if fx not in EFFECTS:
        fx = "none"

    chars = [{
        "who": "kevin",
        "emotion": emotion,
        "action": "huss" if kind == "huss" else "idle",
    }]
    if beat.get("pigeon"):
        chars.append({"who": "pigeon", "emotion": "neutral", "action": "idle"})

    return {
        "label": str(beat.get("label", "STORY"))[:24],
        "kind": kind,
        "text": text,
        "scene": scene,
        "chars": chars,
        "activity": activity,
        "party": bool(beat.get("party", True)),
        "fx": fx,
        "visual": str(beat.get("visual", ""))[:80],
    }


def write_episode(episode_number, state):
    trends = trending_topics()
    recent = state.get("topics", [])[-40:]
    twist = random.choice(trends) if trends else "an everyday problem"
    activity = random.choice(sorted(ACTIVITIES))

    prompt = f"""
You write viral vertical YouTube Shorts in the "Huss Huss Huss" meme format:
a deadpan blocky Roblox-style avatar calmly tells the story of a town where
everybody goes out partying except him. Between his lines he is shown doing
one repetitive skill and making a heavy-breathing sound: "Huss, huss, huss."

FORMAT RULES
- One narrator avatar, flat deadpan delivery, quietly sad-funny.
- Beats are either kind "line" (a spoken sentence) or kind "huss" (the avatar
  does his skill and the narration is ONLY a breathing sound such as
  "Huss, huss, huss." or "Huss, huss, huss, huss.").
- Pattern: HOOK line, line, huss, line, line, huss, line, bigger escalation,
  huss, a twist or payoff, then a final short line or huss that loops back to
  the hook. Use at least 3 huss beats.
- The hook is the signature setup, reworded for this episode, for example:
  "In this town, everybody goes out partying except me."
- This episode's skill (activity): {activity}. Lines must make sense for it.
- Every line escalates the absurdity: why he trains, what he gave up, how good
  he is getting, what the town thinks. Lines are 6 to 14 words.
- A silent pigeon watches from the side in 2 to 3 beats (set "pigeon": true).
- Optional comedic twist inspiration (use loosely; no real people or brands):
  {twist}
- Avoid repeating these topics: {json.dumps(recent[-25:])}
- Family friendly. Original wording. No copyrighted characters, lyrics, brands
  or real people's names.

Return ONLY valid JSON in this exact shape:
{{
 "title":"short clickable title, under 70 chars",
 "description":"one sentence",
 "summary":"one sentence plot summary",
 "topic":"short topic label for this episode",
 "activity":"{activity}",
 "tags":["huss","shorts","funny"],
 "beats":[
  {{
   "label":"HOOK",
   "kind":"line",
   "text":"In this town, everybody goes out partying except me.",
   "scene":"city_night",
   "party":true,
   "emotion":"sad",
   "activity":"{activity}",
   "pigeon":false,
   "fx":"none",
   "visual":"short visible action"
  }}
 ]
}}

Make 10 to 14 beats and 35 to 110 spoken words in total.
Use only these values:
Scenes: {sorted(SCENES)}
Emotions: {sorted(EMOTIONS)}
Activities: {sorted(ACTIVITIES)}
Effects: {sorted(EFFECTS)}
"""

    last_error = None

    for attempt in range(4):
        try:
            data = llm_json(prompt)
            default_activity = data.get("activity")
            if default_activity not in ACTIVITIES:
                default_activity = activity

            beats = [clean_beat(x, default_activity) for x in data.get("beats", [])]
            words = sum(len(x["text"].split()) for x in beats)
            huss_count = sum(1 for x in beats if x["kind"] == "huss")

            if not 8 <= len(beats) <= 16 or not 30 <= words <= 140:
                raise ValueError(
                    f"Story length outside limits: {len(beats)} beats, {words} words"
                )
            if huss_count < 2:
                raise ValueError("Story needs at least 2 huss beats.")

            data["beats"] = beats
            data["activity"] = default_activity
            data["title"] = str(
                data.get("title", f"In This Town #{episode_number}")
            )[:90]
            data["topic"] = str(data.get("topic", f"{default_activity} grind"))[:100]
            data["summary"] = str(data.get("summary", ""))[:300]
            data["description"] = str(data.get("description", ""))[:500]
            data["tags"] = [str(t)[:30] for t in data.get("tags", [])[:12]]
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

    # Piper's CLI flag spelling differs between releases; try each variant.
    variants = [
        ["--length_scale", "0.94"],
        ["--length-scale", "0.94"],
        [],
    ]
    last_error = ""
    for extra in variants:
        output_path.unlink(missing_ok=True)
        result = subprocess.run(
            [binary, "--model", str(model), *extra,
             "--output_file", str(output_path)],
            input=text, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if (
            result.returncode == 0
            and output_path.exists()
            and output_path.stat().st_size > 1000
        ):
            return
        last_error = result.stderr[-1500:]

    raise RuntimeError("Piper failed: " + last_error)


def voice_beats(beats, folder):
    processed_files = []
    durations = []

    for index, beat in enumerate(beats):
        raw = folder / f"voice_raw_{index:02d}.wav"
        synthesize_piper(beat["text"], raw)

        # Huss beats get a minimum length so the animation can actually play.
        pad = "apad=whole_dur=1.7" if beat["kind"] == "huss" else "apad=pad_dur=0.18"

        processed = folder / f"voice_{index:02d}.wav"
        run([
            "ffmpeg", "-y", "-i", str(raw),
            "-af",
            "highpass=f=75,lowpass=f=11000,"
            "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=100,"
            "loudnorm=I=-16:TP=-1.5:LRA=11," + pad,
            "-ar", "24000", "-ac", "1", str(processed),
        ])

        raw.unlink(missing_ok=True)
        processed_files.append(processed)
        durations.append(probe_duration(processed))

    concat_list = folder / "voice_list.txt"
    concat_list.write_text(
        "".join(f"file '{p.name}'\n" for p in processed_files), encoding="utf-8"
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
    """Spread each beat's script words across that beat's exact duration."""
    result = []
    current = 0.0
    for beat, duration in zip(beats, durations):
        words = beat["text"].split()
        weights = [max(1, len(re.sub(r"\W", "", w))) for w in words]
        total = sum(weights) or 1
        # Leave the padded tail silent.
        speech = max(0.3, duration - (0.18 if beat["kind"] == "line" else 0.2))
        cursor = current
        for word, weight in zip(words, weights):
            span = speech * weight / total
            result.append((cursor, cursor + span, word))
            cursor += span
        current += duration
    return result


def transcribe_words(audio_path, beats, durations):
    """Script-accurate captions. Whisper timing only if it matches the script."""
    script_words = [w for b in beats for w in b["text"].split()]
    fallback = fallback_word_times(beats, durations)

    if not CFG["use_whisper"]:
        return fallback

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8"
        )
        segments, _ = model.transcribe(
            str(audio_path), word_timestamps=True, vad_filter=True
        )
        heard = []
        for segment in segments:
            for item in (segment.words or []):
                if item.word.strip():
                    heard.append((float(item.start), float(item.end)))

        if len(heard) == len(script_words):
            return [(s, e, w) for (s, e), w in zip(heard, script_words)]
        print("Whisper word count differs from script; using script timing.")
    except Exception as exc:
        print("Whisper unavailable; using script timing:", exc)

    return fallback


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
    series = SERIES_NAME.replace(",", "")

    lines.append(
        f"Dialogue: 0,0:00:00.00,{ass_time(min(2, total))},"
        f"Top,,0,0,0,,{series}  •  EPISODE {episode_number}\n"
    )

    for start, end, word in words:
        clean = word.replace("{", "").replace("}", "").replace("\\", "")
        if not clean:
            continue
        lines.append(
            f"Dialogue: 1,{ass_time(start)},{ass_time(max(start + 0.08, end))},"
            f"Default,,0,0,0,,{clean}\n"
        )

    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# CARTOON ("toon") VISUALS
# ---------------------------------------------------------------------

def mix(a, b, t):
    t = clamp(t, 0, 1)
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def ease(t):
    t = clamp(t, 0, 1)
    return t * t * (3 - 2 * t)


PALETTES = {
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


def make_bg(scene):
    top, bottom = PALETTES.get(scene, PALETTES["city_night"])
    img = Image.new("RGB", (TW, TH))
    draw = ImageDraw.Draw(img)

    for y in range(TH):
        draw.line((0, y, TW, y), fill=mix(top, bottom, y / TH))

    if scene == "space":
        rng = random.Random(12)
        for _ in range(90):
            x, y = rng.randrange(TW), rng.randrange(TH)
            r = rng.choice((1, 2, 3))
            draw.ellipse((x - r, y - r, x + r, y + r), fill="white")
        draw.ellipse((480, 150, 700, 370), fill=(100, 115, 205))
    else:
        draw.rectangle((0, 950, TW, TH), fill=(75, 100, 95))

        if scene in {"street", "city_night"}:
            rng = random.Random(scene)
            for x in range(0, TW, 90):
                height = rng.randint(250, 570)
                draw.rectangle((x, 950 - height, x + 75, 950), fill=(55, 65, 90))
                for wy in range(1000 - height, 900, 65):
                    for wx in range(x + 15, x + 65, 28):
                        draw.rectangle((wx, wy, wx + 10, wy + 18), fill=(255, 220, 120))
        else:
            draw.rectangle((40, 180, TW - 40, 850), fill=(235, 230, 205), outline=INK, width=8)
            draw.rectangle((75, 220, TW - 75, 800), fill=(180, 220, 230))
            draw.rectangle((0, 900, TW, 930), fill=(145, 105, 75))

    return img


def draw_party(draw, t):
    """Flashing party lights plus tiny dancing silhouettes in the background."""
    rng = random.Random(int(t / 0.25) + 5)
    colors = [(255, 60, 160), (60, 220, 255), (255, 225, 60),
              (150, 90, 255), (80, 255, 140)]
    for _ in range(9):
        x, y = rng.randint(30, TW - 30), rng.randint(220, 520)
        r = rng.randint(20, 46)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=rng.choice(colors))

    for i in range(6):
        x = 50 + i * 125
        hop = abs(math.sin(t * 6 + i)) * 20
        base = 940 - hop
        draw.rounded_rectangle((x - 16, base - 70, x + 16, base), radius=8, fill=(30, 28, 55))
        draw.ellipse((x - 14, base - 102, x + 14, base - 74), fill=(30, 28, 55))


ROSTER = {
    "kevin": {
        "shirt": (235, 75, 65),
        "pants": (45, 65, 105),
        "hair": (65, 40, 25),
    },
}


def draw_pigeon(draw, x, y, scale=1.0, beat_t=0.0):
    y += math.sin(beat_t * 6) * 5 * scale
    s = scale
    draw.ellipse((x - 48*s, y - 45*s, x + 48*s, y + 38*s), fill=(115, 125, 140), outline=INK, width=4)
    draw.ellipse((x + 10*s, y - 78*s, x + 58*s, y - 25*s), fill=(125, 135, 150), outline=INK, width=4)
    draw.polygon(
        [(x + 45*s, y - 55*s), (x + 80*s, y - 43*s), (x + 45*s, y - 35*s)],
        fill=(240, 155, 45), outline=INK,
    )
    draw.ellipse((x + 35*s, y - 62*s, x + 43*s, y - 54*s), fill=INK)
    draw.line((x - 12*s, y + 30*s, x - 15*s, y + 58*s), fill=(220, 145, 45), width=4)
    draw.line((x + 18*s, y + 30*s, x + 20*s, y + 58*s), fill=(220, 145, 45), width=4)


def draw_character(draw, x, ground_y, scale, emotion, action, activity, beat_t):
    spec = ROSTER["kevin"]
    s = scale
    phase = (beat_t % PERIOD) / PERIOD
    swing = math.sin(math.tau * phase)
    lift = (1 - math.cos(math.tau * phase)) / 2  # 0..1..0 each repetition

    bob = 0.0
    ldy = rdy = 0.0  # arm end offsets
    huss = action == "huss"

    if huss:
        amp = 60 if activity == "jump_rope" else 22
        bob = -abs(math.sin(math.pi * phase)) * amp * s
        if activity == "basketball":
            rdy = swing * 40
        elif activity == "weights":
            ldy = rdy = -lift * 170
        elif activity == "running":
            ldy, rdy = swing * 45, -swing * 45
        elif activity == "typing":
            ldy, rdy = swing * 10, -swing * 10
        elif activity == "jump_rope":
            ldy, rdy = swing * 25, swing * 25
    elif action in {"jump", "panic"}:
        bob = -abs(math.sin(beat_t * 8)) * 45 * s

    y = ground_y + bob
    head_size = 100 * s
    body_top = y - 250 * s
    head_top = body_top - head_size + 10 * s

    leg_swing = swing * 18 * s if (huss and activity == "running") else 0
    draw.line((x - 25*s, y - 75*s, x - 42*s + leg_swing, ground_y), fill=spec["pants"], width=max(1, int(24*s)))
    draw.line((x + 25*s, y - 75*s, x + 42*s - leg_swing, ground_y), fill=spec["pants"], width=max(1, int(24*s)))

    draw.rounded_rectangle(
        (x - 58*s, body_top, x + 58*s, y - 65*s),
        radius=max(1, int(25*s)), fill=spec["shirt"], outline=INK, width=4,
    )

    draw.line((x - 50*s, body_top + 25*s, x - 95*s, body_top + 100*s + ldy*s), fill=spec["shirt"], width=max(1, int(22*s)))
    draw.line((x + 50*s, body_top + 25*s, x + 95*s, body_top + 100*s + rdy*s), fill=spec["shirt"], width=max(1, int(22*s)))

    draw.ellipse((x - head_size/2, head_top, x + head_size/2, head_top + head_size),
                 fill=(245, 190, 145), outline=INK, width=4)
    draw.pieslice((x - head_size/2 - 3, head_top - 12*s, x + head_size/2 + 3, head_top + 48*s),
                  180, 360, fill=spec["hair"], outline=INK)

    face_y = head_top + head_size * .56
    eye_r = (17 if emotion in {"shock", "scared"} else 13) * s

    for side in (-1, 1):
        ex = x + side * 24 * s
        ey = face_y - 8 * s
        draw.ellipse((ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r), fill="white", outline=INK, width=3)
        pupil = eye_r * .4
        draw.ellipse((ex - pupil, ey - pupil, ex + pupil, ey + pupil), fill=INK)
        if emotion in {"angry", "sad"}:
            tilt = 8 * s if emotion == "angry" else -8 * s
            draw.line((ex - side*eye_r, ey - eye_r - 6*s - side*tilt,
                       ex + side*eye_r, ey - eye_r - 6*s + side*tilt), fill=INK, width=4)

    my = face_y + 30 * s
    if emotion == "happy":
        draw.arc((x - 24*s, my - 24*s, x + 24*s, my + 14*s), 15, 165, fill=INK, width=5)
    elif emotion in {"shock", "scared"}:
        draw.ellipse((x - 14*s, my - 5*s, x + 14*s, my + 28*s), fill=(70, 12, 24), outline=INK, width=3)
    elif huss:
        # Open "huss" mouth that pulses with each breath.
        mh = (6 + 16 * lift) * s
        draw.ellipse((x - 14*s, my - 4*s, x + 14*s, my - 4*s + mh), fill=(70, 12, 24), outline=INK, width=3)
    elif emotion in {"angry", "sad"}:
        draw.arc((x - 20*s, my - 2*s, x + 20*s, my + 30*s), 200, 340, fill=INK, width=5)
    else:
        draw.line((x - 14*s, my, x + 14*s, my), fill=INK, width=5)

    return y  # so props can follow the body


def draw_activity(draw, activity, x, ground_y, scale, t, animate):
    s = scale
    phase = (t % PERIOD) / PERIOD if animate else 0.0
    lift = (1 - math.cos(math.tau * phase)) / 2
    swing = math.sin(math.tau * phase)

    if activity == "basketball":
        cx = x + 120 * s
        cy = (ground_y - 30*s - abs(math.sin(math.pi * phase)) * 190 * s) if animate else ground_y - 30*s
        r = 34 * s
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(235, 125, 40), outline=INK, width=4)
        draw.line((cx - r, cy, cx + r, cy), fill=INK, width=3)
        draw.line((cx, cy - r, cx, cy + r), fill=INK, width=3)

    elif activity == "jump_rope":
        if animate:
            squash = 0.25 + 0.75 * lift
            top = ground_y - 330*s
            bottom = ground_y + 10*s
            mid = (top + bottom) / 2
            half = (bottom - top) / 2 * squash
            draw.ellipse((x - 150*s, mid - half, x + 150*s, mid + half), outline=(250, 235, 90), width=6)
        else:
            draw.arc((x - 150*s, ground_y - 120*s, x + 150*s, ground_y + 20*s), 0, 180, fill=(250, 235, 90), width=6)

    elif activity == "weights":
        yy = ground_y - 160*s - (lift * 170 * s if animate else 0)
        draw.line((x - 170*s, yy, x + 170*s, yy), fill=(70, 70, 80), width=max(2, int(10*s)))
        for sx in (-1, 1):
            draw.rounded_rectangle((x + sx*150*s - 16*s, yy - 55*s, x + sx*150*s + 16*s, yy + 55*s),
                                   radius=8, fill=(45, 45, 60), outline=INK, width=3)

    elif activity == "boxing":
        for k, sx in enumerate((-1, 1)):
            punch = max(0.0, math.sin(math.tau * phase + k * math.pi)) if animate else 0.0
            gx = x + sx * (105 - 60 * punch) * s
            gy = ground_y - 190*s - punch * 50 * s
            r = (32 + 12 * punch) * s
            draw.ellipse((gx - r, gy - r, gx + r, gy + r), fill=(215, 40, 50), outline=INK, width=4)

    elif activity == "running":
        if animate:
            for i in range(5):
                lx = (x + 260*s - ((t * 900 + i * 170) % 560) * s)
                ly = ground_y - (60 + i * 55) * s
                draw.line((lx, ly, lx + 90*s, ly), fill="white", width=4)

    elif activity == "typing":
        top = ground_y - 110*s
        draw.rounded_rectangle((x - 140*s, top, x + 140*s, top + 20*s), radius=6, fill=(150, 95, 55), outline=INK, width=3)
        draw.rectangle((x - 80*s, top - 80*s, x + 80*s, top), fill=(35, 40, 55), outline=INK, width=4)
        draw.rectangle((x - 66*s, top - 68*s, x + 66*s, top - 10*s), fill=(80, 190, 240))
        if animate and swing > 0:
            draw.rectangle((x - 40*s, top - 50*s, x + 40*s, top - 38*s), fill=(250, 250, 250))


def draw_effect(draw, effect, beat_t):
    cx, cy = TW / 2, 520
    if effect == "explosion":
        radius = 35 + ease(beat_t / .35) * 220
        points = []
        for i in range(24):
            r = radius * (1 if i % 2 == 0 else .55)
            angle = i / 24 * math.tau
            points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        draw.polygon(points, fill=(255, 130, 20), outline=(220, 40, 40))
    elif effect == "fire":
        for i in range(6):
            x = 45 + i * 120
            height = 140 + 55 * math.sin(beat_t * 10 + i)
            draw.polygon(((x - 35, 1050), (x, 1050 - height), (x + 35, 1050)), fill=(255, 100, 15))
    elif effect == "shock" and beat_t < .12:
        draw.rectangle((0, 0, TW, TH), fill="white")
    elif effect == "sparkle":
        rng = random.Random(7)
        for i in range(12):
            x, y = rng.randint(35, TW - 35), rng.randint(150, 900)
            r = 8 + 10 * abs(math.sin(beat_t * 7 + i))
            draw.line((x - r, y, x + r, y), fill=(255, 245, 140), width=4)
            draw.line((x, y - r, x, y + r), fill=(255, 245, 140), width=4)


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
            "fps": CFG["blender_fps"],
        }),
        encoding="utf-8",
    )

    command = [
        blender, "-b", "-t", "2",
        "--python-exit-code", "1",
        "--python", str(scene_script),
        "--", str(payload), str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=5400)

    # Blender prints Python tracebacks to stdout and may still exit 0 on some
    # builds, so check for our completion marker too.
    if result.returncode != 0 or "3D_RENDER_COMPLETE" not in result.stdout:
        raise RuntimeError(
            "Blender rendering failed:\n"
            + result.stdout[-4000:] + result.stderr[-2000:]
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
        ImageFont.truetype(font_path, 22)
        if Path(font_path).exists() else ImageFont.load_default()
    )

    output = folder / "bg.mp4"
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{TW}x{TH}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(output),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    frame_number = 0
    exact = 0.0  # keeps video length locked to the audio (no per-beat rounding drift)

    try:
        for index, (beat, duration) in enumerate(zip(beats, durations)):
            scene = beat["scene"]
            if scene not in bgs:
                bgs[scene] = make_bg(scene)

            main = beat["chars"][0]
            has_pigeon = any(c["who"] == "pigeon" for c in beat["chars"])
            animate = main["action"] == "huss"

            exact += duration * FPS
            start_frame = frame_number
            end_frame = max(start_frame + 1, round(exact))

            for frame in range(start_frame, end_frame):
                beat_t = (frame - start_frame) / FPS
                img = bgs[scene].copy()
                draw = ImageDraw.Draw(img)

                if beat.get("party"):
                    draw_party(draw, frame / FPS)
                draw_effect(draw, beat["fx"], beat_t)

                if has_pigeon:
                    draw_pigeon(draw, TW * .84, 1005, 1.0, beat_t)

                cx, ground, scale = TW * .5, 1030, 1.45
                body_y = draw_character(
                    draw, cx, ground, scale, main["emotion"], main["action"],
                    beat["activity"], beat_t,
                )
                draw_activity(draw, beat["activity"], cx, ground, scale, beat_t, animate)

                progress = (index + beat_t / max(duration, .1)) / len(beats)
                draw.text((TW / 2, 150), "GRIND LEVEL", font=font, fill="white",
                          anchor="mm", stroke_width=2, stroke_fill=INK)
                draw.rounded_rectangle((90, 175, 630, 205), radius=14,
                                       fill=(25, 25, 40), outline="white", width=2)
                draw.rounded_rectangle(
                    (94, 179, 94 + max(5, int(532 * progress)), 201), radius=11,
                    fill=mix((65, 220, 100), (245, 55, 55), progress),
                )

                # Punch-in on every cut, extra shake on big effects.
                zoom = 1.0 + .075 * (1 - ease(beat_t / .25)) + .012 * math.sin(beat_t * 2)
                shake = (
                    8 * max(0, 1 - beat_t / .5)
                    if beat["fx"] in {"explosion", "shock"} else 0
                )
                crop_w, crop_h = TW / zoom, TH / zoom
                left = clamp(TW/2 + math.sin(frame*2.7)*shake - crop_w/2, 0, TW - crop_w)
                top = clamp(TH/2 + math.cos(frame*3.1)*shake - crop_h/2, 0, TH - crop_h)
                crop = img.crop((
                    int(left), int(top),
                    max(int(left) + 1, int(math.ceil(clamp(left + crop_w, 1, TW)))),
                    max(int(top) + 1, int(math.ceil(clamp(top + crop_h, 1, TH)))),
                )).resize((TW, TH), Image.Resampling.BILINEAR)

                process.stdin.write(crop.tobytes())

            frame_number = end_frame

    except Exception:
        try:
            process.stdin.close()
        except Exception:
            pass
        process.kill()
        process.wait()
        raise

    process.stdin.close()
    if process.wait():
        raise RuntimeError("Cartoon video encode failed (ffmpeg returned an error).")

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
        params={"key": key, "q": query[:100], "per_page": 15,
                "safesearch": "true", "video_type": "film"},
        timeout=30,
    )
    response.raise_for_status()
    results = []
    for hit in response.json().get("hits", []):
        videos = hit.get("videos", {})
        chosen = next(
            (videos[k] for k in ("medium", "small", "large")
             if videos.get(k, {}).get("url")),
            None,
        )
        if chosen:
            results.append((chosen["url"], hit.get("pageURL", "")))
    return results


def pexels_candidates(query):
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return []
    response = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": key},
        params={"query": query, "orientation": "portrait", "per_page": 12},
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
            chosen = min(files, key=lambda f: abs((f.get("width") or 720) - 720))
            results.append((chosen["link"], video.get("url", "")))
    return results


def download(url, path):
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with path.open("wb") as out:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    out.write(chunk)


def make_seg(raw, duration, output):
    raw_duration = probe_duration(raw)
    start = random.uniform(0, max(0, raw_duration - duration - .2))
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,eq=saturation=1.15:contrast=1.05,setsar=1,fps=24"
    )
    run([
        "ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(raw), "-an",
        "-t", f"{duration:.2f}", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", str(output),
    ])


def build_background(beats, durations, folder):
    local = pick_files(ROOT / "footage", VIDEO_EXT)
    api_search = pixabay_candidates if os.getenv("PIXABAY_API_KEY") else pexels_candidates
    api_enabled = bool(os.getenv("PIXABAY_API_KEY") or os.getenv("PEXELS_API_KEY"))

    pool, sources, segments, used = [], [], [], set()

    for index, (beat, duration) in enumerate(zip(beats, durations)):
        raw = None

        if api_enabled:
            try:
                candidates = [
                    x for x in api_search(beat["visual"] or beat["scene"])
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
            raise RuntimeError("Stock visuals selected, but no footage was available.")

        segment = folder / f"segment_{index}.mp4"
        make_seg(raw, duration, segment)
        segments.append(segment)

    concat = folder / "segments.txt"
    concat.write_text("".join(f"file '{p.name}'\n" for p in segments), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat.name,
         "-c", "copy", "bg.mp4"], cwd=folder)

    return folder / "bg.mp4", sources


# ---------------------------------------------------------------------
# FINAL VIDEO COMPOSITION
# ---------------------------------------------------------------------

def render(background, voice, workdir, output, duration):
    music_files = pick_files(ROOT / "music", AUDIO_EXT)
    music = random.choice(music_files) if music_files else None

    command = ["ffmpeg", "-y", "-i", str(background), "-i", str(voice)]

    # Hold the last frame if the video is a hair short, scale, then burn
    # captions at the final 1080x1920 resolution.
    video_filter = (
        "[0:v]tpad=stop_mode=clone:stop_duration=2,"
        "scale=1080:1920:flags=lanczos,ass=captions.ass[v];"
    )

    if music:
        command += ["-stream_loop", "-1", "-i", str(music)]
        audio_filter = (
            f"[2:a]volume={CFG['music_volume']}[music];"
            "[1:a][music]amix=inputs=2:duration=first:"
            "dropout_transition=0:normalize=0[a]"
        )
    else:
        audio_filter = "[1:a]anull[a]"

    command += [
        "-filter_complex", video_filter + audio_filter,
        "-map", "[v]", "-map", "[a]",
        "-t", f"{duration + .3:.2f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output),
    ]
    run(command, cwd=workdir)

    return music.name if music else None


# ---------------------------------------------------------------------
# PRODUCE ONE EPISODE
# ---------------------------------------------------------------------

def make_video(index):
    state = load_state()
    episode_number = max([e.get("ep", 0) for e in state["episodes"]] + [0]) + 1

    data = write_episode(episode_number, state)

    folder = OUT_DAY / f"{index:02d}_ep{episode_number}_{safe_name(data['title'])}"
    folder.mkdir(parents=True, exist_ok=True)

    beats = data["beats"]

    (folder / "script.txt").write_text(
        "\n".join(f"[{b['label']}|{b['kind']}] {b['text']}" for b in beats),
        encoding="utf-8",
    )

    print("Generating narration...")
    durations = voice_beats(beats, folder)
    total_duration = sum(durations)

    print("Generating captions...")
    words = transcribe_words(folder / "voice.wav", beats, durations)
    write_ass(words, beats, durations, episode_number, folder / "captions.ass")

    print("Rendering animated scenes...")
    if CFG["visuals"] == "blender":
        background, sources = blender_background(beats, durations, folder)
    elif CFG["visuals"] == "toon":
        background, sources = toon_background(beats, durations, folder)
    elif CFG["visuals"] == "stock":
        background, sources = build_background(beats, durations, folder)
    else:
        raise ValueError("VISUALS must be 'blender', 'toon', or 'stock'.")

    print("Compositing final Short...")
    music = render(background, folder / "voice.wav", folder,
                   folder / "video.mp4", total_duration)

    for pattern in ("voice_raw_*.wav", "voice_[0-9]*.wav", "raw_*.mp4",
                    "segment_*.mp4", "segments.txt", "voice_list.txt",
                    "bg.mp4", "blender_beats.json"):
        for path in folder.glob(pattern):
            path.unlink(missing_ok=True)

    metadata = {
        "episode": episode_number,
        "series": SERIES_NAME,
        "title": data["title"],
        "description": data.get("description", "") + "\n\nFictional comedy series. #Shorts",
        "summary": data.get("summary", ""),
        "tags": data.get("tags", []),
        "activity": data.get("activity"),
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
    (folder / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Only record the topic once the video actually succeeded.
    if data["topic"] not in state["topics"]:
        state["topics"].append(data["topic"])
    state["episodes"].append({
        "ep": episode_number,
        "topic": data["topic"],
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
            json={"chat_id": chat, "text": message[:4000]},
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
                data={"chat_id": chat, "caption": caption[:900]},
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
        errors.append("VISUALS must be 'blender', 'toon', or 'stock'.")

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
        errors.append(f"Piper model is missing: {CFG['piper_model']}")

    if CFG["visuals"] == "stock" and not (
        os.getenv("PIXABAY_API_KEY") or os.getenv("PEXELS_API_KEY")
        or pick_files(ROOT / "footage", VIDEO_EXT)
    ):
        errors.append("Stock mode requires an API key or footage in ./footage.")

    if not os.getenv("GEMINI_API_KEY"):
        try:
            requests.get(CFG["ollama_url"].rsplit("/api", 1)[0], timeout=4)
        except Exception:
            errors.append("Set GEMINI_API_KEY or run a reachable Ollama server.")

    return errors


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    OUT_DAY.mkdir(parents=True, exist_ok=True)

    print("Starting Shorts Factory v4 (Huss format)...")
    print("Visual engine:", CFG["visuals"])
    print("Videos requested:", CFG["videos_per_run"])

    init_tts()

    errors = preflight()
    completed = []

    if not errors:
        start_index = len([p for p in OUT_DAY.iterdir() if p.is_dir()]) + 1

        for offset in range(CFG["videos_per_run"]):
            try:
                folder, title = make_video(start_index + offset)
                completed.append((title, folder))
                notify_video(Path(folder) / "video.mp4", title)
                print(f"Completed: {title}")
            except Exception as exc:
                message = f"Video {offset + 1}: {exc}"
                errors.append(message)
                print(message)
                with (OUT_DAY / "errors.log").open("a", encoding="utf-8") as log:
                    log.write(traceback.format_exc() + "\n")
    else:
        for error in errors:
            print("PREFLIGHT:", error)

    report = [
        f"# Daily report - {TODAY}",
        f"Completed: {len(completed)}",
        f"Errors: {len(errors)}",
        f"Voice: Piper / {CFG['voice']}",
        f"Visuals: {CFG['visuals']}",
        "",
    ]
    report.extend(f"- COMPLETE: {title} -> {folder}" for title, folder in completed)
    report.extend(f"- ERROR: {error}" for error in errors)
    report.append("\nReview each video's video.mp4 and meta.json before uploading.")

    report_text = "\n".join(report)
    with (OUT_DAY / "report.md").open("a", encoding="utf-8") as out:
        out.write(report_text + "\n\n")

    print(report_text)
    notify(report_text)

    # Make the GitHub Action fail visibly when nothing was produced.
    if errors and not completed:
        sys.exit(1)


if __name__ == "__main__":
    main()
