#!/usr/bin/env python3
"""Shorts Factory v2: connected "brain-rot" style series.
episode idea -> beats (voice line + visual + label) -> per-beat voice -> word captions
-> fast-cut stock footage -> 1080x1920 video -> daily report. All free tools."""
import datetime as dt
import json
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

ROOT = Path(__file__).parent
SERIES_NAME = os.getenv("SERIES_NAME") or "KEVIN'S WORST IDEAS"
DEFAULT_PREMISE = (
    "Kevin is an overconfident, clueless guy who tries one absurd 'what if' experiment per episode "
    "(example scale: what if he did X at 1000x). Each episode escalates through levels of chaos, "
    "every beat bigger than the last. His friend Dave warns him and is always ignored. "
    "Running gag: a pigeon is somehow always watching. Every episode ends on a cliffhanger "
    "teasing Kevin's next terrible idea.")
CFG = {
    "videos_per_run": int(os.getenv("VIDEOS_PER_RUN") or 3),
    "premise": os.getenv("NICHE") or DEFAULT_PREMISE,
    "voice": os.getenv("VOICE") or "am_michael",
    "speed": float(os.getenv("VOICE_SPEED") or 1.15),
    "engine": "kokoro",  # falls back to piper automatically
    "visuals": os.getenv("VISUALS") or "toon",  # "toon" = our own cartoon scenes, "stock" = stock footage
    "ollama_url": "http://localhost:11434/api/generate",
    "ollama_model": "llama3.1:8b",
    "piper_bin": os.getenv("PIPER_BIN") or "piper",
    "piper_model": str(ROOT / "models" / "en_US-lessac-medium.onnx"),
    "font": os.getenv("CAPTION_FONT") or "DejaVu Sans",
    "whisper_model": "small",
    "music_volume": 0.10,
    "words_min": 80,
    "words_max": 150,
}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg"}
TODAY = dt.date.today().isoformat()
OUT_DAY = ROOT / "output" / TODAY
STATE_FILE = ROOT / "used_topics.json"  # holds the episode history (kept for workflow compatibility)


# ---------- helpers ----------
def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def probe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    return float(r.stdout.strip())


def pick_files(folder, exts):
    return [p for p in folder.iterdir() if p.suffix.lower() in exts] if folder.exists() else []


# ---------- LLM (Gemini free tier, auto-picks a working model) ----------
_GEMINI_OK = None


def gemini_models(key):
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
    r.raise_for_status()
    names = []
    for m in r.json().get("models", []):
        n = m["name"].split("/")[-1]
        if ("generateContent" in m.get("supportedGenerationMethods", []) and "flash" in n
                and not re.search(r"image|tts|live|audio|embed|robotics", n)):
            names.append(n)

    def rank(n):
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return ("preview" in n or "exp" in n, "lite" not in n, -(float(v.group(1)) if v else 0))
    return sorted(names, key=rank)


def llm_json(prompt):
    global _GEMINI_OK
    key = os.getenv("GEMINI_API_KEY")
    if key:
        if _GEMINI_OK:
            cands = [_GEMINI_OK]
        else:
            cands = [os.getenv("GEMINI_MODEL")] if os.getenv("GEMINI_MODEL") else []
            try:
                cands += gemini_models(key)
            except Exception:
                pass
            cands += ["gemini-flash-latest", "gemini-2.5-flash"]
            cands = list(dict.fromkeys(cands))
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0}}
        last = "no models found"
        for model in cands:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            r = None
            for _ in range(3):
                try:
                    r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=75)
                except requests.exceptions.RequestException as e:
                    last = f"{model}: {type(e).__name__}"
                    print(f"Gemini {last}, trying next model")
                    r = None
                    break
                if r.status_code == 429:
                    time.sleep(25)
                    continue
                break
            if r is None:
                continue
            if r.status_code in (400, 403, 404, 429, 500, 503):
                last = f"{model}: HTTP {r.status_code}"
                print(f"Gemini {last}, trying next model")
                continue
            r.raise_for_status()
            _GEMINI_OK = model
            print(f"Using Gemini model: {model}")
            return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        raise RuntimeError("No Gemini model worked (" + last + ")")
    r = requests.post(CFG["ollama_url"], json={
        "model": CFG["ollama_model"], "prompt": prompt, "stream": False, "format": "json"}, timeout=900)
    r.raise_for_status()
    return json.loads(r.json()["response"])


# ---------- series memory + episode writing ----------
def load_state():
    if STATE_FILE.exists():
        try:
            d = json.loads(STATE_FILE.read_text())
            if isinstance(d, dict) and "episodes" in d:
                return d
        except Exception:
            pass
    return {"episodes": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=1))


def trending_hints():
    try:
        r = requests.get("https://trends.google.com/trending/rss?geo=US", timeout=20)
        return [i.findtext("title") for i in ET.fromstring(r.content).iter("item")][:10]
    except Exception:
        return []


SCENE_NAMES = ["classroom", "bedroom", "kitchen", "office", "lab", "stage", "street",
               "city_night", "park", "space"]
EMOTIONS = ["neutral", "happy", "angry", "shock", "scared"]
ACTIONS = ["idle", "jump", "shake", "slide", "panic"]
FX_NAMES = ["none", "explosion", "fire", "sparkle", "shock"]


def clean_beat(b):
    """Make sure every beat has valid scene/characters even if the LLM was sloppy."""
    b["label"] = (b.get("label") or "").upper()[:28]
    if b.get("scene") not in SCENE_NAMES:
        b["scene"] = "street"
    if b.get("fx") not in FX_NAMES:
        b["fx"] = "none"
    chars = []
    for c in (b.get("chars") or [])[:3]:
        if not isinstance(c, dict):
            continue
        who = c.get("who") if c.get("who") in ROSTER else "kevin"
        chars.append({"who": who,
                      "emotion": c.get("emotion") if c.get("emotion") in EMOTIONS else "neutral",
                      "action": c.get("action") if c.get("action") in ACTIONS else "idle"})
    b["chars"] = chars or [{"who": "kevin", "emotion": "neutral", "action": "idle"}]
    b["visual"] = b.get("visual") or b["scene"].replace("_", " ")


def write_episode(n, state):
    history = "\n".join(f"EP {e['ep']}: {e['title']} - {e.get('summary', '')}"
                        for e in state["episodes"][-6:]) or "none yet (this is the first episode)"
    prompt = (
        "You write viral YouTube Shorts scripts in a fast, chaotic, funny 'brain rot' storytelling "
        "style: every line is a mini-cliffhanger, stakes escalate every beat, Gen-Z voice, zero filler.\n"
        f"SERIES: {SERIES_NAME}\nPREMISE: {CFG['premise']}\n"
        f"PREVIOUS EPISODES (stay consistent; call back to one earlier event or the running gag):\n{history}\n"
        f"Trending today (use ONLY if it fits naturally as a scenario): {trending_hints()}\n\n"
        f"Write EPISODE {n}: a brand-new scenario, clearly different from every previous episode.\n"
        "RULES:\n"
        "- 9 to 13 beats. Each beat is ONE spoken line of 6-16 words.\n"
        "- Beat 1 is a hook of 10 words or fewer that drops the viewer into the chaos or asks an "
        "irresistible question.\n"
        "- Each beat escalates the stakes or adds a twist. Keep it moving; never explain slowly.\n"
        "- Casual, punchy, funny. PG only: no profanity, no real people, no brands, no dangerous "
        "real-world instructions. Any real fact or number you state must be true; exaggeration "
        "must be obviously a joke.\n"
        "- Last beat: a cliffhanger teasing Kevin's NEXT idea, then asks viewers to comment what "
        "Kevin should try next.\n"
        f"- Total spoken words between {CFG['words_min'] + 10} and {CFG['words_max'] - 10}.\n"
        "- For each beat describe the cartoon scene:\n"
        "  \"scene\": one of classroom, bedroom, kitchen, office, lab, stage, street, city_night, park, space.\n"
        "  \"chars\": 1 to 3 characters on screen, from: kevin, dave, cop, boss, kid, pigeon. Each is "
        "{\"who\": name, \"emotion\": neutral|happy|angry|shock|scared, "
        "\"action\": idle|jump|shake|slide|panic}. Kevin is in most beats; the pigeon pops up as the running gag.\n"
        "  \"fx\": none, explosion, fire, sparkle or shock (use explosion/fire for big chaos moments).\n"
        "  \"label\": short ALL-CAPS on-screen tag, max 4 words (a level, stat or reaction, "
        "e.g. 'CHAOS LEVEL 3', '210 DB', 'DAVE IS WORRIED').\n"
        'Return JSON only: {"title": "under 60 chars, no emoji", "description": "1-2 sentences", '
        '"tags": ["..."], "summary": "one sentence of what happened", '
        '"beats": [{"text": "...", "label": "...", "scene": "street", "fx": "none", '
        '"chars": [{"who": "kevin", "emotion": "happy", "action": "idle"}]}]}')
    for _ in range(3):
        d = llm_json(prompt)
        beats = d.get("beats") or []
        words = sum(len(b.get("text", "").split()) for b in beats)
        ok = (8 <= len(beats) <= 14 and CFG["words_min"] <= words <= CFG["words_max"] + 20
              and all(b.get("text") for b in beats))
        if ok:
            for b in beats:
                clean_beat(b)
            return d
    raise RuntimeError("LLM could not produce a valid episode")


# ---------- voice ----------
_KP = None


def init_tts():
    """Try Kokoro (natural, free, local). Fall back to Piper if it can't load."""
    try:
        import kokoro  # noqa: F401
    except ImportError:
        if os.getenv("GITHUB_ACTIONS"):
            try:
                subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "espeak-ng"], check=False)
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch",
                                "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kokoro>=0.9.4",
                                "soundfile"], check=True)
            except Exception as e:
                print("Kokoro install failed:", e)
    try:
        from kokoro import KPipeline
        global _KP
        _KP = KPipeline(lang_code="a")
        CFG["engine"] = "kokoro"
    except Exception as e:
        print("Kokoro unavailable, using Piper:", e)
        CFG["engine"] = "piper"


def tts(text, wav):
    if CFG["engine"] == "kokoro":
        import numpy as np
        import soundfile as sf
        chunks = []
        for _, _, audio in _KP(text, voice=CFG["voice"], speed=CFG["speed"]):
            chunks.append(audio.numpy() if hasattr(audio, "numpy") else audio)
        sf.write(str(wav), np.concatenate(chunks), 24000)
    else:
        run([CFG["piper_bin"], "--model", CFG["piper_model"], "--length_scale", "0.85",
             "--output_file", str(wav)], input=text)


def voice_beats(beats, d):
    """One clip per beat so each visual cut lines up exactly with its line."""
    durs, files = [], []
    for i, b in enumerate(beats):
        raw, fin = d / f"b{i}_raw.wav", d / f"b{i}.wav"
        tts(b["text"], raw)
        run(["ffmpeg", "-y", "-i", str(raw), "-ar", "24000", "-ac", "1",
             "-af", "apad=pad_dur=0.06", str(fin)])
        raw.unlink()
        durs.append(probe_duration(fin))
        files.append(fin)
    (d / "vlist.txt").write_text("".join(f"file '{f.name}'\n" for f in files))
    run(["ffmpeg", "-y", "-f", "concat", "-i", "vlist.txt", "-c", "copy", "voice.wav"], cwd=d)
    return durs


# ---------- captions: one big word at a time ----------
def transcribe_words(wav):
    from faster_whisper import WhisperModel
    model = WhisperModel(CFG["whisper_model"], device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(wav), word_timestamps=True)
    return [(w.word.strip(), w.start, w.end) for s in segs for w in s.words if w.word.strip()]


def ass_time(s):
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"


def write_ass(words, beats, durs, ep, path):
    head = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,"
        "Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Word,@F,118,&H0000E5FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,10,3,5,60,60,60,1\n"
        "Style: Head,@F,46,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,6,0,8,60,60,60,1\n"
        "Style: Label,@F,74,&H00FFFFFF,&H000000FF,&H003030E0,&H00000000,-1,0,0,0,100,100,0,0,3,14,0,8,60,60,60,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    ).replace("@F", CFG["font"])
    total = sum(durs)
    ev = [f"Dialogue: 0,{ass_time(0)},{ass_time(total)},Head,,0,0,0,,"
          f"{{\\an8\\pos(540,170)}}{SERIES_NAME.upper()}  |  EP {ep}"]
    t = 0.0
    for b, d in zip(beats, durs):
        if b.get("label"):
            ev.append(f"Dialogue: 0,{ass_time(t)},{ass_time(t + d)},Label,,0,0,0,,"
                      f"{{\\an8\\pos(540,290)}}{b['label']}")
        t += d
    pop = "\\fscx85\\fscy85\\t(0,90,\\fscx108\\fscy108)\\t(90,150,\\fscx100\\fscy100)"
    for i, (w, s, e) in enumerate(words):
        nxt = words[i + 1][1] if i + 1 < len(words) else None
        end = nxt if nxt is not None and nxt - e < 0.4 else e + 0.05
        if end <= s:
            end = s + 0.1
        txt = re.sub(r"[{}\\]", "", w).upper()
        ev.append(f"Dialogue: 1,{ass_time(s)},{ass_time(end)},Word,,0,0,0,,"
                  f"{{\\an5\\pos(540,1130){pop}}}{txt}")
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


# ---------- visuals: a new stock clip for every beat ----------
def pexels_candidates(query):
    r = requests.get("https://api.pexels.com/videos/search",
                     headers={"Authorization": os.environ["PEXELS_API_KEY"]},
                     params={"query": query, "orientation": "portrait", "per_page": 15}, timeout=30)
    r.raise_for_status()
    out = []
    for v in r.json().get("videos", []):
        files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4"
                 and (f.get("width") or 0) >= 720 and (f.get("height") or 0) > (f.get("width") or 0)]
        if files:
            f = min(files, key=lambda f: abs(f["width"] - 1080))
            out.append((f["link"], v.get("url")))
    return out


def pixabay_candidates(query):
    r = requests.get("https://pixabay.com/api/videos/",
                     params={"key": os.environ["PIXABAY_API_KEY"], "q": query[:100],
                             "per_page": 20, "safesearch": "true", "video_type": "film"}, timeout=30)
    r.raise_for_status()
    out = []
    for h in r.json().get("hits", []):
        vids = h.get("videos", {})
        pick = next((vids[k] for k in ("medium", "small", "large") if vids.get(k, {}).get("url")), None)
        if pick:
            out.append((pick["url"], h.get("pageURL")))
    return out


def download(link, path):
    with requests.get(link, stream=True, timeout=90) as resp:
        resp.raise_for_status()
        with path.open("wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)


def make_seg(raw, dur, out):
    """Crop to 9:16 with a slight random punch-in, boosted colour, exact beat length."""
    rd = probe_duration(raw)
    start = random.uniform(0, max(0.0, rd - dur - 0.2))
    jx, jy = random.randint(-50, 50), random.randint(-90, 90)
    vf = (f"scale=1188:2112:force_original_aspect_ratio=increase,"
          f"crop=1080:1920:(iw-1080)/2+({jx}):(ih-1920)/2+({jy}),"
          "eq=saturation=1.3:contrast=1.08,setsar=1,fps=30")
    run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-stream_loop", "-1", "-i", str(raw), "-an",
         "-t", f"{dur:.2f}", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
         "-pix_fmt", "yuv420p", str(out)])


def build_background(beats, durs, d):
    use_api = bool(os.getenv("PIXABAY_API_KEY") or os.getenv("PEXELS_API_KEY"))
    search = pixabay_candidates if os.getenv("PIXABAY_API_KEY") else pexels_candidates
    local = pick_files(ROOT / "footage", VIDEO_EXT)
    pool, used, sources, segs = [], set(), [], []
    for i, (b, dur) in enumerate(zip(beats, durs)):
        raw = None
        if use_api:
            try:
                cands = [c for c in search(b["visual"]) if c[0] not in used]
                if not cands:
                    cands = [c for c in search(b["visual"].split()[0]) if c[0] not in used]
                if cands:
                    link, page = random.choice(cands[:8])
                    raw = d / f"raw{i}.mp4"
                    download(link, raw)
                    used.add(link)
                    pool.append(raw)
                    sources.append(page)
            except Exception as e:
                print(f"Footage search failed for '{b['visual']}': {e}")
                raw = None
        if raw is None and pool:
            raw = random.choice(pool)
        if raw is None and local:
            raw = random.choice(local)
        if raw is None:
            raise RuntimeError(f"No footage for beat {i + 1} ('{b['visual']}')")
        seg = d / f"seg{i}.mp4"
        make_seg(raw, dur, seg)
        segs.append(seg)
    (d / "list.txt").write_text("".join(f"file '{s.name}'\n" for s in segs))
    run(["ffmpeg", "-y", "-f", "concat", "-i", "list.txt", "-c", "copy", "bg.mp4"], cwd=d)
    return d / "bg.mp4", sources


def render(bg, start, wav, workdir, out_mp4, dur):
    music = pick_files(ROOT / "music", AUDIO_EXT)
    track = random.choice(music) if music else None
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-stream_loop", "-1", "-i", str(bg), "-i", str(wav)]
    vf = "[0:v]ass=captions.ass[v];"
    if track:
        cmd += ["-stream_loop", "-1", "-i", str(track)]
        af = (f"[2:a]volume={CFG['music_volume']}[m];[1:a][m]amix=inputs=2:duration=first:"
              "dropout_transition=0:normalize=0[a]")
    else:
        af = "[1:a]anull[a]"
    cmd += ["-filter_complex", vf + af, "-map", "[v]", "-map", "[a]", "-t", f"{dur + 0.3:.2f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_mp4)]
    run(cmd, cwd=workdir)  # cwd=workdir so captions.ass needs no path escaping
    return track.name if track else None


# ---------- toon engine: our own animated cartoon scenes (free, original, no stock footage) ----------
import math  # noqa: E402

TW, TH = 720, 1280  # drawn small for speed, scaled to 1080x1920 by ffmpeg
INK = (22, 22, 34)
ROSTER = {
    "kevin": dict(skin=(255, 205, 160), shirt=(255, 140, 40), pants=(50, 60, 140), hair=(90, 55, 30)),
    "dave": dict(skin=(225, 170, 130), shirt=(60, 140, 230), pants=(60, 60, 70), hair=(25, 25, 30), glasses=True),
    "cop": dict(skin=(240, 190, 150), shirt=(35, 55, 120), pants=(25, 35, 80), hair=(40, 30, 20), hat=(25, 40, 100)),
    "boss": dict(skin=(210, 160, 120), shirt=(95, 95, 105), pants=(60, 60, 70), hair=(170, 170, 175), tie=(210, 40, 40)),
    "kid": dict(skin=(250, 215, 180), shirt=(170, 90, 220), pants=(40, 40, 60), hair=(240, 200, 70)),
    "pigeon": {},
}
SCENES = {
    "classroom": dict(kind="room", wall=(238, 216, 150), floor=(150, 105, 70), prop="board"),
    "bedroom": dict(kind="room", wall=(176, 198, 240), floor=(120, 90, 70), prop="bed"),
    "kitchen": dict(kind="room", wall=(252, 238, 200), floor=(196, 200, 212), prop="counter"),
    "office": dict(kind="room", wall=(186, 212, 200), floor=(90, 100, 122), prop="desk"),
    "lab": dict(kind="room", wall=(58, 70, 96), floor=(36, 42, 60), prop="lab"),
    "stage": dict(kind="room", wall=(88, 28, 112), floor=(62, 20, 72), prop="stage"),
    "street": dict(kind="street", night=False),
    "city_night": dict(kind="street", night=True),
    "park": dict(kind="park"),
    "space": dict(kind="space"),
}


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def _grad(top, bot):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (TW, TH))
    d = ImageDraw.Draw(img)
    for y in range(TH):
        d.line([(0, y), (TW, y)], fill=mix(top, bot, y / TH))
    return img


def make_bg(scene):
    from PIL import ImageDraw
    spec = SCENES.get(scene, SCENES["street"])
    rnd = random.Random(sum(map(ord, scene)))
    kind = spec["kind"]
    if kind == "room":
        img = _grad(spec["wall"], mix(spec["wall"], (0, 0, 0), 0.15))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 870, TW, TH], fill=spec["floor"])
        d.rectangle([0, 858, TW, 878], fill=mix(spec["floor"], (255, 255, 255), 0.3), outline=INK, width=3)
        for x in range(-100, TW + 100, 120):
            d.line([(x, 878), (x - 70, TH)], fill=mix(spec["floor"], (0, 0, 0), 0.22), width=3)
        p = spec["prop"]
        if p == "board":
            d.rectangle([70, 170, 650, 470], fill=(40, 92, 72), outline=(130, 85, 40), width=14)
            for i in range(5):
                d.line([(110, 225 + i * 48), (110 + rnd.randint(180, 430), 225 + i * 48)],
                       fill=(235, 235, 235), width=5)
            for x in (70, 320, 540):
                d.rectangle([x, 930, x + 110, 1000], fill=(196, 146, 84), outline=INK, width=4)
        elif p == "bed":
            d.rectangle([470, 150, 690, 420], fill=(150, 212, 255), outline=INK, width=7)
            d.line([(580, 150), (580, 420)], fill=INK, width=5)
            d.line([(470, 285), (690, 285)], fill=INK, width=5)
            d.rectangle([20, 700, 330, 880], fill=(235, 235, 252), outline=INK, width=5)
            d.rectangle([20, 650, 110, 880], fill=(112, 156, 230), outline=INK, width=5)
        elif p == "counter":
            d.rectangle([0, 700, TW, 780], fill=(250, 250, 250), outline=INK, width=4)
            d.rectangle([0, 780, TW, 870], fill=(140, 100, 70), outline=INK, width=4)
            for x in (90, 300, 520):
                d.rectangle([x, 560, x + 90, 700], fill=(235, 90, 90), outline=INK, width=4)
        elif p == "desk":
            d.rectangle([60, 800, 660, 870], fill=(150, 105, 70), outline=INK, width=5)
            d.rectangle([250, 640, 470, 790], fill=(50, 55, 70), outline=INK, width=6)
            d.rectangle([265, 655, 455, 775], fill=(120, 220, 250))
            d.rectangle([90, 120, 330, 300], fill=(255, 255, 255), outline=INK, width=5)
        elif p == "lab":
            d.rectangle([40, 770, 680, 860], fill=(120, 130, 150), outline=INK, width=5)
            for x, c in ((120, (90, 255, 140)), (330, (255, 90, 200)), (540, (90, 200, 255))):
                d.rectangle([x, 660, x + 70, 770], fill=c, outline=INK, width=4)
                d.ellipse([x - 5, 630, x + 75, 690], fill=c, outline=INK, width=4)
            d.rectangle([60, 150, 300, 330], fill=(30, 35, 50), outline=(90, 255, 140), width=6)
        elif p == "stage":
            d.polygon([(0, 0), (190, 0), (90, 870), (0, 870)], fill=(190, 30, 50), outline=INK)
            d.polygon([(TW, 0), (TW - 190, 0), (TW - 90, 870), (TW, 870)], fill=(190, 30, 50), outline=INK)
            d.polygon([(360, 0), (200, 860), (520, 860)], fill=mix(spec["wall"], (255, 240, 160), 0.45))
        return img
    if kind == "street":
        night = spec["night"]
        img = _grad((14, 14, 52), (70, 40, 96)) if night else _grad((110, 196, 255), (226, 242, 255))
        d = ImageDraw.Draw(img)
        x = -10
        while x < TW:
            w, h = rnd.randint(90, 170), rnd.randint(320, 700)
            col = mix((40, 40, 80), (90, 70, 120), rnd.random()) if night else \
                mix((240, 120, 120), (255, 215, 120), rnd.random())
            d.rectangle([x, 860 - h, x + w, 860], fill=col, outline=INK, width=4)
            for wy in range(860 - h + 30, 820, 60):
                for wx in range(x + 16, x + w - 30, 40):
                    lit = night and rnd.random() < 0.55
                    d.rectangle([wx, wy, wx + 20, wy + 30], fill=(255, 235, 120) if lit else
                                ((50, 50, 90) if night else (190, 230, 255)))
            x += w + rnd.randint(0, 10)
        d.rectangle([0, 840, TW, 884], fill=(184, 184, 194), outline=INK, width=3)
        d.rectangle([0, 884, TW, TH], fill=(70, 70, 82))
        for x in range(10, TW, 150):
            d.rectangle([x, 1130, x + 80, 1148], fill=(245, 245, 245))
        return img
    if kind == "park":
        img = _grad((130, 210, 255), (232, 246, 255))
        d = ImageDraw.Draw(img)
        d.ellipse([-200, 640, 520, 1000], fill=(110, 200, 110), outline=INK, width=4)
        d.ellipse([260, 600, 940, 1000], fill=(90, 184, 100), outline=INK, width=4)
        d.rectangle([0, 860, TW, TH], fill=(96, 196, 96))
        for x, h in ((90, 230), (590, 280), (430, 190)):
            d.rectangle([x - 18, 860 - h, x + 18, 870], fill=(120, 80, 50), outline=INK, width=4)
            d.ellipse([x - 90, 860 - h - 130, x + 90, 860 - h + 50], fill=(60, 160, 70), outline=INK, width=4)
        d.ellipse([560, 80, 660, 180], fill=(255, 230, 90), outline=INK, width=4)
        return img
    img = _grad((4, 4, 24), (34, 10, 64))  # space
    d = ImageDraw.Draw(img)
    for _ in range(150):
        x, y, r = rnd.randint(0, TW), rnd.randint(0, TH), rnd.choice((1, 1, 2, 3))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
    d.ellipse([360, 180, 680, 500], fill=(232, 124, 72), outline=INK, width=5)
    d.ellipse([320, 300, 720, 380], outline=(255, 220, 150), width=8)
    d.ellipse([80, 200, 190, 310], fill=(190, 190, 205), outline=INK, width=4)
    d.ellipse([40, 990, TW - 40, 1110], fill=(92, 80, 116), outline=INK, width=5)
    return img


def _rot(pivot, pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(pivot[0] + x * c - y * s, pivot[1] + x * s + y * c) for x, y in pts]


def draw_pigeon(d, cx, base, u, t, act, seed):
    bob = math.sin(t * 8) * 5 * u
    hb = math.sin(t * 16) * 6 * u
    if act in ("shake", "panic"):
        cx += random.Random(seed).uniform(-7, 7)
    d.line([(cx - 12 * u, base - 30 * u + bob), (cx - 12 * u, base)], fill=(255, 150, 40), width=int(6 * u))
    d.line([(cx + 16 * u, base - 30 * u + bob), (cx + 16 * u, base)], fill=(255, 150, 40), width=int(6 * u))
    d.polygon([(cx - 60 * u, base - 80 * u + bob), (cx - 130 * u, base - 60 * u + bob),
               (cx - 60 * u, base - 50 * u + bob)], fill=(120, 120, 140), outline=INK)
    d.ellipse([cx - 72 * u, base - 118 * u + bob, cx + 72 * u, base - 22 * u + bob],
              fill=(162, 162, 178), outline=INK, width=4)
    d.ellipse([cx - 52 * u, base - 100 * u + bob, cx + 22 * u, base - 46 * u + bob],
              fill=(122, 122, 142), outline=INK, width=3)
    hx, hy = cx + 62 * u, base - 124 * u + bob + hb
    d.ellipse([hx - 32 * u, hy - 32 * u, hx + 32 * u, hy + 32 * u], fill=(142, 142, 162), outline=INK, width=4)
    d.ellipse([hx - 30 * u, hy + 12 * u, hx + 6 * u, hy + 40 * u], fill=(110, 190, 170))
    d.polygon([(hx + 26 * u, hy - 8 * u), (hx + 62 * u, hy + 4 * u), (hx + 26 * u, hy + 12 * u)],
              fill=(255, 172, 40), outline=INK)
    d.ellipse([hx + 4 * u, hy - 18 * u, hx + 24 * u, hy + 2 * u], fill=(255, 255, 255), outline=INK, width=2)
    d.ellipse([hx + 11 * u, hy - 12 * u, hx + 19 * u, hy - 4 * u], fill=INK)


def draw_char(d, who, cx, base, u, emo, act, t, tb, side, seed):
    if who == "pigeon":
        return draw_pigeon(d, cx, base, u * 0.9, t, act, seed)
    sp = ROSTER.get(who, ROSTER["kevin"])
    dx = dy = 0.0
    rnd = random.Random(seed)
    if act == "jump":
        dy = -abs(math.sin(t * 5.0)) * 120
    elif act == "shake":
        dx = rnd.uniform(-9, 9)
    elif act == "panic":
        dx, dy = rnd.uniform(-7, 7), -abs(math.sin(t * 14)) * 30
    else:
        dy = math.sin(t * 6) * 5
    if act == "slide":
        dx += (1 - ease(tb / 0.35)) * (-560 if side <= 0 else 560)
    x, yb = cx + dx, base + dy
    leg_h, tor_h, tor_w, head = 88 * u, 138 * u, 124 * u, 112 * u
    d.ellipse([cx - 72 * u, base - 10 * u, cx + 72 * u, base + 16 * u], fill=(34, 34, 46))
    for sx in (-1, 1):
        lx = x + sx * 24 * u
        d.rectangle([lx - 19 * u, yb - leg_h, lx + 19 * u, yb - 8 * u], fill=sp["pants"], outline=INK, width=4)
        d.rounded_rectangle([lx - 24 * u, yb - 14 * u, lx + 24 * u, yb + 2 * u], radius=int(6 * u),
                            fill=(245, 245, 245), outline=INK, width=3)
    top = yb - leg_h - tor_h
    d.rounded_rectangle([x - tor_w / 2, top, x + tor_w / 2, yb - leg_h + 6 * u], radius=int(14 * u),
                        fill=sp["shirt"], outline=INK, width=4)
    if sp.get("tie"):
        d.polygon([(x - 9 * u, top + 4 * u), (x + 9 * u, top + 4 * u), (x + 6 * u, top + 90 * u),
                   (x, top + 104 * u), (x - 6 * u, top + 90 * u)], fill=sp["tie"], outline=INK)
    up = emo in ("shock", "scared") or act in ("panic", "jump")
    swing = math.sin(t * 6)
    angs = (2.55 + 0.25 * math.sin(t * 18), -2.55 - 0.25 * math.sin(t * 18 + 1)) if up else \
        (0.22 + 0.22 * swing, -0.22 - 0.22 * swing)
    for sx, ang in ((-1, angs[0]), (1, angs[1])):
        pv = (x + sx * (tor_w / 2 + 2 * u), top + 22 * u)
        aw, ah = 36 * u, 120 * u
        pts = _rot(pv, [(-aw / 2, 0), (aw / 2, 0), (aw / 2, ah), (-aw / 2, ah)], ang)
        d.polygon(pts, fill=sp["shirt"], outline=INK)
        ex, ey = _rot(pv, [(0, ah + 4 * u)], ang)[0]
        d.ellipse([ex - 19 * u, ey - 19 * u, ex + 19 * u, ey + 19 * u], fill=sp["skin"], outline=INK, width=3)
    htop = top - 4 * u - head
    d.rounded_rectangle([x - head / 2, htop, x + head / 2, htop + head], radius=int(22 * u),
                        fill=sp["skin"], outline=INK, width=4)
    d.rounded_rectangle([x - head / 2, htop, x + head / 2, htop + 34 * u], radius=int(18 * u),
                        fill=sp["hair"], outline=INK, width=4)
    if sp.get("hat"):
        d.rectangle([x - head / 2 - 8 * u, htop - 16 * u, x + head / 2 + 8 * u, htop + 24 * u],
                    fill=sp["hat"], outline=INK, width=4)
        d.ellipse([x - 9 * u, htop - 8 * u, x + 9 * u, htop + 10 * u], fill=(255, 215, 60), outline=INK, width=2)
    fy = htop + head * 0.56
    big = emo in ("shock", "scared")
    er = (17 if big else 13) * u
    for sx in (-1, 1):
        ex = x + sx * 24 * u
        d.ellipse([ex - er, fy - 8 * u - er, ex + er, fy - 8 * u + er], fill=(255, 255, 255), outline=INK, width=3)
        pr = er * (0.3 if big else 0.45)
        d.ellipse([ex - pr, fy - 8 * u - pr, ex + pr, fy - 8 * u + pr], fill=INK)
        if sp.get("glasses"):
            d.ellipse([ex - er - 5 * u, fy - 8 * u - er - 5 * u, ex + er + 5 * u, fy - 8 * u + er + 5 * u],
                      outline=INK, width=3)
    my = fy + 30 * u
    if emo == "happy":
        d.arc([x - 24 * u, my - 24 * u, x + 24 * u, my + 14 * u], 15, 165, fill=INK, width=5)
    elif emo == "angry":
        d.arc([x - 20 * u, my - 2 * u, x + 20 * u, my + 30 * u], 200, 340, fill=INK, width=5)
        d.line([(x - 42 * u, fy - 34 * u), (x - 8 * u, fy - 20 * u)], fill=INK, width=7)
        d.line([(x + 8 * u, fy - 20 * u), (x + 42 * u, fy - 34 * u)], fill=INK, width=7)
    elif emo == "shock":
        d.ellipse([x - 15 * u, my - 6 * u, x + 15 * u, my + 28 * u], fill=(70, 12, 24), outline=INK, width=3)
    elif emo == "scared":
        pts = [(x - 20 * u + i * 8 * u, my + (6 * u if i % 2 else -2 * u)) for i in range(6)]
        d.line(pts, fill=INK, width=5)
        d.ellipse([x + head / 2 - 4 * u, htop + 10 * u, x + head / 2 + 14 * u, htop + 40 * u],
                  fill=(120, 200, 255), outline=INK, width=2)
    else:
        d.line([(x - 14 * u, my), (x + 14 * u, my)], fill=INK, width=5)


def draw_fx(d, kind, tb, seed, front):
    cx, cy = TW / 2, 520
    if kind == "explosion" and not front:
        r = lerp(60, 330, ease(tb / 0.3)) + math.sin(tb * 30) * 12
        for scale, col in ((1.0, (230, 40, 40)), (0.8, (255, 150, 30)), (0.58, (255, 230, 70)), (0.3, (255, 255, 255))):
            pts = []
            for i in range(28):
                rr = r * scale * (1 if i % 2 == 0 else 0.55)
                a = i / 28 * math.tau
                pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
            d.polygon(pts, fill=col, outline=INK if scale == 1.0 else None)
    elif kind == "fire" and not front:
        for i in range(8):
            fx = 40 + i * 90
            h = 200 + 90 * math.sin(tb * 9 + i * 1.7)
            d.polygon([(fx - 55, 1000), (fx, 1000 - h), (fx + 55, 1000)], fill=(255, 120, 20), outline=INK)
            d.polygon([(fx - 28, 1000), (fx, 1000 - h * 0.62), (fx + 28, 1000)], fill=(255, 220, 70))
    elif kind == "shock" and not front:
        rnd = random.Random(seed // 2)
        for _ in range(26):
            a = rnd.uniform(0, math.tau)
            r0, r1 = rnd.uniform(260, 340), rnd.uniform(700, 900)
            w = 0.03
            d.polygon([(cx + math.cos(a - w) * r0, cy + math.sin(a - w) * r0),
                       (cx + math.cos(a + w) * r0, cy + math.sin(a + w) * r0),
                       (cx + math.cos(a) * r1, cy + math.sin(a) * r1)], fill=(255, 255, 255))
    elif kind == "sparkle" and front:
        rnd = random.Random(7)
        for i in range(9):
            x, y = rnd.randint(40, TW - 40), rnd.randint(160, 900)
            r = 16 + 14 * abs(math.sin(tb * 6 + i))
            d.polygon([(x, y - r), (x + r * 0.28, y - r * 0.28), (x + r, y), (x + r * 0.28, y + r * 0.28),
                       (x, y + r), (x - r * 0.28, y + r * 0.28), (x - r, y), (x - r * 0.28, y - r * 0.28)],
                      fill=(255, 245, 140), outline=INK)


def toon_background(beats, durs, d):
    """Render every beat as animated cartoon scenes -> bg.mp4 (1080x1920)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow"], check=True)
        from PIL import Image, ImageDraw, ImageFont
    fps = 30
    font = None
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if Path(p).exists():
            font = ImageFont.truetype(p, 24)
    bgs = {}
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{TW}x{TH}", "-r", str(fps),
           "-i", "-", "-vf", "scale=1080:1920:flags=bicubic", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", str(d / "bg.mp4")]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n, cum = len(beats), 0.0
    for i, (b, dur) in enumerate(zip(beats, durs)):
        f0 = round(cum * fps)
        cum += dur
        f1 = round(cum * fps)
        scene = b["scene"]
        if scene not in bgs:
            bgs[scene] = make_bg(scene)
        chars = b["chars"]
        cnt = len(chars)
        u = {1: 1.5, 2: 1.28, 3: 1.05}[cnt]
        xs = {1: [0.5], 2: [0.3, 0.7], 3: [0.2, 0.5, 0.8]}[cnt]
        for f in range(f0, f1):
            tb, t = (f - f0) / fps, f / fps
            img = bgs[scene].copy()
            dr = ImageDraw.Draw(img)
            draw_fx(dr, b["fx"], tb, f, False)
            for c, xr in zip(chars, xs):
                draw_char(dr, c["who"], TW * xr, 1010, u, c["emotion"], c["action"], t, tb,
                          -1 if xr < 0.5 else 1 if xr > 0.5 else 0, f * 7 + int(xr * 10))
            draw_fx(dr, b["fx"], tb, f, True)
            z = 1 + 0.12 * (1 - ease(tb / 0.22)) + 0.04 * (tb / max(dur, 0.1))
            amp = 14 * max(0.0, 1 - tb / 0.6) if b["fx"] in ("explosion", "shock") else 0.0
            rr = random.Random(f)
            sx, sy = rr.uniform(-amp, amp), rr.uniform(-amp, amp)
            w, h = TW / z, TH / z
            l = min(max(TW / 2 + sx - w / 2, 0), TW - w)
            tp = min(max(TH / 2 + sy - h / 2, 0), TH - h)
            img = img.resize((TW, TH), Image.BILINEAR, box=(l, tp, l + w, tp + h))
            dr = ImageDraw.Draw(img)
            frac = (i + tb / max(dur, 0.1)) / n
            dr.rounded_rectangle([110, 436, 610, 470], radius=17, fill=(28, 28, 40), outline=(255, 255, 255), width=3)
            dr.rounded_rectangle([113, 439, 113 + max(8, int(494 * frac)), 467], radius=14,
                                 fill=mix((60, 220, 90), (240, 50, 50), frac))
            if font:
                dr.text((TW / 2, 418), "CHAOS LEVEL", font=font, fill=(255, 255, 255), anchor="mm",
                        stroke_width=3, stroke_fill=INK)
            proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("cartoon render failed")
    return d / "bg.mp4", []


# ---------- one full episode ----------
def make_video(idx):
    state = load_state()
    n = len(state["episodes"]) + 1
    data = write_episode(n, state)
    beats = data["beats"]
    d = OUT_DAY / f"{idx:02d}_ep{n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "script.txt").write_text("\n".join(f"[{b['label']}] {b['text']}" for b in beats),
                                  encoding="utf-8")
    durs = voice_beats(beats, d)
    total = sum(durs)
    words = transcribe_words(d / "voice.wav")
    write_ass(words, beats, durs, n, d / "captions.ass")
    bg, sources = (toon_background(beats, durs, d) if CFG["visuals"] == "toon"
                   else build_background(beats, durs, d))
    track = render(bg, 0.0, d / "voice.wav", d, d / "video.mp4", total)
    for pattern in ("raw*.mp4", "seg*.mp4", "b*.wav", "list.txt", "vlist.txt", "bg.mp4"):
        for p in d.glob(pattern):
            p.unlink()
    meta = {"episode": n, "series": SERIES_NAME, "title": data["title"],
            "description": data.get("description", "") + "\n\nFictional comedy series. #Shorts",
            "tags": data.get("tags", []), "duration_sec": round(total, 1),
            "voice_engine": CFG["engine"], "footage_sources": sources, "music_used": track,
            "review_checklist": ["Fact-check any real facts", "Footage/music license OK",
                                 "Turn on YouTube's altered/synthetic content disclosure if needed"]}
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    state["episodes"].append({"ep": n, "title": data["title"], "summary": data.get("summary", "")})
    save_state(state)
    return str(d), data["title"]


# ---------- run, report, notify ----------
def preflight():
    errs = []
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            errs.append(f"'{tool}' not found in PATH")
    if CFG["engine"] == "piper":
        if not (shutil.which(CFG["piper_bin"]) or Path(CFG["piper_bin"]).exists()):
            errs.append(f"Kokoro failed and Piper not found: {CFG['piper_bin']}")
        elif not Path(CFG["piper_model"]).exists():
            errs.append(f"Piper voice model missing: {CFG['piper_model']}")
    if CFG["visuals"] == "stock" and not (os.getenv("PIXABAY_API_KEY") or os.getenv("PEXELS_API_KEY")) \
            and not pick_files(ROOT / "footage", VIDEO_EXT):
        errs.append("No footage: set PIXABAY_API_KEY or add licensed clips to /footage")
    if not os.getenv("GEMINI_API_KEY"):
        try:
            requests.get(CFG["ollama_url"].rsplit("/api", 1)[0], timeout=5)
        except Exception:
            errs.append("No GEMINI_API_KEY and Ollama is not running")
    return errs


def notify(text):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if tok and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": text[:4000]}, timeout=15)
        except Exception:
            pass


def notify_video(path, caption):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if tok and chat:
        try:
            with open(path, "rb") as fh:
                requests.post(f"https://api.telegram.org/bot{tok}/sendVideo",
                              data={"chat_id": chat, "caption": caption[:900]},
                              files={"video": fh}, timeout=180)
        except Exception:
            pass


def main():
    OUT_DAY.mkdir(parents=True, exist_ok=True)
    done, errors = [], []
    init_tts()
    errors += preflight()
    if not errors:
        start_idx = len([p for p in OUT_DAY.iterdir() if p.is_dir()]) + 1
        for i in range(CFG["videos_per_run"]):
            try:
                path, title = make_video(start_idx + i)
                done.append((title, path))
                notify_video(Path(path) / "video.mp4", title)
            except Exception as e:
                errors.append(f"video {i + 1}: {e}")
                (OUT_DAY / "errors.log").open("a").write(traceback.format_exc() + "\n")
    report = [f"# Daily report {TODAY}",
              f"Completed: {len(done)}  |  Errors: {len(errors)}  |  Voice: {CFG['engine']}", ""]
    report += [f"- OK: {t} -> {p}" for t, p in done]
    report += [f"- ERROR: {e}" for e in errors]
    report.append("\nNext step: review each folder's video.mp4 + meta.json, then upload manually.")
    text = "\n".join(report)
    with (OUT_DAY / "report.md").open("a", encoding="utf-8") as f:
        f.write(text + "\n\n")
    print(text)
    notify(text)


if __name__ == "__main__":
    main()
