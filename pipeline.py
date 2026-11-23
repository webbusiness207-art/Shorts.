cd /home/claude/shorts_factory && cat > pipeline.py << 'EOF'
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
        "- For each beat give \"visual\": 2-3 plain words to search free stock video that fits "
        "(common, filmable things such as 'city traffic', 'storm clouds', 'shocked man', "
        "'crowd running', 'rocket launch').\n"
        "- For each beat give \"label\": a short ALL-CAPS on-screen tag, max 4 words (a level, stat "
        "or reaction, e.g. 'CHAOS LEVEL 3', '210 DB', 'DAVE IS WORRIED').\n"
        'Return JSON only: {"title": "under 60 chars, no emoji", "description": "1-2 sentences", '
        '"tags": ["..."], "summary": "one sentence of what happened", '
        '"beats": [{"text": "...", "visual": "...", "label": "..."}]}')
    for _ in range(3):
        d = llm_json(prompt)
        beats = d.get("beats") or []
        words = sum(len(b.get("text", "").split()) for b in beats)
        ok = (8 <= len(beats) <= 14 and CFG["words_min"] <= words <= CFG["words_max"] + 20
              and all(b.get("text") and b.get("visual") for b in beats))
        if ok:
            for b in beats:
                b["label"] = (b.get("label") or "").upper()[:28]
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
    bg, sources = build_background(beats, durs, d)
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
    if not (os.getenv("PIXABAY_API_KEY") or os.getenv("PEXELS_API_KEY")) \
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
EOF
python3 -m py_compile pipeline.py && echo COMPILED
