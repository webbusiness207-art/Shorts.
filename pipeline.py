#!/usr/bin/env python3
"""Trend-driven Shorts Factory: discovers topics, writes original scripts, renders 3 Shorts."""
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
SERIES_NAME = os.getenv("SERIES_NAME", "TRENDING STORIES")
CFG = {
    "videos_per_run": int(os.getenv("VIDEOS_PER_RUN") or "3"),
    "voice": os.getenv("VOICE", "am_michael"),
    "speed": float(os.getenv("VOICE_SPEED", "1.12")),
    "engine": "kokoro",
    "ollama_url": "http://localhost:11434/api/generate",
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    "piper_bin": os.getenv("PIPER_BIN", "piper"),
    "piper_model": str(ROOT / "models" / "en_US-lessac-medium.onnx"),
    "font": os.getenv("CAPTION_FONT", "DejaVu Sans"),
    "whisper_model": os.getenv("WHISPER_MODEL", "small"),
    "music_volume": 0.10,
    "words_min": 75,
    "words_max": 135,
}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg"}
TODAY = dt.date.today().isoformat()
OUT_DAY = ROOT / "output" / TODAY
STATE_FILE = ROOT / "used_topics.json"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ShortsFactory/3.0 (trend research; contact: local-run)"})


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def probe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    return float(r.stdout.strip())


def pick_files(folder, exts):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts] if folder.exists() else []


# ---------- topic discovery ----------
def google_trends():
    """Google's public daily trending-search RSS; availability/coverage can vary."""
    urls = [
        "https://trends.google.com/trending/rss?geo=US",
        "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US",
    ]
    for url in urls:
        try:
            resp = SESSION.get(url, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = []
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                traffic = (item.findtext("{https://trends.google.com/trending/rss}approx_traffic") or "").strip()
                if title:
                    items.append({"topic": title, "source": "Google Trends", "signal": traffic})
            if items:
                return items[:20]
        except Exception as exc:
            print(f"Google Trends unavailable: {exc}")
    return []


def youtube_trends():
    """Optional YouTube Data API signal: recent high-view videos for currently popular queries.
    This is a discovery signal, not an official Shorts trending feed."""
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        return []
    terms = ["trending", "viral", "new technology", "gaming", "science", "internet culture"]
    found = []
    published_after = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)).isoformat(timespec="seconds").replace("+00:00", "Z")
    for term in terms:
        try:
            r = SESSION.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={"part": "snippet", "type": "video", "order": "viewCount",
                        "publishedAfter": published_after, "maxResults": 5,
                        "q": term, "key": key},
                timeout=20,
            )
            r.raise_for_status()
            for item in r.json().get("items", []):
                title = item.get("snippet", {}).get("title", "").strip()
                if title:
                    found.append({"topic": title, "source": "YouTube recent high-view search", "signal": term})
        except Exception as exc:
            print(f"YouTube trend lookup failed for {term}: {exc}")
            break
    return found[:20]


def get_trend_candidates():
    candidates = google_trends() + youtube_trends()
    # Deduplicate while preserving source order.
    seen, clean = set(), []
    for item in candidates:
        key = re.sub(r"\W+", " ", item["topic"].lower()).strip()
        if key and key not in seen:
            seen.add(key)
            clean.append(item)
    return clean[:30]


# ---------- LLM ----------
_GEMINI_OK = None


def gemini_models(key):
    r = SESSION.get("https://generativelanguage.googleapis.com/v1beta/models",
                    headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
    r.raise_for_status()
    names = []
    for model in r.json().get("models", []):
        name = model["name"].split("/")[-1]
        if ("generateContent" in model.get("supportedGenerationMethods", [])
                and "flash" in name
                and not re.search(r"image|tts|live|audio|embed|robotics", name)):
            names.append(name)
    def rank(name):
        version = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        return ("preview" in name or "exp" in name, "lite" not in name,
                -(float(version.group(1)) if version else 0))
    return sorted(names, key=rank)


def llm_json(prompt):
    global _GEMINI_OK
    key = os.getenv("GEMINI_API_KEY")
    if key:
        if _GEMINI_OK:
            candidates = [_GEMINI_OK]
        else:
            candidates = [os.getenv("GEMINI_MODEL")] if os.getenv("GEMINI_MODEL") else []
            try:
                candidates += gemini_models(key)
            except Exception as exc:
                print(f"Gemini model discovery failed: {exc}")
            candidates += ["gemini-flash-latest", "gemini-2.5-flash"]
            candidates = list(dict.fromkeys(x for x in candidates if x))
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.9}}
        last_error = "no models found"
        for model in candidates:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            for attempt in range(3):
                try:
                    response = SESSION.post(url, headers={"x-goog-api-key": key},
                                            json=body, timeout=90)
                    if response.status_code == 429 and attempt < 2:
                        time.sleep(15 * (attempt + 1))
                        continue
                    if response.status_code in (400, 403, 404, 429, 500, 503):
                        last_error = f"{model}: HTTP {response.status_code}"
                        break
                    response.raise_for_status()
                    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                    _GEMINI_OK = model
                    print(f"Using Gemini model: {model}")
                    return json.loads(text)
                except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as exc:
                    last_error = f"{model}: {type(exc).__name__}: {exc}"
                    break
        raise RuntimeError("No Gemini model worked (" + last_error + ")")
    response = SESSION.post(CFG["ollama_url"], json={
        "model": CFG["ollama_model"], "prompt": prompt, "stream": False, "format": "json"
    }, timeout=900)
    response.raise_for_status()
    return json.loads(response.json()["response"])


# ---------- persistent history ----------
def load_state():
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            # Backward compatibility: the old file was sometimes a plain list of topics.
            if isinstance(data, list):
                return {"episodes": [], "used_topics": [str(x) for x in data]}
            if isinstance(data, dict):
                data.setdefault("episodes", [])
                data.setdefault("used_topics", [])
                return data
        except Exception as exc:
            print(f"Could not read topic history; starting fresh: {exc}")
    return {"episodes": [], "used_topics": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_topic(candidates, state):
    used = {str(x).strip().lower() for x in state.get("used_topics", [])}
    recent_titles = {e.get("topic", e.get("title", "")).strip().lower()
                     for e in state.get("episodes", [])[-30:]}
    fresh = [x for x in candidates if x["topic"].strip().lower() not in used
             and x["topic"].strip().lower() not in recent_titles]
    if not fresh:
        fresh = candidates
    if not fresh:
        # Fallback only if live trend feeds are unavailable.
        fresh = [
            {"topic": "Why people are fascinated by strange deep-sea discoveries", "source": "fallback", "signal": ""},
            {"topic": "The surprising science behind everyday technology", "source": "fallback", "signal": ""},
            {"topic": "An unexplained historical mystery people are discussing", "source": "fallback", "signal": ""},
        ]
    prompt = (
        "You are a trend editor for an original YouTube Shorts channel. Select ONE topic that has "
        "the strongest short-form storytelling potential from the candidate list. Prefer timely, "
        "widely understandable topics with a surprising angle. Avoid tragedy, graphic violence, "
        "politics, medical misinformation, dangerous challenges, and unverified allegations. "
        "Do not claim a topic is viral merely because it appears in this list. "
        "Return JSON only: {\"topic\":\"...\",\"angle\":\"one specific original angle\","
        "\"reason\":\"brief reason\"}.\nCANDIDATES:\n" +
        json.dumps(fresh[:25], ensure_ascii=False)
    )
    result = llm_json(prompt)
    topic = str(result.get("topic", "")).strip()
    if not topic:
        topic = fresh[0]["topic"]
    return {"topic": topic, "angle": str(result.get("angle", "")).strip(),
            "source": next((x["source"] for x in fresh if x["topic"].lower() == topic.lower()), "AI-selected trend")}


def write_episode(n, state, trend):
    history = "\n".join(
        f"EP {e.get('ep')}: {e.get('topic', e.get('title', ''))} - {e.get('summary', '')}"
        for e in state["episodes"][-8:]
    ) or "No previous episodes."
    prompt = (
        "Write an original, high-retention YouTube Short script based on the selected topic. "
        "The style should match strong current short-form storytelling: immediate hook, curiosity, "
        "clear escalating information, punchy conversational narration, and a satisfying payoff. "
        "Do not imitate a specific creator. Do not invent current events, quotes, statistics, or facts. "
        "If a fact cannot be stated confidently, phrase it cautiously or omit it. "
        "Avoid dangerous real-world instructions, graphic content, medical misinformation, and "
        "unverified claims about real people.\n"
        f"CHANNEL: {SERIES_NAME}\nSELECTED TOPIC: {trend['topic']}\n"
        f"ORIGINAL ANGLE: {trend['angle']}\nTREND SIGNAL SOURCE: {trend['source']}\n"
        f"RECENT EPISODES (avoid repeating):\n{history}\n"
        "RULES:\n"
        "- 8 to 12 beats, each one spoken line of 7-16 words.\n"
        "- First beat is a powerful hook, 10 words maximum.\n"
        "- Every beat adds a new fact, reveal, escalation, or question.\n"
        "- Explain enough context that a new viewer understands the story.\n"
        "- End with a memorable payoff and a natural comment prompt, not fake engagement bait.\n"
        "- Keep it suitable for a general audience.\n"
        "- Each beat has visual: 2-4 searchable visual keywords for relevant stock footage.\n"
        "- Each beat has label: short ALL-CAPS caption tag, maximum 4 words.\n"
        f"- Total narration: {CFG['words_min']} to {CFG['words_max']} words.\n"
        'Return JSON only: {"title":"under 60 characters, no emoji",'
        '"description":"1-2 accurate sentences", "tags":["..."],'
        '"summary":"one sentence", "beats":[{"text":"...","visual":"...","label":"..."}]}'
    )
    for _ in range(3):
        data = llm_json(prompt)
        beats = data.get("beats") or []
        words = sum(len(b.get("text", "").split()) for b in beats)
        valid = (8 <= len(beats) <= 12 and CFG["words_min"] <= words <= CFG["words_max"]
                 and all(b.get("text") and b.get("visual") for b in beats))
        if valid:
            for beat in beats:
                beat["label"] = (beat.get("label") or "").upper()[:28]
            data["topic"] = trend["topic"]
            data["trend_source"] = trend["source"]
            return data
    raise RuntimeError("AI could not produce a valid script after 3 attempts")


# ---------- voice ----------
_KP = None


def init_tts():
    try:
        from kokoro import KPipeline
        global _KP
        _KP = KPipeline(lang_code="a")
        CFG["engine"] = "kokoro"
        return
    except Exception as exc:
        print("Kokoro unavailable; trying Piper:", exc)
    CFG["engine"] = "piper"


def tts(text, wav):
    if CFG["engine"] == "kokoro":
        import numpy as np
        import soundfile as sf
        chunks = []
        for _, _, audio in _KP(text, voice=CFG["voice"], speed=CFG["speed"]):
            chunks.append(audio.numpy() if hasattr(audio, "numpy") else audio)
        if not chunks:
            raise RuntimeError("Kokoro returned no audio")
        sf.write(str(wav), np.concatenate(chunks), 24000)
    else:
        run([CFG["piper_bin"], "--model", CFG["piper_model"], "--length_scale", "0.88",
             "--output_file", str(wav)], input=text)


def voice_beats(beats, workdir):
    durations, files = [], []
    for i, beat in enumerate(beats):
        raw, final = workdir / f"b{i}_raw.wav", workdir / f"b{i}.wav"
        tts(beat["text"], raw)
        run(["ffmpeg", "-y", "-i", str(raw), "-ar", "24000", "-ac", "1",
             "-af", "apad=pad_dur=0.06", str(final)])
        raw.unlink(missing_ok=True)
        durations.append(probe_duration(final))
        files.append(final)
    (workdir / "vlist.txt").write_text(
        "".join(f"file '{f.name}'\n" for f in files), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "vlist.txt",
         "-c", "copy", "voice.wav"], cwd=workdir)
    return durations


# ---------- captions ----------
def transcribe_words(wav):
    from faster_whisper import WhisperModel
    model = WhisperModel(CFG["whisper_model"], device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav), word_timestamps=True)
    return [(w.word.strip(), w.start, w.end)
            for segment in segments for w in (segment.words or []) if w.word.strip()]


def ass_time(seconds):
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{seconds % 60:05.2f}"


def write_ass(words, beats, durations, ep, path):
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,"
        "Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Word,@F,118,&H0000E5FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,10,3,5,60,60,60,1\n"
        "Style: Head,@F,46,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,6,0,8,60,60,60,1\n"
        "Style: Label,@F,74,&H00FFFFFF,&H000000FF,&H003030E0,&H00000000,-1,0,0,0,100,100,0,0,3,14,0,8,60,60,60,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    ).replace("@F", CFG["font"])
    total = sum(durations)
    events = [f"Dialogue: 0,{ass_time(0)},{ass_time(total)},Head,,0,0,0,,"
              f"{{\\an8\\pos(540,170)}}{SERIES_NAME.upper()}  |  EP {ep}"]
    current = 0.0
    for beat, duration in zip(beats, durations):
        if beat.get("label"):
            safe_label = re.sub(r"[{}\\]", "", beat["label"])
            events.append(f"Dialogue: 0,{ass_time(current)},{ass_time(current + duration)},Label,,0,0,0,,"
                          f"{{\\an8\\pos(540,290)}}{safe_label}")
        current += duration
    pop = "\\fscx85\\fscy85\\t(0,90,\\fscx108\\fscy108)\\t(90,150,\\fscx100\\fscy100)"
    for i, (word, start, end) in enumerate(words):
        next_start = words[i + 1][1] if i + 1 < len(words) else None
        finish = next_start if next_start is not None and next_start - end < 0.4 else end + 0.05
        finish = max(finish, start + 0.1)
        safe_word = re.sub(r"[{}\\]", "", word).upper()
        events.append(f"Dialogue: 1,{ass_time(start)},{ass_time(finish)},Word,,0,0,0,,"
                      f"{{\\an5\\pos(540,1130){pop}}}{safe_word}")
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


# ---------- stock footage ----------
def pexels_candidates(query):
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return []
    response = SESSION.get("https://api.pexels.com/videos/search",
                          headers={"Authorization": key},
                          params={"query": query, "orientation": "portrait", "per_page": 15},
                          timeout=30)
    response.raise_for_status()
    results = []
    for video in response.json().get("videos", []):
        files = [f for f in video.get("video_files", [])
                 if f.get("file_type") == "video/mp4" and (f.get("height") or 0) > (f.get("width") or 0)]
        if files:
            chosen = min(files, key=lambda f: abs((f.get("width") or 0) - 1080))
            results.append((chosen["link"], video.get("url")))
    return results


def pixabay_candidates(query):
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        return []
    response = SESSION.get("https://pixabay.com/api/videos/",
                           params={"key": key, "q": query[:100], "per_page": 20,
                                   "safesearch": "true", "video_type": "film"},
                           timeout=30)
    response.raise_for_status()
    results = []
    for hit in response.json().get("hits", []):
        videos = hit.get("videos", {})
        chosen = next((videos[k] for k in ("medium", "small", "large")
                       if videos.get(k, {}).get("url")), None)
        if chosen:
            results.append((chosen["url"], hit.get("pageURL")))
    return results


def download(url, path):
    with SESSION.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    handle.write(chunk)
    if not path.exists() or path.stat().st_size < 10_000:
        raise RuntimeError(f"Downloaded footage is empty or too small: {path.name}")


def make_segment(raw, duration, output):
    raw_duration = probe_duration(raw)
    if raw_duration <= 0:
        raise RuntimeError(f"Invalid source video duration: {raw}")
    start = random.uniform(0, max(0.0, raw_duration - min(duration, raw_duration)))
    x, y = random.randint(-35, 35), random.randint(-60, 60)
    vf = (f"scale=1188:2112:force_original_aspect_ratio=increase,"
          f"crop=1080:1920:(iw-1080)/2+({x}):(ih-1920)/2+({y}),"
          "eq=saturation=1.18:contrast=1.06,setsar=1,fps=30")
    run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-stream_loop", "-1", "-i", str(raw),
         "-an", "-t", f"{duration:.2f}", "-vf", vf, "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", str(output)])


def build_background(beats, durations, workdir):
    local = pick_files(ROOT / "footage", VIDEO_EXT)
    downloaded, used_urls, sources, segments = [], set(), [], []
    providers = []
    if os.getenv("PIXABAY_API_KEY"):
        providers.append(pixabay_candidates)
    if os.getenv("PEXELS_API_KEY"):
        providers.append(pexels_candidates)

    for i, (beat, duration) in enumerate(zip(beats, durations)):
        raw = None
        for provider in providers:
            try:
                candidates = [item for item in provider(beat["visual"]) if item[0] not in used_urls]
                if not candidates:
                    candidates = [item for item in provider(beat["visual"].split()[0]) if item[0] not in used_urls]
                if candidates:
                    url, page = random.choice(candidates[:8])
                    candidate_path = workdir / f"raw{i}.mp4"
                    download(url, candidate_path)
                    raw = candidate_path
                    downloaded.append(raw)
                    used_urls.add(url)
                    sources.append(page)
                    break
            except Exception as exc:
                print(f"Footage provider failed for '{beat['visual']}': {exc}")
        if raw is None and downloaded:
            raw = random.choice(downloaded)
        if raw is None and local:
            raw = random.choice(local)
        if raw is None:
            raise RuntimeError(
                f"No footage for beat {i + 1}. Add a working Pixabay/PEXELS key or licensed clips in /footage."
            )
        segment = workdir / f"seg{i}.mp4"
        make_segment(raw, duration, segment)
        segments.append(segment)

    (workdir / "segments.txt").write_text(
        "".join(f"file '{segment.name}'\n" for segment in segments), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "segments.txt",
         "-c", "copy", "background.mp4"], cwd=workdir)
    return workdir / "background.mp4", sources


def render(background, voice, workdir, output, duration):
    music_files = pick_files(ROOT / "music", AUDIO_EXT)
    music = random.choice(music_files) if music_files else None
    command = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(background), "-i", str(voice)]
    video_filter = "[0:v]ass=captions.ass[v];"
    if music:
        command += ["-stream_loop", "-1", "-i", str(music)]
        audio_filter = (f"[2:a]volume={CFG['music_volume']}[m];"
                        "[1:a][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
    else:
        audio_filter = "[1:a]anull[a]"
    command += ["-filter_complex", video_filter + audio_filter,
                "-map", "[v]", "-map", "[a]", "-t", f"{duration + 0.25:.2f}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", str(output)]
    run(command, cwd=workdir)
    return music.name if music else None


# ---------- episode ----------
def make_video(index, state, candidates):
    trend = choose_topic(candidates, state)
    episode_number = len(state["episodes"]) + 1
    data = write_episode(episode_number, state, trend)
    workdir = OUT_DAY / f"{index:02d}_ep{episode_number}"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "script.txt").write_text(
        "\n".join(f"[{b['label']}] {b['text']}" for b in data["beats"]), encoding="utf-8")
    durations = voice_beats(data["beats"], workdir)
    total = sum(durations)
    words = transcribe_words(workdir / "voice.wav")
    write_ass(words, data["beats"], durations, episode_number, workdir / "captions.ass")
    background, sources = build_background(data["beats"], durations, workdir)
    music = render(background, workdir / "voice.wav", workdir,
                   workdir / "video.mp4", total)
    video_path = workdir / "video.mp4"
    if not video_path.exists() or video_path.stat().st_size < 50_000:
        raise RuntimeError("Rendered MP4 is missing or unexpectedly small")

    metadata = {
        "episode": episode_number, "series": SERIES_NAME,
        "topic": data["topic"], "trend_signal_source": data["trend_source"],
        "title": data["title"], "description": data.get("description", "") +
        "\n\nOriginal AI-assisted commentary. Verify facts and footage rights before publishing. #Shorts",
        "tags": data.get("tags", []), "duration_sec": round(total, 1),
        "voice_engine": CFG["engine"], "footage_sources": sources, "music_used": music,
        "review_checklist": [
            "Check factual claims and spelling",
            "Confirm stock footage and music license terms",
            "Review platform rules and synthetic-content disclosure requirements",
            "Review the finished video before publishing",
        ],
    }
    (workdir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Only mark a topic used after the MP4 has successfully rendered.
    state["used_topics"].append(data["topic"])
    state["episodes"].append({
        "ep": episode_number, "topic": data["topic"],
        "title": data["title"], "summary": data.get("summary", ""),
    })
    save_state(state)

    # Remove large intermediate video/audio segments; retain final video and review files.
    for pattern in ("raw*.mp4", "seg*.mp4", "b*.wav", "segments.txt",
                    "vlist.txt", "background.mp4", "voice.wav"):
        for path in workdir.glob(pattern):
            path.unlink(missing_ok=True)
    return str(workdir), data["title"]


# ---------- reports and notifications ----------
def preflight():
    errors = []
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            errors.append(f"'{tool}' not found in PATH")
    if not os.getenv("GEMINI_API_KEY"):
        try:
            SESSION.get(CFG["ollama_url"].rsplit("/api", 1)[0], timeout=5)
        except Exception:
            errors.append("Missing GEMINI_API_KEY and Ollama is not running")
    if not os.getenv("PIXABAY_API_KEY") and not os.getenv("PEXELS_API_KEY") \
            and not pick_files(ROOT / "footage", VIDEO_EXT):
        errors.append("No footage source configured: add PIXABAY_API_KEY/PEXELS_API_KEY or clips in /footage")
    if CFG["engine"] == "piper":
        if not (shutil.which(CFG["piper_bin"]) or Path(CFG["piper_bin"]).exists()):
            errors.append(f"Piper executable not found: {CFG['piper_bin']}")
        if not Path(CFG["piper_model"]).exists():
            errors.append(f"Piper model missing: {CFG['piper_model']}")
    return errors


def notify(message):
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            SESSION.post(f"https://api.telegram.org/bot{token}/sendMessage",
                         json={"chat_id": chat, "text": message[:4000]}, timeout=20).raise_for_status()
        except Exception as exc:
            print(f"Telegram text notification failed: {exc}")


def notify_video(path, caption):
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            with open(path, "rb") as handle:
                response = SESSION.post(
                    f"https://api.telegram.org/bot{token}/sendVideo",
                    data={"chat_id": chat, "caption": caption[:900]},
                    files={"video": handle}, timeout=(30, 240))
            response.raise_for_status()
        except Exception as exc:
            print(f"Telegram video delivery failed: {exc}")


def main():
    OUT_DAY.mkdir(parents=True, exist_ok=True)
    state = load_state()
    done, errors = [], []
    init_tts()
    errors.extend(preflight())
    if not errors:
        candidates = get_trend_candidates()
        print(f"Found {len(candidates)} live trend candidates.")
        start_index = len([p for p in OUT_DAY.iterdir() if p.is_dir()]) + 1
        for offset in range(CFG["videos_per_run"]):
            try:
                path, title = make_video(start_index + offset, state, candidates)
                done.append((title, path))
                notify_video(Path(path) / "video.mp4", title)
            except Exception as exc:
                errors.append(f"Video {offset + 1}: {type(exc).__name__}: {exc}")
                with (OUT_DAY / "errors.log").open("a", encoding="utf-8") as handle:
                    handle.write(traceback.format_exc() + "\n")
    report_lines = [
        f"# Daily report {TODAY}",
        f"Completed: {len(done)} | Errors: {len(errors)} | Voice: {CFG['engine']}",
        f"Trend candidates discovered: {len(get_trend_candidates()) if not errors else 'not checked'}",
        "",
    ]
    report_lines += [f"- OK: {title} -> {path}" for title, path in done]
    report_lines += [f"- ERROR: {error}" for error in errors]
    report_lines.append("\nNext step: review each video's MP4 and meta.json before publishing.")
    report = "\n".join(report_lines)
    with (OUT_DAY / "report.md").open("a", encoding="utf-8") as handle:
        handle.write(report + "\n\n")
    print(report)
    notify(report)
    if errors and not done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
