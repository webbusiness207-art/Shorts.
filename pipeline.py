#!/usr/bin/env python3
"""Shorts Factory: topic -> script -> voice -> captions -> 1080x1920 video -> daily report.
Everything runs locally and free (Ollama, Piper, faster-whisper, FFmpeg)."""
import datetime as dt
import json
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

ROOT = Path(__file__).parent
CFG = {
    "videos_per_run": int(os.getenv("VIDEOS_PER_RUN") or 3),
    "niche": os.getenv("NICHE") or "mind-blowing space and science facts",
    "ollama_url": "http://localhost:11434/api/generate",
    "ollama_model": "llama3.1:8b",
    "piper_bin": os.getenv("PIPER_BIN") or "piper",
    "font": os.getenv("CAPTION_FONT") or "DejaVu Sans",
    "piper_model": str(ROOT / "models" / "en_US-lessac-medium.onnx"),
    "whisper_model": "small",
    "music_volume": 0.10,
    "words_min": 90,
    "words_max": 160,
}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg"}
TODAY = dt.date.today().isoformat()
OUT_DAY = ROOT / "output" / TODAY
USED_FILE = ROOT / "used_topics.json"


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def probe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    return float(r.stdout.strip())


_GEMINI_OK = None


def gemini_models(key):
    """Ask Google which Flash models this key can use, newest non-lite first."""
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
        # stable before preview; lite first (fast, no slow "thinking"), then newest version
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
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.9}}
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
                if r.status_code == 429:  # free-tier rate limit: wait and retry
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
        "model": CFG["ollama_model"], "prompt": prompt,
        "stream": False, "format": "json"}, timeout=900)
    r.raise_for_status()
    return json.loads(r.json()["response"])


def trending_hints():
    try:
        r = requests.get("https://trends.google.com/trending/rss?geo=US", timeout=20)
        root = ET.fromstring(r.content)
        return [i.findtext("title") for i in root.iter("item")][:15]
    except Exception:
        return []


def load_used():
    return json.loads(USED_FILE.read_text()) if USED_FILE.exists() else []


def get_topics(n):
    used = {t.lower() for t in load_used()}
    topics = []
    queue = ROOT / "topics.txt"  # your own hand-picked ideas run first
    if queue.exists():
        topics += [l.strip() for l in queue.read_text().splitlines() if l.strip()]
    topics = [t for t in topics if t.lower() not in used]
    if len(topics) < n:
        hints = trending_hints()
        data = llm_json(
            f"You are a YouTube Shorts researcher. Niche: {CFG['niche']}.\n"
            f"Trending searches today (use only if they genuinely fit the niche): {hints}\n"
            f"Already covered, do not repeat: {sorted(used)[-40:]}\n"
            f"Give {n - len(topics) + 2} specific, verifiable, curiosity-driven video topics. "
            'Return JSON: {"topics": ["...", "..."]}')
        topics += [t for t in data.get("topics", []) if t.lower() not in used]
    return topics[:n]


def write_script(topic):
    prompt = (
        f"Write a YouTube Shorts narration about: {topic}\nNiche: {CFG['niche']}.\n"
        f"Rules: {CFG['words_min']}-{CFG['words_max']} words. The FIRST sentence is a hook of 12 words "
        "or fewer that creates curiosity or tension. Conversational, punchy, short sentences, "
        "one surprising payoff, end with a line that makes viewers want to follow. "
        "ONLY state facts you are highly confident are true; no invented statistics or quotes; "
        "nothing about real named people that could be defamatory. No stage directions, no emojis.\n"
        'Also give "footage_query": 1-3 plain words describing generic stock footage that fits the '
        'topic visually (e.g. "galaxy stars", "ocean waves", "city night").\n'
        'Return JSON: {"title": "under 70 chars", "script": "...", "description": "...", '
        '"tags": ["..."], "footage_query": "..."}')
    for _ in range(3):
        d = llm_json(prompt)
        n = len(d.get("script", "").split())
        if CFG["words_min"] <= n <= CFG["words_max"] + 20:
            return d
    raise RuntimeError("LLM could not produce a script of valid length")


def tts(text, wav):
    run([CFG["piper_bin"], "--model", CFG["piper_model"], "--output_file", str(wav)], input=text)


def ass_time(s):
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"


def make_captions(wav, ass_path):
    from faster_whisper import WhisperModel
    model = WhisperModel(CFG["whisper_model"], device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(wav), word_timestamps=True)
    words = [w for s in segs for w in s.words]
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,"
        "Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,{font},96,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,640,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    ).replace("{font}", CFG["font"])
    lines = []
    for i in range(0, len(words), 3):
        grp = words[i:i + 3]
        text = " ".join(w.word.strip() for w in grp).upper().replace("{", "").replace("}", "")
        end = words[i + 3].start if i + 3 < len(words) else grp[-1].end
        lines.append(f"Dialogue: 0,{ass_time(grp[0].start)},{ass_time(end)},Default,,0,0,0,,{text}")
    ass_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def pick_files(folder, exts):
    return [p for p in folder.iterdir() if p.suffix.lower() in exts] if folder.exists() else []


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


def fetch_stock(query, workdir, dur):
    """Download licensed stock clips (Pixabay or Pexels), crop to 9:16, concat into bg.mp4."""
    cands = pixabay_candidates(query) if os.getenv("PIXABAY_API_KEY") else pexels_candidates(query)
    random.shuffle(cands)
    segs, urls, total = [], [], 0.0
    for link, page in cands:
        raw, seg = workdir / f"raw{len(segs)}.mp4", workdir / f"seg{len(segs)}.mp4"
        with requests.get(link, stream=True, timeout=90) as resp:
            resp.raise_for_status()
            with raw.open("wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
        run(["ffmpeg", "-y", "-i", str(raw), "-an", "-t", "12", "-vf",
             "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(seg)])
        raw.unlink()
        segs.append(seg)
        urls.append(page)
        total += probe_duration(seg)
        if total >= dur + 2 or len(segs) >= 6:
            break
    if not segs:
        raise RuntimeError(f"No stock footage found for '{query}'")
    (workdir / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in segs))
    run(["ffmpeg", "-y", "-f", "concat", "-i", "list.txt", "-c", "copy", "bg.mp4"], cwd=workdir)
    for p in segs:
        p.unlink()
    return workdir / "bg.mp4", urls


def get_background(query, workdir, dur):
    if os.getenv("PIXABAY_API_KEY") or os.getenv("PEXELS_API_KEY"):
        bg, urls = fetch_stock(query, workdir, dur)
        return bg, 0.0, urls
    clip = random.choice(pick_files(ROOT / "footage", VIDEO_EXT))
    return clip, random.uniform(0, max(0.0, probe_duration(clip) - dur - 1)), [clip.name]


def render(bg, start, wav, workdir, out_mp4, dur):
    music = pick_files(ROOT / "music", AUDIO_EXT)
    track = random.choice(music) if music else None
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-stream_loop", "-1", "-i", str(bg),
           "-i", str(wav)]
    vf = ("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
          "setsar=1,fps=30,ass=captions.ass[v];")
    if track:
        cmd += ["-stream_loop", "-1", "-i", str(track)]
        af = (f"[2:a]volume={CFG['music_volume']}[m];[1:a][m]amix=inputs=2:duration=first:"
              "dropout_transition=0:normalize=0[a]")
    else:
        af = "[1:a]anull[a]"
    cmd += ["-filter_complex", vf + af, "-map", "[v]", "-map", "[a]", "-t", f"{dur + 0.4:.2f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_mp4)]
    run(cmd, cwd=workdir)  # cwd=workdir so captions.ass needs no path escaping
    return track.name if track else None


def make_video(idx, topic):
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:40].strip("-")
    d = OUT_DAY / f"{idx:02d}_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    s = write_script(topic)
    (d / "script.txt").write_text(s["script"], encoding="utf-8")
    tts(s["script"], d / "voice.wav")
    dur = probe_duration(d / "voice.wav")
    make_captions(d / "voice.wav", d / "captions.ass")
    bg, start, sources = get_background(s.get("footage_query") or CFG["niche"], d, dur)
    track = render(bg, start, d / "voice.wav", d, d / "video.mp4", dur)
    for junk in ("bg.mp4", "list.txt"):
        (d / junk).unlink(missing_ok=True)
    meta = {"topic": topic, "title": s["title"], "description": s["description"],
            "tags": s.get("tags", []), "duration_sec": round(dur, 1),
            "footage_sources": sources, "music_used": track,
            "review_checklist": ["Fact-check every claim", "Footage/music license OK",
                                 "Disclose AI/synthetic content in YouTube if it looks realistic"]}
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(d)


def preflight():
    errs = []
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            errs.append(f"'{tool}' not found in PATH")
    if not (shutil.which(CFG["piper_bin"]) or Path(CFG["piper_bin"]).exists()):
        errs.append(f"Piper not found: {CFG['piper_bin']}")
    if not Path(CFG["piper_model"]).exists():
        errs.append(f"Piper voice model missing: {CFG['piper_model']}")
    if not (os.getenv("PIXABAY_API_KEY") or os.getenv("PEXELS_API_KEY")) and not pick_files(ROOT / "footage", VIDEO_EXT):
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
    done, failed, errors = [], [], []
    errors += preflight()
    if not errors:
        try:
            topics = get_topics(CFG["videos_per_run"])
        except Exception as e:
            topics = []
            errors.append(f"Topic research failed: {e}")
        start_idx = len([p for p in OUT_DAY.iterdir() if p.is_dir()]) + 1
        for i, topic in enumerate(topics):
            try:
                path = make_video(start_idx + i, topic)
                done.append((topic, path))
                notify_video(Path(path) / 'video.mp4', topic)
                used = load_used() + [topic]
                USED_FILE.write_text(json.dumps(used, indent=1))
            except Exception as e:
                failed.append(topic)
                errors.append(f"'{topic}': {e}")
                (OUT_DAY / "errors.log").open("a").write(traceback.format_exc() + "\n")
    report = [f"# Daily report {TODAY}", f"Completed: {len(done)}  |  Failed: {len(failed)}", ""]
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
