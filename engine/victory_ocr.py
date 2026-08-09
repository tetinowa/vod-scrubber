import os
import subprocess

FRAME_SAMPLE_INTERVAL_SEC = 1.0
SCAN_WINDOW_AFTER_SEC = 90.0

TESSERACT_CMD_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SCAN_SOURCE_TAGS = ["is_boss_rage_cussing_beat", "is_nonono_or_death_beat"]

OCR_SCORE = 0.9


def format_hms(seconds):
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def configure_tesseract():
    import pytesseract
    if os.path.exists(TESSERACT_CMD_WINDOWS):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_WINDOWS


def extract_frame(video_path, t_sec, out_path):
    cmd = ["ffmpeg", "-y", "-ss", str(t_sec), "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ocr_frame_text(image_path):
    import pytesseract
    from PIL import Image
    configure_tesseract()
    return pytesseract.image_to_string(Image.open(image_path)).lower()


def is_victory_text(text):
    return "victory" in text and "achiev" in text


def find_victory_in_window(video_path, start_sec, end_sec, frame_path, sample_interval_sec=FRAME_SAMPLE_INTERVAL_SEC):
    t = start_sec
    while t < end_sec:
        try:
            extract_frame(video_path, t, frame_path)
            text = ocr_frame_text(frame_path)
        except Exception:
            t += sample_interval_sec
            continue
        if is_victory_text(text):
            return t
        t += sample_interval_sec
    return None


def scan_for_boss_victories(video_path, candidates):
    seed_times = sorted({c["time"] for c in candidates if any(c.get(tag) for tag in SCAN_SOURCE_TAGS)})
    if not seed_times:
        return []

    print(f"Scanning {len(seed_times)} flagged moments for VICTORY ACHIEVED banners...", flush=True)
    frame_path = os.path.join(os.path.dirname(os.path.abspath(video_path)) or ".", "_victory_frame.jpg")

    victories = []
    try:
        for i, t in enumerate(seed_times):
            print(f"  [{i + 1}/{len(seed_times)}] scanning near {format_hms(t)}...", flush=True)
            found = find_victory_in_window(video_path, t, t + SCAN_WINDOW_AFTER_SEC, frame_path)
            if found is not None:
                print(f"    victory banner found at {format_hms(found)}", flush=True)
                victories.append(found)
    finally:
        if os.path.exists(frame_path):
            os.remove(frame_path)

    return sorted(set(victories))


def victories_to_candidates(victory_times, pad_before=10.0, pad_after=6.0):
    return [
        {
            "time": round(t, 2),
            "score": OCR_SCORE,
            "reason": "Boss victory banner detected on screen (OCR)",
            "start": round(max(0.0, t - pad_before), 2),
            "end": round(t + pad_after, 2),
            "categories": ["skill_highlight"],
            "shareability_score": round(OCR_SCORE * 100),
            "hype_score": 80,
            "controversy_level": "none",
            "moderation_risk": "none",
            "has_setup_punchline_shape": False,
            "confidence": "high",
            "uncertainty_reason": "",
            "sub_beats": [],
            "is_nonono_or_death_beat": False,
            "is_zweihander_beat": False,
            "is_boss_rage_cussing_beat": False,
            "is_boss_victory_beat": True,
        }
        for t in victory_times
    ]
