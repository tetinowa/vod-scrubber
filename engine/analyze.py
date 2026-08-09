#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import namedtuple

import numpy as np

WINDOW_SEC = 5.0
HOP_SEC = 2.0

WEIGHT_AUDIO_ENERGY = 0.15
WEIGHT_PROFANITY = 0.05
WEIGHT_SPEAKING_RATE = 0.10
WEIGHT_REPETITION = 0.15
WEIGHT_PAYOFF = 0.10
WEIGHT_CHAT = 0.00

LLM_MERGE_GAP_SEC = 30.0

AUDIO_ENERGY_SPEECH_GATE_FLOOR = 0.35

RATE_BASELINE_WINDOWS = 30
PAYOFF_LOOKAHEAD_SEC = 15.0

MIN_WORD_CONFIDENCE = 0.4
HALLUCINATION_RUN_LIMIT = 5

DEFAULT_MIN_SCORE = 0.35
DEFAULT_TOP_N = None
DEFAULT_MIN_GAP_SEC = 30.0
DEFAULT_PAD_BEFORE_SEC = 5.0
DEFAULT_PAD_AFTER_SEC = 15.0

WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

AUDIO_SAMPLE_RATE = 22050

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

HYPE_PHRASES = [
    "no way", "let's go", "lets go", "let's fucking go", "lets fucking go",
    "oh my god", "holy shit", "are you kidding me", "no fucking way",
    "got him", "gotcha", "we did it", "i did it", "clutch", "insane",
    "unbelievable", "yesss", "yessss", "finally", "gg",
]

PANIC_PHRASES = [
    "oh no", "oh god", "oh shit", "i'm dead", "im dead", "no no no",
    "wait wait", "watch out", "what the hell", "fuck fuck", "shit shit",
    "help me", "run run",
]

PROFANITY_WORDS = ["fuck", "fucking", "shit", "damn", "hell"]

SIGNAL_LABELS = {
    "audio_energy": "loud",
    "profanity": "raging/cursing",
    "speaking_rate": "rapid speech",
    "repetition": "repeated exclamation",
    "payoff": "leads somewhere",
    "chat": "chat surge",
}


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Install it and make sure 'ffmpeg' runs from a shell.")


def extract_audio(video_path, out_wav, clip_duration_sec=None, start_sec=0.0):
    print(f"Extracting audio from {video_path} starting at {format_hms(start_sec)}...", flush=True)
    cmd = ["ffmpeg", "-y"]
    if start_sec:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", video_path]
    if clip_duration_sec:
        cmd += ["-t", str(clip_duration_sec)]
    cmd += ["-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), out_wav]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def normalize(a):
    a = np.asarray(a, dtype=float)
    if a.size and a.max() > a.min():
        return (a - a.min()) / (a.max() - a.min())
    return np.zeros_like(a)


def audio_energy_scores(wav_path, window_sec, hop_sec):
    import librosa
    print("Scoring audio energy...", flush=True)
    y, sr = librosa.load(wav_path, sr=None)
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    times, energies = [], []
    for start in range(0, max(1, len(y) - win), hop):
        chunk = y[start:start + win]
        energies.append(float(np.sqrt(np.mean(chunk ** 2))))
        times.append(start / sr)
    return np.array(times), normalize(energies)


def format_hms(seconds):
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


Word = namedtuple("Word", "start end text probability")


def transcribe_words(audio_path, start_sec=0.0):
    from faster_whisper import WhisperModel
    print(f"Transcribing with faster-whisper ({WHISPER_MODEL_SIZE}, {WHISPER_DEVICE})...", flush=True)
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    words = []
    for segment in segments:
        print(f"[{format_hms(start_sec + segment.start)}] {segment.text.strip()}", flush=True)
        for word in segment.words:
            words.append(Word(start_sec + word.start, start_sec + word.end, word.word.strip().lower(), word.probability))
    return words


def raw_signals_cache_path(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    safe_base = re.sub(r"[^\w-]", "_", base)[:80]
    model_tag = re.sub(r"[^\w-]", "_", WHISPER_MODEL_SIZE)
    return os.path.join(CACHE_DIR, f"{safe_base}.raw.{model_tag}.json")


def cache_covers_duration(cached_duration_sec, requested_duration_sec):
    if cached_duration_sec is None:
        return True
    if requested_duration_sec is None:
        return False
    return cached_duration_sec >= requested_duration_sec


def trim_to_duration(times, audio_e, words, duration_sec):
    if duration_sec is None:
        return times, audio_e, words
    mask = times < duration_sec
    return times[mask], audio_e[mask], [w for w in words if w.start < duration_sec]


def load_raw_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        data = json.load(f)
    return {
        "duration_sec": data["duration_sec"],
        "times": np.array(data["times"]),
        "audio_e": np.array(data["audio_e"]),
        "words": [Word(*w) for w in data["words"]],
    }


def save_raw_cache(cache_path, duration_sec, times, audio_e, words):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({
            "duration_sec": duration_sec,
            "times": times.tolist(),
            "audio_e": audio_e.tolist(),
            "words": words,
        }, f)


def compute_raw_signals(video_path, duration_sec, start_sec=0.0):
    require_ffmpeg()
    wav_path = os.path.join(os.path.dirname(os.path.abspath(video_path)) or ".", "_analyze_audio.wav")
    clip_duration_sec = (duration_sec - start_sec) if duration_sec is not None else None
    try:
        extract_audio(video_path, wav_path, clip_duration_sec, start_sec)
        times, audio_e = audio_energy_scores(wav_path, WINDOW_SEC, HOP_SEC)
        words = transcribe_words(wav_path, start_sec)
        times = times + start_sec
        return times, audio_e, words
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def get_raw_signals(video_path, duration_sec, use_cache):
    cache_path = raw_signals_cache_path(video_path)
    cached = load_raw_cache(cache_path) if use_cache else None

    if cached is not None and cache_covers_duration(cached["duration_sec"], duration_sec):
        print(f"Using cached audio+transcription: {cache_path}", flush=True)
        return trim_to_duration(cached["times"], cached["audio_e"], cached["words"], duration_sec)

    if cached is not None and cached["duration_sec"] is not None:
        print(f"Extending cached transcription from {format_hms(cached['duration_sec'])} onward...", flush=True)
        new_times, new_audio_e, new_words = compute_raw_signals(video_path, duration_sec, start_sec=cached["duration_sec"])
        times = np.concatenate([cached["times"], new_times])
        audio_e = np.concatenate([cached["audio_e"], new_audio_e])
        words = cached["words"] + new_words
    else:
        times, audio_e, words = compute_raw_signals(video_path, duration_sec)

    save_raw_cache(cache_path, duration_sec, times, audio_e, words)
    return trim_to_duration(times, audio_e, words, duration_sec)


def confident_words(words):
    return [w for w in words if w.probability >= MIN_WORD_CONFIDENCE]


def drop_hallucination_loops(words, run_limit=HALLUCINATION_RUN_LIMIT):
    result = []
    run_token = None
    run_count = 0
    for w in words:
        token = w.text.strip(".,!?")
        if token and token == run_token:
            run_count += 1
        else:
            run_token = token
            run_count = 1
        if run_count <= run_limit:
            result.append(w)
    return result


def words_in_range(words, start_t, end_t):
    return [w for w in words if start_t <= w.start < end_t]


def words_in_window(words, t, window_sec):
    return words_in_range(words, t, t + window_sec)


def speech_presence_scores(words, times, window_sec):
    return np.array([1.0 if words_in_window(words, t, window_sec) else 0.0 for t in times])


def gate_audio_energy(audio_e, speech_presence):
    gate = AUDIO_ENERGY_SPEECH_GATE_FLOOR + (1 - AUDIO_ENERGY_SPEECH_GATE_FLOOR) * speech_presence
    return audio_e * gate


def profanity_density_scores(words, times, window_sec):
    hits = np.zeros(len(times))
    for i, t in enumerate(times):
        tokens = [w.text.strip(".,!?") for w in words_in_window(words, t, window_sec)]
        hits[i] = sum(1 for token in tokens if token in PROFANITY_WORDS)
    return normalize(hits)


def rolling_mean(a, k):
    if len(a) == 0:
        return a
    kernel = np.ones(k) / k
    return np.convolve(a, kernel, mode="same")


def speaking_rate_spike_scores(words, times, window_sec):
    print("Scoring speaking rate...", flush=True)
    raw_rate = np.array([len(words_in_window(words, t, window_sec)) / window_sec for t in times])
    baseline = rolling_mean(raw_rate, min(RATE_BASELINE_WINDOWS, max(1, len(raw_rate))))
    spike = np.clip(raw_rate - baseline, 0, None)
    return normalize(spike)


def repetition_scores(words, times, window_sec):
    print("Scoring repetition bursts...", flush=True)
    bursts = np.zeros(len(times))
    for i, t in enumerate(times):
        tokens = [w.text.strip(".,!?") for w in words_in_window(words, t, window_sec)]
        best_run = 1
        run = 1
        for j in range(1, len(tokens)):
            if tokens[j] and tokens[j] == tokens[j - 1]:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 1
        bursts[i] = max(0, best_run - 1)
    return normalize(bursts)


def payoff_scores(words, times, window_sec, lookahead_sec):
    hits = np.zeros(len(times))
    for i, t in enumerate(times):
        text = " ".join(w.text for w in words_in_range(words, t + window_sec, t + window_sec + lookahead_sec))
        hits[i] = sum(text.count(phrase) for phrase in HYPE_PHRASES + PANIC_PHRASES)
    return normalize(hits)


def chat_scores_placeholder(times):
    return np.zeros(len(times))


def build_reason(contributions):
    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    top = [SIGNAL_LABELS[name] for name, value in ranked if value > 0.03][:2]
    return " + ".join(top) if top else "elevated activity"


def fuse_scores(times, signals):
    combined = np.zeros(len(times))
    reasons = []
    for i in range(len(times)):
        contributions = {
            "audio_energy": WEIGHT_AUDIO_ENERGY * signals["audio_energy"][i],
            "profanity": WEIGHT_PROFANITY * signals["profanity"][i],
            "speaking_rate": WEIGHT_SPEAKING_RATE * signals["speaking_rate"][i],
            "repetition": WEIGHT_REPETITION * signals["repetition"][i],
            "payoff": WEIGHT_PAYOFF * signals["payoff"][i],
            "chat": WEIGHT_CHAT * signals["chat"][i],
        }
        combined[i] = sum(contributions.values())
        reasons.append(build_reason(contributions))
    return combined, reasons


def pick_top_moments(times, scores, reasons, min_score, min_gap_sec, max_candidates=None):
    order = np.argsort(scores)[::-1]
    chosen = []
    for idx in order:
        if scores[idx] < min_score:
            break
        t = times[idx]
        if all(abs(t - c[0]) > min_gap_sec for c in chosen):
            chosen.append((t, scores[idx], reasons[idx]))
        if max_candidates is not None and len(chosen) >= max_candidates:
            break
    chosen.sort()
    return chosen


def to_candidates(chosen, pad_before, pad_after):
    return [
        {
            "time": round(t, 2),
            "score": round(float(score), 4),
            "reason": reason,
            "start": round(max(0.0, t - pad_before), 2),
            "end": round(t + pad_after, 2),
        }
        for t, score, reason in chosen
    ]


def merge_candidates(heuristic_candidates, llm_candidates, min_gap_sec):
    merged = list(llm_candidates)
    for c in heuristic_candidates:
        if any(abs(c["time"] - m["time"]) < min_gap_sec for m in llm_candidates):
            continue
        merged.append(c)
    merged.sort(key=lambda c: c["time"])
    return merged


def analyze(video_path, min_score, min_gap_sec, pad_before, pad_after, duration_sec=None, use_cache=True, max_candidates=None, use_llm=True):
    times, audio_e, words = get_raw_signals(video_path, duration_sec, use_cache)
    clean_words = drop_hallucination_loops(confident_words(words))

    print("Scoring audio and speech signals...", flush=True)
    speech_presence = speech_presence_scores(words, times, WINDOW_SEC)
    signals = {
        "audio_energy": gate_audio_energy(audio_e, speech_presence),
        "profanity": profanity_density_scores(clean_words, times, WINDOW_SEC),
        "speaking_rate": speaking_rate_spike_scores(words, times, WINDOW_SEC),
        "repetition": repetition_scores(clean_words, times, WINDOW_SEC),
        "payoff": payoff_scores(clean_words, times, WINDOW_SEC, PAYOFF_LOOKAHEAD_SEC),
        "chat": chat_scores_placeholder(times),
    }

    combined, reasons = fuse_scores(times, signals)
    chosen = pick_top_moments(times, combined, reasons, min_score, min_gap_sec, max_candidates)
    heuristic_candidates = to_candidates(chosen, pad_before, pad_after)

    if not use_llm:
        return heuristic_candidates

    import llm_review
    llm_candidates = llm_review.get_llm_candidates(video_path, words, duration_sec, use_cache)
    return merge_candidates(heuristic_candidates, llm_candidates, LLM_MERGE_GAP_SEC)


def main():
    parser = argparse.ArgumentParser(description="Detect clip-worthy moments in a Twitch VOD.")
    parser.add_argument("video", help="Path to the downloaded VOD file")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Only keep candidates scoring at least this high")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="Optional cap on how many candidates to return")
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP_SEC)
    parser.add_argument("--pad-before", type=float, default=DEFAULT_PAD_BEFORE_SEC)
    parser.add_argument("--pad-after", type=float, default=DEFAULT_PAD_AFTER_SEC)
    parser.add_argument("--out", default="candidates.json")
    parser.add_argument("--duration-min", type=float, default=None, help="Only process the first N minutes of the VOD")
    parser.add_argument("--no-cache", action="store_true", help="Force re-transcription even if a cached run exists")
    parser.add_argument("--no-llm", action="store_true", help="Skip the Claude semantic review pass, heuristics only")
    args = parser.parse_args()

    duration_sec = args.duration_min * 60 if args.duration_min else None
    candidates = analyze(
        args.video, args.min_score, args.min_gap, args.pad_before, args.pad_after,
        duration_sec=duration_sec, use_cache=not args.no_cache, max_candidates=args.top, use_llm=not args.no_llm,
    )

    with open(args.out, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"\nDone. {len(candidates)} candidates written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
