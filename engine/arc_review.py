import argparse
import glob
import json
import os
import re

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

ARC_MODEL = "claude-sonnet-5"
ARC_MAX_TOKENS = 8192

CONNECTIVE_MODEL = "claude-haiku-4-5-20251001"
CONNECTIVE_MAX_TOKENS = 16384
CONNECTIVE_MIN_GAP_SEC = 20.0
CONNECTIVE_WORD_CONFIDENCE_FLOOR = 0.4
CONNECTIVE_HALLUCINATION_RUN_LIMIT = 5
CONNECTIVE_MIN_GAP_WORDS = 12
CONNECTIVE_MIN_WPM = 15.0
CONNECTIVE_CHUNK_WORDS = 6000

DEFAULT_GAME = "ds2"
BOSS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boss_data")

ARC_ROLES = ["hook", "rising_action", "attempt", "setback", "climax", "resolution"]

FLAG_FIELDS = [
    "is_boss_victory_beat",
    "is_nonono_or_death_beat",
    "is_boss_rage_cussing_beat",
    "is_zweihander_beat",
]

ARC_PROMPT = """Below is an ordered list of clip-worthy moments already detected in a Dark Souls Twitch VOD \
by an earlier review pass. Each moment is independent and was scored for standalone shareability — your \
job is different: find which of these moments, strung together in order, form a coherent long-form \
throughline (setup, escalation, failed attempts, a turning point, a payoff), and label each with its \
narrative role.

Not every moment belongs in the arc. Skip ones that are isolated bits, banter, or asides with no connection \
to a struggle/attempt/resolution shape. Include only moments that genuinely build toward or come from a \
throughline.

is_boss_victory_beat is a strong signal for climax/resolution. is_nonono_or_death_beat and \
is_boss_rage_cussing_beat are strong signals for attempt/setback. Use them, but judge from the reason text \
too — flags can be noisy.

Roles, in the order they'd typically appear (a real arc can repeat attempt/setback many times, or skip some):
- hook: the moment that establishes what's at stake or draws the viewer in
- rising_action: tension building before a real attempt
- attempt: a genuine try at the thing (boss fight, section, puzzle)
- setback: a failure, death, or step backward
- climax: the decisive moment things turn
- resolution: the payoff/aftermath once it's over

Streams don't usually end at an arbitrary point — they tend to end on a payoff (a boss kill, reaching a \
new area, some accomplishment). Give real weight to that when you're near the end of the list: if one of \
the last few moments plausibly reads as a stopping point, lean toward labeling it climax/resolution rather \
than treating resolution-finding as a cold search across the whole list. But don't force it — if the VOD \
genuinely just cuts off mid-attempt with no payoff, don't invent one; leave the tail as attempt/setback.

For each moment you include, also give your best guess at what boss or area it belongs to, from context \
in its reason text — a short free-text label (e.g. "Ruin Sentinels", "Heide's Tower"). If you can't tell, \
use null. This is a rough guess for later filtering, not an authoritative source.

Moments (index, timestamp, categories, scores, flags, reason):
{moments}

Respond with ONLY a JSON object of this exact shape, no other text:
{{
  "segments": [
    {{
      "source_moment_index": <int, index from the list above>,
      "role": "hook" | "rising_action" | "attempt" | "setback" | "climax" | "resolution",
      "reason": "one line on why this moment plays this role in the throughline",
      "boss_or_area": "<short label or null>"
    }}
  ]
}}

Order segments the way they should appear in the final edit. Empty "segments" array if nothing here forms \
a real throughline."""

CONNECTIVE_PROMPT = """Below are stretches of transcript from a Dark Souls Twitch VOD that an earlier \
clip-detection pass skipped over — they weren't independently shareable as standalone clips, so they were \
never scored. Each stretch is timestamped hh:mm:ss and shown separately; stretches are NOT continuous with \
each other, don't assume anything connects across a "--- GAP ---" break.

This is for a long-form edit, not a shorts reel. Find material here worth keeping for narrative continuity \
— quiet strategizing, between-attempt commentary, reactions right after a death or a near-miss, setup chatter \
before an attempt, brief narration of what she's about to try. The bar is low: it doesn't need a punchline or \
be independently impressive, it just needs to help a viewer follow what's happening and why between the \
bigger moments.

Do not flag: dead air, loading screens, isolated hallucinated words scattered with long gaps between them \
(faster-whisper noise, not real speech), or stretches with no actual sentence-level content.

For each stretch worth keeping, report its real start/end boundaries as they appear in the transcript (not \
the full gap window, just the part with content), a one-line reason, and your best guess at what boss or \
area it's about (short label, or null if you can't tell).

Transcript stretches:
{stretches}

Respond with ONLY a JSON array of objects: {{"start": "hh:mm:ss", "end": "hh:mm:ss", "reason": "...", \
"boss_or_area": "<label or null>"}}, sorted by start. Empty array if nothing qualifies."""


def format_hms(seconds):
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_hms(hms):
    h, m, s = (int(part) for part in str(hms).split(":"))
    return h * 3600 + m * 60 + s


def get_text_block(response):
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def load_boss_lookup(game):
    path = os.path.join(BOSS_DATA_DIR, f"{game}_bosses.json")
    if not os.path.exists(path):
        print(f"No boss database at {path}, boss_or_area guesses won't be normalized.", flush=True)
        return []
    with open(path) as f:
        boss_db = json.load(f)
    boss_entries, area_entries = [], []
    for area in boss_db.get("areas", []):
        area_entries.append((area["area_name"], {
            "area_id": area["area_id"],
            "area_name": area["area_name"],
            "boss_id": None,
            "boss_name": None,
        }))
        for boss in area.get("bosses", []):
            for name in [boss["boss_name"]] + boss.get("aliases", []):
                boss_entries.append((name, {
                    "area_id": area["area_id"],
                    "area_name": area["area_name"],
                    "boss_id": boss["boss_id"],
                    "boss_name": boss["boss_name"],
                }))
    boss_entries.sort(key=lambda pair: -len(pair[0]))
    area_entries.sort(key=lambda pair: -len(pair[0]))
    return [(name.lower(), match) for name, match in boss_entries + area_entries]


def match_boss_or_area(guess, lookup):
    if not guess:
        return None
    guess_lower = guess.lower()
    for name, match in lookup:
        if name in guess_lower or guess_lower in name:
            return match
    return None


def normalize_segments(segments, lookup):
    for seg in segments:
        guess = seg.get("boss_or_area")
        seg["boss_or_area"] = {"guess": guess, "matched": match_boss_or_area(guess, lookup)}
    return segments


def load_moments_cache(cache_path):
    if not os.path.exists(cache_path):
        raise SystemExit(f"No cache file at {cache_path}")
    with open(cache_path) as f:
        data = json.load(f)
    return sorted(data.get("moments", []), key=lambda m: m["anchor_sec"])


def find_raw_cache_path(llm_cache_path):
    cache_dir = os.path.dirname(os.path.abspath(llm_cache_path))
    match = re.match(r"(.+)\.llm\..+\.json$", os.path.basename(llm_cache_path))
    if not match:
        return None
    candidates = sorted(glob.glob(os.path.join(cache_dir, f"{match.group(1)}.raw.*.json")))
    return candidates[0] if candidates else None


def load_raw_cache(raw_cache_path):
    with open(raw_cache_path) as f:
        data = json.load(f)
    words = [{"start": w[0], "end": w[1], "text": w[2], "probability": w[3]} for w in data["words"]]
    duration_sec = data.get("duration_sec")
    if duration_sec is None and words:
        duration_sec = words[-1]["end"]
    return duration_sec, words


def words_to_transcript_text(words):
    lines = []
    line = None
    last_t = None
    for w in words:
        if last_t is None or w["start"] - last_t > 2:
            if line:
                lines.append(line)
            line = f"[{format_hms(w['start'])}] "
        line += w["text"] + " "
        last_t = w["start"]
    if line:
        lines.append(line)
    return "\n".join(lines)


def covered_intervals(moments):
    intervals = sorted(
        (max(0.0, m["anchor_sec"] + m["clip_start_offset_sec"]), m["anchor_sec"] + m["clip_end_offset_sec"])
        for m in moments
    )
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def find_gaps(covered, duration_sec, min_gap_sec):
    gaps = []
    cursor = 0.0
    for start, end in covered:
        if start - cursor >= min_gap_sec:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration_sec is not None and duration_sec - cursor >= min_gap_sec:
        gaps.append((cursor, duration_sec))
    return gaps


def clean_words(words):
    confident = [w for w in words if w["probability"] >= CONNECTIVE_WORD_CONFIDENCE_FLOOR]
    result = []
    run_token = None
    run = 0
    for w in confident:
        token = w["text"].strip(".,!?")
        if token and token == run_token:
            run += 1
        else:
            run_token = token
            run = 1
        if run <= CONNECTIVE_HALLUCINATION_RUN_LIMIT:
            result.append(w)
    return result


def slice_words_by_gaps(gaps, words):
    sliced = []
    i = 0
    n = len(words)
    for start, end in gaps:
        while i < n and words[i]["start"] < start:
            i += 1
        j = i
        while j < n and words[j]["start"] < end:
            j += 1
        sliced.append((start, end, words[i:j]))
        i = j
    return sliced


def qualifying_gap_blocks(gaps, words):
    blocks = []
    for start, end, gap_words in slice_words_by_gaps(gaps, words):
        if len(gap_words) < CONNECTIVE_MIN_GAP_WORDS:
            continue
        wpm = len(gap_words) / max((end - start) / 60.0, 1e-6)
        if wpm < CONNECTIVE_MIN_WPM:
            continue
        text = f"--- GAP {format_hms(start)}-{format_hms(end)} ---\n{words_to_transcript_text(gap_words)}"
        blocks.append((len(gap_words), text))
    return blocks


def chunk_blocks(blocks, max_words):
    chunks = []
    current = []
    count = 0
    for word_count, text in blocks:
        if current and count + word_count > max_words:
            chunks.append(current)
            current = []
            count = 0
        current.append(text)
        count += word_count
    if current:
        chunks.append(current)
    return chunks


def classify_connective_chunk(client, stretches):
    prompt = CONNECTIVE_PROMPT.format(stretches=stretches)
    response = client.messages.create(
        model=CONNECTIVE_MODEL,
        max_tokens=CONNECTIVE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = get_text_block(response)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    segments = []
    for item in items:
        try:
            start_sec = round(float(parse_hms(item["start"])), 2)
            end_sec = round(float(parse_hms(item["end"])), 2)
        except (KeyError, ValueError):
            continue
        if end_sec <= start_sec:
            continue
        segments.append({
            "role": "connective",
            "start_sec": start_sec,
            "end_sec": end_sec,
            "reason": item.get("reason", ""),
            "source_moment_anchor_sec": None,
            "boss_or_area": item.get("boss_or_area"),
        })
    return segments


def find_connective_segments(client, gaps, words):
    blocks = qualifying_gap_blocks(gaps, clean_words(words))
    if not blocks:
        return []
    segments = []
    for chunk in chunk_blocks(blocks, CONNECTIVE_CHUNK_WORDS):
        segments.extend(classify_connective_chunk(client, "\n\n".join(chunk)))
    return segments


def moments_to_prompt_text(moments):
    lines = []
    for i, m in enumerate(moments):
        flags = [tag for tag in FLAG_FIELDS if m.get(tag)]
        lines.append(
            f"[{i}] {format_hms(m['anchor_sec'])} categories={m.get('categories', [])} "
            f"shareability={m.get('shareability_score')} hype={m.get('hype_score')} "
            f"flags={flags or 'none'}\n    reason: {m.get('reason', '')}"
        )
    return "\n".join(lines)


def group_into_arc(client, moments):
    prompt = ARC_PROMPT.format(moments=moments_to_prompt_text(moments))
    response = client.messages.create(
        model=ARC_MODEL,
        max_tokens=ARC_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = get_text_block(response)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return data.get("segments", [])


def resolve_segments(raw_segments, moments):
    resolved = []
    for seg in raw_segments:
        idx = seg.get("source_moment_index")
        if not isinstance(idx, int) or not (0 <= idx < len(moments)):
            continue
        role = seg.get("role")
        if role not in ARC_ROLES:
            continue
        m = moments[idx]
        resolved.append({
            "role": role,
            "start_sec": round(max(0.0, m["anchor_sec"] + m["clip_start_offset_sec"]), 2),
            "end_sec": round(m["anchor_sec"] + m["clip_end_offset_sec"], 2),
            "reason": seg.get("reason", ""),
            "source_moment_anchor_sec": m["anchor_sec"],
            "boss_or_area": seg.get("boss_or_area"),
        })
    resolved.sort(key=lambda s: s["start_sec"])
    return resolved


def build_storyline(cache_path, include_connective=False, game=DEFAULT_GAME):
    moments = load_moments_cache(cache_path)

    if not moments and not include_connective:
        return {"segments": []}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("No ANTHROPIC_API_KEY set.")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    segments = []
    if moments:
        raw_segments = group_into_arc(client, moments)
        segments.extend(resolve_segments(raw_segments, moments))

    if include_connective:
        raw_cache_path = find_raw_cache_path(cache_path)
        if raw_cache_path is None:
            raise SystemExit(f"--fill-connective needs a matching *.raw.*.json cache next to {cache_path}, none found.")
        duration_sec, words = load_raw_cache(raw_cache_path)
        gaps = find_gaps(covered_intervals(moments), duration_sec, CONNECTIVE_MIN_GAP_SEC)
        segments.extend(find_connective_segments(client, gaps, words))

    segments.sort(key=lambda s: s["start_sec"])
    normalize_segments(segments, load_boss_lookup(game))
    return {"segments": segments}


def main():
    parser = argparse.ArgumentParser(description="Group cached LLM moments into a long-form story arc.")
    parser.add_argument("cache_file", help="Path to an engine/cache/*.llm.v2-two-tier.json file")
    parser.add_argument("--out", default="storyline.json")
    parser.add_argument(
        "--fill-connective", action="store_true",
        help="Also run a low-threshold pass over transcript gaps the shorts pipeline skipped (needs a matching raw cache)",
    )
    parser.add_argument(
        "--game", choices=["ds1", "ds2"], default=DEFAULT_GAME,
        help="Which boss_data/<game>_bosses.json to normalize boss_or_area guesses against",
    )
    args = parser.parse_args()

    storyline = build_storyline(args.cache_file, include_connective=args.fill_connective, game=args.game)
    with open(args.out, "w") as f:
        json.dump(storyline, f, indent=2)
    print(f"Done. {len(storyline['segments'])} segments written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
