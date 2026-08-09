import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

PIPELINE_VERSION = "v2-two-tier"

SWEEP_MODEL = "claude-haiku-4-5-20251001"
SWEEP_MAX_TOKENS = 16384
SWEEP_MERGE_GAP_SEC = 60.0

CLASSIFY_MODEL = "claude-sonnet-5"
CLASSIFY_MAX_TOKENS = 4096
CLASSIFY_CONCURRENCY = 5
CONTEXT_BEFORE_SEC = 60.0
CONTEXT_AFTER_SEC = 20.0

CATEGORIES = [
    "skill_highlight", "death_comedy", "banter_wit",
    "innuendo_character_bit", "wholesome_genuine", "chaos_unscripted", "cute_charming",
]
CONTROVERSY_LEVELS = ["none", "mild", "edgy", "high"]
MODERATION_RISKS = ["none", "slur_or_flaggable_language", "sexual_content", "other"]

PERSONA_BRIEF = """Dry, melancholic default delivery. Dark/edgy humor, doesn't self-censor. Jokes often \
land BECAUSE of flat/deadpan delivery, not despite it — there's usually no vocal tone-shift signaling \
"this is a joke," so don't require one to flag something as a bit.

Two recurring comedic patterns:
- Panic-spiral (helpless direction): repetition-as-comedy under pressure, e.g. rapid "no no no no" or \
"let me flask" when something goes wrong.
- Panic-spiral (feral direction): same trigger (dying/losing progress), but flips into mock-aggressive \
trash talk at the boss/game instead of helplessness.
- She falls a lot and has a specific "nononono" scream pattern tied specifically to falling — this is a \
recognizable callback bit for regulars, not just generic panic. Worth tagging when it's clearly the \
falling-specific version.

Also actively look for cute_charming moments — small, low-key, endearing shenanigans that make her come \
across as cute/likeable, not just funny or hype. These can be quiet: a silly voice, a little quirk, a soft \
aside, small dorky enthusiasm. Don't skip these just because they're not loud or don't have a punchline — \
this category isn't about being impressive, it's about being charming in a small way.

Stick close to what's literally in the transcript for your reasoning — don't invent surrounding narrative \
or causality you can't actually verify from the text and context given. If you're inferring what triggered \
something rather than reading it directly, that's a sign confidence should be low, not a place to guess \
confidently.

Known transcription noise, two distinct patterns — both garbage, don't classify either:
- Tight loops: the same single word repeated 6+ times back-to-back with no variation (e.g. "no no no no \
no no no no"). A genuine repeated-phrase callback (words vary slightly, phrase carries meaning each \
repetition) is NOT this — treat that as real.
- Sparse dead-air noise: isolated single words (often "you") scattered ~20-30+ seconds apart over what is \
actually silence or near-silence — e.g. pre-stream waiting room, AFK/bathroom breaks, quiet exploration. \
Individually each word looks like normal transcription, but across a stretch of several minutes it's \
periodic noise, not speech. If a segment is mostly gaps with a lone word every 20-30s and no real sentence \
forms, do not invent a narrative around it (this has previously caused false positives like reading \
"self-censoring" into what was actually just hallucinated noise before the stream had even started)."""

FEW_SHOT_EXAMPLES = [
    {
        "label": "panic-spiral, helpless direction",
        "transcript": (
            "[00:16:24] What the hell is that?\n"
            "[00:16:26] What is that? What is that?\n"
            "[00:16:27] No, no, no, no, no!\n"
            "[00:16:29] What, what, what?\n"
            "[00:16:37] What?\n"
            "[00:16:41] Oh no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no!"
        ),
        "classification": {
            "categories": ["chaos_unscripted"],
            "shareability_score": 70,
            "hype_score": 90,
            "controversy_level": "none",
            "moderation_risk": "none",
            "has_setup_punchline_shape": False,
            "confidence": "high",
            "uncertainty_reason": "",
            "anchor_sec": 976,
            "clip_start_offset_sec": -12.0,
            "clip_end_offset_sec": 8.0,
            "reason": "Genuine unscripted panic-spiral, helpless direction — real escalating dread, not a bit but authentic chaos.",
            "sub_beats": [],
            "is_nonono_or_death_beat": True,
            "is_zweihander_beat": False,
            "is_boss_rage_cussing_beat": False,
            "is_boss_victory_beat": False,
        },
    },
    {
        "label": "panic-spiral, feral direction",
        "transcript": (
            "[00:27:41] i don't want to beat you\n"
            "[00:27:47] bro can you just stay in one damn place\n"
            "[00:27:50] quickly\n"
            "[00:27:55] you bastard you bat\n"
            "[00:27:58] you bastard you bath-\n"
            "[00:27:59] you bastard"
        ),
        "classification": {
            "categories": ["banter_wit", "chaos_unscripted"],
            "shareability_score": 65,
            "hype_score": 60,
            "controversy_level": "none",
            "moderation_risk": "none",
            "has_setup_punchline_shape": True,
            "confidence": "high",
            "uncertainty_reason": "",
            "anchor_sec": 1667,
            "clip_start_offset_sec": -10.0,
            "clip_end_offset_sec": 5.0,
            "reason": "Same panic trigger as the helpless pattern, but flips feral — mock-aggressive trash talk at the boss instead of helplessness, funny escalation ending on a stammered insult.",
            "sub_beats": [],
            "is_nonono_or_death_beat": False,
            "is_zweihander_beat": False,
            "is_boss_rage_cussing_beat": True,
            "is_boss_victory_beat": False,
        },
    },
    {
        "label": "chat-adjacent repeated-wordplay bit (positive)",
        "transcript": (
            "[02:21:14] a one sip and i go down what's the next song match into water nah nope nope ah i i love this song i love this song\n"
            "[02:21:34] i love this song i love sex it's a song name by the way i love sex it's a song name by the way i love\n"
            "[02:21:48] sex it's a\n"
            "[02:21:52] song name by the way\n"
            "[02:22:17] i love sex it's a song name by the way i love sex it's a song name by the way i love sex it's a song name by the way i love sex it's a song name by the way"
        ),
        "classification": {
            "categories": ["innuendo_character_bit", "banter_wit"],
            "shareability_score": 75,
            "hype_score": 30,
            "controversy_level": "mild",
            "moderation_risk": "none",
            "has_setup_punchline_shape": True,
            "confidence": "high",
            "uncertainty_reason": "",
            "anchor_sec": 8494,
            "clip_start_offset_sec": -5.0,
            "clip_end_offset_sec": 25.0,
            "reason": "Realizes a song is literally titled \"Sex,\" turns it into a repeated bit riding the double meaning — the repetition here is a deliberate callback, not transcription noise, because the phrase stays coherent and meaningful each time.",
            "sub_beats": [],
            "is_nonono_or_death_beat": False,
            "is_zweihander_beat": False,
            "is_boss_rage_cussing_beat": False,
            "is_boss_victory_beat": False,
        },
    },
    {
        "label": "routine verbal tic that superficially resembles hype (negative)",
        "transcript": (
            "[01:57:37] alright.\n"
            "[01:57:40] okay.\n"
            "[01:57:43] zwei. let's go, my zwei. let's go, my zwei. let's go. let's go, zwei. we can do this.\n"
            "[01:57:55] we can do this, zwei."
        ),
        "classification": None,
        "classification_note": "Not clip-worthy. This exact phrasing (\"let's go, my zwei\") is a habitual pre-attempt ritual said the same way most attempts — despite matching hype phrasing, there's no genuine excitement here. Do not flag routine tics like this just because they use hype-coded words. (This is a confident NO, not a low-confidence case — she says this the same way nearly every attempt, there's nothing ambiguous about it.)",
    },
    {
        "label": "sparse dead-air hallucination noise (negative)",
        "transcript": (
            "[00:00:00] you\n"
            "[00:00:30] you\n"
            "[00:01:00] you\n"
            "[00:01:30] you\n"
            "[00:02:00] you\n"
            "[00:02:30] you\n"
            "[00:03:00] you\n"
            "[00:03:30] you\n"
            "[00:04:21] oh oh shit wait um i fucked something up"
        ),
        "classification": None,
        "classification_note": "Not clip-worthy, and not real speech. This is pre-stream dead air — a single hallucinated word (\"you\") every ~30 seconds is faster-whisper noise, not someone actually talking. Don't invent a narrative (e.g. \"self-censoring,\" \"trailing off\") to explain sparse isolated words like this. The real content only starts at 00:04:21 once actual sentences appear.",
    },
    {
        "label": "genuinely ambiguous — low confidence example",
        "transcript": (
            "[02:23:35] wait, i just had this crazy thought, like, sometimes i think like, if, if, if some unfortunate things happen and i get assaulted and the dude like forces me on his knees or something, i would, i would just cut, like, i would just cut him in the balls. just bite on his balls, like, hard as i can and run off while he's clutching his balls in pain.\n"
            "[02:24:07] doesn't help that i'm also a dragon.\n"
            "[02:24:12] that got dark.\n"
            "[02:24:15] i mean, come on.\n"
            "[02:24:20] i mean, it's not completely impossible. that would actually work. yeah. cause like, i don't know. i would imagine, uh, women\n"
            "[02:24:33] might be like, too disgusted to bite on, oh fuck, to\n"
            "[02:24:40] bite on the balls, but then it's like probably men's, um, weakest, like weak points, like the biggest weak points. so yeah. oh fuck. shut the fuck up. shut the fuck up, laura. shut the fuck up, laura. what the fuck are you talking about?"
        ),
        "classification": {
            "categories": ["banter_wit", "chaos_unscripted"],
            "shareability_score": 55,
            "hype_score": 20,
            "controversy_level": "edgy",
            "moderation_risk": "sexual_content",
            "has_setup_punchline_shape": True,
            "confidence": "low",
            "uncertainty_reason": "The dark hypothetical (sexual assault as setup for a self-defense bit) is delivered in her usual dry style and she self-flags it (\"that got dark\") the same way she does with other bits, which reads as comedic self-awareness — but the subject matter is heavy enough that whether this actually lands as funny vs. just uncomfortable is a genuine taste call that depends on delivery/tone I can't fully judge from text alone. Best guess given below, but this one's really hers to decide.",
            "anchor_sec": 8632,
            "clip_start_offset_sec": -8.0,
            "clip_end_offset_sec": 25.0,
            "reason": "Absurd dark-humor tangent with a self-aware \"that got dark\" beat and an escalating self-own ending (\"shut the fuck up, laura\") — has real comedic shape, but the setup is heavy enough that confidence is low rather than a clean flag.",
            "sub_beats": [],
            "is_nonono_or_death_beat": False,
            "is_zweihander_beat": False,
            "is_boss_rage_cussing_beat": False,
            "is_boss_victory_beat": False,
        },
    },
]

SWEEP_PROMPT = """Below is the full transcript of a Twitch VOD (Dark Souls playthrough). Each line is \
timestamped hh:mm:ss (video-time from VOD start).

Find every stretch of transcript that contains a joke, comedic bit, innuendo, dark humor, or a notable \
tonal shift. This is a coarse first-pass filter feeding a more expensive review step — bias hard toward \
recall. If you're at all unsure whether something counts, include it.

For each stretch, report its actual start and end time as you see it in the transcript — the real \
boundaries of the moment itself, not a fixed window, however long or short that turns out to be.

Respond with ONLY a JSON array of {{"start": "hh:mm:ss", "end": "hh:mm:ss"}} objects, sorted by start. \
Empty array if nothing qualifies.

Transcript:
{transcript}"""

CLASSIFY_PROMPT = """You're the expensive, careful review step in a two-tier clip-detection pipeline for \
a Dark Souls Twitch VOD. A cheap first-pass model already flagged the segment below as possibly containing \
something clip-worthy — your job is to actually judge it with real context.

Streamer persona:
{persona}

Examples of how to classify (few-shot calibration):
{few_shot}

Now classify this segment. You're given context before and after the flagged window — use it, don't \
just look at the middle. Timestamps in brackets are absolute video-time in hh:mm:ss.

--- CONTEXT BEFORE ---
{context_before}
--- FLAGGED SEGMENT ---
{segment}
--- CONTEXT AFTER ---
{context_after}

Find zero or more genuinely independently-clippable moments in the FLAGGED SEGMENT (context is for \
understanding only, don't report moments that are only in the context). Do not force a quota — if nothing \
here is actually clip-worthy despite the sweep flag, return an empty array.

She edits clips herself and would rather trim excess than have something feel cut off, so err toward more \
generous clip_start_offset_sec/clip_end_offset_sec rather than cutting tight — better to include a few \
extra seconds of setup or a follow-up reaction than to clip it right at the punchline.

For each moment, respond with an object in this exact shape:
{{
  "categories": [...],  // subset of {categories}
  "shareability_score": 0-100,
  "hype_score": 0-100,
  "controversy_level": "none" | "mild" | "edgy" | "high",
  "moderation_risk": "none" | "slur_or_flaggable_language" | "sexual_content" | "other",
  "has_setup_punchline_shape": true | false,
  "confidence": "high" | "low",
  "uncertainty_reason": "why this is a hard call, or empty string if confidence is high",
  "anchor_sec": <absolute video-time float of the SPECIFIC line that makes this clip-worthy — copy the exact \
timestamp of that line from the transcript, don't estimate a rough midpoint of the broader moment. If a \
setup and its punchline are both essential, anchor on the punchline and let clip_start_offset_sec reach \
back to cover the setup>,
  "clip_start_offset_sec": <float, negative, seconds before anchor_sec the clip should start>,
  "clip_end_offset_sec": <float, positive, seconds after anchor_sec the clip should end>,
  "reason": "one line why this is clip-worthy",
  "sub_beats": [{{"offset_sec": <absolute video-time float>, "note": "why this sub-moment is independently clippable"}}],
  "is_nonono_or_death_beat": true | false,
  "is_zweihander_beat": true | false,
  "is_boss_rage_cussing_beat": true | false,
  "is_boss_victory_beat": true | false
}}

is_nonono_or_death_beat feeds a separate montage of just her panic-scream/death moments strung together — set \
true if this moment IS (or centers on) a "nononono" panic-spiral scream (either direction, including the \
falling-specific callback) or an actual in-game death/character death reaction. false for everything else, \
even other panic-spiral variants that aren't the classic nononono scream (e.g. rapid-fire cursing without \
"no" isn't this).

is_boss_rage_cussing_beat feeds a separate montage of her cussing/raging directly at a boss or enemy — this \
is the feral-direction panic-spiral specifically (mock-aggressive trash talk, swearing AT the boss/enemy), \
not cussing at herself, at game mechanics in general, or in unrelated bits. If it's the helpless-direction \
panic-spiral (dread, not aggression) this is false even if there's swearing in it.

is_boss_victory_beat feeds a separate montage of the moments she actually beats a boss — set true for the \
genuine victory reaction (relief, triumph, disbelief that it's finally over) right after a boss kill, not \
mid-fight optimism or a false alarm where the boss wasn't actually dead. This is usually a real payoff — \
if you can tell from context (before/after) that a prolonged struggle with a specific boss just ended, \
anchor on the reaction line itself.

is_zweihander_beat feeds a separate montage of moments about her sword (called Zwei/Zweihander) specifically \
— set true if the moment is genuinely ABOUT the sword itself: affection/mock-romance toward it, worship-style \
bits, jokes or wordplay tied to its name, treating it as a character. false for just casually mentioning it \
in passing, and false for the routine pre-attempt "let's go, my zwei" ritual on its own (that's a habitual \
tic, not appreciation content) — unless that exact moment is already part of a larger genuine bit about the \
sword.

controversy_level is a taste/timing call (she filters this herself at review). moderation_risk is a \
platform-policy flag (slurs etc. can get clips struck regardless of context) — keep these separate, don't \
conflate them.

confidence/uncertainty_reason is a separate escape hatch from controversy_level, not limited to taste calls \
— use it for ANY hard call: ambiguous category, unclear whether it's a bit or genuine reaction, fuzzy \
setup/punchline boundary, whatever. Don't force a confident verdict just to fill the fields — if you're \
genuinely unsure, mark confidence "low" and say specifically what's ambiguous in uncertainty_reason. Still \
give your best-guess categories/scores/reason even when confidence is low — she's correcting a guess at \
review time, not classifying from scratch.

Respond with ONLY a JSON array of these objects, no other text. Empty array if nothing qualifies."""


def format_hms(seconds):
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def words_to_transcript_text(words):
    lines = []
    line = None
    last_t = None
    for w in words:
        if last_t is None or w.start - last_t > 2:
            if line:
                lines.append(line)
            line = f"[{format_hms(w.start)}] "
        line += w.text + " "
        last_t = w.start
    if line:
        lines.append(line)
    return "\n".join(lines)


def parse_hms(hms):
    h, m, s = (int(part) for part in str(hms).split(":"))
    return h * 3600 + m * 60 + s


def get_text_block(response):
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def sweep_full(client, words):
    transcript = words_to_transcript_text(words)
    prompt = SWEEP_PROMPT.format(transcript=transcript)
    response = client.messages.create(
        model=SWEEP_MODEL,
        max_tokens=SWEEP_MAX_TOKENS,
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
    ranges = []
    for item in items:
        try:
            ranges.append((parse_hms(item["start"]), parse_hms(item["end"])))
        except (KeyError, ValueError):
            continue
    return sorted(ranges)


def merge_flagged_ranges(ranges, merge_gap_sec=SWEEP_MERGE_GAP_SEC):
    if not ranges:
        return []
    merged = [list(ranges[0])]
    for start, end in ranges[1:]:
        if start <= merged[-1][1] + merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(r) for r in merged]


def build_few_shot_text():
    blocks = []
    for ex in FEW_SHOT_EXAMPLES:
        classification = ex["classification"] if ex["classification"] is not None else {"clip_worthy": False, "note": ex.get("classification_note", "")}
        blocks.append(f"[{ex['label']}]\nTranscript:\n{ex['transcript']}\nExpected classification:\n{json.dumps(classification, indent=2)}")
    return "\n\n".join(blocks)


def classify_segment(client, words, context_before_words, segment_words, context_after_words):
    prompt = CLASSIFY_PROMPT.format(
        persona=PERSONA_BRIEF,
        few_shot=build_few_shot_text(),
        categories=CATEGORIES,
        context_before=words_to_transcript_text(context_before_words) or "(none, start of VOD)",
        segment=words_to_transcript_text(segment_words),
        context_after=words_to_transcript_text(context_after_words) or "(none, end of VOD)",
    )
    response = client.messages.create(
        model=CLASSIFY_MODEL,
        max_tokens=CLASSIFY_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return get_text_block(response)


def parse_classify_response(raw_text):
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    moments = []
    for item in items:
        try:
            moments.append({
                "categories": item.get("categories", []),
                "shareability_score": item["shareability_score"],
                "hype_score": item["hype_score"],
                "controversy_level": item.get("controversy_level", "none"),
                "moderation_risk": item.get("moderation_risk", "none"),
                "has_setup_punchline_shape": item.get("has_setup_punchline_shape", False),
                "confidence": item.get("confidence", "high"),
                "uncertainty_reason": item.get("uncertainty_reason", ""),
                "anchor_sec": float(item["anchor_sec"]),
                "clip_start_offset_sec": float(item.get("clip_start_offset_sec", -5.0)),
                "clip_end_offset_sec": float(item.get("clip_end_offset_sec", 15.0)),
                "reason": item.get("reason", ""),
                "sub_beats": item.get("sub_beats", []),
                "is_nonono_or_death_beat": item.get("is_nonono_or_death_beat", False),
                "is_zweihander_beat": item.get("is_zweihander_beat", False),
                "is_boss_rage_cussing_beat": item.get("is_boss_rage_cussing_beat", False),
                "is_boss_victory_beat": item.get("is_boss_victory_beat", False),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return moments


def context_window(words, center_start, center_end, before_sec, after_sec):
    before = [w for w in words if center_start - before_sec <= w.start < center_start]
    segment = [w for w in words if center_start <= w.start < center_end]
    after = [w for w in words if center_end <= w.start < center_end + after_sec]
    return before, segment, after


def review_words(words, start_sec=0.0):
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("No ANTHROPIC_API_KEY set, skipping LLM review.", flush=True)
        return []

    scoped = [w for w in words if w.start >= start_sec]
    if not scoped:
        return []

    client = Anthropic(api_key=api_key)
    print(f"Sweeping {len(scoped)} words in a single Haiku call...", flush=True)
    try:
        raw_ranges = sweep_full(client, scoped)
    except Exception as e:
        print(f"Sweep failed, classifying the whole remaining transcript as one range (fail open): {e}", flush=True)
        raw_ranges = [(scoped[0].start, scoped[-1].start)]

    flagged_ranges = merge_flagged_ranges(raw_ranges)
    print(f"Sweep found {len(raw_ranges)} raw hits, merged into {len(flagged_ranges)} ranges to classify.", flush=True)

    def classify_range(r_start, r_end):
        before, segment, after = context_window(words, r_start, r_end, CONTEXT_BEFORE_SEC, CONTEXT_AFTER_SEC)
        raw_text = classify_segment(client, words, before, segment, after)
        return parse_classify_response(raw_text)

    moments = []
    print(f"Classifying {len(flagged_ranges)} ranges with {CLASSIFY_CONCURRENCY} concurrent workers...", flush=True)
    with ThreadPoolExecutor(max_workers=CLASSIFY_CONCURRENCY) as pool:
        future_to_range = {pool.submit(classify_range, r_start, r_end): (r_start, r_end) for r_start, r_end in flagged_ranges}
        for done, future in enumerate(as_completed(future_to_range), start=1):
            r_start, r_end = future_to_range[future]
            label = f"{format_hms(r_start)}-{format_hms(r_end)}"
            try:
                range_moments = future.result()
            except Exception as e:
                print(f"  [{done}/{len(flagged_ranges)}] {label} classify failed: {e}", flush=True)
                continue
            print(f"  [{done}/{len(flagged_ranges)}] {label}: found {len(range_moments)} moment(s).", flush=True)
            moments.extend(range_moments)

    return dedupe_moments(moments)


def dedupe_moments(moments):
    seen_times = set()
    deduped = []
    for m in sorted(moments, key=lambda m: m["anchor_sec"]):
        if any(abs(m["anchor_sec"] - t) < 10 for t in seen_times):
            continue
        seen_times.add(m["anchor_sec"])
        deduped.append(m)
    return deduped


def moments_to_candidates(moments):
    return [
        {
            "time": round(m["anchor_sec"], 2),
            "score": round(m["shareability_score"] / 100.0, 4),
            "reason": f"llm: {m['reason']}",
            "start": round(max(0.0, m["anchor_sec"] + m["clip_start_offset_sec"]), 2),
            "end": round(m["anchor_sec"] + m["clip_end_offset_sec"], 2),
            "categories": m["categories"],
            "shareability_score": m["shareability_score"],
            "hype_score": m["hype_score"],
            "controversy_level": m["controversy_level"],
            "moderation_risk": m["moderation_risk"],
            "has_setup_punchline_shape": m["has_setup_punchline_shape"],
            "confidence": m.get("confidence", "high"),
            "uncertainty_reason": m.get("uncertainty_reason", ""),
            "sub_beats": m["sub_beats"],
            "is_nonono_or_death_beat": m.get("is_nonono_or_death_beat", False),
            "is_zweihander_beat": m.get("is_zweihander_beat", False),
            "is_boss_rage_cussing_beat": m.get("is_boss_rage_cussing_beat", False),
            "is_boss_victory_beat": m.get("is_boss_victory_beat", False),
        }
        for m in moments
    ]


def llm_cache_path(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    safe_base = re.sub(r"[^\w-]", "_", base)[:80]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    return os.path.join(cache_dir, f"{safe_base}.llm.{PIPELINE_VERSION}.json")


def cache_covers_duration(cached_duration_sec, requested_duration_sec):
    if cached_duration_sec is None:
        return True
    if requested_duration_sec is None:
        return False
    return cached_duration_sec >= requested_duration_sec


def load_llm_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        return json.load(f)


def save_llm_cache(cache_path, duration_sec, moments):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"duration_sec": duration_sec, "moments": moments}, f, indent=2)


def get_llm_candidates(video_path, words, duration_sec, use_cache=True):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY set, skipping LLM review.", flush=True)
        return []

    cache_path = llm_cache_path(video_path)
    cached = load_llm_cache(cache_path) if use_cache else None
    cached_duration = cached["duration_sec"] if cached else None
    cached_moments = cached["moments"] if cached else []

    if cached is not None and cache_covers_duration(cached_duration, duration_sec):
        print(f"Using cached LLM review: {cache_path}", flush=True)
        moments = [m for m in cached_moments if duration_sec is None or m["anchor_sec"] < duration_sec]
        return moments_to_candidates(moments)

    resume_from = cached_duration if cached_duration else 0.0
    scoped_words = [w for w in words if duration_sec is None or w.start < duration_sec]
    new_moments = review_words(scoped_words, start_sec=resume_from)

    all_moments = dedupe_moments(cached_moments + new_moments)
    save_llm_cache(cache_path, duration_sec, all_moments)
    return moments_to_candidates(all_moments)
