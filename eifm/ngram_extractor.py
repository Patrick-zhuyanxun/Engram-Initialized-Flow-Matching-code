"""
Verb-phrase N-gram extraction for EI-FM Static Engram keys.

Uses spaCy POS tagging to identify motor primitives ("pick up", "push", "open")
from natural-language robot instructions.
"""

from __future__ import annotations

import spacy


def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.Language:
    """Load a spaCy model (cached after first call)."""
    return spacy.load(model_name)


def extract_verb_ngrams(instruction: str, nlp: spacy.Language) -> list[str]:
    """
    Extract verb-phrase N-grams from a language instruction.

    Algorithm:
    1. Parse instruction with spaCy.
    2. For each VERB token, collect the verb itself plus any immediately
       following particles (PART / ADP that form phrasal verbs like "pick up",
       "turn on", "put down").
    3. Return a list of lowercased verb-phrase strings.

    Examples
    --------
    >>> nlp = load_spacy_model()
    >>> extract_verb_ngrams("pick up the black bowl and place it on the plate", nlp)
    ['pick up', 'place']
    >>> extract_verb_ngrams("open the top drawer", nlp)
    ['open']
    >>> extract_verb_ngrams("turn on the stove and put the frying pan on it", nlp)
    ['turn on', 'put']
    """
    doc = nlp(instruction.lower().strip())
    ngrams: list[str] = []

    i = 0
    tokens = list(doc)
    while i < len(tokens):
        tok = tokens[i]
        if tok.pos_ == "VERB":
            phrase_parts = [tok.text]
            # Collect following particles / adverbs that form phrasal verbs
            j = i + 1
            while j < len(tokens):
                next_tok = tokens[j]
                # Phrasal verb particles: "up", "on", "off", "down", "out", "in", "away"
                KNOWN_PARTICLES = {"up", "down", "on", "off", "out", "in", "away", "over"}
                is_particle_by_dep = next_tok.dep_ in ("prt", "compound", "advmod")
                is_known_particle = (
                    next_tok.text in KNOWN_PARTICLES
                    and next_tok.pos_ in ("ADP", "PART", "ADV")
                    and next_tok.head == tok  # must be dependent on the verb
                )
                if is_particle_by_dep or is_known_particle:
                    phrase_parts.append(next_tok.text)
                    j += 1
                else:
                    break
            ngram = " ".join(phrase_parts)
            ngrams.append(ngram)
            i = j
        else:
            i += 1

    return ngrams


def compute_engram_key(ngrams: list[str], mode: str = "multi") -> str | None:
    """
    Compute a deterministic engram hash key from extracted N-grams.

    Parameters
    ----------
    ngrams : list[str]
        Verb-phrase N-grams, e.g. ["pick up", "place"].
    mode : str
        "single" — use only the first verb phrase.
        "multi"  — sort and join all verb phrases with "_".

    Returns
    -------
    str or None
        Key string like "pick_up", "pick_up_place", "open".
        None if ngrams is empty.
    """
    if not ngrams:
        return None

    if mode == "single":
        selected = [ngrams[0]]
    else:  # multi
        selected = sorted(set(ngrams))

    # Normalize: spaces within each phrase → "_", then join phrases with "_"
    parts = [phrase.replace(" ", "_") for phrase in selected]
    return "_".join(parts)


# ─────────────────────────────────────────────────────────────
# CLI: test on all LIBERO instructions
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import os
    from collections import Counter

    from config import LIBERO_DATA_DIR, LIBERO_SUITES, ENGRAM_KEY_MODE

    nlp = load_spacy_model()

    all_keys: list[str] = []
    instruction_key_map: dict[str, str | None] = {}
    miss_count = 0
    total_count = 0

    for suite in LIBERO_SUITES:
        suite_dir = LIBERO_DATA_DIR / suite
        if not suite_dir.exists():
            print(f"  [SKIP] {suite_dir} not found")
            continue

        hdf5_files = sorted(suite_dir.glob("*.hdf5"))
        for fpath in hdf5_files:
            import h5py

            with h5py.File(fpath, "r") as f:
                problem_info = json.loads(f["data"].attrs["problem_info"])
                instruction = problem_info["language_instruction"]

            if instruction in instruction_key_map:
                continue  # already processed this instruction

            total_count += 1
            ngrams = extract_verb_ngrams(instruction, nlp)
            key = compute_engram_key(ngrams, mode=ENGRAM_KEY_MODE)

            instruction_key_map[instruction] = key
            if key is None:
                miss_count += 1
                print(f"  [MISS] {instruction!r}  →  ngrams={ngrams}")
            else:
                all_keys.append(key)
                print(f"  [OK]   {instruction!r}  →  ngrams={ngrams}  →  key={key!r}")

    print("\n" + "=" * 70)
    print(f"Total unique instructions: {total_count}")
    print(f"Cache misses (no verb found): {miss_count}")
    print(f"Cache hit rate: {(total_count - miss_count) / total_count * 100:.1f}%")
    print(f"\nUnique engram keys ({len(set(all_keys))}):")
    for key, count in sorted(Counter(all_keys).items(), key=lambda x: -x[1]):
        print(f"  {key:30s}  ({count} instructions)")

    # Save instruction→key mapping for later use
    import torch

    out_path = LIBERO_DATA_DIR.parent.parent.parent / "eifm" / "instruction_key_map.pt"
    torch.save(instruction_key_map, out_path)
    print(f"\nSaved instruction→key mapping to {out_path}")
