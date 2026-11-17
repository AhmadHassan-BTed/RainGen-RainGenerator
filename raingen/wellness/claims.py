"""Claims linter for presets, docs and store copy (MASTER_PLAN 8.4).

FDA General Wellness guidance (6 Jan 2026) allows low-risk wellness language:
"relax", "unwind", "manage stress", "sleep routine", "mask distractions",
"focus". It does not allow disease claims ("treats insomnia / anxiety / ADHD /
tinnitus / depression"), physiological-outcome claims ("increases deep sleep",
"improves memory") or medical framing ("clinical", "therapy", "prescribed").
Tinnitus maskers are regulated Class II devices, and there must be no infant or
toddler sleep claims (Basner et al. 2026). CI fails on any 'error' finding.

A line containing ``claims: ignore`` is skipped (for documents that discuss
the rules themselves, such as this one).
"""

from __future__ import annotations

import re
import sys
from typing import Iterable, Optional

ALLOWED_TERMS = (
    "relax", "unwind", "wind down", "manage stress", "sleep routine", "bedtime routine", "mask distractions",
    "masking", "focus", "comfort", "cozy", "calm", "ambience", "soothing", "peaceful", "rest", "background sound",
)

_DISEASES = (
    r"insomnia", r"sleep\s+disorders?", r"sleep\s+apn(?:o)?ea", r"anxiety(?:\s+disorders?)?", r"panic\s+attacks?",
    r"adhd", r"attention[- ]deficit", r"tinnitus", r"depression", r"depressive", r"ptsd", r"trauma",
    r"autism", r"dementia", r"alzheimer'?s", r"migraines?", r"hypertension", r"high\s+blood\s+pressure",
    r"chronic\s+pain", r"ocd", r"bipolar", r"stress\s+disorder", r"burnout\s+syndrome",
)
_DISEASE_RE = r"(?:" + "|".join(_DISEASES) + r")"
# Conditions named by the accessibility hearing profiles ("Misophonia-safe", "Sensitive hearing"):
# allowed as a neutral profile name, an error when paired with a treatment verb.
_SENSITIVE = (r"misophonia", r"misophonic", r"hyperacusis")
_SENSITIVE_RE = r"(?:" + "|".join(_SENSITIVE) + r")"
_ANY_CONDITION_RE = r"(?:" + "|".join(_DISEASES + _SENSITIVE) + r")"
_TREAT_VERBS = (r"treat(?:s|ing|ment|ments)?|cure(?:s|d)?|curing|heal(?:s|ing)?|fix(?:es)?|eliminat(?:e|es|ing)|"
                r"reliev(?:e|es|ing)|relief\s+(?:for|from)|remed(?:y|ies)|prevent(?:s|ing)?|reduc(?:e|es|ing)|"
                r"alleviat(?:e|es|ing)|combat(?:s|ing)?|fight(?:s|ing)?|manag(?:e|es|ing)|therapy\s+for|beat(?:s|ing)?|"
                r"overcome|stop(?:s|ping)?|end(?:s|ing)?|help(?:s|ing)?\s+(?:with|against)")

RULES: list[dict] = [
    {"id": "disease-treatment", "severity": "error",
     "pattern": r"\b(?:" + _TREAT_VERBS + r")\b[^.!?\n]{0,40}?\b" + _ANY_CONDITION_RE + r"\b",
     "message": "Disease or disorder treatment claim (not allowed under FDA General Wellness guidance).",
     "suggestion": "Use wellness language: 'helps you relax', 'part of a sleep routine', 'masks distractions'."},
    {"id": "disease-for", "severity": "error",
     "pattern": r"\b(?:(?:against|suffer(?:ing|s)?\s+from|people\s+with|patients\s+with)\s+(?:your\s+|their\s+|my\s+)?"
                + _ANY_CONDITION_RE + r"|(?:for|with)\s+(?:your\s+|their\s+|my\s+)?" + _DISEASE_RE + r")\b",
     "message": "Positions the product for a disease or disorder.",
     "suggestion": "Describe the experience instead: 'for winding down', 'for focus', 'to mask background noise'."},
    {"id": "disease-word", "severity": "error",
     "pattern": r"\b" + _DISEASE_RE + r"\b",
     "message": "Disease or disorder word in product copy (CI rejects disease words in presets, docs and store copy).",
     "suggestion": "Remove the condition name; keep to relaxation, routine, comfort and masking language."},
    {"id": "condition-word", "severity": "warning",
     "pattern": r"\b" + _SENSITIVE_RE + r"\b",
     "message": "Condition name used as a profile label; keep it neutral (a listening preference, never a remedy).",
     "suggestion": "Prefer 'Gentle on sensitive ears' or 'No tapping or clicking sounds'."},
    {"id": "tinnitus-masker", "severity": "error",
     "pattern": r"\btinnitus\s+(?:masker|masking|relief|sound|therapy|retraining)\b|\bmask(?:s|ing)?\s+(?:your\s+)?tinnitus\b",
     "message": "Tinnitus maskers are regulated Class II medical devices.",
     "suggestion": "Say 'masks distracting background noise'."},
    {"id": "deep-sleep", "severity": "error",
     "pattern": r"\b(?:deep[- ]sleep|slow[- ]wave(?:\s+sleep)?|sws|rem(?:\s+sleep)?|sleep\s+quality|sleep\s+architecture)\s*"
                r"(?:boost(?:s|er|ing)?|enhanc\w*|increas\w*|improv\w*|more|longer|better|deeper)\b|"
                r"\b(?:boost(?:s|ing)?|enhanc\w*|increas\w*|improv\w*|more|longer|deepen\w*|extend\w*|maximi[sz]\w*)\s+"
                r"(?:your\s+)?(?:deep[- ]sleep|slow[- ]wave(?:\s+sleep)?|rem(?:\s+sleep)?|sleep\s+quality|sleep\s+cycles?)\b",
     "message": "Claims to change sleep physiology (only shown with EEG-triggered stimulation, never for continuous sound).",
     "suggestion": "Say 'a calm sound for your bedtime routine'."},
    {"id": "cognition", "severity": "error",
     "pattern": r"\b(?:improv\w*|boost\w*|enhanc\w*|increas\w*|sharpen\w*)\s+(?:your\s+)?(?:memory|cognition|cognitive\s+\w+|iq|brain\s*power|"
                r"concentration\s+span|productivity\s+by\s+\d+|learning)\b|\bmemory\s+boost\w*\b|\bcognitive\s+enhancement\b|"
                r"\bbrain\s*waves?\b|\bbinaural\s+beats?\s+(?:for|to)\b|\bentrain\w*\b",
     "message": "Cognitive or brain-state claim.",
     "suggestion": "Say 'masks distractions so you can focus'."},
    {"id": "medical-framing", "severity": "error",
     "pattern": r"\b(?:medical(?:ly)?|clinical(?:ly)?(?:\s+(?:proven|tested|validated))?|therapeutic|therap(?:y|ies)|"
                r"prescri(?:bed|ption|be)|diagnos\w*|doctor[- ]?(?:recommended|approved)|fda[- ](?:approved|cleared)|"
                r"medicine|cure|symptoms?|patients?|disorders?|treatment)\b",
     "message": "Medical framing is not allowed in wellness copy.",
     "suggestion": "Use everyday wellness language."},
    {"id": "infant-sleep", "severity": "error",
     "pattern": r"\b(?:babies|baby|infants?|newborns?|toddlers?|nursery|crib|cot)\b[^.!?\n]{0,30}\bsleep\w*\b|"
                r"\bsleep\w*\b[^.!?\n]{0,30}\b(?:babies|baby|infants?|newborns?|toddlers?)\b",
     "message": "No infant or toddler sleep presets or claims (Basner et al. 2026 caution).",
     "suggestion": "Remove the infant sleep reference."},
    {"id": "physiology", "severity": "error",
     "pattern": r"\b(?:lowers?|reduc\w*|decreas\w*)\s+(?:your\s+)?(?:blood\s+pressure|heart\s+rate|cortisol|stress\s+hormones?)\b",
     "message": "Physiological outcome claim.",
     "suggestion": "Say 'helps you relax' or 'helps you manage stress'."},
    {"id": "absolute-claim", "severity": "warning",
     "pattern": r"\b(?:guaranteed?|scientifically\s+proven|proven\s+to|100\s*%\s+(?:effective|safe)|instantly\s+(?:fall\s+asleep|calm)|"
                r"works\s+for\s+everyone|miracle)\b",
     "message": "Absolute or efficacy claim; the evidence for sound and sleep is very low (GRADE).",
     "suggestion": "Soften: 'many people find it relaxing'."},
    {"id": "safe-volume", "severity": "warning",
     "pattern": r"\b(?:safe|harmless)\s+(?:all\s+night|at\s+any\s+volume|for\s+hours)\b|\ball[- ]night\s+(?:safe|harmless)\b",
     "message": "All-night or any-volume safety claim (50 dBA pink noise reduced REM in Basner et al. 2026).",
     "suggestion": "Recommend low levels and the sleep timer."},
    {"id": "stress-reduction", "severity": "warning",
     "pattern": r"\b(?:reduc\w*|lower\w*|eliminat\w*|cur\w*)\s+(?:your\s+)?stress\b|\bstress[- ]free\b",
     "message": "Outcome wording on stress; the allowed wellness phrasing is 'relax' / 'manage stress'.",
     "suggestion": "Say 'helps you relax' or 'helps you manage stress'."},
]

_COMPILED = [(r, re.compile(r["pattern"], re.IGNORECASE)) for r in RULES]
_IGNORE = re.compile(r"claims:\s*ignore", re.IGNORECASE)
_SPECIFIC = ("disease-treatment", "disease-for", "tinnitus-masker")


def lint_text(text: str, rules: Optional[Iterable[str]] = None) -> list[dict]:
    """Findings for `text`: dicts with rule, severity, match, start, end, line, column, message, suggestion.

    A generic 'disease-word' finding is dropped where a more specific rule already covers
    the same span."""
    wanted = set(rules) if rules is not None else None
    findings: list[dict] = []
    line_starts = [0]
    for m in re.finditer(r"\n", text):
        line_starts.append(m.end())
    ignored_lines = set()
    for i, start in enumerate(line_starts):
        end = line_starts[i + 1] if i + 1 < len(line_starts) else len(text)
        if _IGNORE.search(text[start:end]):
            ignored_lines.add(i + 1)

    def _line_col(pos: int) -> tuple[int, int]:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1, pos - line_starts[lo] + 1

    for rule, rx in _COMPILED:
        if wanted is not None and rule["id"] not in wanted:
            continue
        for m in rx.finditer(text):
            line, col = _line_col(m.start())
            if line in ignored_lines:
                continue
            findings.append({"rule": rule["id"], "severity": rule["severity"], "match": m.group(0),
                             "start": m.start(), "end": m.end(), "line": line, "column": col,
                             "message": rule["message"], "suggestion": rule["suggestion"]})
    specific = [(f["start"], f["end"]) for f in findings if f["rule"] in _SPECIFIC]
    out = []
    for f in findings:
        if f["rule"] in ("disease-word", "condition-word", "medical-framing") and any(s <= f["start"] and f["end"] <= e for s, e in specific):
            continue
        out.append(f)
    out.sort(key=lambda f: (f["start"], f["rule"]))
    return out


def wellness_terms(text: str) -> list[str]:
    """Allowed wellness vocabulary found in `text` (for the 'what this copy promises' summary)."""
    low = text.lower()
    return [t for t in ALLOWED_TERMS if re.search(r"\b" + re.escape(t) + r"\b", low)]


def is_clean(text: str) -> bool:
    return not any(f["severity"] == "error" for f in lint_text(text))


def lint_presets() -> list[dict]:
    """Lint every preset title and description."""
    from ..scene.presets import PRESETS
    out = []
    for name, spec in PRESETS.items():
        for field in ("title", "description"):
            for f in lint_text(spec[field]):
                f["preset"] = name
                f["field"] = field
                out.append(f)
    return out


def lint_files(paths: Iterable[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            found = lint_text(fh.read())
        if found:
            out[path] = found
    return out


def main(argv: Optional[list[str]] = None) -> int:
    """CI entry point: python -m raingen.wellness.claims [files...]; exits 1 on any error."""
    argv = list(sys.argv[1:] if argv is None else argv)
    errors = 0
    results = lint_files(argv) if argv else {}
    preset_findings = lint_presets()
    if preset_findings:
        results["<presets>"] = preset_findings
    for path, found in results.items():
        for f in found:
            where = f"{path}:{f.get('line', 0)}:{f.get('column', 0)}"
            if "preset" in f:
                where = f"preset {f['preset']}.{f['field']}"
            print(f"{where}: {f['severity']}: [{f['rule']}] {f['match']!r}: {f['message']} Suggestion: {f['suggestion']}")
            errors += f["severity"] == "error"
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
