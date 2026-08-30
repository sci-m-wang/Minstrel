from __future__ import annotations

import re

from .schema import CharacterSpec


def _patterns(spec: CharacterSpec) -> list[re.Pattern[str]]:
    values = [*spec.aliases, spec.character_name, spec.work]
    unique = sorted({value.strip() for value in values if value.strip()}, key=len, reverse=True)
    return [re.compile(re.escape(value), flags=re.IGNORECASE) for value in unique]


def anonymize_text(text: str, spec: CharacterSpec, marker: str = "[TARGET]") -> str:
    result = text
    for pattern in _patterns(spec):
        result = pattern.sub(marker, result)
    return result


def find_identity_leaks(text: str, spec: CharacterSpec) -> list[str]:
    leaks = []
    for value in [*spec.aliases, spec.character_name, spec.work]:
        if value and re.search(re.escape(value), text, flags=re.IGNORECASE):
            leaks.append(value)
    return sorted(set(leaks))

