"""
Ground truth number-plate labels for benchmarking.

This module exposes:
- PLATES: ordered list of plate strings (as labelled by a human)
- split_plate(plate): returns a list of characters for a single plate (whitespace removed)
- all_plate_chars(plates): mapping plate -> list of characters
- all_chars_flat(plates): flattened list of characters across all plates (per-tile order)
- GT_CHARS_FLAT: default flattened list for PLATES
- pretty_print(plates): prints each plate and its spaced characters
- pretty_print_flat(plates): prints the flattened characters with 1-based indices

Use this to compare your model predictions (per-tile, left-to-right)
against the known ground-truth characters.
"""

from typing import Dict, Iterable, List, Sequence


# Ordered list of labelled plates (exactly as provided)
PLATES: List[str] = [
    "BMIOG",
    "DM52YX",
    "DFG9OL", 
    "EE", "None", "X16G",
    "FBH7OF",
    "AQWO1S",
    "DN18GK",
    "AZD96K",
    "EQJ63F",
    "BMYO8T",
    "CWI96E",
    "E","None", "KC6N",
    "ENS2OG",
    "CDI92S",
    "FJR34M",
    "DZN41K",
    "B", "None", "9U",
    "ELV24Z",
    "BLDO5R",
]


def split_plate(plate: str) -> List[str]:
    """Return a list of characters for a single plate string.

    - Removes any whitespace so indexing aligns per tile.
    - Example: "BM IO1G" -> ["B", "M", "I", "O", "1", "G"]
    """
    if not isinstance(plate, str):
        raise TypeError("plate must be a string")
    cleaned = "".join(ch for ch in plate if not ch.isspace())
    # Special case: allow the literal string "None" to represent a single-tile placeholder
    # This helps align ground truth when extraction yields an extra/misaligned tile.
    if cleaned.lower() == "none":
        return ["None"]
    return list(cleaned)


def all_plate_chars(plates: Sequence[str] | None = None) -> Dict[str, List[str]]:
    """Return a mapping of plate -> list of characters.

    If `plates` is None, uses the default PLATES list above.
    """
    src = PLATES if plates is None else list(plates)
    return {p: split_plate(p) for p in src}


def all_chars_flat(plates: Sequence[str] | None = None) -> List[str]:
    """Flatten all plate characters into a single per-tile list.

    Useful when consuming one tile at a time in order across all images.
    """
    src = PLATES if plates is None else list(plates)
    out: List[str] = []
    for p in src:
        out.extend(split_plate(p))
    return out


# Default flattened ground-truth characters for PLATES
GT_CHARS_FLAT: List[str] = all_chars_flat()


def pretty_print(plates: Sequence[str] | None = None) -> None:
    """Print plates and their characters spaced for visual inspection."""
    for p in (PLATES if plates is None else plates):
        chars = " ".join(split_plate(p))
        print(f"{p}  ->  {chars}")


def pretty_print_flat(plates: Sequence[str] | None = None) -> None:
    """Print flattened characters with 1-based indices (per tile)."""
    chars = all_chars_flat(plates)
    for i, ch in enumerate(chars, start=1):
        print(f"{i}: {ch}")


if __name__ == "__main__":
    # Simple visual dump when run directly
    pretty_print()
    print("\nFlat per-tile listing:\n----------------------")
    pretty_print_flat()
