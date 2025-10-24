"""Utility helpers for decoding config strings into feature attributes."""
from dataclasses import dataclass
from typing import Dict, Tuple

# Normalized string labels the rest of the pipeline expects.
DENSITY_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}

INTERFERENCE_MAP = {
    "N": "nonsensical",
    "P": "paraphrased",
    "T": "thematic",
}

POSITION_MAP = {
    "beg": "beginning",
    "mid": "middle",
    "end": "end",
}

UNKNOWN_VALUE = "unknown"

# Numeric encodings for downstream numeric pipelines.
DENSITY_TO_CODE = {value: idx for idx, value in enumerate(["low", "medium", "high", UNKNOWN_VALUE])}
INTERFERENCE_TO_CODE = {value: idx for idx, value in enumerate(["nonsensical", "paraphrased", "thematic", UNKNOWN_VALUE])}
POSITION_TO_CODE = {value: idx for idx, value in enumerate(["beginning", "middle", "end", UNKNOWN_VALUE])}


@dataclass(frozen=True)
class ConfigAttributes:
    """Structured representation of the parsed config metadata."""
    distractor_density: str
    interference_type: str
    evidence_position: str

    @property
    def density_code(self) -> int:
        return DENSITY_TO_CODE.get(self.distractor_density, DENSITY_TO_CODE[UNKNOWN_VALUE])

    @property
    def interference_code(self) -> int:
        return INTERFERENCE_TO_CODE.get(self.interference_type, INTERFERENCE_TO_CODE[UNKNOWN_VALUE])

    @property
    def position_code(self) -> int:
        return POSITION_TO_CODE.get(self.evidence_position, POSITION_TO_CODE[UNKNOWN_VALUE])

    def to_feature_dict(self) -> Dict[str, int]:
        """Return numeric features suitable for the time-series matrix."""
        return {
            "distractor_density_code": self.density_code,
            "interference_type_code": self.interference_code,
            "evidence_position_code": self.position_code,
            f"density_is_{self.distractor_density}": 1,
            f"interference_is_{self.interference_type}": 1,
            f"evidence_pos_is_{self.evidence_position}": 1,
        }


def parse_config(config_value: str) -> ConfigAttributes:
    """Decode a config string such as 'Low-N-Beg' into structured attributes."""
    if not isinstance(config_value, str):
        return ConfigAttributes(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)

    parts = [part.strip() for part in config_value.split("-") if part.strip()]
    density_key = parts[0].lower() if len(parts) > 0 else ""
    interference_key = parts[1].upper() if len(parts) > 1 else ""
    position_key = parts[2].lower() if len(parts) > 2 else ""

    density = DENSITY_MAP.get(density_key, UNKNOWN_VALUE)
    interference = INTERFERENCE_MAP.get(interference_key, UNKNOWN_VALUE)
    position = POSITION_MAP.get(position_key, UNKNOWN_VALUE)

    return ConfigAttributes(density, interference, position)


def derive_config_features(config_value: str) -> Dict[str, int]:
    """
    Convenience helper returning numeric one-hot style features derived from config.
    Missing indicators are set to zero; unknown values are captured explicitly.
    """
    attributes = parse_config(config_value)
    feature_dict = attributes.to_feature_dict()
    # Ensure canonical keys exist even if attribute resolves to unknown.
    for key in [
        "density_is_low",
        "density_is_medium",
        "density_is_high",
        "interference_is_nonsensical",
        "interference_is_paraphrased",
        "interference_is_thematic",
        "evidence_pos_is_beginning",
        "evidence_pos_is_middle",
        "evidence_pos_is_end",
    ]:
        feature_dict.setdefault(key, 0)
    return feature_dict


def add_config_columns(df, config_column: str = "config"):
    """Mutate a DataFrame in place with decoded config metadata columns."""
    densities = []
    interferences = []
    positions = []
    for value in df.get(config_column, []):
        attributes = parse_config(value)
        densities.append(attributes.distractor_density)
        interferences.append(attributes.interference_type)
        positions.append(attributes.evidence_position)
    df["distractor_density"] = densities
    df["interference_type"] = interferences
    df["evidence_position"] = positions
    return df
