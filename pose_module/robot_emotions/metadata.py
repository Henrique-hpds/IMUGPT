"""Static metadata for the RobotEmotions dataset."""

from __future__ import annotations

from typing import Any, Sequence

PARTICIPANTS_BY_DOMAIN: dict[str, dict[int, dict[str, Any]]] = {
    "30ms": {
        1: {"name": "Pasquale", "age": 24, "gender": "M"},
        2: {"name": "Luigi", "age": 26, "gender": "M"},
        3: {"name": "Alessandro", "age": 24, "gender": "M"},
        4: {"name": "Carmine", "age": 24, "gender": "M"},
        5: {"name": "Andrea", "age": 24, "gender": "M"},
        6: {"name": "Rosaria", "age": 24, "gender": "F"},
        8: {"name": "Sara", "age": 26, "gender": "F"},
    },
    "10ms": {
        2: {"name": "Fulvio", "age": 24, "gender": "M"},
        3: {"name": "Manuel", "age": 23, "gender": "M"},
        4: {"name": "Andrea", "age": 24, "gender": "M"},
        5: {"name": "Gennaro", "age": 24, "gender": "M"},
    },
}

PROTOCOL_ROWS: list[dict[str, Any]] = [
    {
        "emotion": "Neutrality",
        "action": "Standing",
        "stimulus": "N.A.",
        "stimulus_details": "N.A.",
        "tag_30ms": 1,
        "tag_10ms": 1,
        "notes": "",
    },
    {
        "emotion": "Neutrality",
        "action": "Sitting",
        "stimulus": "N.A.",
        "stimulus_details": "N.A.",
        "tag_30ms": 2,
        "tag_10ms": 2,
        "notes": "",
    },
    {
        "emotion": "Neutrality",
        "action": "Walking + arms outstretched",
        "stimulus": "N.A.",
        "stimulus_details": "N.A.",
        "tag_30ms": 14,
        "tag_10ms": 3,
        "notes": "",
    },
    {
        "emotion": "Sadness",
        "action": "Slow walking + free arm movement",
        "stimulus": "AUDIO",
        "stimulus_details": "Have the participant choose a sad song",
        "tag_30ms": 15,
        "tag_10ms": 4,
        "notes": "",
    },
    {
        "emotion": "Sadness",
        "action": "Sitting + free arm movement",
        "stimulus": "VISUAL",
        "stimulus_details": "Show/have the participant choose sad video clips",
        "tag_30ms": 7,
        "tag_10ms": 5,
        "notes": "",
    },
    {
        "emotion": "Sadness",
        "action": "Free arm movement + sitting",
        "stimulus": "AUTOBIOGRAPHICAL RECALL",
        "stimulus_details": "Ask the participant to remember a sad episode",
        "tag_30ms": 8,
        "tag_10ms": 6,
        "notes": "",
    },
    {
        "emotion": "Sadness",
        "action": "Free arm movement + standing",
        "stimulus": "AUTOBIOGRAPHICAL RECALL",
        "stimulus_details": "Ask the participant to remember a sad episode",
        "tag_30ms": 17,
        "tag_10ms": 7,
        "notes": "",
    },
    {
        "emotion": "Happiness",
        "action": "Fast walking + free arm movement",
        "stimulus": "AUDIO",
        "stimulus_details": "Have the participant choose a song they like",
        "tag_30ms": 16,
        "tag_10ms": 8,
        "notes": "",
    },
    {
        "emotion": "Happiness",
        "action": "Sitting + free arm movement",
        "stimulus": "VISUAL",
        "stimulus_details": "Show/have the participant choose happy video clips",
        "tag_30ms": 5,
        "tag_10ms": 9,
        "notes": "",
    },
    {
        "emotion": "Happiness",
        "action": "Gesturing with arms raised + sitting",
        "stimulus": "AUTOBIOGRAPHICAL RECALL / SIMULATING A CONVERSATION",
        "stimulus_details": "Ask the participant to recall a happy episode/funny anecdote",
        "tag_30ms": 3,
        "tag_10ms": 10,
        "notes": "",
    },
    {
        "emotion": "Happiness",
        "action": "Gesturing with arms raised + raised",
        "stimulus": "AUTOBIOGRAPHICAL RECALL / SIMULATING A CONVERSATION",
        "stimulus_details": "Ask the participant to imagine a happy episode/funny anecdote",
        "tag_30ms": 4,
        "tag_10ms": 11,
        "notes": "",
    },
    {
        "emotion": "Anger",
        "action": "Standing + free arm movement",
        "stimulus": "AUTOBIOGRAPHICAL RECALL / SIMULATING AN UNPLEASANT CONVERSATION",
        "stimulus_details": (
            "Ask the participant to imagine an unfair episode towards him/her/"
            "a moment of argument"
        ),
        "tag_30ms": 18,
        "tag_10ms": 12,
        "notes": "",
    },
    {
        "emotion": "Fear",
        "action": "Bounce + free arm movement",
        "stimulus": "AUDIOVISUAL/ SIMULATION",
        "stimulus_details": (
            "Show the participant an unexpected, scary video/"
            "Ask the participant to simulate the action"
        ),
        "tag_30ms": 9,
        "tag_10ms": 14,
        "notes": "30ms table indicates '9 - sitting'",
    },
    {
        "emotion": "Fear",
        "action": "Escape",
        "stimulus": "SIMULATION",
        "stimulus_details": (
            "Ask the participant to simulate a dangerous situation in which he turns "
            "and runs away"
        ),
        "tag_30ms": 13,
        "tag_10ms": 15,
        "notes": "",
    },
    {
        "emotion": "Boredom",
        "action": "Sitting + free arm movement",
        "stimulus": "AUDIOVISUAL/ SIMULATION",
        "stimulus_details": (
            "Show the participant a video that is not interesting to him/her/"
            "Ask the participant to simulate a moment of boredom"
        ),
        "tag_30ms": 6,
        "tag_10ms": 16,
        "notes": "",
    },
    {
        "emotion": "Boredom",
        "action": "Standing + free arm movement",
        "stimulus": "AUDIOVISUAL/ SIMULATION",
        "stimulus_details": (
            "Show the participant a video that is not interesting to him/her/"
            "Ask the participant to simulate a moment of boredom"
        ),
        "tag_30ms": 12,
        "tag_10ms": 17,
        "notes": "",
    },
    {
        "emotion": "Curiosity",
        "action": "Free movement of the still body",
        "stimulus": "SIMULATION",
        "stimulus_details": "Ask the participant to find a hidden object",
        "tag_30ms": 10,
        "tag_10ms": 18,
        "notes": "",
    },
    {
        "emotion": "Curiosity",
        "action": "Free body movement + walking",
        "stimulus": "SIMULATION",
        "stimulus_details": "Ask the participant to find a hidden object",
        "tag_30ms": 11,
        "tag_10ms": 19,
        "notes": "",
    },
]

PROTOCOL_BY_TAG: dict[str, dict[int, dict[str, Any]]] = {"10ms": {}, "30ms": {}}
for _entry in PROTOCOL_ROWS:
    PROTOCOL_BY_TAG["10ms"][int(_entry["tag_10ms"])] = _entry
    PROTOCOL_BY_TAG["30ms"][int(_entry["tag_30ms"])] = _entry
del _entry

CHANNEL_AXIS_ORDER: tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")

SENSOR_NAME_BY_ID: dict[int, str] = {
    1: "waist",
    2: "head",
    3: "left_forearm",
    4: "right_forearm",
}

SENSOR_MAPPING_POLICY_NOTE = (
    "CSV sensor ids are mapped as 1-waist, 2-head, 3-left_forearm, 4-right_forearm."
)


def get_user_profile(domain: str, user_id: int) -> dict[str, Any]:
    profile = PARTICIPANTS_BY_DOMAIN.get(domain, {}).get(user_id)
    if profile is None:
        return {"name": None, "age": None, "gender": None}
    return dict(profile)


def get_protocol_info(domain: str, tag_number: int) -> dict[str, Any] | None:
    info = PROTOCOL_BY_TAG.get(domain, {}).get(tag_number)
    return dict(info) if info is not None else None


def get_sensor_name(sensor_id: int) -> str:
    return SENSOR_NAME_BY_ID.get(int(sensor_id), f"sensor_{int(sensor_id)}")


def resolve_sensor_names(sensor_ids: Sequence[int]) -> list[str]:
    return [get_sensor_name(int(sensor_id)) for sensor_id in sensor_ids]
