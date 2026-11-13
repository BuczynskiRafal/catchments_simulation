"""The parameter guide: one entry per sweepable feature, tied to the browser model."""

import json
import re
from pathlib import Path

from django.conf import settings

from catchment_simulation.features import FEATURES
from main.guide import PARAMETER_GUIDE, parameter_guide_payload

RUNOFF_MODEL_JS = (
    Path(settings.BASE_DIR) / "main" / "static" / "main" / "js" / "model" / "runoff-model.js"
)


def _model_parameter_keys() -> set[str]:
    """Keys of CS.runoffModel.DEFAULTS, read from the script."""
    source = RUNOFF_MODEL_JS.read_text(encoding="utf-8")
    block = re.search(r"var DEFAULTS = \{(?P<body>.*?)\n    \};", source, flags=re.DOTALL)
    assert block, "DEFAULTS not found in runoff-model.js"
    return set(re.findall(r"^\s+(\w+):", block.group("body"), flags=re.MULTILINE))


def test_guide_covers_every_feature():
    assert set(PARAMETER_GUIDE) == set(FEATURES)


def test_lab_keys_are_parameters_of_the_model():
    keys = _model_parameter_keys()
    assert {"area", "imperviousness", "width", "slope", "conductivity"} <= keys
    for guide in PARAMETER_GUIDE.values():
        assert guide.lab_key is None or guide.lab_key in keys, guide.feature


def test_anchor_names_the_feature():
    for feature, guide in PARAMETER_GUIDE.items():
        assert guide.feature == feature
        assert guide.anchor == f"parameter-{feature}"


def test_payload_is_json_with_anchors():
    payload = json.loads(json.dumps(parameter_guide_payload()))
    assert set(payload) == set(PARAMETER_GUIDE)
    for feature, entry in payload.items():
        assert entry["anchor"] == f"parameter-{feature}"
        assert entry["name"] == PARAMETER_GUIDE[feature].name
