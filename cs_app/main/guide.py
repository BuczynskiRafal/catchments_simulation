"""
What each sweepable subcatchment parameter does in SWMM's runoff model.

One entry per feature of ``catchment_simulation.features.FEATURES``. The sweep
forms show an entry next to the chosen parameter, and the "How it works" page
lists them all in its parameter guide; both link to ``anchor`` on that page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ParameterGuide:
    """
    How one feature enters the runoff model.

    Attributes
    ----------
    feature : str
        Column of the SWMM input file, e.g. ``"PercSlope"``.
    name : str
        Display name without a unit, e.g. ``"Slope"``.
    symbol : str
        Symbol used in the equations, as plain text.
    tex : str
        The same symbol in TeX, for the rendered equations.
    section : str
        Section of the input file that holds the column.
    where : str
        The term of the model the feature appears in, as plain text.
    role : str
        What the feature is and how the model uses it.
    effect : str
        What happens to the runoff when the feature grows.
    lab_key : str | None
        The matching parameter of the interactive model
        (``CS.runoffModel.DEFAULTS``), or None when it has none.
    """

    feature: str
    name: str
    symbol: str
    tex: str
    section: str
    where: str
    role: str
    effect: str
    lab_key: str | None

    @property
    def anchor(self) -> str:
        """Fragment identifier of the feature in the parameter guide."""
        return f"parameter-{self.feature}"


_OUTFLOW_RATE = "Outflow rate α = W·√S / (A·n)"

PARAMETER_GUIDE: dict[str, ParameterGuide] = {
    guide.feature: guide
    for guide in (
        ParameterGuide(
            feature="Area",
            name="Area",
            symbol="A",
            tex="A",
            section="[SUBCATCHMENTS]",
            where="Rain volume i·A and outflow rate α = W·√S / (A·n)",
            role=(
                "The surface that collects rain. The rain volume grows in proportion to it, "
                "while the width that drains it stays the same."
            ),
            effect=(
                "A larger area gives more runoff and a higher peak, but less runoff per "
                "hectare: the same width has to drain more surface, so water stays longer "
                "and more of it infiltrates."
            ),
            lab_key="area",
        ),
        ParameterGuide(
            feature="PercImperv",
            name="Imperviousness",
            symbol="I",
            tex="I",
            section="[SUBCATCHMENTS]",
            where="Sub-area split: impervious I·A, pervious (1 − I)·A",
            role=(
                "The share of the area that sheds water without infiltration, such as roofs "
                "and pavements. It sets the size of the impervious and pervious sub-areas."
            ),
            effect=(
                "More impervious surface means less infiltration and faster drainage: "
                "runoff volume and peak rate both rise."
            ),
            lab_key="imperviousness",
        ),
        ParameterGuide(
            feature="Width",
            name="Width",
            symbol="W",
            tex="W",
            section="[SUBCATCHMENTS]",
            where=_OUTFLOW_RATE,
            role=(
                "The characteristic width of the overland flow: the area divided by the "
                "length of the flow path. It multiplies the outflow rate."
            ),
            effect=(
                "A wider subcatchment has a shorter flow path and drains faster: the peak "
                "is higher and earlier, and less water is left on the surface to infiltrate."
            ),
            lab_key="width",
        ),
        ParameterGuide(
            feature="PercSlope",
            name="Slope",
            symbol="S",
            tex="S",
            section="[SUBCATCHMENTS]",
            where=_OUTFLOW_RATE,
            role=(
                "The average slope of the surface. It enters the outflow rate as a square "
                "root, as in Manning's equation."
            ),
            effect=(
                "Steeper surfaces drain faster: the peak rises and a little more water runs "
                "off instead of infiltrating. The square root flattens the response at "
                "steep slopes."
            ),
            lab_key="slope",
        ),
        ParameterGuide(
            feature="CurbLength",
            name="Curb length",
            symbol="L",
            tex="L_c",
            section="[SUBCATCHMENTS]",
            where="Pollutant build-up only",
            role=(
                "The total length of curbs. SWMM uses it only to scale the build-up of "
                "pollutants on streets."
            ),
            effect=(
                "It does not enter the runoff equations, so every run returns the same "
                "runoff. Sweep it only in water-quality studies."
            ),
            lab_key=None,
        ),
        ParameterGuide(
            feature="N-Imperv",
            name="Manning's n, impervious",
            symbol="n",
            tex=r"n_{\mathrm{imp}}",
            section="[SUBAREAS]",
            where=_OUTFLOW_RATE,
            role=(
                "The roughness of the impervious surface. It divides the outflow rate of "
                "the impervious sub-areas."
            ),
            effect=(
                "Rougher pavement slows the flow, so the hydrograph rises and falls more "
                "gently. The volume barely changes, because impervious surfaces lose "
                "almost nothing."
            ),
            lab_key="nImperv",
        ),
        ParameterGuide(
            feature="N-Perv",
            name="Manning's n, pervious",
            symbol="n",
            tex=r"n_{\mathrm{perv}}",
            section="[SUBAREAS]",
            where=_OUTFLOW_RATE,
            role=(
                "The roughness of the pervious surface, such as grass or bare soil. It "
                "divides the outflow rate of the pervious sub-area."
            ),
            effect=(
                "Rougher ground holds water longer, so more of it infiltrates: the peak "
                "and the runoff volume both fall."
            ),
            lab_key="nPerv",
        ),
        ParameterGuide(
            feature="S-Imperv",
            name="Depression storage, impervious",
            symbol="ds",
            tex=r"d_{s,\mathrm{imp}}",
            section="[SUBAREAS]",
            where="Runoff starts when the depth d exceeds ds",
            role=(
                "The depth of water that puddles and surface wetting hold back on the "
                "impervious surface. Runoff starts only after it has filled."
            ),
            effect=(
                "Deeper storage delays the start of runoff and takes that depth out of the "
                "runoff volume."
            ),
            lab_key="storageImperv",
        ),
        ParameterGuide(
            feature="S-Perv",
            name="Depression storage, pervious",
            symbol="ds",
            tex=r"d_{s,\mathrm{perv}}",
            section="[SUBAREAS]",
            where="Runoff starts when the depth d exceeds ds",
            role=(
                "The depth of water held back on the pervious surface. Runoff starts only "
                "after it has filled, and the stored water keeps infiltrating."
            ),
            effect=(
                "Deeper storage delays runoff from the pervious sub-area, lowers its "
                "volume and leaves more water to infiltrate."
            ),
            lab_key="storagePerv",
        ),
        ParameterGuide(
            feature="PctZero",
            name="Impervious area without storage",
            symbol="z",
            tex="z",
            section="[SUBAREAS]",
            where="Sub-area split: I·z·A has no depression storage",
            role=(
                "The share of the impervious area that has no depression storage and "
                "runs off from the first drop of rain."
            ),
            effect=(
                "A larger share makes runoff start sooner. The volume changes little "
                "unless the depression storage is deep."
            ),
            lab_key="zeroImperv",
        ),
    )
}


def parameter_guide_payload() -> dict[str, dict[str, str | None]]:
    """The guide as JSON-ready dicts keyed by feature, each with its ``anchor``."""
    return {
        feature: {**asdict(guide), "anchor": guide.anchor}
        for feature, guide in PARAMETER_GUIDE.items()
    }
