"""The "How it works" page."""

import re

import pytest
from django.urls import reverse
from django.utils.html import escape

from main.guide import PARAMETER_GUIDE

SECTION_ANCHORS = (
    "pipeline",
    "model",
    "lab",
    "parameters",
    "infiltration",
    "numerics",
    "sweep",
    "network",
    "references",
)


@pytest.fixture
def page(client):
    return client.get(reverse("main:how_it_works"))


@pytest.fixture
def html(page):
    return page.content.decode()


@pytest.mark.django_db
def test_anonymous_visitor_gets_the_page(page):
    assert page.status_code == 200
    assert "main/how_it_works.html" in [template.name for template in page.templates]


@pytest.mark.django_db
@pytest.mark.parametrize(
    "anchor", [*SECTION_ANCHORS, *(g.anchor for g in PARAMETER_GUIDE.values())]
)
def test_every_anchor_is_present_once(html, anchor):
    assert len(re.findall(rf'\bid="{re.escape(anchor)}"', html)) == 1


@pytest.mark.django_db
@pytest.mark.parametrize("feature", list(PARAMETER_GUIDE))
def test_every_parameter_is_explained(html, feature):
    guide = PARAMETER_GUIDE[feature]
    entry = re.search(rf'<article[^>]*id="{guide.anchor}".*?</article>', html, flags=re.DOTALL)
    assert entry, feature
    text = entry.group(0)
    for value in (guide.name, guide.feature, guide.section, guide.where, guide.role, guide.effect):
        assert escape(value) in text, (feature, value)
    assert ('data-hiw-try="' + feature + '"' in text) is (guide.lab_key is not None)


@pytest.mark.django_db
def test_sweep_picker_lists_every_parameter_of_the_model(html):
    options = dict(re.findall(r'<option value="([^"]+)" data-hiw-param="([^"]+)"', html))
    assert options == {f: g.lab_key for f, g in PARAMETER_GUIDE.items() if g.lab_key}


@pytest.mark.django_db
def test_nav_marks_the_page_as_current(html):
    current = re.findall(r'<a class="nav-link"[^>]*aria-current="page"[^>]*>([^<]+)</a>', html)
    assert current == ["How it works"]


@pytest.mark.django_db
def test_equations_are_numbered_in_order(html):
    numbers = re.findall(r'class="hiw-equation__number">\((\d+)\)<', html)
    assert numbers == [str(n) for n in range(1, 19)]
    assert re.findall(r'class="hiw-equation" id="eq-(\d+)"', html) == numbers


@pytest.mark.django_db
def test_tex_reaches_the_page_verbatim(html):
    # {% verbatim %} keeps TeX braces away from the template engine.
    assert r"\frac{\mathrm{d}d}{\mathrm{d}t} = i - e - f - q" in html
    assert "{#" not in html
    assert "{%" not in html


@pytest.mark.django_db
def test_no_inline_scripts_or_styles(html):
    for attrs, _body in re.findall(
        r"<script(?P<attrs>[^>]*)>(.*?)</script>", html, flags=re.DOTALL
    ):
        assert re.search(r"\bsrc\s*=", attrs) or "application/json" in attrs
    assert not re.search(r"\sstyle\s*=", html)


@pytest.mark.django_db
def test_katex_loads_with_subresource_integrity(html):
    for asset in ("katex.min.js", "katex.min.css"):
        tag = re.search(
            rf"<(?:script|link)[^>]*katex@0\.19\.1/dist/{asset}[^>]*>", html, flags=re.DOTALL
        )
        assert tag, asset
        assert 'integrity="sha384-' in tag.group(0)
        assert 'crossorigin="anonymous"' in tag.group(0)


@pytest.mark.django_db
def test_lab_needs_javascript_and_says_so(html):
    assert re.search(r'<div class="hiw-lab" id="hiw-lab" hidden>', html)
    assert "<noscript>" in html
    assert "needs JavaScript" in html


@pytest.mark.django_db
def test_calls_to_action_link_to_the_tools(html):
    for name in ("main:simulation", "main:timeseries", "main:calculations"):
        assert f'href="{reverse(name)}"' in html
