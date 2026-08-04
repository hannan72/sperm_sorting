"""API tests for the demo service.

Kept rather than thrown away: the endpoints encode three claims that are easy to
break silently and expensive to break in public --

* ``/generate`` is deterministic in its seed, and the labels it returns really
  are the generator's own (a flipped aspect must flip the overall label),
* ``/classify`` announces that it is untrained, in a field a machine can read,
* ``/decide`` reproduces the five mandated cases exactly, including the two
  boundary cases where exactly 60% must REJECT and drive ``FIELD_ON``.

The decision cases in particular are the reason this file exists. They have been
got wrong before, in both directions, and an assertion is cheaper than a recall.

Run with ``pytest web/test_api.py`` (pyproject puts the repository root on
``pythonpath``, so ``import web.app`` resolves).
"""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sperm_sorting.constants import LABEL_ABNORMAL, LABEL_NORMAL, MORPHOLOGY_ASPECTS
from web.app import DISCLAIMER, app

pytestmark = pytest.mark.web


@pytest.fixture(scope="module")
def client() -> Any:
    """A client that runs the lifespan, so ``app.state`` is populated.

    ``TestClient`` as a context manager is what triggers startup/shutdown;
    instantiating it without ``with`` would leave every handler reading an empty
    ``app.state`` and failing for the wrong reason.
    """
    with TestClient(app) as test_client:
        yield test_client


def _generate(client: Any, **overrides: Any) -> dict[str, Any]:
    """POST /generate, asserting success so callers can index the body."""
    response = client.post("/generate", json=overrides)
    assert response.status_code == 200, response.text
    return response.json()


# ==========================================================================
# 1. Liveness and the disclaimer
# ==========================================================================


def test_health_is_200_and_reports_the_engine(client: Any) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # No weights exist yet, so a health check claiming a trained model would be
    # the first place the demo lied.
    assert body["trained"] is False


def test_index_is_200_and_carries_the_disclaimer_verbatim(client: Any) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert DISCLAIMER in response.text, (
        "the served page must contain the disclaimer as one uninterrupted "
        "string; if this fails, check that index.html has not been reflowed"
    )
    assert "DNA integrity" in response.text
    assert "apoptosis" in response.text


def test_static_assets_are_served_locally(client: Any) -> None:
    for path in ("/static/app.js", "/static/style.css"):
        assert client.get(path).status_code == 200, path


def test_page_makes_no_external_requests(client: Any) -> None:
    """The device is offline; a single CDN reference would break the demo."""
    page = client.get("/").text
    for marker in ("http://", "https://", "//cdn", "integrity="):
        assert marker not in page, f"index.html references something external: {marker}"


# ==========================================================================
# 2. /generate: shape, determinism, and the same-sperm guarantee
# ==========================================================================


def test_generate_returns_a_decodable_png_of_the_documented_shape(client: Any) -> None:
    body = _generate(client, seed=11, image_size=128, n_points=64)

    assert body["image_format"] == "png"
    assert body["image_shape"] == [128, 128]
    raw = base64.b64decode(body["image"])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.format == "PNG"
        assert image.size == (128, 128)
        assert image.mode == "L"

    assert len(body["trajectory"]) == 64
    assert all(len(point) == 2 for point in body["trajectory"])
    assert body["trajectory_units"] == "pixels"


@pytest.mark.parametrize("size", [64, 128])
def test_generate_honours_both_supported_crop_sizes(client: Any, size: int) -> None:
    body = _generate(client, seed=3, image_size=size)
    with Image.open(io.BytesIO(base64.b64decode(body["image"]))) as image:
        assert image.size == (size, size)


@pytest.mark.parametrize("n_points", [8, 32, 96, 257])
def test_generate_returns_exactly_the_requested_number_of_points(
    client: Any, n_points: int
) -> None:
    body = _generate(client, seed=5, n_points=n_points)
    assert len(body["trajectory"]) == n_points


def test_generate_rejects_an_unsupported_crop_size(client: Any) -> None:
    assert client.post("/generate", json={"image_size": 96}).status_code == 422


def test_same_seed_is_byte_identical_and_a_different_seed_is_not(client: Any) -> None:
    first = _generate(client, seed=99)
    again = _generate(client, seed=99)
    other = _generate(client, seed=100)

    assert first["image"] == again["image"]
    assert first["trajectory"] == again["trajectory"]
    assert first == again, "the whole response must be reproducible from the seed"

    assert other["image"] != first["image"]
    assert other["trajectory"] != first["trajectory"]


def test_the_image_and_the_track_describe_the_same_sperm(client: Any) -> None:
    """A forced grade must show up in the kinematics, not only in the label."""
    fast = _generate(client, seed=4, motility="rapid_progressive")
    dead = _generate(client, seed=4, motility="immotile")
    assert fast["true_motility"] == "rapid_progressive"
    assert dead["true_motility"] == "immotile"
    assert fast["casa"]["vsl"] > dead["casa"]["vsl"]


# ==========================================================================
# 3. /generate: the health rule, aspect by aspect
# ==========================================================================


ALL_NORMAL = [LABEL_NORMAL] * len(MORPHOLOGY_ASPECTS)


@pytest.mark.parametrize("motility", ["rapid_progressive", "slow_progressive"])
def test_all_normal_and_progressive_is_healthy(client: Any, motility: str) -> None:
    body = _generate(client, seed=21, aspects=ALL_NORMAL, motility=motility)
    assert body["true_aspects"] == dict.fromkeys(MORPHOLOGY_ASPECTS, LABEL_NORMAL)
    assert body["true_label"] == LABEL_NORMAL
    assert body["true_label_name"] == "healthy"
    assert body["true_motility_label_name"] == "progressive"


@pytest.mark.parametrize("index,aspect", list(enumerate(MORPHOLOGY_ASPECTS)))
def test_flipping_any_single_aspect_makes_it_unhealthy(
    client: Any, index: int, aspect: str
) -> None:
    """All four aspects, independently. One defect is enough to disqualify."""
    flags = list(ALL_NORMAL)
    flags[index] = LABEL_ABNORMAL
    body = _generate(client, seed=21, aspects=flags, motility="rapid_progressive")

    assert body["true_aspects"][aspect] == LABEL_ABNORMAL
    for other in MORPHOLOGY_ASPECTS:
        if other != aspect:
            assert body["true_aspects"][other] == LABEL_NORMAL
    assert body["true_label"] == LABEL_ABNORMAL, f"{aspect} defect must disqualify"
    assert body["true_label_name"] == "unhealthy"


@pytest.mark.parametrize("motility", ["non_progressive", "immotile"])
def test_perfect_morphology_without_progression_is_unhealthy(
    client: Any, motility: str
) -> None:
    body = _generate(client, seed=21, aspects=ALL_NORMAL, motility=motility)
    assert body["true_morphology_label"] == LABEL_NORMAL
    assert body["true_label"] == LABEL_ABNORMAL


def test_overriding_a_knob_is_announced(client: Any) -> None:
    """A hand-set knob breaks the label/pixel link and must say so."""
    plain = _generate(client, seed=7, aspects=ALL_NORMAL, motility="rapid_progressive")
    assert plain["overridden_knobs"] == []
    assert plain["label_pixel_link_intact"] is True

    forced = _generate(
        client,
        seed=7,
        aspects=ALL_NORMAL,
        motility="rapid_progressive",
        knobs={"head_axis_ratio": 2.8},
    )
    assert forced["overridden_knobs"] == ["head_axis_ratio"]
    assert forced["label_pixel_link_intact"] is False
    assert forced["state"]["head_axis_ratio"] == pytest.approx(2.8)
    # The label is unchanged: overriding the evidence does not rewrite history.
    assert forced["true_aspects"]["head"] == LABEL_NORMAL


def test_generate_rejects_an_unknown_knob(client: Any) -> None:
    response = client.post("/generate", json={"knobs": {"nose_length": 1.0}})
    assert response.status_code == 422


# ==========================================================================
# 4. /classify: probabilities, and an unmissable untrained flag
# ==========================================================================


def test_classify_returns_four_probabilities_and_flags_itself_untrained(
    client: Any,
) -> None:
    response = client.post("/classify", json={"seed": 42})
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body["probs"]) == set(MORPHOLOGY_ASPECTS)
    for aspect, probability in body["probs"].items():
        assert 0.0 <= probability <= 1.0, aspect

    for aspect in MORPHOLOGY_ASPECTS:
        detail = body["aspect_detail"][aspect]
        assert 0.0 <= detail["p_normal"] <= 1.0
        assert 0.0 <= detail["p_abnormal"] <= 1.0
        assert detail["p_normal"] + detail["p_abnormal"] == pytest.approx(1.0)
        assert detail["label"] in (LABEL_NORMAL, LABEL_ABNORMAL)

    model = body["model"]
    assert model["untrained"] is True
    assert model["trained"] is False
    assert model["provenance"] == "random-test-engine"
    assert "untrained" in model["untrained_warning"].lower()
    assert model["reads_the_image"] is False
    assert model["engine_class"] == "RandomMorphologyEngine"


def test_classify_accepts_a_png_and_says_it_cannot_grade_motility(client: Any) -> None:
    generated = _generate(client, seed=17)
    response = client.post("/classify", json={"image": generated["image"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["input_kind"] == "image"
    assert body["pred_motility"] == "undetermined"
    assert body["ineligibility_reason"] == "motility_undetermined"
    assert body["pred_label"] == LABEL_ABNORMAL
    assert "no trajectory" in body["pred_motility_reason"]


def test_classify_accepts_a_data_uri(client: Any) -> None:
    generated = _generate(client, seed=17)
    response = client.post(
        "/classify", json={"image": "data:image/png;base64," + generated["image"]}
    )
    assert response.status_code == 200, response.text


def test_classify_grades_motility_with_the_production_rule(client: Any) -> None:
    """The motility half is not random; it is the real WHO-threshold classifier."""
    body = client.post(
        "/classify",
        json={"seed": 8, "params": {"seed": 8, "motility": "rapid_progressive"}},
    ).json()
    assert body["motility_source"] == "casa_rule"
    assert body["pred_motility_progressive"] is True
    assert "classify_motility" in body["motility_rule"]

    dead = client.post(
        "/classify", json={"seed": 8, "params": {"seed": 8, "motility": "immotile"}}
    ).json()
    assert dead["pred_motility_progressive"] is False
    assert dead["ineligibility_reason"] == "not_progressive"


def test_classify_rejects_an_empty_body_and_a_broken_image(client: Any) -> None:
    assert client.post("/classify", json={}).status_code == 422
    assert client.post("/classify", json={"image": "not base64!!"}).status_code == 422
    assert (
        client.post("/classify", json={"image": base64.b64encode(b"nope").decode()}).status_code
        == 422
    )


# ==========================================================================
# 5. /decide: the five mandated cases, exactly
# ==========================================================================


#: (eligible, trackable, status, field command). These four rows and the fifth
#: below are the specification, transcribed. Do not "simplify" 15/25.
MANDATED = [
    (15, 25, "reject", "FIELD_ON"),
    (16, 25, "accept", "FIELD_OFF"),
    (12, 20, "reject", "FIELD_ON"),
    (13, 20, "accept", "FIELD_OFF"),
]


@pytest.mark.parametrize("eligible,trackable,status,field", MANDATED)
def test_mandated_decision_cases(
    client: Any, eligible: int, trackable: int, status: str, field: str
) -> None:
    response = client.post(
        "/decide", json={"ai_eligible_count": eligible, "trackable_count": trackable}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == status, f"{eligible}/{trackable}"
    assert body["field_command"] == field, f"{eligible}/{trackable}"
    assert body["ai_eligible_count"] == eligible
    assert body["trackable_count"] == trackable


def test_nineteen_trackable_at_timeout_is_indeterminate(client: Any) -> None:
    """Even a perfect 19/19 is INDETERMINATE: the minimum is 20, not the ratio."""
    body = client.post(
        "/decide", json={"ai_eligible_count": 19, "trackable_count": 19}
    ).json()
    assert body["status"] == "indeterminate"
    assert body["field_command"] == "FIELD_OFF"
    assert body["minimum_trackable"] == 20
    assert body["is_rejection"] is False


def test_exactly_sixty_percent_rejects_and_energises_the_field(client: Any) -> None:
    for eligible, trackable in [(15, 25), (12, 20), (18, 30)]:
        body = client.post(
            "/decide",
            json={"ai_eligible_count": eligible, "trackable_count": trackable},
        ).json()
        assert body["exactly_at_threshold"] is True, f"{eligible}/{trackable}"
        assert body["status"] == "reject"
        assert body["field_command"] == "FIELD_ON"
        assert body["is_rejection"] is True
        assert "REJECT" in body["boundary_rule"]


def test_field_on_is_described_as_the_rejection(client: Any) -> None:
    reject = client.post(
        "/decide", json={"ai_eligible_count": 15, "trackable_count": 25}
    ).json()
    accept = client.post(
        "/decide", json={"ai_eligible_count": 16, "trackable_count": 25}
    ).json()
    assert "waste" in reject["field_command_meaning"]
    assert "rejection" in reject["field_command_meaning"].lower()
    assert "collection" in accept["field_command_meaning"]


def test_decide_uses_the_real_engine_and_returns_its_rationale(client: Any) -> None:
    body = client.post(
        "/decide", json={"ai_eligible_count": 15, "trackable_count": 25}
    ).json()
    assert body["engine"] == "sperm_sorting.decision.engine.decide"
    assert "does not exceed" in body["rationale"]
    assert "energised" in body["rationale"]


# ==========================================================================
# 6. /decide: bad input is a 4xx, never a 500
# ==========================================================================


def test_numerator_larger_than_denominator_is_a_4xx(client: Any) -> None:
    response = client.post(
        "/decide", json={"ai_eligible_count": 26, "trackable_count": 25}
    )
    assert 400 <= response.status_code < 500, response.status_code
    assert "exceeds" in response.json()["detail"]


def test_negative_counts_are_a_4xx(client: Any) -> None:
    response = client.post(
        "/decide", json={"ai_eligible_count": -1, "trackable_count": 25}
    )
    assert 400 <= response.status_code < 500


def test_zero_trackable_is_indeterminate_not_a_division_error(client: Any) -> None:
    body = client.post(
        "/decide", json={"ai_eligible_count": 0, "trackable_count": 0}
    ).json()
    assert body["status"] == "indeterminate"
    assert body["ratio"] == 0.0


# ==========================================================================
# 7. /config and /aspects
# ==========================================================================


def test_config_includes_the_feasibility_report_and_the_um_per_px_figure(
    client: Any,
) -> None:
    response = client.get("/config")
    assert response.status_code == 200, response.text
    body = response.json()

    feasibility = body["feasibility"]
    assert feasibility["um_per_px"] > 0.0
    assert feasibility["um_per_px"] == pytest.approx(0.0345, abs=1e-6)
    assert feasibility["um_per_px_is_measured"] is False
    assert feasibility["field_width_um"] > 0.0
    assert feasibility["field_height_um"] > 0.0
    assert feasibility["head_span_px"] > 1.0
    assert isinstance(feasibility["warnings"], list)
    assert "sampling" in feasibility["formatted"]

    # The headline optical fact this panel exists to state.
    assert feasibility["whole_sperm_fits_across_flow"] is False
    assert any("does not fit across" in warning for warning in feasibility["warnings"])

    assert body["summary"]["decision_threshold"] == 0.60
    assert body["decision"]["minimum_trackable"] == 20
    assert body["morphology"]["untrained"] is True
    assert body["disclaimer"] == DISCLAIMER


def test_aspects_publishes_the_canonical_order_and_convention(client: Any) -> None:
    body = client.get("/aspects").json()
    assert body["aspects"] == list(MORPHOLOGY_ASPECTS)
    assert body["label_normal"] == LABEL_NORMAL == 0
    assert body["label_abnormal"] == LABEL_ABNORMAL == 1
    assert body["label_names"] == {"0": "normal", "1": "abnormal"}
    assert body["overall_label_names"] == ["healthy", "unhealthy"]
    assert body["motility_label_names"] == ["progressive", "non_progressive", "immotile"]
    assert body["progressive_classes"] == ["rapid_progressive", "slow_progressive"]
    assert "undetermined" not in body["motility_classes"]
    assert body["polarity"]["logit_means"] == "P(abnormal)"
    assert {knob["name"] for knob in body["knobs"]} >= {"head_axis_ratio", "contrast"}


def test_aspects_carries_all_five_mandated_cases(client: Any) -> None:
    """The page's preset buttons come from here, so the set must be complete."""
    cases = client.get("/aspects").json()["mandated_decision_cases"]
    pairs = {(case["ai_eligible_count"], case["trackable_count"]) for case in cases}
    assert pairs == {(15, 25), (16, 25), (12, 20), (13, 20), (19, 19)}
