import pytest

from model_router.routing.estimator import (
    CodeBlockFeatureExtractor,
    ComplexityEstimator,
    LengthFeatureExtractor,
    ReasoningKeywordExtractor,
    TechnicalTermExtractor,
    default_complexity_estimator,
)


@pytest.fixture
def estimator() -> ComplexityEstimator:
    return default_complexity_estimator()


def test_estimator_requires_features():
    with pytest.raises(ValueError):
        ComplexityEstimator([])


def test_blank_prompt_scores_zero(estimator):
    assert estimator.estimate("   ") == 0.0


PROMPTS = [
    ("Hi", "low"),
    ("List three fun facts about cats.", "low"),
    ("Provide a short grocery checklist with five items.", "low"),
    ("Write a friendly thank-you note to my neighbor.", "low"),
    (
        "Explain the API schema fields for a basic inventory database, including the purpose of each column.",
        "medium",
    ),
    (
        "Summarize the latency and throughput trade-offs for this service, analyze the risks, and recommend an approach.",
        "medium",
    ),
    (
        "Evaluate and compare two classroom seating arrangements and justify which one optimizes collaboration.",
        "medium",
    ),
    (
        "Explain step-by-step how to derive Bayes theorem, analyze prior probability choices, and relate them to tensor gradients.",
        "high",
    ),
    (
        "Analyze and compare the AES encryption algorithm with lattice-based cryptography, explain step-by-step probability trade-offs, and derive tensor gradients.",
        "high",
    ),
    (
        "Explain why the following code works: ```python\ndef score(prompt):\n    return prompt.count('data')\n```",
        "medium",
    ),
    (
        "Compose a SQL query that joins orders and customers, explain step-by-step why the algorithm works, and discuss database schema latency impacts.",
        "high",
    ),
]


RANGES = {
    "low": (0.0, 0.3),
    "medium": (0.3, 0.6),
    "high": (0.6, 1.01),
}


@pytest.mark.parametrize("prompt,band", PROMPTS)
def test_estimator_places_prompts_into_expected_bands(estimator, prompt, band):
    score = estimator.estimate(prompt)
    low, high = RANGES[band]
    assert low <= score <= high


def test_feature_extractors_output_normalized_scores():
    features = [
        LengthFeatureExtractor(target_chars=10),
        CodeBlockFeatureExtractor(),
        ReasoningKeywordExtractor(),
        TechnicalTermExtractor(),
    ]
    prompt = "Explain tensor gradients inside ```python``` blocks"
    for feature in features:
        score = feature.extract(prompt)
        assert 0.0 <= score <= 1.0
