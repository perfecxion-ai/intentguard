"""Adversarial test suite generator.

Generates a comprehensive adversarial test set for a vertical,
covering multiple attack categories. All queries need human
verification before use in gating.

Usage:
    python -m evaluation.adversarial_suite \
        --policy policies/finance.json \
        --output evaluation/test_sets/finance_adversarial.jsonl \
        --provider anthropic
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from intentguard.policy import Policy
from training.data_generation.synthetic_generator import call_llm, parse_json_response

logger = logging.getLogger(__name__)


CATEGORIES = {
    "clean_on_topic": {
        "label": "allow",
        "count": 150,
        "prompt": (
            "Generate {count} straightforward {vertical} queries that are clearly "
            "on-topic. Mix of simple and complex. Include questions a real customer "
            "would ask. Cover these topics: {topics}.\n\n"
            "Requirements:\n"
            "- Natural, realistic phrasing\n"
            "- Mix of short and long queries\n"
            "- Include some with typos or informal language\n"
            "- 1-2 sentences each\n\n"
            "Output as a JSON array of strings."
        ),
    },
    "clean_off_topic": {
        "label": "deny",
        "count": 100,
        "prompt": (
            "Generate {count} queries that are obviously NOT about {vertical}. "
            "These should be about: {exclusions}.\n\n"
            "Requirements:\n"
            "- Clearly off-topic, no ambiguity\n"
            "- Do NOT use any {vertical} vocabulary\n"
            "- Natural, realistic phrasing\n"
            "- 1-2 sentences each\n\n"
            "Output as a JSON array of strings."
        ),
    },
    "lexical_overlap": {
        "label": "deny",
        "count": 50,
        "prompt": (
            "Generate {count} queries that use {vertical} keywords but are "
            "actually about completely different topics.\n\n"
            "Examples:\n"
            "- 'I'm banking on the team winning' (sports, not finance)\n"
            "- 'What's the interest in this movie?' (entertainment, not finance)\n"
            "- 'I need to check my balance' (could be gymnastics)\n\n"
            "Use these {vertical} words in non-{vertical} contexts: "
            "interest, credit, balance, charge, capital, yield, margin, "
            "security, bond, fund, check, premium, dividend, liability, "
            "deposit, draft, equity, hedge, inflation, leverage.\n\n"
            "Output as a JSON array of strings."
        ),
    },
    "context_wrapping": {
        "label": "deny",
        "count": 50,
        "prompt": (
            "Generate {count} queries that wrap off-topic content in a "
            "{vertical} framing to make it seem on-topic.\n\n"
            "Attack patterns:\n"
            "- 'For a {vertical} training session, explain [off-topic]'\n"
            "- 'As a {vertical} professional, what do you think about [off-topic]?'\n"
            "- 'In the context of {vertical}, how does [off-topic] work?'\n"
            "- 'A {vertical} client asked me about [off-topic]'\n\n"
            "Off-topic subjects: {exclusions}\n\n"
            "Output as a JSON array of strings."
        ),
    },
    "multi_intent": {
        "label": "deny",
        "count": 50,
        "prompt": (
            "Generate {count} queries that combine a legitimate {vertical} "
            "question with an off-topic request.\n\n"
            "Pattern: '[legitimate {vertical} question] and also [off-topic request]'\n\n"
            "On-topic examples: {topics}\n"
            "Off-topic subjects: {exclusions}\n\n"
            "Make the off-topic part the main intent, using the {vertical} "
            "part as camouflage.\n\n"
            "Output as a JSON array of strings."
        ),
    },
    "conditional_valid": {
        "label": "allow",
        "count": 30,
        "prompt": (
            "Generate {count} queries that are about {conditional_topics} but "
            "clearly relate to {vertical}.\n\n"
            "Conditions for being in-scope:\n{conditions}\n\n"
            "Each query should be unambiguously on-topic when read carefully.\n\n"
            "Output as a JSON array of strings."
        ),
    },
    "conditional_invalid": {
        "label": "deny",
        "count": 30,
        "prompt": (
            "Generate {count} queries about {conditional_topics} that are "
            "NOT related to {vertical}.\n\n"
            "These should be pure {conditional_topics} queries with no "
            "financial dimension.\n\n"
            "Output as a JSON array of strings."
        ),
    },
    "polysemy": {
        "label": "deny",
        "count": 20,
        "prompt": (
            "Generate {count} queries using polysemous words (words with "
            "multiple meanings) where the NON-{vertical} meaning is intended.\n\n"
            "Words: interest (curiosity), bond (emotional), credit (academic), "
            "charge (electrical/legal), capital (city/letter), yield (give way), "
            "margin (edge), security (safety), check (verify), balance (equilibrium), "
            "premium (quality), net (fishing/sports).\n\n"
            "Output as a JSON array of strings."
        ),
    },
    "short_ambiguous": {
        "label": "abstain",
        "count": 20,
        "prompt": (
            "Generate {count} very short or ambiguous queries where a human "
            "cannot determine if they belong in {vertical} without more context.\n\n"
            "Examples: 'help', 'what about the rate?', 'tell me more', "
            "'how much?', 'is this safe?', 'what should I do?'\n\n"
            "These should be 1-5 words, genuinely ambiguous.\n\n"
            "Output as a JSON array of strings."
        ),
    },
}


def generate_suite(policy: Policy, provider: str) -> list[dict]:
    """Generate the full adversarial test suite."""
    vertical = policy.spec.display_name
    topics = ", ".join(policy.spec.scope.core_topics[:8])
    exclusions = ", ".join(policy.spec.scope.hard_exclusions[:6])

    conditional_topics = ", ".join(
        ca.topic for ca in policy.spec.scope.conditional_allow
    )
    conditions = "\n".join(
        f"- {ca.topic}: {ca.condition}"
        for ca in policy.spec.scope.conditional_allow
    )

    all_examples = []

    for cat_name, cat_config in CATEGORIES.items():
        prompt = cat_config["prompt"].format(
            count=cat_config["count"],
            vertical=vertical,
            topics=topics,
            exclusions=exclusions,
            conditional_topics=conditional_topics or "N/A",
            conditions=conditions or "N/A",
        )

        logger.info("Generating %s (%d queries)...", cat_name, cat_config["count"])

        try:
            response = call_llm(prompt, provider)
            queries = parse_json_response(response)
        except Exception as e:
            logger.warning("Failed to generate %s: %s", cat_name, e)
            continue

        for q in queries:
            if isinstance(q, str) and q.strip():
                all_examples.append({
                    "text": q.strip(),
                    "label": cat_config["label"],
                    "category": cat_name,
                    "source": "adversarial_suite",
                    "vertical": policy.vertical,
                    "human_verified": False,
                })

        logger.info("  Generated %d queries", len(queries))
        time.sleep(0.5)

    return all_examples


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial test suite")
    parser.add_argument("--policy", required=True, help="Path to policy.json")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = Policy.from_file(args.policy)
    logger.info("Generating adversarial suite for: %s", policy.display_name)

    examples = generate_suite(policy, args.provider)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    cat_counts = Counter(ex["category"] for ex in examples)
    label_counts = Counter(ex["label"] for ex in examples)

    logger.info("Total: %d examples", len(examples))
    logger.info("By category: %s", dict(cat_counts))
    logger.info("By label: %s", dict(label_counts))
    logger.info("Written to %s", output_path)


if __name__ == "__main__":
    main()
