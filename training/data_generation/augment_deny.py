"""Deny-side data augmentation.

Generates diverse off-topic queries across many categories to
strengthen the DENY boundary. The main gap in synthetic data is
that LLM-generated "hard negatives" tend to cluster around the
same patterns. This script generates broad off-topic coverage.

Usage:
    python -m training.data_generation.augment_deny \
        --policy policies/finance.json \
        --output data/finance/deny_augment.jsonl \
        --provider openai \
        --count 1500
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

# Broad off-topic categories not covered well by policy exclusions
GENERAL_OFF_TOPIC = [
    "weather and climate",
    "pets and animals",
    "gardening and plants",
    "DIY home projects",
    "car maintenance and repair",
    "parenting and childcare",
    "dating and relationships",
    "social media and influencers",
    "music and concerts",
    "art and painting",
    "photography tips",
    "history and historical events",
    "science experiments",
    "space and astronomy",
    "philosophy and ethics",
    "religion and spirituality",
    "politics and elections",
    "education and school",
    "job interview tips",
    "resume writing",
    "programming and coding",
    "web design",
    "mobile app development",
    "cybersecurity basics",
    "math homework",
    "language learning",
    "translation help",
    "trivia and general knowledge",
    "jokes and humor",
    "greetings and small talk",
]


def generate_broad_deny(
    policy: Policy, provider: str, count_per_topic: int = 15
) -> list[dict]:
    """Generate diverse off-topic queries across many categories."""
    examples = []

    # Batch topics to reduce API calls
    batch_size = 6
    topics = GENERAL_OFF_TOPIC
    for i in range(0, len(topics), batch_size):
        batch = topics[i : i + batch_size]
        topic_list = ", ".join(batch)

        prompt = (
            f"Generate {count_per_topic * len(batch)} user queries across these topics: "
            f"{topic_list}.\n\n"
            f"These queries should be clearly NOT about {policy.spec.display_name}.\n"
            f"Mix of questions, statements, and requests.\n"
            f"Include casual, formal, and fragmented styles.\n"
            f"Include some with typos or shorthand.\n"
            f"1-2 sentences each.\n\n"
            f"Output as a JSON array of strings."
        )

        try:
            response = call_llm(prompt, provider)
            queries = parse_json_response(response)
        except Exception as e:
            logger.warning("Failed for topics %s: %s", topic_list, e)
            continue

        for q in queries:
            if isinstance(q, str) and q.strip():
                examples.append({
                    "text": q.strip(),
                    "label": "deny",
                    "category": "broad_off_topic",
                    "source": "augmented",
                    "vertical": policy.vertical,
                })

        logger.info("Generated %d queries for: %s", len(queries), topic_list)
        time.sleep(0.5)

    return examples


def generate_edge_cases(policy: Policy, provider: str, count: int = 200) -> list[dict]:
    """Generate queries that are close to the vertical but still off-topic."""
    topics = ", ".join(policy.spec.scope.core_topics[:8])

    prompt = (
        f"Generate {count} queries that are NEAR the topic of "
        f"{policy.spec.display_name} but are NOT actually in scope.\n\n"
        f"In-scope topics: {topics}\n\n"
        f"Types of near-miss queries to generate:\n"
        f"- Questions about the industry but not the service "
        f"(e.g., for finance: 'How much does a banker earn?' is about careers, not financial services)\n"
        f"- Questions using domain jargon in metaphors or slang\n"
        f"- Questions about related fields that are out of scope\n"
        f"- General curiosity questions that mention domain terms\n"
        f"- Questions about news/current events in the domain "
        f"(e.g., 'Did you hear about the bank robbery?' is news, not banking)\n"
        f"- Pop culture references using domain vocabulary\n\n"
        f"Each query should be clearly off-topic to a human but could fool a keyword filter.\n\n"
        f"Output as a JSON array of strings."
    )

    try:
        response = call_llm(prompt, provider)
        queries = parse_json_response(response)
    except Exception as e:
        logger.warning("Edge case generation failed: %s", e)
        return []

    examples = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            examples.append({
                "text": q.strip(),
                "label": "deny",
                "category": "edge_case",
                "source": "augmented",
                "vertical": policy.vertical,
            })

    logger.info("Generated %d edge case queries", len(examples))
    return examples


def generate_more_allow(policy: Policy, provider: str, count: int = 500) -> list[dict]:
    """Generate additional allow queries with diverse phrasing."""
    topics = ", ".join(policy.spec.scope.core_topics)

    prompt = (
        f"Generate {count} diverse user queries about {policy.spec.display_name}.\n"
        f"Topics: {topics}\n\n"
        f"Requirements:\n"
        f"- Mix of simple, complex, and expert-level queries\n"
        f"- Include casual/informal phrasing (no punctuation, lowercase, shorthand)\n"
        f"- Include queries with typos\n"
        f"- Include multi-sentence queries\n"
        f"- Include fragments ('401k rollover?', 'deductible limit')\n"
        f"- Include questions with context ('My employer offers a 401k match...')\n"
        f"- Include emotional/urgent queries ('I'm worried about my debt')\n"
        f"- Do NOT make them sound like they were written by an AI\n\n"
        f"Output as a JSON array of strings."
    )

    try:
        response = call_llm(prompt, provider)
        queries = parse_json_response(response)
    except Exception as e:
        logger.warning("Allow augmentation failed: %s", e)
        return []

    examples = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            examples.append({
                "text": q.strip(),
                "label": "allow",
                "category": "allow_augmented",
                "source": "augmented",
                "vertical": policy.vertical,
            })

    logger.info("Generated %d additional allow queries", len(examples))
    return examples


def generate_more_abstain(policy: Policy, provider: str, count: int = 200) -> list[dict]:
    """Generate additional abstain queries — genuinely ambiguous."""
    topics = ", ".join(policy.spec.scope.core_topics[:6])

    prompt = (
        f"Generate {count} queries that are genuinely ambiguous — a human "
        f"cannot tell if they belong in {policy.spec.display_name} without "
        f"asking a follow-up question.\n\n"
        f"In-scope topics: {topics}\n\n"
        f"Types:\n"
        f"- Very short fragments (1-3 words)\n"
        f"- Polysemous words used ambiguously\n"
        f"- Context-dependent queries ('what about the rate?')\n"
        f"- Vague requests ('help me with this')\n"
        f"- Questions that could belong to multiple domains\n\n"
        f"Output as a JSON array of strings."
    )

    try:
        response = call_llm(prompt, provider)
        queries = parse_json_response(response)
    except Exception as e:
        logger.warning("Abstain augmentation failed: %s", e)
        return []

    examples = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            examples.append({
                "text": q.strip(),
                "label": "abstain",
                "category": "abstain_augmented",
                "source": "augmented",
                "vertical": policy.vertical,
            })

    logger.info("Generated %d additional abstain queries", len(examples))
    return examples


def main():
    parser = argparse.ArgumentParser(description="Augment training data with deny-focused examples")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default="openai", choices=["anthropic", "openai"])
    parser.add_argument("--count", type=int, default=3000, help="Target total additional examples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = Policy.from_file(args.policy)
    logger.info("Augmenting data for: %s", policy.display_name)

    all_examples = []

    # ~50% deny (broad + edge), ~30% allow, ~20% abstain
    deny_count = int(args.count * 0.5)
    allow_count = int(args.count * 0.3)
    abstain_count = int(args.count * 0.2)

    per_topic = max(5, deny_count // (len(GENERAL_OFF_TOPIC) + 5))

    logger.info("Generating broad off-topic deny (%d per topic batch)...", per_topic)
    all_examples.extend(generate_broad_deny(policy, args.provider, per_topic))

    logger.info("Generating edge case deny (200)...")
    all_examples.extend(generate_edge_cases(policy, args.provider, 200))
    time.sleep(1)

    logger.info("Generating additional allow (%d)...", allow_count)
    all_examples.extend(generate_more_allow(policy, args.provider, allow_count))
    time.sleep(1)

    logger.info("Generating additional abstain (%d)...", abstain_count)
    all_examples.extend(generate_more_abstain(policy, args.provider, abstain_count))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    label_counts = Counter(ex["label"] for ex in all_examples)
    logger.info("Generated %d total: %s", len(all_examples), dict(label_counts))
    logger.info("Written to %s", output_path)


if __name__ == "__main__":
    main()
