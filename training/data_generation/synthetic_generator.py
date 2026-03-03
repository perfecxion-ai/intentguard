"""Synthetic training data generator.

Reads a policy.json and generates labeled JSONL training data
using an LLM API (Claude or OpenAI). Produces four categories:
positive (allow), hard negative (deny), ambiguous (abstain),
and adversarial (deny).

Usage:
    python -m training.data_generation.synthetic_generator \
        --policy policies/finance.json \
        --output data/finance/synthetic.jsonl \
        --provider anthropic \
        --count 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from intentguard.policy import Policy

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(name: str) -> str:
    path = TEMPLATE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text()


def call_llm(prompt: str, provider: str, model: str | None = None) -> str:
    """Call an LLM API and return the text response."""
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    elif provider == "openai":
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")


def parse_json_response(text: str) -> list | dict:
    """Extract JSON from an LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def generate_positive(policy: Policy, provider: str, per_topic: int = 15) -> list[dict]:
    """Generate on-topic queries for each core topic."""
    template = load_template("positive_queries.txt")
    examples = []

    for topic in policy.spec.scope.core_topics:
        prompt = template.format(
            count=per_topic,
            vertical=policy.spec.display_name,
            topic=topic,
        )

        try:
            response = call_llm(prompt, provider)
            queries = parse_json_response(response)
        except Exception as e:
            logger.warning("Failed to generate positives for %s: %s", topic, e)
            continue

        for q in queries:
            if isinstance(q, str) and q.strip():
                examples.append({
                    "text": q.strip(),
                    "label": "allow",
                    "category": "positive",
                    "source": "synthetic",
                    "vertical": policy.vertical,
                    "topic": topic,
                })

        logger.info("Generated %d positives for topic: %s", len(queries), topic)
        time.sleep(0.5)  # rate limit courtesy

    return examples


def generate_hard_negatives(policy: Policy, provider: str, per_exclusion: int = 20) -> list[dict]:
    """Generate off-topic queries that use domain vocabulary."""
    template = load_template("hard_negatives.txt")
    examples = []

    for exclusion in policy.spec.scope.hard_exclusions:
        prompt = template.format(
            count=per_exclusion,
            vertical=policy.spec.display_name,
            excluded_topic=exclusion,
        )

        try:
            response = call_llm(prompt, provider)
            queries = parse_json_response(response)
        except Exception as e:
            logger.warning("Failed to generate negatives for %s: %s", exclusion, e)
            continue

        for q in queries:
            if isinstance(q, str) and q.strip():
                examples.append({
                    "text": q.strip(),
                    "label": "deny",
                    "category": "hard_negative",
                    "source": "synthetic",
                    "vertical": policy.vertical,
                    "topic": exclusion,
                })

        logger.info("Generated %d hard negatives for: %s", len(queries), exclusion)
        time.sleep(0.5)

    return examples


def generate_ambiguous(policy: Policy, provider: str, per_conditional: int = 30) -> list[dict]:
    """Generate queries in conditional allow zones — both allow and deny sides."""
    template = load_template("ambiguous_queries.txt")
    examples = []

    for ca in policy.spec.scope.conditional_allow:
        allow_examples_str = "\n".join(f"- {e}" for e in ca.examples_allow) if ca.examples_allow else "None provided"
        deny_examples_str = "\n".join(f"- {e}" for e in ca.examples_deny) if ca.examples_deny else "None provided"

        allow_count = per_conditional // 2
        deny_count = per_conditional - allow_count

        prompt = template.format(
            count=per_conditional,
            conditional_topic=ca.topic,
            vertical=policy.spec.display_name,
            condition=ca.condition,
            allow_count=allow_count,
            deny_count=deny_count,
            allow_examples=allow_examples_str,
            deny_examples=deny_examples_str,
        )

        try:
            response = call_llm(prompt, provider)
            data = parse_json_response(response)
        except Exception as e:
            logger.warning("Failed to generate ambiguous for %s: %s", ca.topic, e)
            continue

        for q in data.get("allow", []):
            if isinstance(q, str) and q.strip():
                examples.append({
                    "text": q.strip(),
                    "label": "allow",
                    "category": "ambiguous_allow",
                    "source": "synthetic",
                    "vertical": policy.vertical,
                    "topic": ca.topic,
                })

        for q in data.get("deny", []):
            if isinstance(q, str) and q.strip():
                examples.append({
                    "text": q.strip(),
                    "label": "deny",
                    "category": "ambiguous_deny",
                    "source": "synthetic",
                    "vertical": policy.vertical,
                    "topic": ca.topic,
                })

        logger.info(
            "Generated %d ambiguous for: %s (%d allow, %d deny)",
            len(data.get("allow", [])) + len(data.get("deny", [])),
            ca.topic,
            len(data.get("allow", [])),
            len(data.get("deny", [])),
        )
        time.sleep(0.5)

    return examples


def generate_adversarial(policy: Policy, provider: str, count: int = 100) -> list[dict]:
    """Generate adversarial cloaking queries."""
    template = load_template("adversarial_cloaking.txt")

    prompt = template.format(
        count=count,
        vertical=policy.spec.display_name,
        excluded_topics=", ".join(policy.spec.scope.hard_exclusions[:6]),
        core_topics=", ".join(policy.spec.scope.core_topics[:6]),
    )

    try:
        response = call_llm(prompt, provider)
        queries = parse_json_response(response)
    except Exception as e:
        logger.warning("Failed to generate adversarial queries: %s", e)
        return []

    examples = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            examples.append({
                "text": q.strip(),
                "label": "deny",
                "category": "adversarial",
                "source": "synthetic",
                "vertical": policy.vertical,
                "topic": "adversarial",
            })

    logger.info("Generated %d adversarial queries", len(examples))
    return examples


def generate_abstain_explicit(policy: Policy, provider: str, count: int = 100) -> list[dict]:
    """Generate genuinely ambiguous queries that should trigger ABSTAIN."""
    prompt = (
        f"Generate {count} user queries that are genuinely ambiguous — a human "
        f"reader cannot confidently determine if they belong in {policy.spec.display_name} "
        f"or not without asking a follow-up question.\n\n"
        f"The {policy.spec.display_name} chatbot covers: {', '.join(policy.spec.scope.core_topics[:8])}\n\n"
        f"Examples of ambiguous queries:\n"
        f"- 'What about the deductible?' (insurance? auto repair?)\n"
        f"- 'Can you help with my claim?' (insurance claim? legal claim? mining claim?)\n"
        f"- 'I need help with interest' (financial interest? academic interest?)\n\n"
        f"Requirements:\n"
        f"- Each query must be genuinely ambiguous without more context\n"
        f"- Include single-word or very short queries\n"
        f"- Include queries that use polysemous words\n"
        f"- 1-2 sentences max\n\n"
        f"Output as a JSON array of strings."
    )

    try:
        response = call_llm(prompt, provider)
        queries = parse_json_response(response)
    except Exception as e:
        logger.warning("Failed to generate abstain queries: %s", e)
        return []

    examples = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            examples.append({
                "text": q.strip(),
                "label": "abstain",
                "category": "ambiguous_explicit",
                "source": "synthetic",
                "vertical": policy.vertical,
                "topic": "ambiguous",
            })

    logger.info("Generated %d explicit abstain queries", len(examples))
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data from a policy")
    parser.add_argument("--policy", required=True, help="Path to policy.json")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--count", type=int, default=2000, help="Target total examples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = Policy.from_file(args.policy)
    logger.info("Loaded policy: %s v%s", policy.vertical, policy.version)

    # Allocate counts: ~40% positive, ~35% negative, ~15% abstain, ~10% adversarial
    target = args.count
    per_topic_positive = max(10, target * 40 // (100 * len(policy.spec.scope.core_topics)))
    per_exclusion_negative = max(10, target * 35 // (100 * max(1, len(policy.spec.scope.hard_exclusions))))
    per_conditional = max(10, target * 15 // (100 * max(1, len(policy.spec.scope.conditional_allow))))
    adversarial_count = max(20, target * 10 // 100)
    abstain_count = max(20, target * 15 // 100)

    all_examples = []

    logger.info("Generating positive queries (%d per topic)...", per_topic_positive)
    all_examples.extend(generate_positive(policy, args.provider, per_topic_positive))

    logger.info("Generating hard negatives (%d per exclusion)...", per_exclusion_negative)
    all_examples.extend(generate_hard_negatives(policy, args.provider, per_exclusion_negative))

    logger.info("Generating ambiguous queries (%d per conditional)...", per_conditional)
    all_examples.extend(generate_ambiguous(policy, args.provider, per_conditional))

    logger.info("Generating adversarial queries (%d)...", adversarial_count)
    all_examples.extend(generate_adversarial(policy, args.provider, adversarial_count))

    logger.info("Generating explicit abstain queries (%d)...", abstain_count)
    all_examples.extend(generate_abstain_explicit(policy, args.provider, abstain_count))

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Summary
    from collections import Counter
    label_counts = Counter(ex["label"] for ex in all_examples)
    category_counts = Counter(ex["category"] for ex in all_examples)

    logger.info("Generated %d total examples", len(all_examples))
    logger.info("By label: %s", dict(label_counts))
    logger.info("By category: %s", dict(category_counts))
    logger.info("Written to %s", output_path)


if __name__ == "__main__":
    main()
