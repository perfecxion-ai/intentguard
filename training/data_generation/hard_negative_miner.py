"""Hard negative miner.

Takes existing training data and generates adversarial variants:
- Keyword swaps: inject domain keywords into off-topic templates
- Semantic adjacency: topics that share vocabulary but differ in scope
- Augmentation: typos, fragments, multilingual mixing

Usage:
    python -m training.data_generation.hard_negative_miner \
        --input data/finance/synthetic.jsonl \
        --policy policies/finance.json \
        --output data/finance/hard_negatives.jsonl \
        --provider anthropic
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

from intentguard.policy import Policy
from training.data_generation.synthetic_generator import call_llm, parse_json_response

logger = logging.getLogger(__name__)


def load_examples(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def mine_keyword_swaps(
    positives: list[dict],
    policy: Policy,
    provider: str,
    count: int = 50,
) -> list[dict]:
    """Take on-topic queries and create off-topic variants that reuse domain keywords."""
    sample = random.sample(positives, min(20, len(positives)))
    sample_texts = [ex["text"] for ex in sample]
    excluded = ", ".join(policy.spec.scope.hard_exclusions[:6])

    prompt = (
        f"Here are {len(sample_texts)} on-topic {policy.spec.display_name} queries:\n\n"
        + "\n".join(f"- {t}" for t in sample_texts)
        + f"\n\nGenerate {count} NEW queries that:\n"
        f"1. Reuse financial keywords from the examples above\n"
        f"2. But are actually about off-topic subjects: {excluded}\n"
        f"3. Should fool a keyword-based filter but not a semantic one\n\n"
        f"Examples:\n"
        f"- Original: 'What are current interest rates?' → "
        f"'I have zero interest in this TV show's plot'\n"
        f"- Original: 'How do I manage my portfolio?' → "
        f"'How do I manage my fantasy football portfolio?'\n\n"
        f"Output as a JSON array of strings."
    )

    try:
        response = call_llm(prompt, provider)
        queries = parse_json_response(response)
    except Exception as e:
        logger.warning("Keyword swap mining failed: %s", e)
        return []

    results = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            results.append({
                "text": q.strip(),
                "label": "deny",
                "category": "keyword_swap",
                "source": "mined",
                "vertical": policy.vertical,
                "topic": "keyword_swap",
            })

    logger.info("Mined %d keyword swap negatives", len(results))
    return results


def mine_polysemy(policy: Policy, provider: str, count: int = 30) -> list[dict]:
    """Generate queries exploiting polysemous words (words with multiple meanings)."""
    prompt = (
        f"Generate {count} queries that use words with multiple meanings, where the "
        f"word has a {policy.spec.display_name} meaning but the query uses a "
        f"different meaning.\n\n"
        f"Polysemous words in {policy.spec.display_name}:\n"
        f"- 'interest' (financial vs curiosity)\n"
        f"- 'bond' (financial vs emotional)\n"
        f"- 'credit' (financial vs academic)\n"
        f"- 'charge' (fee vs electrical/legal)\n"
        f"- 'capital' (money vs city vs letter)\n"
        f"- 'yield' (return vs give way)\n"
        f"- 'margin' (trading vs edge)\n"
        f"- 'security' (financial vs safety)\n"
        f"- 'bull/bear' (market vs animal)\n"
        f"- 'fund' (investment vs to finance)\n"
        f"- 'check' (payment vs verification)\n"
        f"- 'balance' (account vs equilibrium)\n"
        f"- 'liability' (financial vs legal)\n"
        f"- 'premium' (insurance vs quality)\n"
        f"- 'dividend' (payout vs benefit)\n\n"
        f"Each query should clearly use the NON-financial meaning.\n"
        f"Output as a JSON array of strings."
    )

    try:
        response = call_llm(prompt, provider)
        queries = parse_json_response(response)
    except Exception as e:
        logger.warning("Polysemy mining failed: %s", e)
        return []

    results = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            results.append({
                "text": q.strip(),
                "label": "deny",
                "category": "polysemy",
                "source": "mined",
                "vertical": policy.vertical,
                "topic": "polysemy",
            })

    logger.info("Mined %d polysemy negatives", len(results))
    return results


def augment_production_style(
    examples: list[dict],
    count: int = 100,
) -> list[dict]:
    """Create production-like variants of existing examples: typos, fragments, slang."""
    if not examples:
        return []

    sample = random.sample(examples, min(count, len(examples)))
    results = []

    for ex in sample:
        text = ex["text"]
        variant = _add_noise(text)
        if variant != text:
            results.append({
                **ex,
                "text": variant,
                "category": ex["category"] + "_augmented",
                "source": "augmented",
            })

    logger.info("Augmented %d examples with production-style noise", len(results))
    return results


def _add_noise(text: str) -> str:
    """Add realistic noise to a query."""
    noise_type = random.choice(["typo", "fragment", "lowercase", "shorthand"])

    if noise_type == "typo" and len(text) > 10:
        # Swap two adjacent characters
        idx = random.randint(1, len(text) - 2)
        text = text[:idx] + text[idx + 1] + text[idx] + text[idx + 2:]

    elif noise_type == "fragment":
        # Truncate to a fragment
        words = text.split()
        if len(words) > 3:
            cut = random.randint(2, max(3, len(words) - 2))
            text = " ".join(words[:cut])
            # Remove trailing punctuation sometimes
            if random.random() > 0.5:
                text = text.rstrip("?.!")

    elif noise_type == "lowercase":
        text = text.lower()

    elif noise_type == "shorthand":
        replacements = {
            "what is": "whats",
            "What is": "whats",
            "how do I": "how do i",
            "I am": "im",
            "do not": "dont",
            "can not": "cant",
            "should I": "should i",
        }
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new, 1)
                break

    return text


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives from existing training data")
    parser.add_argument("--input", required=True, help="Input JSONL with existing examples")
    parser.add_argument("--policy", required=True, help="Path to policy.json")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = Policy.from_file(args.policy)
    existing = load_examples(Path(args.input))
    positives = [ex for ex in existing if ex["label"] == "allow"]
    logger.info("Loaded %d existing examples (%d positives)", len(existing), len(positives))

    mined = []

    logger.info("Mining keyword swaps...")
    mined.extend(mine_keyword_swaps(positives, policy, args.provider, count=50))
    time.sleep(1)

    logger.info("Mining polysemy negatives...")
    mined.extend(mine_polysemy(policy, args.provider, count=30))
    time.sleep(1)

    logger.info("Augmenting with production-style noise...")
    mined.extend(augment_production_style(existing, count=100))

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in mined:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    cat_counts = Counter(ex["category"] for ex in mined)
    logger.info("Mined %d total examples", len(mined))
    logger.info("By category: %s", dict(cat_counts))
    logger.info("Written to %s", output_path)


if __name__ == "__main__":
    main()
