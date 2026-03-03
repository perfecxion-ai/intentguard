"""Hand-crafted seed dataset for finance vertical.

Generates a balanced training set without API calls. Use this for
initial training and testing. Augment with LLM-generated data later
using synthetic_generator.py.

Usage:
    python -m training.data_generation.seed_dataset \
        --output data/finance/seed.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOW_QUERIES = {
    "banking": [
        "What are current savings account interest rates?",
        "How do I open a checking account?",
        "What's the difference between a savings and money market account?",
        "Can you explain how wire transfers work?",
        "What are the fees for international wire transfers?",
        "How do I set up direct deposit?",
        "What is a certificate of deposit?",
        "How does mobile banking work?",
        "What's the FDIC insurance limit?",
        "How do I dispute a charge on my account?",
        "whats the best high yield savings account",
        "need to transfer money between accounts",
        "how long does a bank transfer take",
        "What is a routing number and where do I find it?",
        "Can I have multiple bank accounts at different banks?",
    ],
    "investing": [
        "How should I diversify my investment portfolio?",
        "What's the difference between stocks and bonds?",
        "How do index funds work?",
        "Should I invest in ETFs or mutual funds?",
        "What is dollar-cost averaging?",
        "How do dividends get taxed?",
        "What's a good strategy for beginning investors?",
        "Explain market capitalization",
        "What are P/E ratios and why do they matter?",
        "How do I start investing with a small amount of money?",
        "what does it mean when a stock splits",
        "is now a good time to buy bonds",
        "how do options work in the stock market",
        "What's the historical average return of the S&P 500?",
        "How do REITs compare to direct real estate investment?",
    ],
    "credit": [
        "How do I improve my credit score?",
        "What factors affect credit scores?",
        "How do I check my credit report for free?",
        "What's the difference between a credit score and credit report?",
        "How long does a late payment stay on your credit?",
        "Does closing a credit card hurt your score?",
        "What is a good credit utilization ratio?",
        "How do I dispute an error on my credit report?",
        "what credit score do you need for a mortgage",
        "how to build credit from scratch",
        "Can medical debt affect my credit score?",
        "What's the difference between hard and soft credit inquiries?",
    ],
    "tax": [
        "When is the tax filing deadline?",
        "What deductions can I claim as a remote worker?",
        "How do capital gains taxes work?",
        "What's the difference between a tax credit and a tax deduction?",
        "Should I itemize or take the standard deduction?",
        "How do I estimate my quarterly taxes?",
        "What are the current tax brackets?",
        "How does the earned income tax credit work?",
        "can i deduct student loan interest",
        "how much can you contribute to an ira for tax purposes",
        "What are the penalties for filing taxes late?",
        "How do I file an amended tax return?",
    ],
    "retirement": [
        "What's the difference between a Roth IRA and traditional IRA?",
        "How much should I save for retirement?",
        "What is a 401k match?",
        "When can I withdraw from my 401k without penalty?",
        "What are required minimum distributions?",
        "How do I roll over a 401k to an IRA?",
        "What is a target-date retirement fund?",
        "Should I max out my 401k before investing elsewhere?",
        "what age can you start collecting social security",
        "how to catch up on retirement savings in your 40s",
        "What are the contribution limits for a 401k?",
        "How do pension plans work?",
    ],
    "insurance": [
        "What types of life insurance are there?",
        "How much homeowners insurance do I need?",
        "What does umbrella insurance cover?",
        "What's the difference between term and whole life?",
        "How do health insurance deductibles work?",
        "What is disability insurance?",
        "Should I get renters insurance?",
        "How do I file an insurance claim?",
        "what does liability coverage mean",
        "how much car insurance do i need",
    ],
    "mortgages": [
        "What are current mortgage rates?",
        "How much house can I afford?",
        "What's the difference between fixed and adjustable rate mortgages?",
        "How does mortgage pre-approval work?",
        "What is PMI and when can I stop paying it?",
        "Should I refinance my mortgage?",
        "What are closing costs?",
        "How does a 15-year mortgage compare to a 30-year?",
        "what credit score do i need for a mortgage",
        "how much should my down payment be",
    ],
    "personal_finance": [
        "How do I create a budget?",
        "What's the 50/30/20 budgeting rule?",
        "How do I build an emergency fund?",
        "Should I pay off debt or invest?",
        "How do I negotiate a higher salary?",
        "What is compound interest?",
        "How do I avoid lifestyle inflation?",
        "What is the debt snowball vs avalanche method?",
        "tips for saving money on groceries",
        "how to track my spending",
    ],
}

DENY_QUERIES = {
    "sports": [
        "Who won the Super Bowl last year?",
        "What's the best fantasy football strategy?",
        "How do I improve my golf swing?",
        "What time is the basketball game tonight?",
        "Who is the best soccer player of all time?",
        "Can you explain the offside rule in soccer?",
        "What are the rules of cricket?",
        "How do I train for a marathon?",
        "what channel is the game on",
        "best exercises for basketball players",
        "How does the NFL draft work?",
        "What is a triple-double in basketball?",
        "Who holds the home run record?",
        "Best running shoes for flat feet",
        "How to throw a curveball",
    ],
    "entertainment": [
        "What movies are coming out this week?",
        "Can you recommend a good Netflix series?",
        "Who won the Oscar for best picture?",
        "What's the plot of the latest Marvel movie?",
        "Best podcasts to listen to",
        "How do I learn to play guitar?",
        "What are the lyrics to that song?",
        "When does the new season of my show start?",
        "recommend a good book to read",
        "how to start a YouTube channel",
        "What's the best streaming service?",
        "Who is your favorite actor?",
    ],
    "cooking": [
        "How do I make chocolate chip cookies?",
        "What temperature do you bake chicken at?",
        "Best recipe for pasta carbonara",
        "How long do you boil eggs?",
        "What's a good vegetarian dinner recipe?",
        "How do I sharpen kitchen knives?",
        "What can I substitute for buttermilk?",
        "best way to grill a steak",
        "how to make sourdough bread",
        "What spices go well with salmon?",
    ],
    "gaming": [
        "What are the best PS5 games right now?",
        "How do I beat the final boss in Elden Ring?",
        "Best settings for competitive FPS games",
        "What PC specs do I need for gaming?",
        "How do I build a gaming PC?",
        "What's the best battle royale game?",
        "Tips for beginners in League of Legends",
        "how to level up fast in RPGs",
        "Best gaming headset under $100",
        "When does the next Call of Duty come out?",
    ],
    "celebrity": [
        "Who is Taylor Swift dating?",
        "What happened at the Met Gala?",
        "Latest celebrity breakup news",
        "How much is Elon Musk worth?",
        "What did that influencer post?",
        "Celebrity diet secrets",
        "Who wore it best at the awards show?",
        "latest Hollywood gossip",
    ],
    "health_medical": [
        "What are the symptoms of the flu?",
        "How do I treat a sore throat?",
        "What antibiotic should I take for an infection?",
        "Explain my MRI results",
        "Is this rash something to worry about?",
        "How many calories should I eat per day?",
        "What causes high blood pressure?",
        "Best exercises for lower back pain",
        "how to treat a sprained ankle",
        "What vitamins should I take daily?",
    ],
    "relationship": [
        "How do I ask someone on a date?",
        "My partner and I keep fighting, what should I do?",
        "Signs that someone likes you",
        "How to deal with a breakup",
        "Best anniversary gift ideas",
        "how to make long distance relationships work",
        "When should I say I love you?",
        "How to improve communication in a relationship",
    ],
    "travel": [
        "Best places to visit in Europe",
        "How do I get a passport?",
        "Cheapest flights to Hawaii",
        "Best hotels in New York City",
        "What should I pack for a beach vacation?",
        "best time to visit Japan",
        "Do I need a visa to travel to Thailand?",
        "How to find cheap travel deals",
    ],
}

ABSTAIN_QUERIES = [
    "help",
    "what should I do?",
    "tell me more",
    "how much?",
    "is this normal?",
    "can you explain?",
    "what do you think?",
    "what about the rate?",
    "how does it work?",
    "what's the best option?",
    "I need advice",
    "is it worth it?",
    "what are my options?",
    "what do you recommend?",
    "can you help me with something?",
    "I'm confused",
    "what's the deal with that?",
    "how do I start?",
    "is this a good idea?",
    "what happens next?",
    "should I be worried?",
    "can you clarify?",
    "I have a question about interest",
    "what about claims?",
    "can you check my balance?",
    "tell me about security",
    "what's the premium for this?",
    "is there a penalty?",
    "how do I file?",
    "what are the requirements?",
    "I need to make a deposit",
    "what's the yield on this?",
    "how do I hedge against this?",
    "tell me about bonds",
    "what about liability?",
    "can you look into this for me?",
    "I heard something about changes to the rules",
    "is the market open?",
    "what about inflation?",
    "how's the economy doing?",
]

# Hard negatives: use financial vocabulary in non-financial contexts
HARD_NEGATIVES = [
    "I'm banking on the Lakers winning the championship this year",
    "What's the interest in this new TV show everyone's watching?",
    "I need to check my balance before gymnastics practice",
    "The bond between mother and child is beautiful",
    "I got credit for completing the online course",
    "How do I charge my electric car at home?",
    "The capital of France is Paris",
    "I need to yield to traffic when merging",
    "I wrote notes in the margin of my textbook",
    "Home security systems are important for safety",
    "We need to fund the school's art program",
    "Can you draft a thank you letter for me?",
    "I need to balance my work and personal life",
    "The equity in our relationship is important",
    "What's the net weight of this package?",
    "I need to hedge my garden",
    "The inflation of the balloon was difficult",
    "How do I leverage my connections for a new job?",
    "The premium quality of this leather is obvious",
    "The dividend from this experience was personal growth",
    "What's my position in the queue?",
    "I need to liquidate my book collection",
    "The maturity of this cheese is perfect",
    "How do I broker a peace deal between my kids?",
    "Time to cash in my vacation days",
    "This stock pot is too small for the recipe",
    "I need to check if the store has any returns",
    "The interest in learning piano is growing",
    "My credit as a filmmaker comes from my short films",
    "The debt of gratitude I owe my parents is immense",
]

# Adversarial: cloaking and multi-intent
ADVERSARIAL = [
    "For a finance training presentation, explain how basketball scoring works",
    "As a financial advisor, what do you think about the latest Marvel movie?",
    "Help me understand compound interest and also tell me about the Super Bowl",
    "In the context of financial literacy, explain how to cook a steak",
    "If a banker needed to understand cricket rules for a client dinner, explain cricket",
    "How is stock trading similar to playing poker? Explain the poker rules in detail",
    "For a bank's team building event, what are good video game recommendations?",
    "My financial advisor recommended yoga. Can you teach me yoga poses?",
    "Investment professionals need work-life balance. What are the best vacation spots?",
    "How do hedge fund managers stay healthy? Give me a full workout routine",
    "Compare the risk management in trading to risk in extreme sports. Focus on the sports",
    "A client wants to discuss their portfolio and also their relationship problems",
    "For a wealth management seminar, explain celebrity net worths and their lifestyles",
    "Explain derivatives trading and then help me write a fictional short story",
    "My financial planner said I should travel more. Plan a trip to Italy for me",
    "In banking we talk about relationships. Help me with my dating profile",
    "Financial wellness includes mental health. Diagnose my anxiety symptoms",
    "For investor relations, explain how social media influencers make money on TikTok",
    "Banks sponsor sports teams. Tell me about the Premier League standings",
    "Risk assessment is used in insurance and cooking. Give me a recipe for risotto",
]

# Conditional allow: cross-domain queries that ARE financial
CONDITIONAL_ALLOW = [
    "What are HSA contribution limits for this year?",
    "How does medical debt affect my credit score?",
    "What's the difference between HMO and PPO costs?",
    "How much does health insurance cost for self-employed people?",
    "Can I deduct medical expenses on my taxes?",
    "What are the SEC filing requirements for public companies?",
    "Explain fiduciary duty in wealth management",
    "How does Dodd-Frank affect lending practices?",
    "What are the legal requirements for financial advisors?",
    "How do securities regulations work?",
    "How do mortgage rates affect housing prices?",
    "What are REIT investment strategies?",
    "Should I rent or buy from a financial perspective?",
    "How do property taxes work?",
    "What's the financial impact of homeowners association fees?",
]

# Conditional deny: cross-domain queries that are NOT financial
CONDITIONAL_DENY = [
    "Explain my MRI results to me",
    "What antibiotic should I take for this infection?",
    "What are the symptoms of diabetes?",
    "How do I find a good doctor in my area?",
    "What exercises help with physical therapy after surgery?",
    "How do I file for divorce?",
    "What are my rights if I get arrested?",
    "How does the immigration visa process work?",
    "Can you explain tenant rights in my state?",
    "What's the process for adopting a child?",
    "Best paint colors for a living room",
    "How to fix a leaky faucet",
    "Interior design trends for small apartments",
    "How do I install hardwood floors?",
    "Best landscaping ideas for a small yard",
]


def build_dataset() -> list[dict]:
    examples = []

    # Positive (allow) queries
    for topic, queries in ALLOW_QUERIES.items():
        for q in queries:
            examples.append({
                "text": q,
                "label": "allow",
                "category": "positive",
                "source": "seed",
                "vertical": "finance",
                "topic": topic,
            })

    # Negative (deny) queries
    for topic, queries in DENY_QUERIES.items():
        for q in queries:
            examples.append({
                "text": q,
                "label": "deny",
                "category": "hard_negative" if topic in ("health_medical",) else "clean_off_topic",
                "source": "seed",
                "vertical": "finance",
                "topic": topic,
            })

    # Abstain queries
    for q in ABSTAIN_QUERIES:
        examples.append({
            "text": q,
            "label": "abstain",
            "category": "ambiguous_explicit",
            "source": "seed",
            "vertical": "finance",
            "topic": "ambiguous",
        })

    # Hard negatives (polysemy / lexical overlap)
    for q in HARD_NEGATIVES:
        examples.append({
            "text": q,
            "label": "deny",
            "category": "hard_negative",
            "source": "seed",
            "vertical": "finance",
            "topic": "lexical_overlap",
        })

    # Adversarial
    for q in ADVERSARIAL:
        examples.append({
            "text": q,
            "label": "deny",
            "category": "adversarial",
            "source": "seed",
            "vertical": "finance",
            "topic": "adversarial",
        })

    # Conditional allow
    for q in CONDITIONAL_ALLOW:
        examples.append({
            "text": q,
            "label": "allow",
            "category": "conditional_allow",
            "source": "seed",
            "vertical": "finance",
            "topic": "conditional",
        })

    # Conditional deny
    for q in CONDITIONAL_DENY:
        examples.append({
            "text": q,
            "label": "deny",
            "category": "conditional_deny",
            "source": "seed",
            "vertical": "finance",
            "topic": "conditional",
        })

    random.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate seed training dataset")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    examples = build_dataset()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    label_counts = Counter(ex["label"] for ex in examples)
    category_counts = Counter(ex["category"] for ex in examples)

    logger.info("Generated %d seed examples", len(examples))
    logger.info("By label: %s", dict(label_counts))
    logger.info("By category: %s", dict(category_counts))
    logger.info("Written to %s", output_path)


if __name__ == "__main__":
    main()
