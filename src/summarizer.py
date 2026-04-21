import anthropic
import logging
from datetime import datetime
from .news_fetcher import Article

logger = logging.getLogger(__name__)

# ─── Stable system prompt — cached with a 1-hour TTL ──────────────────────────
# This stays constant across daily runs, making it a perfect cache candidate.
# At ~2500+ tokens it comfortably exceeds the Sonnet 4.6 minimum (2048 tokens).
SYSTEM_PROMPT = """You are a senior AI industry analyst and technology advisor with deep expertise
across the entire artificial intelligence landscape. Your mission is to produce a
concise, insightful, and highly actionable daily digest of the most important AI
developments, tailored for a professional who wants to stay ahead of AI trends
and immediately apply them in both their work and personal life.

## Your Core Responsibilities

1. ANALYZE — Read all provided articles and identify the 3-5 most significant
   developments. Significance is determined by real-world impact, strategic
   importance, novelty, and practical applicability — not just headline size.

2. ORGANIZE — Group findings by AI provider / topic category. If a section has
   no news, write "No major updates today." rather than skipping it entirely.

3. CONTEXTUALIZE — Go beyond summarizing the headlines. Explain WHY each
   development matters: what problem does it solve? What does it replace?
   Who benefits most?

4. ADVISE — For every piece of news you surface, think: "How can a busy
   professional actually use this today?" Provide specific, concrete, actionable
   suggestions — not vague platitudes.

## Output Structure (follow exactly)

Produce the digest using this exact markdown structure:

---
# 🤖 Daily AI Digest — {DATE}

## 🔥 Top Headlines
*(3-5 bullet points — the most impactful stories of the day, one sentence each)*

---

## 🏢 Provider Updates

### 🟣 Anthropic & Claude
*What's new from Anthropic and the Claude model family*

### 🟢 OpenAI & ChatGPT
*What's new from OpenAI, GPT-4, ChatGPT, Sora, and related products*

### 🔵 Google AI & Gemini
*What's new from Google DeepMind, Gemini, and Google AI research*

### 🟡 Meta AI
*What's new from Meta AI, LLaMA, and Meta's research labs*

### ⚪ Other Providers
*Mistral AI, VortexAI, Hugging Face, Cohere, Stability AI, and other providers*

### 📰 Industry News & Analysis
*Broader trends, funding rounds, regulatory developments, and market analysis*

---

## 💼 How to Use This for Your WORK

Provide 5-7 specific, actionable suggestions for professional use. For each:
- Name the specific AI tool or feature
- Describe the exact work scenario where it helps
- Give a concrete example ("Instead of spending 2 hours on X, you can now...")
- Include any relevant tips for getting started

Focus on categories like: productivity, writing, coding, data analysis,
presentations, research, customer communication, project management, learning.

---

## 🏠 How to Use This in Your PERSONAL LIFE

Provide 4-6 specific, actionable suggestions for personal use. For each:
- Name the specific AI tool or feature
- Describe the personal scenario (learning, creativity, health, finance, travel,
  relationships, hobbies)
- Give a concrete personal example

---

## ⚡ Quick Wins — Try These Today
*(3 things the reader can do RIGHT NOW, within the next hour, based on today's news.
Each should take under 30 minutes and deliver immediate value.)*

---

## Writing Guidelines

- Be direct and confident. Avoid hedging language like "might be useful" or
  "could potentially."
- Use second person ("you can...") for suggestions — make it personal.
- Quantify impact when possible: "saves 30 minutes per document," "reduces
  errors by 90%," etc.
- Highlight free vs. paid tools when relevant.
- For each "Quick Win," start with an action verb: "Open...", "Try...", "Test..."
- Assume the reader is technically comfortable but not an AI researcher.
- If VortexAI news appears in the articles, include it under Other Providers.
- Keep the total digest readable in under 10 minutes.
"""


def _format_articles(articles: list[Article]) -> str:
    """Format article list grouped by category for the user message."""
    if not articles:
        return "No new articles found in the past 24 hours."

    by_category: dict[str, list[Article]] = {}
    for a in articles:
        by_category.setdefault(a.category, []).append(a)

    lines: list[str] = []
    for category, arts in sorted(by_category.items()):
        lines.append(f"\n## {category}\n")
        for i, a in enumerate(arts, 1):
            date_str = (
                a.published.strftime("%Y-%m-%d %H:%M UTC")
                if a.published
                else "date unknown"
            )
            lines.append(f"{i}. **{a.title}**")
            lines.append(f"   Source: {a.source} | Published: {date_str}")
            lines.append(f"   URL: {a.url}")
            if a.summary:
                lines.append(f"   Preview: {a.summary}")
            lines.append("")

    return "\n".join(lines)


def generate_digest(articles: list[Article], api_key: str) -> str:
    """
    Call Claude claude-sonnet-4-6 with a cached system prompt to generate the digest.

    Caching strategy:
    - System prompt: stable across all daily runs → cached with 1h TTL.
      At ~2500 tokens it exceeds Sonnet 4.6's 2048-token minimum for caching.
    - User message (articles): changes every day → never cached.

    On the first request of the day, cache_creation_input_tokens will be non-zero.
    During development/testing (multiple runs within an hour), cache_read_input_tokens
    will show the savings.
    """
    client = anthropic.Anthropic(api_key=api_key)

    today = datetime.now().strftime("%A, %B %d, %Y")
    articles_text = _format_articles(articles)

    user_content = (
        f"Today is {today}.\n\n"
        f"Here are all AI news articles from the past 24 hours:\n\n"
        f"{articles_text}\n\n"
        f"Please generate the full daily digest using the exact structure "
        f"specified in your instructions. Replace {{DATE}} in the heading with {today}."
    )

    logger.info("Calling Claude claude-sonnet-4-6 to generate digest...")

    # Stream the response — output can be long and streaming prevents timeouts
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # 1h TTL: maximises cache reuse during development and testing.
                # For a once-daily production job the 5-min default would always
                # be cold; 1h keeps the cache warm within the same working session.
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        message = stream.get_final_message()

    # Log cache usage to help monitor cost savings
    usage = message.usage
    logger.info(
        "Token usage — input: %d, output: %d, cache_write: %d, cache_read: %d",
        usage.input_tokens,
        usage.output_tokens,
        getattr(usage, "cache_creation_input_tokens", 0),
        getattr(usage, "cache_read_input_tokens", 0),
    )

    return next(b.text for b in message.content if b.type == "text")
