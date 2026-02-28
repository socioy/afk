"""
---
name: Content Moderator — Tools
description: Tool definitions and simulated data for the content moderation system.
tags: [tools]
---
---
All tool definitions, argument schemas, and simulated data for the content moderator.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


# ===========================================================================
# Simulated content databases
# ===========================================================================

CONTENT_DB: list[dict] = []
FLAGGED_CONTENT: list[dict] = []
REJECTED_CONTENT: list[dict] = []
PUBLISHED_CONTENT: list[dict] = []

SENSITIVE_KEYWORDS: list[str] = [
    "violence", "hate", "harassment", "threat", "spam",
    "scam", "phishing", "explicit", "dangerous", "illegal",
]

FLAGGABLE_KEYWORDS: list[str] = [
    "controversial", "political", "medical advice", "financial advice",
    "gambling", "supplement", "miracle", "cure", "guaranteed",
]


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class AnalyzeContentArgs(BaseModel):
    content: str = Field(description="The text content of the post to analyze")
    author: str = Field(description="The username of the post author")


class PublishPostArgs(BaseModel):
    content: str = Field(description="The content to publish")
    author: str = Field(description="The author of the post")
    category: str = Field(description="Content category: general, news, opinion, creative")


class FlagContentArgs(BaseModel):
    content: str = Field(description="The content being flagged")
    author: str = Field(description="The author of the post")
    reason: str = Field(description="Why this content is being flagged for review")


class RejectContentArgs(BaseModel):
    content: str = Field(description="The content being rejected")
    author: str = Field(description="The author of the post")
    reason: str = Field(description="Why this content is being rejected")


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=AnalyzeContentArgs, name="analyze_content", description="Analyze a post's content for policy violations, tone, and category. Returns a detailed analysis report.")
def analyze_content(args: AnalyzeContentArgs) -> str:
    content_lower = args.content.lower()
    found_sensitive = [kw for kw in SENSITIVE_KEYWORDS if kw in content_lower]
    found_flaggable = [kw for kw in FLAGGABLE_KEYWORDS if kw in content_lower]

    word_count = len(args.content.split())
    char_count = len(args.content)
    has_urls = "http://" in content_lower or "https://" in content_lower
    all_caps_ratio = sum(1 for c in args.content if c.isupper()) / max(char_count, 1)

    status = "clean"
    if found_sensitive:
        status = "reject"
    elif found_flaggable:
        status = "flag"
    elif all_caps_ratio > 0.7 and word_count > 5:
        status = "flag"
    elif has_urls and word_count < 10:
        status = "flag"

    report_lines = [
        f"Content Analysis Report",
        f"{'=' * 40}",
        f"Author: {args.author}",
        f"Word count: {word_count}",
        f"Character count: {char_count}",
        f"Contains URLs: {'yes' if has_urls else 'no'}",
        f"ALL-CAPS ratio: {all_caps_ratio:.0%}",
        f"Sensitive keywords found: {', '.join(found_sensitive) if found_sensitive else 'none'}",
        f"Flaggable keywords found: {', '.join(found_flaggable) if found_flaggable else 'none'}",
        f"",
        f"Recommendation: {status.upper()}",
    ]

    if status == "reject":
        report_lines.append(f"Reason: Contains policy-violating content ({', '.join(found_sensitive)})")
    elif status == "flag":
        reasons = []
        if found_flaggable:
            reasons.append(f"sensitive topics ({', '.join(found_flaggable)})")
        if all_caps_ratio > 0.7:
            reasons.append("excessive capitalization")
        if has_urls and word_count < 10:
            reasons.append("short post with URLs (possible spam)")
        report_lines.append(f"Reason: Requires review — {', '.join(reasons)}")
    else:
        report_lines.append("Reason: Content appears safe for publication")

    return "\n".join(report_lines)


@tool(args_model=PublishPostArgs, name="publish_post", description="Publish a post to the content feed. Gated by PolicyEngine and PolicyRole — may be denied.")
def publish_post(args: PublishPostArgs) -> str:
    post = {"content": args.content, "author": args.author, "category": args.category, "status": "published"}
    PUBLISHED_CONTENT.append(post)
    CONTENT_DB.append(post)
    return (
        f"Post published successfully!\n"
        f"Author: {args.author}\n"
        f"Category: {args.category}\n"
        f"Content preview: {args.content[:100]}{'...' if len(args.content) > 100 else ''}\n"
        f"Total published posts: {len(PUBLISHED_CONTENT)}"
    )


@tool(args_model=FlagContentArgs, name="flag_content", description="Flag content for human moderator review. Use when content is borderline.")
def flag_content(args: FlagContentArgs) -> str:
    entry = {"content": args.content, "author": args.author, "reason": args.reason, "status": "flagged"}
    FLAGGED_CONTENT.append(entry)
    CONTENT_DB.append(entry)
    return f"Content flagged for review.\nAuthor: {args.author}\nReason: {args.reason}\nFlagged posts in queue: {len(FLAGGED_CONTENT)}"


@tool(args_model=RejectContentArgs, name="reject_content", description="Reject content that clearly violates content policies.")
def reject_content(args: RejectContentArgs) -> str:
    entry = {"content": args.content, "author": args.author, "reason": args.reason, "status": "rejected"}
    REJECTED_CONTENT.append(entry)
    CONTENT_DB.append(entry)
    return f"Content rejected.\nAuthor: {args.author}\nReason: {args.reason}\nTotal rejected posts: {len(REJECTED_CONTENT)}"
