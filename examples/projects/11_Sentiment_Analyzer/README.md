
# Sentiment Analyzer

A conversational text analysis agent built on afk using ChatAgent. The agent can analyze sentiment, count words, extract keywords, and summarize text through a multi-turn conversation.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/11_Sentiment_Analyzer

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/11_Sentiment_Analyzer

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/11_Sentiment_Analyzer

Expected interaction
User: Analyze "I absolutely love this product, it's amazing and wonderful!"
Agent: The sentiment is positive (confidence: 0.85). The text has 10 words...

The agent maintains conversation history so you can ask follow-up questions about previously analyzed text.

