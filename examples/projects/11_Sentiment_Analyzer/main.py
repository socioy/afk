"""
---
name: Sentiment Analyzer
description: A conversational text analysis agent that can analyze sentiment, count words, extract keywords, and summarize text.
tags: [chatagent, runner, tools, conversation]
---
---
This example introduces ChatAgent - a specialized agent type designed for conversational interactions.
Unlike the base Agent which is task-oriented (single request -> single response), ChatAgent is built
for sustained dialogue with multi-turn memory. Here we combine it with four text analysis tools to
build a practical conversational text analyzer.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic is used to define structured argument models for tools. This lets you specify exactly what inputs each tool expects, with types, descriptions, and validation built in.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import ChatAgent  # <- ChatAgent is for conversational interactions, vs Agent for task-based ones. It requires a user_message on every call, making it ideal for multi-turn dialogue where context builds over time.
from afk.tools import tool  # <- The @tool decorator turns a plain Python function into a tool that an agent can call. You give it a name, description, and an args_model so the LLM knows when and how to use it.


# --- Tool argument schema ---

class TextInput(BaseModel):  # <- A Pydantic model that defines the arguments for our text analysis tools. All four tools share the same schema: a single text string to analyze. Using a shared model avoids duplication.
    text: str = Field(description="The text to analyze")  # <- Field lets you attach metadata like descriptions so the LLM understands what each argument means.


# --- Sentiment word lists (keyword-based heuristics) ---

POSITIVE_WORDS = {  # <- A set of words commonly associated with positive sentiment. This is a simple heuristic approach - production systems would use ML models instead.
    "love", "great", "amazing", "wonderful", "fantastic", "excellent", "good",
    "happy", "joy", "beautiful", "best", "brilliant", "awesome", "perfect",
    "outstanding", "superb", "delightful", "impressive", "enjoy", "pleased",
    "glad", "nice", "terrific", "marvelous", "positive", "incredible",
    "fabulous", "magnificent", "thrilled", "satisfied", "cheerful", "excited",
}

NEGATIVE_WORDS = {  # <- A matching set for negative sentiment. The overlap between these two sets determines the confidence score.
    "hate", "terrible", "awful", "horrible", "bad", "worst", "ugly", "sad",
    "angry", "disgusting", "dreadful", "poor", "annoying", "disappointing",
    "frustrating", "miserable", "painful", "unpleasant", "boring", "mediocre",
    "unhappy", "upset", "furious", "dislike", "pathetic", "nasty", "atrocious",
    "depressing", "lousy", "inferior", "failed", "regret",
}


# --- Tool definitions ---

@tool(args_model=TextInput, name="analyze_sentiment", description="Analyze the sentiment of a given text. Returns whether the text is positive, negative, or neutral along with a confidence score.")  # <- Each @tool call registers a tool the agent can use. The name and description help the LLM decide which tool to call based on the user's message.
def analyze_sentiment(args: TextInput) -> str:
    words = args.text.lower().split()  # <- Tokenize by splitting on whitespace. Simple but effective for this heuristic approach.
    word_set = {word.strip(".,!?;:\"'()[]{}") for word in words}  # <- Strip punctuation from each word so "amazing!" matches "amazing" in our word lists.

    positive_matches = word_set & POSITIVE_WORDS  # <- Set intersection gives us all words in the text that appear in our positive word list.
    negative_matches = word_set & NEGATIVE_WORDS  # <- Same for negative words.

    pos_count = len(positive_matches)  # <- Count how many positive vs negative words we found.
    neg_count = len(negative_matches)
    total_sentiment_words = pos_count + neg_count

    if total_sentiment_words == 0:  # <- If no sentiment words were found, we can't make a confident judgment.
        return "Sentiment: neutral | Confidence: 0.50 | No strong sentiment words detected."

    if pos_count > neg_count:  # <- More positive than negative words means overall positive sentiment.
        label = "positive"
        confidence = round(pos_count / total_sentiment_words, 2)  # <- Confidence is the ratio of the dominant sentiment to total sentiment words. Higher ratio = more confident.
        detail = f"Positive words found: {', '.join(sorted(positive_matches))}"
    elif neg_count > pos_count:  # <- More negative words means overall negative sentiment.
        label = "negative"
        confidence = round(neg_count / total_sentiment_words, 2)
        detail = f"Negative words found: {', '.join(sorted(negative_matches))}"
    else:  # <- Equal counts means mixed/neutral sentiment.
        label = "neutral (mixed)"
        confidence = 0.50
        detail = f"Positive: {', '.join(sorted(positive_matches))} | Negative: {', '.join(sorted(negative_matches))}"

    return f"Sentiment: {label} | Confidence: {confidence} | {detail}"


@tool(args_model=TextInput, name="count_words", description="Count the number of words, sentences, and characters in the given text.")
def count_words(args: TextInput) -> str:
    text = args.text
    words = text.split()  # <- Split on whitespace to get word tokens.
    word_count = len(words)
    char_count = len(text)  # <- Total characters including spaces and punctuation.
    char_no_spaces = len(text.replace(" ", ""))  # <- Characters excluding spaces, useful for density analysis.
    sentence_count = sum(1 for ch in text if ch in ".!?") or 1  # <- Count sentence-ending punctuation marks. Default to 1 if none found (treat entire text as one sentence).
    avg_word_length = round(sum(len(w) for w in words) / max(word_count, 1), 1)  # <- Average word length can indicate text complexity. Guard against division by zero with max().

    return (
        f"Words: {word_count} | "
        f"Sentences: {sentence_count} | "
        f"Characters: {char_count} (without spaces: {char_no_spaces}) | "
        f"Avg word length: {avg_word_length}"
    )


@tool(args_model=TextInput, name="extract_keywords", description="Extract the most important keywords from the given text based on word frequency, filtering out common stop words.")
def extract_keywords(args: TextInput) -> str:
    stop_words = {  # <- Common English stop words that don't carry meaningful content. Filtering these out lets us focus on the words that actually matter.
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very", "just", "about",
        "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
        "they", "them", "this", "that", "these", "those", "what", "which",
        "who", "whom", "how", "when", "where", "why",
    }

    words = args.text.lower().split()
    cleaned = [word.strip(".,!?;:\"'()[]{}") for word in words]  # <- Strip punctuation so "amazing!" and "amazing" are treated as the same word.
    filtered = [w for w in cleaned if w and w not in stop_words]  # <- Remove empty strings and stop words to keep only meaningful content words.

    freq: dict[str, int] = {}  # <- Build a frequency map to find the most common meaningful words.
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1

    if not freq:  # <- Handle edge case where text is entirely stop words or empty.
        return "No significant keywords found in the text."

    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]  # <- Take the top 10 most frequent meaningful words as our keywords.
    keyword_list = [f"{word} ({count})" for word, count in sorted_keywords]

    return f"Top keywords: {', '.join(keyword_list)}"


@tool(args_model=TextInput, name="summarize_text", description="Generate a brief extractive summary of the given text by selecting the most important sentences.")
def summarize_text(args: TextInput) -> str:
    import re  # <- Import here since only this tool needs regex for sentence splitting.

    sentences = re.split(r'(?<=[.!?])\s+', args.text.strip())  # <- Split text into sentences using punctuation followed by whitespace as delimiters.
    sentences = [s.strip() for s in sentences if s.strip()]  # <- Clean up whitespace and remove empty strings.

    if len(sentences) <= 2:  # <- If the text is already very short, there's nothing to summarize.
        return f"Text is already brief ({len(sentences)} sentence(s)). Full text: {args.text}"

    # Score each sentence by counting meaningful (non-stop) words.
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "to", "of", "in", "for", "on", "with", "at",
        "by", "from", "as", "and", "but", "or", "not", "so", "it", "its",
        "this", "that", "i", "me", "my", "we", "you", "he", "she", "they",
    }

    scored = []  # <- We'll score each sentence by how many meaningful words it contains. Sentences with more content words are assumed to be more informative.
    for sentence in sentences:
        words = sentence.lower().split()
        meaningful = [w.strip(".,!?;:\"'()[]{}") for w in words if w.strip(".,!?;:\"'()[]{}") not in stop_words]
        scored.append((len(meaningful), sentence))

    scored.sort(key=lambda x: x[0], reverse=True)  # <- Sort by score (most meaningful words first).

    # Take the top sentences (up to 3) but preserve their original order in the text.
    top_count = min(3, len(scored))  # <- Limit summary to at most 3 sentences to keep it concise.
    top_sentences = {scored[i][1] for i in range(top_count)}
    summary = [s for s in sentences if s in top_sentences]  # <- Re-order selected sentences to match their original position in the text. This makes the summary read naturally.

    return f"Summary ({len(summary)} of {len(sentences)} sentences): {' '.join(summary)}"


# --- Agent setup ---

analyzer = ChatAgent(  # <- ChatAgent instead of Agent! ChatAgent requires a user_message on every call, enforcing the conversational pattern. Under the hood it extends Agent, so all the same features (tools, instructions, model) work exactly the same.
    name="sentiment-analyzer",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a text analysis assistant. You can analyze sentiment, count words,
    extract keywords, and summarize text. When the user provides text, analyze it
    using the appropriate tool(s) and explain the results conversationally.

    If the user asks for a full analysis, use all four tools. If they ask for
    something specific (e.g. "what's the sentiment?"), use only the relevant tool.

    Always be friendly, clear, and explain your findings in plain language.
    When presenting results, highlight the most interesting aspects of the analysis.
    """,  # <- Instructions that guide the agent's behavior. The agent will choose the right tool based on these instructions and the user's message.
    tools=[analyze_sentiment, count_words, extract_keywords, summarize_text],  # <- Pass all four tools to the agent. The LLM will automatically pick the right one (or several) based on what the user asks.
)
runner = Runner()

if __name__ == "__main__":
    print(
        "[sentiment-analyzer] > Hello! I'm your text analysis assistant. "
        "Paste any text and I can analyze its sentiment, count words, extract "
        "keywords, or summarize it. Type 'quit' to exit."
    )  # <- Welcome message to orient the user. Since ChatAgent is conversational, we set up a loop rather than a single exchange.

    thread_id = "sentiment-session"  # <- A thread ID groups messages into a single conversation. The Runner uses this to maintain context across turns so the agent remembers what was discussed earlier.

    while True:  # <- Conversation loop: ChatAgent shines in multi-turn interactions where context builds over time. This is the key difference from Agent, which is designed for single request-response exchanges.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.strip().lower() in ("quit", "exit", "q"):  # <- Let the user exit the conversation loop gracefully.
            print("[sentiment-analyzer] > Goodbye! Happy analyzing!")
            break

        response = runner.run_sync(
            analyzer, user_message=user_input, thread_id=thread_id
        )  # <- Run the agent synchronously using the Runner. We pass the user's input and the thread_id so the agent maintains conversation history across turns.

        print(
            f"[sentiment-analyzer] > {response.final_text}"
        )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.



"""
---
Tl;dr: This example creates a conversational text analysis agent using ChatAgent instead of Agent. ChatAgent is designed for multi-turn dialogue where context builds over time, while Agent is best for single request-response tasks. The agent has four tools: analyze_sentiment (keyword-based heuristic that scores text as positive/negative/neutral with a confidence score), count_words (counts words, sentences, and characters), extract_keywords (frequency-based keyword extraction with stop word filtering), and summarize_text (extractive summarization by selecting the most information-dense sentences). The conversation loop uses a thread_id to maintain memory across turns.
---
---
What's next?
- Try sending multiple texts in a row and asking the agent to compare them. Since ChatAgent maintains conversation history via thread_id, it can reference earlier analyses.
- Experiment with the sentiment word lists: add domain-specific words (e.g. financial terms like "bullish"/"bearish") to customize the analyzer for your use case.
- Swap ChatAgent for Agent and notice the difference: without the conversation loop pattern, the agent treats each message independently with no memory of previous turns.
- Check out the other examples in the library to see how to build multi-agent systems where a ChatAgent could delegate specialized analysis to sub-agents!
---
"""
