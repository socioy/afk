"""
---
name: Personal Wiki
description: A personal wiki agent that uses long-term memory with text search for storing and retrieving knowledge articles.
tags: [agent, runner, memory, long-term-memory, search, async]
---
---
This example demonstrates AFK's long-term memory system for storing and retrieving knowledge
articles. Unlike short-term state (get_state/put_state), long-term memory supports text search
to find relevant content by similarity. The agent acts as a personal wiki where you can save
articles on any topic and later search for them using natural language queries. This pattern
is essential for building knowledge bases, RAG systems, and agents with persistent knowledge.
---
"""

import asyncio  # <- Async required because memory operations are async.
import uuid  # <- For generating unique memory IDs.
from datetime import datetime, timezone  # <- For timestamping articles.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner executes agents with memory store integration.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool, ToolContext  # <- @tool decorator and ToolContext for accessing memory.
from afk.memory import InMemoryMemoryStore  # <- InMemoryMemoryStore with long-term memory support. Swap to SQLiteMemoryStore for persistence across restarts.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class SaveArticleArgs(BaseModel):  # <- Schema for saving a knowledge article.
    title: str = Field(description="Title of the article")
    content: str = Field(description="The full article content")
    tags: list[str] = Field(default_factory=list, description="Optional tags for categorization (e.g., ['python', 'tutorial'])")


class SearchArgs(BaseModel):  # <- Schema for searching the wiki.
    query: str = Field(description="Search query — a topic, keyword, or natural language question")
    limit: int = Field(default=3, description="Maximum number of results to return")


class ArticleIdArgs(BaseModel):  # <- Schema for retrieving a specific article.
    article_id: str = Field(description="The unique ID of the article to retrieve")


class EmptyArgs(BaseModel):
    pass


# ===========================================================================
# In-memory article index (supplements long-term memory with metadata)
# ===========================================================================

article_index: dict[str, dict] = {}  # <- Index for quick lookups by ID. Long-term memory handles search; this handles direct access.


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=SaveArticleArgs, name="save_article", description="Save a knowledge article to the wiki with searchable content")
async def save_article(args: SaveArticleArgs, ctx: ToolContext) -> str:  # <- ToolContext provides access to the memory store via ctx.memory.
    memory = ctx.memory  # <- Access the memory store from the tool context.
    thread_id = ctx.thread_id  # <- Thread ID scopes all memory operations.
    article_id = f"article-{uuid.uuid4().hex[:8]}"  # <- Generate a unique ID for this article.

    # --- Store in long-term memory for text search ---
    await memory.upsert_long_term_memory(  # <- upsert_long_term_memory stores content that can be searched later via text or vector search. It supports content, metadata, and optional embeddings.
        thread_id=thread_id,
        memory_id=article_id,  # <- Unique identifier for this memory entry. Using the same ID will update an existing entry.
        content=f"{args.title}\n\n{args.content}",  # <- The searchable content. Text search will match against this field.
        metadata={  # <- Metadata stored alongside the content. Not searched by default, but returned with results.
            "title": args.title,
            "tags": args.tags,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "wiki_article",
        },
    )

    # --- Also store in our quick-access index ---
    article_index[article_id] = {
        "id": article_id,
        "title": args.title,
        "content": args.content,
        "tags": args.tags,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return f"Article saved: '{args.title}' (ID: {article_id}). Tags: {', '.join(args.tags) if args.tags else 'none'}"


@tool(args_model=SearchArgs, name="search_wiki", description="Search the wiki for articles matching a query using text search")
async def search_wiki(args: SearchArgs, ctx: ToolContext) -> str:
    memory = ctx.memory
    thread_id = ctx.thread_id

    results = await memory.search_long_term_memory_text(  # <- search_long_term_memory_text performs text-based search against stored content. Returns a list of (LongTermMemory, score) tuples ranked by relevance.
        thread_id=thread_id,
        query=args.query,  # <- The search query. The store matches this against the content field.
        limit=args.limit,  # <- Maximum results to return.
    )

    if not results:
        return f"No articles found for query: '{args.query}'. Try different keywords or save some articles first!"

    lines = [f"Search results for '{args.query}' ({len(results)} found):"]
    for mem, score in results:  # <- Each result is a tuple of (LongTermMemory object, relevance score).
        title = mem.metadata.get("title", "Untitled") if mem.metadata else "Untitled"
        tags = mem.metadata.get("tags", []) if mem.metadata else []
        preview = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content  # <- Show a preview of the content.
        lines.append(
            f"\n  [{mem.memory_id}] {title} (score: {score:.2f})\n"
            f"    Tags: {', '.join(tags) if tags else 'none'}\n"
            f"    Preview: {preview}"
        )
    return "\n".join(lines)


@tool(args_model=ArticleIdArgs, name="get_article", description="Get the full content of a specific article by its ID")
async def get_article(args: ArticleIdArgs, ctx: ToolContext) -> str:
    article = article_index.get(args.article_id)
    if article is None:
        return f"Article '{args.article_id}' not found. Use search_wiki to find articles first."
    return (
        f"--- {article['title']} ---\n"
        f"ID: {article['id']}\n"
        f"Tags: {', '.join(article['tags']) if article['tags'] else 'none'}\n"
        f"Created: {article['created_at']}\n\n"
        f"{article['content']}"
    )


@tool(args_model=EmptyArgs, name="list_articles", description="List all articles in the wiki with their titles and IDs")
async def list_articles(args: EmptyArgs, ctx: ToolContext) -> str:
    if not article_index:
        return "The wiki is empty. Save some articles to get started!"
    lines = ["Articles in your wiki:"]
    for aid, article in article_index.items():
        tags = ", ".join(article["tags"]) if article["tags"] else "none"
        lines.append(f"  [{aid}] {article['title']} (tags: {tags})")
    return "\n".join(lines)


# ===========================================================================
# Agent and runner setup
# ===========================================================================

wiki_agent = Agent(
    name="personal-wiki",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a personal wiki assistant. You help users build and search their knowledge base.

    When the user wants to save information:
    1. Use save_article with a clear title, the content, and relevant tags.

    When the user wants to find something:
    1. Use search_wiki with their query to find relevant articles.
    2. If they want the full article, use get_article with the ID from search results.

    When the user wants an overview:
    1. Use list_articles to show everything in the wiki.

    Be helpful in suggesting tags and organizing knowledge. Encourage the user to save
    useful information for later retrieval.

    **NOTE**: Articles are searchable by their content — encourage detailed, keyword-rich articles!
    """,
    tools=[save_article, search_wiki, get_article, list_articles],
)

THREAD_ID = "wiki-main"  # <- Fixed thread_id for consistent memory scope.


async def main():
    memory = InMemoryMemoryStore()  # <- Create an InMemoryMemoryStore. For persistence, swap to SQLiteMemoryStore(db_path="wiki.db").
    await memory.setup()  # <- Initialize the store. Always call setup() before use.

    runner = Runner(memory_store=memory)  # <- Pass the memory store to the Runner. Tools access it via ToolContext.memory.

    print("Personal Wiki Agent (type 'quit' to exit)")
    print("=" * 45)
    print("Save articles, search your knowledge base, or list everything.\n")
    print("Try: 'Save an article about Python async/await'")
    print("     'Search for Python'")
    print("     'List all articles'\n")

    try:
        while True:
            user_input = input("[] > ")

            if user_input.strip().lower() in ("quit", "exit", "q"):
                print("Goodbye! Your wiki had", len(article_index), "articles.")
                break

            response = await runner.run(
                wiki_agent,
                user_message=user_input,
                thread_id=THREAD_ID,  # <- Same thread_id for all calls means all articles are in the same wiki scope.
            )

            print(f"[personal-wiki] > {response.final_text}\n")
    finally:
        await memory.close()  # <- Clean up the memory store.


if __name__ == "__main__":
    asyncio.run(main())



"""
---
Tl;dr: This example creates a personal wiki agent with long-term memory for storing and searching
knowledge articles. Articles are saved via upsert_long_term_memory and searched via
search_long_term_memory_text, which returns ranked results by relevance. Unlike short-term state
(get_state/put_state), long-term memory is designed for content that needs to be discoverable through
search. This pattern is the foundation for RAG systems, knowledge bases, and agents with persistent
searchable knowledge.
---
---
What's next?
- Try saving several articles on related topics and see how text search ranks them.
- Swap InMemoryMemoryStore for SQLiteMemoryStore to persist your wiki across restarts.
- Add an "update_article" tool that uses upsert_long_term_memory with an existing memory_id.
- Experiment with search_long_term_memory_vector for embedding-based semantic search (requires an embedding provider).
- Build a tag-based filtering system using metadata fields.
- Combine the wiki with a ChatAgent for a conversational knowledge assistant!
---
"""
