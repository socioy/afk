"""
---
name: Shopping Assistant
description: A shopping assistant with subagents that demonstrates context_defaults and inherit_context_keys for passing context to child agents.
tags: [agent, runner, subagents, context-defaults, inherit-context-keys, tool-context]
---
---
This example demonstrates how context flows from a parent agent to its subagents using two
complementary features: context_defaults and inherit_context_keys. The parent coordinator
agent defines context_defaults — a dict of key-value pairs that are automatically merged into
the run context before every execution. Subagents declare inherit_context_keys — a list of
context key names they want to receive from their parent. When the runner executes a subagent,
it copies ONLY the keys listed in inherit_context_keys from the parent's context into the
subagent's context. This is a deliberate security/isolation pattern: subagents only see the
context they explicitly ask for, not the entire parent context. Tools can read these context
values via ToolContext.metadata. This pattern is ideal for multi-agent systems where shared
configuration (currency, locale, user preferences) needs to flow through the agent hierarchy
without passing it manually in every tool call.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner orchestrates agent execution and context propagation.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool, ToolContext  # <- @tool decorator and ToolContext for reading runtime context inside tools.


# ===========================================================================
# Simulated product catalog
# ===========================================================================

PRODUCT_CATALOG: list[dict] = [  # <- Simulated product database. In production this would be a real inventory system. Each product has prices in multiple currencies to demonstrate context-driven currency selection.
    {
        "id": "LAPTOP-001",
        "name": "ProBook Ultra 15",
        "category": "electronics",
        "prices": {"USD": 999.99, "EUR": 919.99, "GBP": 789.99, "JPY": 149999},
        "rating": 4.5,
        "tags": ["laptop", "ultrabook", "productivity"],
    },
    {
        "id": "PHONE-001",
        "name": "Galaxy Nova X",
        "category": "electronics",
        "prices": {"USD": 799.99, "EUR": 739.99, "GBP": 639.99, "JPY": 119999},
        "rating": 4.3,
        "tags": ["phone", "smartphone", "android"],
    },
    {
        "id": "HEADPHONES-001",
        "name": "SoundWave Pro ANC",
        "category": "electronics",
        "prices": {"USD": 349.99, "EUR": 319.99, "GBP": 279.99, "JPY": 52499},
        "rating": 4.7,
        "tags": ["headphones", "noise-cancelling", "wireless"],
    },
    {
        "id": "CHAIR-001",
        "name": "ErgoMax Office Chair",
        "category": "furniture",
        "prices": {"USD": 599.99, "EUR": 549.99, "GBP": 479.99, "JPY": 89999},
        "rating": 4.6,
        "tags": ["chair", "ergonomic", "office"],
    },
    {
        "id": "KEYBOARD-001",
        "name": "MechType 75 Wireless",
        "category": "electronics",
        "prices": {"USD": 159.99, "EUR": 147.99, "GBP": 127.99, "JPY": 23999},
        "rating": 4.4,
        "tags": ["keyboard", "mechanical", "wireless"],
    },
    {
        "id": "BACKPACK-001",
        "name": "TravelPro Carry-On Pack",
        "category": "accessories",
        "prices": {"USD": 89.99, "EUR": 82.99, "GBP": 71.99, "JPY": 13499},
        "rating": 4.2,
        "tags": ["backpack", "travel", "laptop"],
    },
    {
        "id": "MONITOR-001",
        "name": "UltraView 27 4K",
        "category": "electronics",
        "prices": {"USD": 449.99, "EUR": 414.99, "GBP": 359.99, "JPY": 67499},
        "rating": 4.8,
        "tags": ["monitor", "4k", "display"],
    },
    {
        "id": "DESK-001",
        "name": "StandUp Pro Adjustable Desk",
        "category": "furniture",
        "prices": {"USD": 749.99, "EUR": 689.99, "GBP": 599.99, "JPY": 112499},
        "rating": 4.5,
        "tags": ["desk", "standing-desk", "adjustable"],
    },
]

CURRENCY_SYMBOLS: dict[str, str] = {  # <- Currency display symbols.
    "USD": "$",
    "EUR": "\u20ac",
    "GBP": "\u00a3",
    "JPY": "\u00a5",
}


# ===========================================================================
# Product finder tools — used by the product_finder subagent
# ===========================================================================

class SearchProductsArgs(BaseModel):  # <- Schema for product search.
    query: str = Field(description="Search query — matches product name, category, or tags")


@tool(  # <- Product search tool. It reads the user's preferred currency from ToolContext.metadata to display prices in the right currency.
    args_model=SearchProductsArgs,
    name="search_products",
    description="Search the product catalog by name, category, or tags. Returns matching products with prices in the user's preferred currency.",
)
def search_products(args: SearchProductsArgs, ctx: ToolContext) -> str:  # <- The second parameter `ctx` is automatically injected by the runner. ToolContext.metadata contains the context values passed down from the parent agent's context_defaults (via inherit_context_keys).
    currency = ctx.metadata.get("currency", "USD")  # <- Read the currency from context. This was set by the parent's context_defaults and inherited by this subagent via inherit_context_keys. Falls back to USD if not set.
    user_pref = ctx.metadata.get("user_preference", "any")  # <- Read user preference (e.g., "budget", "premium", "any") from inherited context.
    symbol = CURRENCY_SYMBOLS.get(currency, currency)

    query_lower = args.query.lower()
    matches = []  # <- Filter products by matching query against name, category, and tags.

    for product in PRODUCT_CATALOG:
        name_match = query_lower in product["name"].lower()
        cat_match = query_lower in product["category"].lower()
        tag_match = any(query_lower in tag for tag in product["tags"])
        if name_match or cat_match or tag_match:
            matches.append(product)

    # --- Apply user preference filter ---
    if user_pref == "budget":  # <- Budget preference: only show products under the median price.
        if matches:
            price_vals = [p["prices"].get(currency, 0) for p in matches]
            median_price = sorted(price_vals)[len(price_vals) // 2]
            matches = [p for p in matches if p["prices"].get(currency, 0) <= median_price]
    elif user_pref == "premium":  # <- Premium preference: only show highly-rated products.
        matches = [p for p in matches if p["rating"] >= 4.5]

    if not matches:
        return f"No products found for '{args.query}' (currency: {currency}, preference: {user_pref})"

    lines = [f"Found {len(matches)} product(s) for '{args.query}' (showing {currency} prices):"]
    for p in matches:
        price = p["prices"].get(currency, "N/A")
        price_str = f"{symbol}{price:,.2f}" if isinstance(price, (int, float)) else str(price)
        lines.append(
            f"  [{p['id']}] {p['name']}\n"
            f"    Category: {p['category']} | Price: {price_str} | Rating: {p['rating']}/5\n"
            f"    Tags: {', '.join(p['tags'])}"
        )

    lines.append(f"\n(Currency: {currency} | Preference: {user_pref})")
    return "\n".join(lines)


class ProductDetailArgs(BaseModel):  # <- Schema for getting details about a specific product.
    product_id: str = Field(description="The product ID to look up (e.g., LAPTOP-001)")


@tool(  # <- Product detail tool. Also uses ToolContext for currency.
    args_model=ProductDetailArgs,
    name="get_product_details",
    description="Get detailed information about a specific product by its ID.",
)
def get_product_details(args: ProductDetailArgs, ctx: ToolContext) -> str:
    currency = ctx.metadata.get("currency", "USD")  # <- Same pattern: read currency from inherited context.
    symbol = CURRENCY_SYMBOLS.get(currency, currency)

    product = next((p for p in PRODUCT_CATALOG if p["id"] == args.product_id.upper()), None)
    if not product:
        return f"Product '{args.product_id}' not found. Check the ID and try again."

    price = product["prices"].get(currency, "N/A")
    price_str = f"{symbol}{price:,.2f}" if isinstance(price, (int, float)) else str(price)

    # --- Show prices in all currencies for comparison ---
    all_prices = []
    for curr, val in product["prices"].items():
        sym = CURRENCY_SYMBOLS.get(curr, curr)
        all_prices.append(f"  {curr}: {sym}{val:,.2f}")

    return (
        f"Product Details: {product['name']}\n"
        f"{'=' * 40}\n"
        f"ID: {product['id']}\n"
        f"Category: {product['category']}\n"
        f"Your price ({currency}): {price_str}\n"
        f"Rating: {product['rating']}/5\n"
        f"Tags: {', '.join(product['tags'])}\n"
        f"\nAll prices:\n" + "\n".join(all_prices)
    )


# ===========================================================================
# Price checker tools — used by the price_checker subagent
# ===========================================================================

class ComparePricesArgs(BaseModel):  # <- Schema for comparing prices across products.
    product_ids: list[str] = Field(description="List of product IDs to compare prices for")


@tool(  # <- Price comparison tool. Uses ToolContext for currency and preference.
    args_model=ComparePricesArgs,
    name="compare_prices",
    description="Compare prices of multiple products side by side in the user's preferred currency.",
)
def compare_prices(args: ComparePricesArgs, ctx: ToolContext) -> str:
    currency = ctx.metadata.get("currency", "USD")  # <- Read currency from inherited context.
    user_pref = ctx.metadata.get("user_preference", "any")  # <- Read preference for recommendation.
    symbol = CURRENCY_SYMBOLS.get(currency, currency)

    products = []
    for pid in args.product_ids:
        product = next((p for p in PRODUCT_CATALOG if p["id"] == pid.upper()), None)
        if product:
            products.append(product)

    if not products:
        return "No valid products found for comparison. Check the product IDs."

    lines = [f"Price Comparison ({currency}):", "=" * 40]
    for p in sorted(products, key=lambda x: x["prices"].get(currency, 0)):  # <- Sort by price ascending.
        price = p["prices"].get(currency, 0)
        price_str = f"{symbol}{price:,.2f}"
        lines.append(f"  {p['name']}: {price_str} (rating: {p['rating']}/5)")

    # --- Recommendation based on preference ---
    if user_pref == "budget":
        cheapest = min(products, key=lambda x: x["prices"].get(currency, 0))
        lines.append(f"\nBudget pick: {cheapest['name']} at {symbol}{cheapest['prices'].get(currency, 0):,.2f}")
    elif user_pref == "premium":
        best_rated = max(products, key=lambda x: x["rating"])
        lines.append(f"\nPremium pick: {best_rated['name']} (rating: {best_rated['rating']}/5)")
    else:
        best_value = max(products, key=lambda x: x["rating"] / max(x["prices"].get(currency, 1), 1))
        lines.append(f"\nBest value: {best_value['name']} (highest rating-to-price ratio)")

    lines.append(f"\n(Currency: {currency} | Preference: {user_pref})")
    return "\n".join(lines)


class FindDealsArgs(BaseModel):  # <- Schema for finding deals.
    category: str = Field(description="Product category to search for deals in: electronics, furniture, accessories")
    max_price: float = Field(description="Maximum price in the user's preferred currency")


@tool(  # <- Deal finder tool. Uses ToolContext for currency.
    args_model=FindDealsArgs,
    name="find_deals",
    description="Find products under a given price threshold in a specific category.",
)
def find_deals(args: FindDealsArgs, ctx: ToolContext) -> str:
    currency = ctx.metadata.get("currency", "USD")  # <- Read currency from inherited context.
    symbol = CURRENCY_SYMBOLS.get(currency, currency)

    matches = [
        p for p in PRODUCT_CATALOG
        if p["category"].lower() == args.category.lower()
        and p["prices"].get(currency, float("inf")) <= args.max_price
    ]

    if not matches:
        return f"No {args.category} products found under {symbol}{args.max_price:,.2f} ({currency})"

    matches.sort(key=lambda x: x["prices"].get(currency, 0))  # <- Sort by price ascending.

    lines = [f"Deals in '{args.category}' under {symbol}{args.max_price:,.2f} ({currency}):"]
    for p in matches:
        price = p["prices"].get(currency, 0)
        lines.append(f"  [{p['id']}] {p['name']} — {symbol}{price:,.2f} (rating: {p['rating']}/5)")

    return "\n".join(lines)


# ===========================================================================
# Subagents — each inherits specific context keys from the coordinator
# ===========================================================================

product_finder = Agent(  # <- The product finder subagent. It searches and retrieves product information.
    name="product-finder",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a product search specialist. Help users find products from our catalog.
    Use search_products to search by name, category, or tags.
    Use get_product_details to show full details for a specific product.

    Important: Product prices are shown in the user's preferred currency, which is
    automatically set from context. You don't need to ask the user for their currency.
    """,
    tools=[search_products, get_product_details],
    inherit_context_keys=["currency", "user_preference"],  # <- This subagent ONLY inherits "currency" and "user_preference" from its parent. It does NOT see other parent context keys (like "session_id" or "user_name"). This is intentional isolation — subagents only get what they need.
)

price_checker = Agent(  # <- The price checker subagent. It compares prices and finds deals.
    name="price-checker",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a price comparison specialist. Help users compare prices and find deals.
    Use compare_prices to show side-by-side comparisons of multiple products.
    Use find_deals to discover products under a specific price threshold in a category.

    Prices are automatically shown in the user's preferred currency from context.
    Tailor your recommendations based on the user's preference (budget, premium, or any).
    """,
    tools=[compare_prices, find_deals],
    inherit_context_keys=["currency", "user_preference"],  # <- Same inherited keys. Both subagents need currency and preference to do their job. They don't need session_id, user_name, or other coordinator-level context.
)


# ===========================================================================
# Coordinator agent — defines context_defaults that flow to subagents
# ===========================================================================

shopping_coordinator = Agent(
    name="shopping-assistant",  # <- The top-level agent the user interacts with.
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a friendly shopping assistant coordinator. Help users find and compare products
    from our catalog.

    You have two specialist teams:
    - product-finder: searches our catalog and shows product details
    - price-checker: compares prices and finds deals within a budget

    Route user requests to the appropriate specialist:
    - Product search/browse requests → product-finder
    - Price comparisons and deal hunting → price-checker

    The user's currency and shopping preference are set automatically from your context.
    You don't need to ask for them — they're passed to your subagents automatically.
    """,
    subagents=[product_finder, price_checker],  # <- Both subagents are registered. The coordinator decides which to invoke.
    context_defaults={  # <- These key-value pairs are merged into the run context BEFORE execution. They act as "default configuration" for this agent and all subagents that inherit them.
        "currency": "USD",  # <- Default currency. Subagents with inherit_context_keys=["currency"] will receive this.
        "user_preference": "any",  # <- Default shopping preference. "budget" shows cheap options, "premium" shows top-rated, "any" shows everything.
        "session_id": "shop-session-001",  # <- Session tracking. Note: subagents do NOT inherit this because it's not in their inherit_context_keys list. Only the coordinator sees it.
        "user_name": "Guest",  # <- User display name. Also NOT inherited by subagents — coordinator only.
    },
)

runner = Runner()  # <- A single Runner handles everything.


# ===========================================================================
# Main entry point — interactive conversation loop
# ===========================================================================

if __name__ == "__main__":
    print("Shopping Assistant (type 'quit' to exit)")
    print("=" * 50)
    print("Search products, compare prices, and find deals!")
    print("Default currency: USD | Preference: any")
    print("\nTry: 'Find me a laptop', 'Compare headphones and keyboard prices', 'Deals under $200'\n")

    while True:  # <- Conversation loop for the shopping interaction.
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Happy shopping! Goodbye!")
            break

        # --- You can override context_defaults at call time ---
        response = runner.run_sync(  # <- run_sync is the synchronous wrapper. Internally it creates an event loop and awaits runner.run().
            shopping_coordinator,
            user_message=user_input,
            context={  # <- This context is merged WITH context_defaults. Values here override defaults. For example, you could change currency to "EUR" at runtime.
                "currency": "USD",  # <- Using the default. Try changing to "EUR" or "GBP" to see prices change.
                "user_preference": "any",  # <- Try "budget" or "premium" to see filtered results.
            },
        )

        print(f"[shopping] > {response.final_text}\n")



"""
---
Tl;dr: This example creates a shopping assistant with a coordinator agent and two subagents
(product-finder, price-checker). The coordinator uses context_defaults to define shared
configuration (currency, user_preference, session_id, user_name) that is automatically merged
into the run context. Subagents use inherit_context_keys=["currency", "user_preference"] to
selectively inherit ONLY the keys they need — they never see session_id or user_name. Inside
tools, ToolContext.metadata provides access to these inherited values (e.g., ctx.metadata.get("currency")).
This pattern enables clean, secure context propagation through agent hierarchies without manual
parameter passing or exposing unnecessary data to subagents.
---
---
What's next?
- Try changing the currency in the run context to "EUR" or "JPY" to see all prices update automatically across both subagents.
- Set user_preference to "budget" to see the product finder filter out expensive items and the price checker recommend the cheapest option.
- Add a new context key (e.g., "region") to context_defaults and inherit it in a subagent to demonstrate region-specific product filtering.
- Create a third subagent (e.g., "review-checker") that inherits a different set of context keys to show selective inheritance in action.
- Experiment with NOT listing a key in inherit_context_keys to verify that subagents truly cannot see parent context they don't declare.
- Check out the PolicyEngine example to see how context values can also influence which tools are allowed or denied!
---
"""
