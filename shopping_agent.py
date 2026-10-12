"""LangGraph shopping workflow using deterministic catalog-backed tools."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from llm_config import LLMUnavailableError, get_chat_model, redact_sensitive_text
from shopping_tools import (
    CatalogError,
    check_inventory,
    compare_products,
    estimate_delivery,
    load_catalog,
    search_products,
    validate_budget,
)


LOGGER = logging.getLogger(__name__)


class ShoppingState(TypedDict, total=False):
    original_query: str
    form_filters: dict[str, Any]
    preferences: dict[str, Any]
    matching_products: list[dict[str, Any]]
    inventory_results: list[dict[str, Any]]
    budget_results: list[dict[str, Any]]
    comparison: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    relaxed_preferences: list[str]
    explanation: str
    messages: list[str]
    error: str
    llm_status: str


def _parse_number(value: Any, minimum: float = 0, maximum: float = 10_000_000) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if minimum <= parsed <= maximum else None


def _rule_based_preferences(query: str, form_filters: dict[str, Any]) -> dict[str, Any]:
    """Extract optional filters while preserving explicit maximum-budget semantics."""
    text = query.casefold()
    catalog = load_catalog()
    categories = sorted(
        {str(category).strip() for category in catalog["category"] if str(category).strip()},
        key=len,
        reverse=True,
    )
    category = str(form_filters.get("category") or "").strip()
    if not category:
        category = next(
            (candidate for candidate in categories if candidate.casefold() in text),
            "",
        )
        if not category:
            for candidate in categories:
                singular = candidate.casefold().removesuffix("s")
                if singular and singular in text:
                    category = candidate
                    break

    budget = _parse_number(form_filters.get("budget"), 0, 10_000_000)
    budget_match = re.search(
        r"(?:under|below|within|budget(?:\s+of)?|less than|max(?:imum)?)\s*"
        r"(?:₹|rs\.?\s*)?([\d,]+(?:\.\d+)?)\s*(k)?",
        text,
    )
    if budget is None and budget_match:
        amount = float(budget_match.group(1).replace(",", ""))
        if budget_match.group(2):
            amount *= 1000
        budget = _parse_number(amount, 0, 10_000_000)

    ram_match = re.search(r"(\d+)\s*gb\s*(?:ram|memory)", text)
    if not ram_match:
        ram_match = re.search(r"(?:ram|memory)\s*(?:of|at least)?\s*(\d+)\s*gb", text)
    storage_match = re.search(r"(\d+)\s*(tb|gb)\s*(?:ssd|storage)", text)
    if not storage_match:
        storage_match = re.search(r"(?:ssd|storage)\s*(?:of|at least)?\s*(\d+)\s*(tb|gb)", text)
    battery_match = re.search(r"(\d+)\s*(?:hours?|hrs?)\s*(?:of\s*)?(?:battery|life)", text)
    if not battery_match:
        battery_match = re.search(r"(?:battery|life)\s*(?:of|at least)?\s*(\d+)\s*(?:hours?|hrs?)", text)

    min_ram = _parse_number(form_filters.get("min_ram_gb"), 0, 1024)
    if min_ram is None and ram_match:
        min_ram = _parse_number(ram_match.group(1), 0, 1024)
    min_storage = _parse_number(form_filters.get("min_storage_gb"), 0, 100_000)
    if min_storage is None and storage_match:
        amount = float(storage_match.group(1))
        unit = storage_match.group(2).casefold()
        if unit == "tb":
            amount *= 1024
        min_storage = _parse_number(amount, 0, 100_000)
    min_battery = _parse_number(form_filters.get("min_battery_hours"), 0, 100)
    if min_battery is None and battery_match:
        min_battery = _parse_number(battery_match.group(1), 0, 100)

    processor = str(form_filters.get("processor") or "").strip()
    if not processor:
        processor_match = re.search(
            r"\b(intel core i[3579](?:-\w+)?|amd ryzen [3579](?:-\w+)?|apple m[1-9])\b",
            query,
            re.IGNORECASE,
        )
        processor = processor_match.group(0) if processor_match else ""

    use_case = "general"
    if any(term in text for term in ("machine learning", "python", "programming", "coding", "developer")):
        use_case = "development"
    elif any(term in text for term in ("gaming", "games", "video editing", "3d")):
        use_case = "performance"
    elif any(term in text for term in ("online class", "online classes", "study", "office")):
        use_case = "productivity"

    return {
        "category": category,
        "budget": budget,
        "min_ram_gb": int(min_ram) if min_ram is not None else None,
        "min_storage_gb": int(min_storage) if min_storage is not None else None,
        "processor": processor,
        "min_battery_hours": int(min_battery) if min_battery is not None else None,
        "use_case": use_case,
    }


def _llm_preferences(query: str) -> dict[str, Any] | None:
    """Optionally extract filters; numeric budget is independently validated."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract only explicit shopping preferences as JSON keys category, "
                "budget, min_ram_gb, min_storage_gb, processor, min_battery_hours. "
                "Use null for absent values. Do not infer product facts. Budget is "
                "a maximum in the customer's currency. Output JSON only.",
            ),
            ("human", "{query}"),
        ]
    )
    output = (prompt | get_chat_model() | StrOutputParser()).invoke(
        {"query": redact_sensitive_text(query)}
    )
    parsed = json.loads(output.strip().removeprefix("```json").removesuffix("```").strip())
    if not isinstance(parsed, dict):
        raise ValueError("Requirement extraction did not return an object.")
    return parsed


def _analyze_requirements(state: ShoppingState) -> dict[str, Any]:
    query = state.get("original_query", "").strip()
    filters = state.get("form_filters", {})
    messages = list(state.get("messages", []))
    try:
        preferences = _rule_based_preferences(query, filters)
        try:
            llm_preferences = _llm_preferences(query)
            if not isinstance(llm_preferences, dict):
                raise ValueError("Requirement extraction did not return an object.")
            for field in (
                "category",
                "min_ram_gb",
                "min_storage_gb",
                "processor",
                "min_battery_hours",
            ):
                if not preferences.get(field) and llm_preferences.get(field) is not None:
                    preferences[field] = llm_preferences[field]
            # The rule parser/user input is the only source allowed to set a budget.
            messages.append("Optional AI requirement extraction was used.")
            status = "AI-assisted preferences"
        except LLMUnavailableError:
            messages.append("Using deterministic requirement parsing (no LLM API key configured).")
            status = "Deterministic mode"
        except Exception as error:
            LOGGER.warning("Optional requirement extraction failed: %s", type(error).__name__)
            messages.append("AI requirement extraction was unavailable; using deterministic parsing.")
            status = "Deterministic fallback"

        preferences["budget"] = _parse_number(preferences.get("budget"), 0, 10_000_000)
        for field, maximum in (
            ("min_ram_gb", 1024),
            ("min_storage_gb", 100_000),
            ("min_battery_hours", 100),
        ):
            number = _parse_number(preferences.get(field), 0, maximum)
            preferences[field] = int(number) if number is not None else None
        preferences["category"] = str(preferences.get("category") or "").strip()
        known_categories = {
            str(category).casefold(): str(category)
            for category in load_catalog()["category"].dropna().unique()
        }
        preferences["category"] = known_categories.get(
            preferences["category"].casefold(),
            "",
        )
        preferences["processor"] = str(preferences.get("processor") or "").strip()
        if len(preferences["processor"]) > 80:
            preferences["processor"] = preferences["processor"][:80]
        return {"preferences": preferences, "messages": messages, "llm_status": status}
    except Exception as error:
        LOGGER.exception("Could not parse shopping requirements.")
        return {"error": str(error), "messages": messages}


def _search_products(state: ShoppingState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    preferences = dict(state.get("preferences", {}))
    search_preferences = dict(preferences)
    messages = list(state.get("messages", []))
    try:
        matches = search_products(search_preferences)
        relaxed_fields: list[str] = []
        # Budget and selected category are hard constraints. Technical preferences
        # are treated as relaxable when the strict catalog search has no results.
        for field in ("min_battery_hours", "processor", "min_storage_gb", "min_ram_gb"):
            if matches:
                break
            if search_preferences.get(field) not in (None, ""):
                search_preferences[field] = None
                relaxed_fields.append(field.replace("_", " "))
                matches = search_products(search_preferences)
        if relaxed_fields:
            messages.append(
                "No exact match was available. Relaxed these optional specifications: "
                + ", ".join(relaxed_fields)
                + ". Category and maximum budget were kept."
            )
        if not matches:
            messages.append(
                "No catalog products match the selected category and maximum budget."
            )
        return {
            "matching_products": matches,
            "relaxed_preferences": relaxed_fields,
            "messages": messages,
        }
    except Exception as error:
        LOGGER.exception("Product search failed.")
        return {"error": "Product search is temporarily unavailable.", "messages": messages}


def _validate_inventory(state: ShoppingState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    matches = list(state.get("matching_products", []))
    preferences = state.get("preferences", {})
    messages = list(state.get("messages", []))
    try:
        inventory = check_inventory([product["product_id"] for product in matches])
        stocked_matches = [
            product for product in matches
            if product["stock_quantity"] > 0
        ]
        if not stocked_matches and preferences.get("category"):
            # Offer other in-stock items in the same hard category/budget scope,
            # but retain the original search findings and explain this relaxation.
            alternatives = search_products(
                {"category": preferences["category"], "budget": preferences.get("budget")}
            )
            stocked_matches = [product for product in alternatives if product["stock_quantity"] > 0]
            if stocked_matches:
                messages.append(
                    "The closest exact matches are out of stock. Showing in-stock "
                    "alternatives in the same category and budget."
                )
                matches = stocked_matches
                inventory = check_inventory([product["product_id"] for product in matches])
        return {
            "matching_products": matches,
            "inventory_results": inventory,
            "messages": messages,
        }
    except Exception:
        LOGGER.exception("Inventory validation failed.")
        return {"error": "Inventory could not be validated.", "messages": messages}


def _validate_budget(state: ShoppingState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    products = state.get("matching_products", [])
    preferences = state.get("preferences", {})
    try:
        results = validate_budget(
            [product["product_id"] for product in products],
            preferences.get("budget"),
        )
        budget_by_id = {item["product_id"]: item for item in results}
        eligible = [
            product for product in products
            if budget_by_id.get(product["product_id"], {}).get("within_budget")
            and product["stock_quantity"] > 0
        ]
        return {
            "budget_results": results,
            "matching_products": eligible,
        }
    except Exception:
        LOGGER.exception("Budget validation failed.")
        return {"error": "Budget validation failed.", "messages": state.get("messages", [])}


def _compare_products(state: ShoppingState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    products = state.get("matching_products", [])
    preferences = state.get("preferences", {})
    try:
        top_candidates = products[:5]
        ids = [product["product_id"] for product in top_candidates]
        comparison = compare_products(ids, budget=preferences.get("budget"))
        delivery_by_id = {
            item["product_id"]: item for item in estimate_delivery(ids)
        }
        for product in comparison:
            product["estimated_delivery_days"] = delivery_by_id[
                product["product_id"]
            ]["delivery_days"]
        return {"comparison": comparison}
    except Exception:
        LOGGER.exception("Product comparison failed.")
        return {"error": "Product comparison is temporarily unavailable."}


def _score_product(product: dict[str, Any], preferences: dict[str, Any]) -> float:
    score = 0.0
    budget = preferences.get("budget")
    score += 35 if product.get("within_budget", True) else 0
    if preferences.get("min_ram_gb"):
        score += 15 if (product.get("ram_gb") or 0) >= preferences["min_ram_gb"] else 0
    if preferences.get("min_storage_gb"):
        score += 15 if (product.get("storage_gb") or 0) >= preferences["min_storage_gb"] else 0
    if preferences.get("min_battery_hours"):
        score += 10 if (product.get("battery_hours") or 0) >= preferences["min_battery_hours"] else 0
    elif product.get("battery_hours"):
        score += min(product["battery_hours"], 15)
    if preferences.get("processor"):
        score += 10 if preferences["processor"].casefold() in product["processor"].casefold() else 0
    if preferences.get("use_case") in {"development", "performance"}:
        processor_name = product.get("processor", "").casefold()
        tier_match = re.search(r"(?:core i|ryzen )([3579])", processor_name)
        if tier_match:
            score += int(tier_match.group(1))
        if re.search(r"(?:h|hs|hx)$", processor_name):
            score += 2
    elif preferences.get("use_case") == "productivity":
        score += min(product.get("battery_hours") or 0, 12) * 0.5
    score += min(max(product.get("rating", 0), 0), 5) * 3
    if budget:
        score += max(0, 10 - (product["price"] / budget) * 10)
    score += max(0, 5 - product.get("delivery_days", 0)) * 0.5
    return score


def _deterministic_explanation(
    recommendations: list[dict[str, Any]],
    preferences: dict[str, Any],
) -> str:
    if not recommendations:
        return (
            "No in-stock product remains within the selected constraints. "
            "Try broadening optional specifications or increasing the budget."
        )
    lead = recommendations[0]
    reason = []
    if preferences.get("min_ram_gb"):
        reason.append(f"{lead['ram_gb']} GB RAM meets the {preferences['min_ram_gb']} GB preference")
    if preferences.get("min_storage_gb"):
        reason.append(
            f"{lead['storage_gb']} GB storage meets the "
            f"{preferences['min_storage_gb']} GB preference"
        )
    if preferences.get("min_battery_hours"):
        reason.append(
            f"{lead['battery_hours']} hours meets the requested "
            f"{preferences['min_battery_hours']} hour battery target"
        )
    if not reason:
        reason.append("it is an in-stock catalog option that fits the selected constraints")
    budget_note = (
        f" It leaves ₹{preferences['budget'] - lead['price']:,.0f} from your budget."
        if preferences.get("budget") is not None
        else ""
    )
    return (
        f"{lead['product_name']} is the highest-ranked eligible sample because "
        + ", ".join(reason)
        + f". It costs ₹{lead['price']:,.0f}, is {lead['stock_status'].lower()}, "
        + f"and has a demonstration delivery estimate of {lead['delivery_days']} days."
        + budget_note
        + " Compare the listed specifications and confirm details with the seller."
    )


def _describe_recommendation(
    product: dict[str, Any],
    preferences: dict[str, Any],
) -> dict[str, Any]:
    """Attach transparent, catalog-only fit notes to each ranked product."""
    reasons = []
    advantages = []
    compromises = []

    if preferences.get("category"):
        reasons.append(f"listed in the {preferences['category']} category")
    if preferences.get("budget") is not None:
        remaining = round(preferences["budget"] - product["price"], 2)
        reasons.append(
            f"₹{product['price']:,.0f} is within your ₹{preferences['budget']:,.0f} "
            f"maximum (₹{remaining:,.0f} remaining)"
        )
        product["remaining_budget"] = remaining
    else:
        product["remaining_budget"] = None

    for field, label, unit in (
        ("min_ram_gb", "RAM", "GB"),
        ("min_storage_gb", "storage", "GB"),
        ("min_battery_hours", "battery life", "hours"),
    ):
        requested = preferences.get(field)
        actual_key = {
            "min_ram_gb": "ram_gb",
            "min_storage_gb": "storage_gb",
            "min_battery_hours": "battery_hours",
        }[field]
        actual = product.get(actual_key)
        if requested is None:
            continue
        if actual is not None and actual >= requested:
            reasons.append(f"{actual} {unit} meets your {requested} {unit} minimum for {label}")
            advantages.append(f"{actual} {unit} listed")
        else:
            compromises.append(
                f"Requested {label} minimum is {requested} {unit}; "
                f"catalog lists {actual if actual is not None else 'no value'}"
            )

    requested_processor = preferences.get("processor", "").strip()
    if requested_processor:
        if requested_processor.casefold() in product["processor"].casefold():
            reasons.append(f"listed processor includes {requested_processor}")
            advantages.append(f"Processor: {product['processor']}")
        else:
            compromises.append(
                f"Requested processor '{requested_processor}' is not listed"
            )
    elif product.get("processor"):
        advantages.append(f"Processor: {product['processor']}")

    if product.get("rating"):
        advantages.append(f"Sample catalog rating: {product['rating']}/5")
    if product.get("battery_hours") and not preferences.get("min_battery_hours"):
        advantages.append(f"Sample catalog battery estimate: {product['battery_hours']} hours")
    if not reasons:
        reasons.append("an in-stock product from the local sample catalog")
    if not compromises:
        compromises.append(
            "Processor and GPU performance are not independently verified by this demo"
        )

    product["fit_reasons"] = reasons
    product["advantages"] = advantages
    product["compromises"] = compromises
    return product


def _recommend(state: ShoppingState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    preferences = state.get("preferences", {})
    eligible = list(state.get("matching_products", []))
    comparison_by_id = {
        product["product_id"]: product
        for product in state.get("comparison", [])
    }
    ranked = []
    for product in eligible:
        comparison_product = comparison_by_id.get(product["product_id"])
        if comparison_product:
            ranked.append(comparison_product)
    ranked.sort(
        key=lambda product: (
            -_score_product(product, preferences),
            product["price"],
            product["delivery_days"],
            product["product_id"],
        )
    )
    recommendations = [
        _describe_recommendation(product, preferences)
        for product in ranked[:3]
    ]
    explanation = _deterministic_explanation(recommendations, preferences)
    messages = list(state.get("messages", []))
    llm_status = state.get("llm_status", "Deterministic mode")
    if recommendations:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Explain the recommendation using only these JSON catalog facts. "
                        "Do not add product specifications, prices, stock, or delivery "
                        "facts. Mention a trade-off if visible. No approval or purchase claim.",
                    ),
                    ("human", "Customer requirements: {preferences}\nCatalog candidates: {products}"),
                ]
            )
            generated = (
                prompt | get_chat_model() | StrOutputParser()
            ).invoke(
                {
                    "preferences": json.dumps(preferences, ensure_ascii=False),
                    "products": json.dumps(recommendations, ensure_ascii=False),
                }
            ).strip()
            if generated:
                explanation = generated
                llm_status = "AI explanation (grounded in catalog results)"
        except LLMUnavailableError:
            pass
        except Exception as error:
            LOGGER.warning("Optional explanation generation failed: %s", type(error).__name__)
            messages.append("AI explanation unavailable; showing the deterministic explanation.")
            llm_status = "Deterministic explanation fallback"
    else:
        messages.append(
            "No available product meets the hard category and budget constraints. "
            "No out-of-stock product is recommended as available."
        )
    return {
        "recommendations": recommendations,
        "explanation": explanation,
        "messages": messages,
        "llm_status": llm_status,
    }


def _fallback(state: ShoppingState) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    if state.get("error"):
        messages.append("The shopping workflow stopped safely because a required step failed.")
        return {
            "recommendations": [],
            "comparison": [],
            "explanation": state["error"],
            "messages": messages,
        }
    messages.append(
        "No products were found in the local catalog for these requirements."
    )
    return {
        "recommendations": [],
        "comparison": [],
        "explanation": "Try changing the category or optional product requirements.",
        "messages": messages,
    }


def _route_after_search(state: ShoppingState) -> str:
    return "fallback" if state.get("error") or not state.get("matching_products") else "inventory"


def _route_after_inventory(state: ShoppingState) -> str:
    return "fallback" if state.get("error") else "budget"


def _route_after_budget(state: ShoppingState) -> str:
    return "fallback" if state.get("error") or not state.get("matching_products") else "comparison"


def _route_after_comparison(state: ShoppingState) -> str:
    return "fallback" if state.get("error") else "recommendation"


def build_shopping_graph():
    """Build the single LangGraph shopping workflow with guarded fallback edges."""
    graph = StateGraph(ShoppingState)
    graph.add_node("requirement_analysis", _analyze_requirements)
    graph.add_node("product_search", _search_products)
    graph.add_node("inventory_validation", _validate_inventory)
    graph.add_node("budget_validation", _validate_budget)
    graph.add_node("product_comparison", _compare_products)
    graph.add_node("recommendation", _recommend)
    graph.add_node("fallback_handling", _fallback)
    graph.add_edge(START, "requirement_analysis")
    graph.add_edge("requirement_analysis", "product_search")
    graph.add_conditional_edges(
        "product_search",
        _route_after_search,
        {"inventory": "inventory_validation", "fallback": "fallback_handling"},
    )
    graph.add_conditional_edges(
        "inventory_validation",
        _route_after_inventory,
        {"budget": "budget_validation", "fallback": "fallback_handling"},
    )
    graph.add_conditional_edges(
        "budget_validation",
        _route_after_budget,
        {"comparison": "product_comparison", "fallback": "fallback_handling"},
    )
    graph.add_conditional_edges(
        "product_comparison",
        _route_after_comparison,
        {"recommendation": "recommendation", "fallback": "fallback_handling"},
    )
    graph.add_edge("recommendation", END)
    graph.add_edge("fallback_handling", END)
    return graph.compile()


shopping_graph = build_shopping_graph()


def run_shopping_workflow(
    query: str,
    form_filters: dict[str, Any] | None = None,
) -> ShoppingState:
    """Execute the shopping graph for one customer request."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a shopping request before searching.")
    try:
        return shopping_graph.invoke(
            {
                "original_query": query.strip(),
                "form_filters": form_filters or {},
                "messages": [],
                "llm_status": "Deterministic mode",
            }
        )
    except CatalogError:
        LOGGER.exception("The product catalog is invalid.")
        raise
