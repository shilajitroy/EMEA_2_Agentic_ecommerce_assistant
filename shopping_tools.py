"""Catalog-backed shopping functions and their LangChain tool wrappers."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
CATALOG_PATH = PROJECT_DIR / "product_data.csv"
REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "brand",
    "price",
    "original_price",
    "discount_percent",
    "ram_gb",
    "storage_gb",
    "processor",
    "battery_hours",
    "rating",
    "stock_quantity",
    "delivery_days",
    "description",
}
NUMERIC_COLUMNS = (
    "price",
    "original_price",
    "discount_percent",
    "ram_gb",
    "storage_gb",
    "battery_hours",
    "rating",
    "stock_quantity",
    "delivery_days",
)


class CatalogError(ValueError):
    """Raised when the product catalog cannot be loaded safely."""


def load_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """Read and validate catalog columns, coercing malformed numeric cells safely."""
    catalog_path = Path(path) if path else CATALOG_PATH
    if not catalog_path.is_file():
        raise CatalogError(f"Product catalog not found: {catalog_path.name}")
    try:
        catalog = pd.read_csv(catalog_path, dtype={"product_id": "string"})
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        raise CatalogError("The product catalog could not be read.") from error

    missing_columns = REQUIRED_COLUMNS.difference(catalog.columns)
    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise CatalogError(f"Product catalog is missing columns: {missing_names}")

    for column in NUMERIC_COLUMNS:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    catalog["category"] = catalog["category"].fillna("").astype(str).str.strip()
    catalog["product_id"] = catalog["product_id"].fillna("").astype(str).str.strip()

    required_values = catalog["product_id"].ne("") & catalog["product_name"].notna()
    required_values &= catalog["category"].ne("") & catalog["price"].notna()
    required_values &= catalog["price"].map(
        lambda value: math.isfinite(value) and value >= 0
    )
    invalid_rows = int((~required_values).sum())
    if invalid_rows:
        LOGGER.warning("Ignoring %d catalog rows with invalid ID/name/category/price.", invalid_rows)
        catalog = catalog.loc[required_values].copy()

    if catalog.empty:
        raise CatalogError("The product catalog has no usable products.")
    if catalog["product_id"].duplicated().any():
        raise CatalogError("Product IDs must be unique in the product catalog.")
    return catalog.reset_index(drop=True)


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _as_int(value: Any, default: int = 0) -> int:
    parsed = _safe_number(value, float(default))
    return int(parsed) if parsed >= 0 else default


def _validated_nonnegative(value: Any, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative number.")
    parsed = _safe_number(value, math.nan)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative number.")
    return parsed


def product_record(row: pd.Series) -> dict[str, Any]:
    """Convert a catalog row to JSON-friendly product facts."""
    stock = _as_int(row.get("stock_quantity"))
    quantity = _safe_number(row.get("stock_quantity"), -1)
    status = (
        "Out of Stock" if quantity <= 0
        else "Low Stock" if quantity <= 2
        else "In Stock"
    )
    price = _safe_number(row.get("price"), math.nan)
    original_price = _safe_number(row.get("original_price"), price)
    declared_discount = _safe_number(row.get("discount_percent"))
    if not math.isfinite(price) or price < 0:
        raise ValueError("This product has an invalid catalog price.")
    if not math.isfinite(original_price) or original_price <= price:
        original_price = price
        discount = 0.0
    else:
        calculated_discount = round((original_price - price) / original_price * 100, 2)
        if (
            0 < declared_discount <= 100
            and abs(declared_discount - calculated_discount) <= 0.1
        ):
            discount = calculated_discount
        else:
            original_price = price
            discount = 0.0

    def optional_int(column: str) -> int | None:
        number = _safe_number(row.get(column), math.nan)
        return int(number) if math.isfinite(number) and number > 0 else None

    return {
        "product_id": str(row.get("product_id", "")),
        "product_name": str(row.get("product_name", "")),
        "category": str(row.get("category", "")),
        "brand": str(row.get("brand", "")),
        "price": round(price, 2),
        "original_price": round(original_price, 2),
        "discount_percent": round(discount, 2),
        "ram_gb": optional_int("ram_gb"),
        "storage_gb": optional_int("storage_gb"),
        "processor": str(row.get("processor", "") or ""),
        "battery_hours": optional_int("battery_hours"),
        "rating": round(_safe_number(row.get("rating")), 1),
        "stock_quantity": stock,
        "stock_status": status,
        "delivery_days": _as_int(row.get("delivery_days")),
        "description": str(row.get("description", "") or ""),
    }


def _matches_minimum(value: Any, minimum: int | float | None) -> bool:
    if minimum is None or minimum <= 0:
        return True
    parsed_value = _safe_number(value, math.nan)
    return math.isfinite(parsed_value) and parsed_value >= minimum


def search_products(
    requirements: dict[str, Any] | None = None,
    catalog: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Filter products by provided hard budget and requested specifications."""
    preferences = requirements or {}
    products = catalog if catalog is not None else load_catalog()
    matches: list[dict[str, Any]] = []
    category = str(preferences.get("category") or "").strip().casefold()
    processor = str(preferences.get("processor") or "").strip().casefold()
    max_budget = _validated_nonnegative(preferences.get("budget"), "Budget")
    minimums = {
        "min_ram_gb": _validated_nonnegative(preferences.get("min_ram_gb"), "Minimum RAM"),
        "min_storage_gb": _validated_nonnegative(
            preferences.get("min_storage_gb"), "Minimum storage"
        ),
        "min_battery_hours": _validated_nonnegative(
            preferences.get("min_battery_hours"), "Minimum battery life"
        ),
    }

    for _, row in products.iterrows():
        record = product_record(row)
        if category and category not in record["category"].casefold():
            continue
        if processor and processor not in record["processor"].casefold():
            continue
        if max_budget is not None and record["price"] > float(max_budget):
            continue
        if not _matches_minimum(record["ram_gb"], minimums["min_ram_gb"]):
            continue
        if not _matches_minimum(
            record["storage_gb"], minimums["min_storage_gb"]
        ):
            continue
        if not _matches_minimum(
            record["battery_hours"], minimums["min_battery_hours"]
        ):
            continue
        matches.append(record)
    return matches


def compare_products(
    product_ids: list[str],
    catalog: pd.DataFrame | None = None,
    budget: float | None = None,
) -> list[dict[str, Any]]:
    """Return comparable catalog facts for valid, distinct product IDs."""
    products = catalog if catalog is not None else load_catalog()
    wanted = list(dict.fromkeys(str(product_id) for product_id in product_ids))
    budget = _validated_nonnegative(budget, "Budget")
    by_id = {
        str(row["product_id"]): row
        for _, row in products.iterrows()
    }
    compared = []
    for product_id in wanted:
        row = by_id.get(product_id)
        if row is None:
            continue
        product = product_record(row)
        product["within_budget"] = budget is None or product["price"] <= float(budget)
        compared.append(product)
    return compared


def check_inventory(
    product_ids: list[str],
    catalog: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Report the stock status and quantity for each known product ID."""
    products = catalog if catalog is not None else load_catalog()
    records = {
        str(row["product_id"]): product_record(row)
        for _, row in products.iterrows()
    }
    result = []
    for product_id in dict.fromkeys(str(item) for item in product_ids):
        product = records.get(product_id)
        result.append(
            {
                "product_id": product_id,
                "known_product": product is not None,
                "stock_quantity": product["stock_quantity"] if product else 0,
                "status": product["stock_status"] if product else "Unknown Product",
            }
        )
    return result


def validate_budget(
    product_ids: list[str],
    budget: float | None,
    catalog: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Check catalog selling prices against the customer-set maximum budget."""
    products = catalog if catalog is not None else load_catalog()
    records = {
        str(row["product_id"]): product_record(row)
        for _, row in products.iterrows()
    }
    result = []
    budget = _validated_nonnegative(budget, "Budget")
    for product_id in dict.fromkeys(str(item) for item in product_ids):
        product = records.get(product_id)
        price = product["price"] if product else None
        result.append(
            {
                "product_id": product_id,
                "known_product": product is not None,
                "price": price,
                "budget": budget,
                "within_budget": (
                    product is not None
                    and (budget is None or price <= float(budget))
                ),
                "remaining_budget": (
                    round(float(budget) - price, 2)
                    if product is not None and budget is not None
                    else None
                ),
            }
        )
    return result


def estimate_delivery(
    product_ids: list[str],
    catalog: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Return demonstration delivery durations from the catalog."""
    products = catalog if catalog is not None else load_catalog()
    records = {
        str(row["product_id"]): product_record(row)
        for _, row in products.iterrows()
    }
    return [
        {
            "product_id": product_id,
            "delivery_days": (
                records[product_id]["delivery_days"] if product_id in records else None
            ),
            "is_demo_estimate": True,
        }
        for product_id in dict.fromkeys(str(item) for item in product_ids)
    ]


def calculate_cart_total(
    cart: dict[str, int],
    catalog: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calculate unit price and subtotal only for available requested quantities."""
    products = catalog if catalog is not None else load_catalog()
    records = {
        str(row["product_id"]): product_record(row)
        for _, row in products.iterrows()
    }
    items = []
    total = 0.0
    for product_id, quantity in cart.items():
        if product_id not in records:
            raise ValueError(f"Unknown product ID: {product_id}")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError(f"Invalid quantity for product {product_id}.")
        product = records[product_id]
        if product["stock_quantity"] <= 0 or quantity > product["stock_quantity"]:
            raise ValueError(f"Quantity exceeds available sample stock for {product_id}.")
        subtotal = round(product["price"] * quantity, 2)
        item = {**product, "quantity": quantity, "subtotal": subtotal}
        items.append(item)
        total += subtotal
    return {"items": items, "total": round(total, 2)}


@tool
def search_product_catalog(
    category: str = "",
    budget: float | None = None,
    min_ram_gb: int | None = None,
    min_storage_gb: int | None = None,
    processor: str = "",
    min_battery_hours: int | None = None,
) -> list[dict[str, Any]]:
    """Search the local product catalog by category, maximum budget, and specifications."""
    return search_products(
        {
            "category": category,
            "budget": budget,
            "min_ram_gb": min_ram_gb,
            "min_storage_gb": min_storage_gb,
            "processor": processor,
            "min_battery_hours": min_battery_hours,
        }
    )


@tool
def compare_catalog_products(product_ids: list[str], budget: float | None = None) -> list[dict[str, Any]]:
    """Compare catalog product IDs using price, specs, inventory, and delivery estimate."""
    return compare_products(product_ids, budget=budget)


@tool
def check_catalog_inventory(product_ids: list[str]) -> list[dict[str, Any]]:
    """Check each catalog product's current demonstration stock status."""
    return check_inventory(product_ids)


@tool
def validate_catalog_budget(product_ids: list[str], budget: float | None = None) -> list[dict[str, Any]]:
    """Check whether catalog selling prices meet a customer's maximum budget."""
    return validate_budget(product_ids, budget)


@tool
def estimate_catalog_delivery(product_ids: list[str]) -> list[dict[str, Any]]:
    """Read estimated demonstration delivery days for catalog products."""
    return estimate_delivery(product_ids)


SHOPPING_TOOLS = [
    search_product_catalog,
    compare_catalog_products,
    check_catalog_inventory,
    validate_catalog_budget,
    estimate_catalog_delivery,
]
