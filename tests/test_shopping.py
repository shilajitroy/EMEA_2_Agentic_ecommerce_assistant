from __future__ import annotations

import pandas as pd
import pytest

from app import app
from shopping_tools import (
    calculate_cart_total,
    check_inventory,
    compare_products,
    load_catalog,
    search_products,
    validate_budget,
)


@pytest.fixture
def catalog():
    return load_catalog()


def test_product_filtering_by_budget_and_ram(catalog):
    matches = search_products(
        {"category": "Laptop", "budget": 60000, "min_ram_gb": 16},
        catalog,
    )
    assert matches
    assert all(item["price"] <= 60000 and item["ram_gb"] >= 16 for item in matches)


def test_product_comparison_returns_catalog_facts(catalog):
    compared = compare_products(["LAP001", "LAP002"], catalog, budget=60000)
    assert [item["product_id"] for item in compared] == ["LAP001", "LAP002"]
    assert compared[0]["price"] == 55990
    assert compared[0]["within_budget"] is True


def test_inventory_validation_identifies_out_of_stock(catalog):
    results = check_inventory(["LAP004", "LAP001", "unknown"], catalog)
    assert [item["status"] for item in results] == [
        "Out of Stock",
        "In Stock",
        "Unknown Product",
    ]


def test_budget_validation_excludes_over_budget(catalog):
    results = validate_budget(["LAP001", "LAP005"], 60000, catalog)
    assert [item["within_budget"] for item in results] == [True, False]
    assert results[0]["remaining_budget"] == 4010


def test_budget_tools_reject_negative_values(catalog):
    with pytest.raises(ValueError, match="Budget must be a non-negative number"):
        search_products({"budget": -1}, catalog)
    with pytest.raises(ValueError, match="Budget must be a non-negative number"):
        validate_budget(["LAP001"], -1, catalog)


def test_no_matching_products_and_optional_zero_filters(catalog):
    assert search_products(
        {"category": "Laptop", "budget": 1, "min_ram_gb": 16},
        catalog,
    ) == []
    # Zero means no minimum, including categories without a RAM specification.
    assert search_products({"category": "Monitor", "min_ram_gb": 0}, catalog)


def test_cart_total_calculation_and_invalid_id(catalog):
    total = calculate_cart_total({"LAP001": 2, "AUD001": 1}, catalog)
    assert total["total"] == 120970
    with pytest.raises(ValueError, match="Unknown product ID"):
        calculate_cart_total({"missing": 1}, catalog)


def test_cart_total_rejects_out_of_stock_and_excess_quantity(catalog):
    with pytest.raises(ValueError, match="exceeds available sample stock"):
        calculate_cart_total({"LAP004": 1}, catalog)
    with pytest.raises(ValueError, match="exceeds available sample stock"):
        calculate_cart_total({"LAP001": 13}, catalog)


def test_catalog_invalid_numeric_values_are_handled_safely(catalog):
    malformed = catalog.copy()
    malformed["stock_quantity"] = malformed["stock_quantity"].astype(object)
    malformed.loc[malformed["product_id"] == "LAP001", "stock_quantity"] = "not-a-number"
    results = check_inventory(["LAP001"], malformed)
    assert results[0]["status"] == "Out of Stock"


def test_search_does_not_mutate_catalog(catalog):
    original = catalog.copy(deep=True)
    search_products({"category": "Laptop", "budget": 60000}, catalog)
    pd.testing.assert_frame_equal(catalog, original)


def test_flask_rejects_empty_search_and_cart_mutation_is_post_only():
    app.config.update(TESTING=True, SECRET_KEY="pytest-test-key")
    client = app.test_client()
    assert client.get("/cart/add").status_code == 405
    response = client.post("/search", data={"query": " "})
    assert response.status_code == 400
    assert b"shopping for before searching" in response.data


def test_flask_search_renders_catalog_recommendations(monkeypatch):
    import shopping_agent

    monkeypatch.setattr(shopping_agent, "_llm_preferences", lambda _query: {})
    app.config.update(TESTING=True, SECRET_KEY="pytest-test-key")
    response = app.test_client().post(
        "/search",
        data={
            "query": "Python laptop under ₹60,000 with 16 GB RAM and 512 GB SSD",
        },
    )
    assert response.status_code == 200
    assert b"Compare recommendations" in response.data
    assert b"Recommendation explanation" in response.data


def test_cart_requires_confirmation_and_validates_stock():
    app.config.update(TESTING=True, SECRET_KEY="pytest-test-key")
    client = app.test_client()
    pending = client.post("/cart/add", data={"product_id": "LAP001", "quantity": "2"})
    assert pending.status_code == 302
    confirm_page = client.get(pending.headers["Location"])
    assert confirm_page.status_code == 200
    assert b"Confirm adding this item" in confirm_page.data

    added = client.post(
        pending.headers["Location"],
        data={"action": "confirm"},
    )
    assert added.status_code == 302
    cart_page = client.get("/cart")
    assert b"Acer Aspire 5" in cart_page.data
    assert b"111,980.00" in cart_page.data

    too_many = client.post(
        "/cart/update",
        data={"product_id": "LAP001", "quantity": "13"},
        follow_redirects=True,
    )
    assert b"exceeds available stock" in too_many.data


def test_flask_rejects_invalid_product_ids_and_out_of_stock_adds():
    app.config.update(TESTING=True, SECRET_KEY="pytest-test-key")
    client = app.test_client()
    bad_id = client.post(
        "/cart/add",
        data={"product_id": "not-a-product", "quantity": "1"},
        follow_redirects=True,
    )
    assert b"not in the catalog" in bad_id.data
    out_of_stock = client.post(
        "/cart/add",
        data={"product_id": "LAP004", "quantity": "1"},
        follow_redirects=True,
    )
    assert b"exceeds the available sample stock" in out_of_stock.data
