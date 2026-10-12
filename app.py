"""Flask routes for the local demonstration shopping assistant."""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for

from rag_assistant import FAQError, answer_faq
from shopping_agent import run_shopping_workflow
from shopping_tools import (
    CatalogError,
    calculate_cart_total,
    load_catalog,
    product_record,
)


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
configured_secret = os.getenv("FLASK_SECRET_KEY", "").strip()
if not configured_secret or configured_secret == "replace-this-with-a-long-random-secret":
    app.secret_key = secrets.token_urlsafe(48)
    LOGGER.warning(
        "FLASK_SECRET_KEY is unset or a placeholder. Using a temporary local-only "
        "session key; configure a persistent secret for deployment."
    )
else:
    app.secret_key = configured_secret
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "false").casefold() == "true",
    MAX_CONTENT_LENGTH=24 * 1024,
)


def _categories() -> list[str]:
    return sorted(set(load_catalog()["category"].dropna().astype(str).str.strip()) - {""})


def _form_filters(form: Any) -> tuple[dict[str, Any], str | None]:
    filters: dict[str, Any] = {
        "category": str(form.get("category", "")).strip(),
        "processor": str(form.get("processor", "")).strip(),
    }
    for field, label, minimum, maximum in (
        ("budget", "Budget", 0, 10_000_000),
        ("min_ram_gb", "Minimum RAM", 0, 1024),
        ("min_storage_gb", "Minimum storage", 0, 100_000),
        ("min_battery_hours", "Minimum battery life", 0, 100),
    ):
        raw_value = str(form.get(field, "")).strip()
        if not raw_value:
            filters[field] = None
            continue
        try:
            parsed = float(raw_value)
        except ValueError:
            return {}, f"{label} must be a number."
        if parsed < minimum or parsed > maximum:
            return {}, f"{label} must be between {minimum} and {maximum}."
        filters[field] = int(parsed) if field != "budget" and parsed.is_integer() else parsed
    category = filters["category"]
    if category and category not in _categories():
        return {}, "Choose a category from the list."
    if len(filters["processor"]) > 80:
        return {}, "Processor filter must be 80 characters or fewer."
    return filters, None


def _product_map():
    return {
        product["product_id"]: product
        for product in (
            product_record(row)
            for _, row in load_catalog().iterrows()
        )
    }


@app.get("/")
def index():
    try:
        categories = _categories()
    except CatalogError as error:
        LOGGER.error("Catalog is unavailable: %s", error)
        categories = []
        flash(str(error), "error")
    return render_template("index.html", categories=categories)


@app.post("/search")
def search():
    query = str(request.form.get("query", "")).strip()
    filters, validation_error = _form_filters(request.form)
    if validation_error:
        flash(validation_error, "error")
        return render_template("index.html", categories=_categories(), query=query), 400
    if not query:
        flash("Describe what you are shopping for before searching.", "error")
        return render_template("index.html", categories=_categories()), 400
    if len(query) > 1500:
        flash("Keep your shopping request to 1,500 characters or fewer.", "error")
        return render_template("index.html", categories=_categories(), query=query[:1500]), 400

    try:
        result = run_shopping_workflow(query, filters)
        return render_template(
            "results.html",
            result=result,
            recommendations=result.get("recommendations", []),
            cart_count=sum(session.get("cart", {}).values()),
        )
    except (CatalogError, ValueError) as error:
        LOGGER.warning("Shopping request rejected: %s", error)
        flash(str(error), "error")
        return render_template("index.html", categories=_categories(), query=query), 400
    except Exception:
        LOGGER.exception("Unexpected shopping search failure.")
        flash("Product search is temporarily unavailable. Please try again.", "error")
        return render_template("index.html", categories=_categories(), query=query), 500


@app.post("/faq")
def faq():
    question = str(request.form.get("faq_question", "")).strip()
    if not question:
        flash("Enter a FAQ question first.", "error")
        return redirect(url_for("index"))
    try:
        result = answer_faq(question)
        return render_template(
            "index.html",
            categories=_categories(),
            faq_question=question,
            faq_result=result,
        )
    except FAQError as error:
        LOGGER.warning("FAQ search unavailable: %s", error)
        flash(str(error), "error")
        return render_template("index.html", categories=_categories(), faq_question=question), 503
    except Exception:
        LOGGER.exception("Unexpected FAQ failure.")
        flash("FAQ search is temporarily unavailable. Please try again.", "error")
        return render_template("index.html", categories=_categories()), 500


@app.get("/cart")
def cart():
    try:
        total = calculate_cart_total(session.get("cart", {}))
        return render_template("cart.html", cart=total, cart_count=len(total["items"]))
    except (CatalogError, ValueError) as error:
        LOGGER.warning("Cart data needs attention: %s", error)
        flash("The cart could not be loaded because an item is no longer valid or available.", "error")
        return render_template("cart.html", cart={"items": [], "total": 0}, cart_count=0), 409
    except Exception:
        LOGGER.exception("Unexpected cart loading failure.")
        flash("The cart could not be loaded. Please try again.", "error")
        return render_template("cart.html", cart={"items": [], "total": 0}, cart_count=0), 500


@app.post("/cart/add")
def cart_add():
    product_id = str(request.form.get("product_id", "")).strip()
    try:
        product = _product_map().get(product_id)
        if product is None:
            flash("That product ID is not in the catalog.", "error")
            return redirect(url_for("cart"))
        raw_quantity = request.form.get("quantity", "1")
        quantity = int(raw_quantity)
        if quantity < 1 or quantity > product["stock_quantity"]:
            flash("The requested quantity exceeds the available sample stock.", "error")
            return redirect(url_for("cart"))
        session["pending_cart_item"] = {"product_id": product_id, "quantity": quantity}
        return redirect(url_for("cart_confirm"))
    except (TypeError, ValueError, CatalogError) as error:
        LOGGER.warning("Invalid add-to-cart request: %s", error)
        flash("Check the product and quantity, then try again.", "error")
        return redirect(url_for("cart"))


@app.route("/cart/confirm", methods=["GET", "POST"])
def cart_confirm():
    pending = session.get("pending_cart_item")
    if not isinstance(pending, dict):
        flash("There is no product waiting for confirmation.", "error")
        return redirect(url_for("index"))
    try:
        product = _product_map().get(str(pending.get("product_id", "")))
        quantity = pending.get("quantity")
        if product is None or not isinstance(quantity, int):
            session.pop("pending_cart_item", None)
            flash("The selected product is no longer valid.", "error")
            return redirect(url_for("index"))
        if quantity < 1 or quantity > product["stock_quantity"]:
            session.pop("pending_cart_item", None)
            flash("The requested quantity is no longer available.", "error")
            return redirect(url_for("index"))
        if request.method == "GET":
            return render_template("confirm.html", product=product, quantity=quantity)
        if request.form.get("action") != "confirm":
            session.pop("pending_cart_item", None)
            flash("Product was not added to the simulated cart.", "info")
            return redirect(url_for("index"))

        cart_items = dict(session.get("cart", {}))
        new_quantity = int(cart_items.get(product["product_id"], 0)) + quantity
        if new_quantity > product["stock_quantity"]:
            session.pop("pending_cart_item", None)
            flash("Cart quantity would exceed available sample stock.", "error")
            return redirect(url_for("cart"))
        cart_items[product["product_id"]] = new_quantity
        calculate_cart_total(cart_items)
        session["cart"] = cart_items
        session.pop("pending_cart_item", None)
        flash("Product added to the simulated cart. No order was created.", "success")
        return redirect(url_for("cart"))
    except (CatalogError, ValueError, TypeError) as error:
        LOGGER.warning("Cart confirmation could not be completed: %s", error)
        session.pop("pending_cart_item", None)
        flash("The product could not be added. Check the available sample stock.", "error")
        return redirect(url_for("cart"))


@app.post("/cart/update")
def cart_update():
    product_id = str(request.form.get("product_id", "")).strip()
    try:
        product = _product_map().get(product_id)
        if product is None:
            raise ValueError("Unknown product ID.")
        quantity = int(request.form.get("quantity", "0"))
        if quantity < 1 or quantity > product["stock_quantity"]:
            raise ValueError("Requested quantity exceeds available stock.")
        cart_items = dict(session.get("cart", {}))
        if product_id not in cart_items:
            raise ValueError("Product is not in the cart.")
        cart_items[product_id] = quantity
        calculate_cart_total(cart_items)
        session["cart"] = cart_items
        flash("Cart quantity updated.", "success")
    except (ValueError, TypeError, CatalogError) as error:
        LOGGER.warning("Cart update rejected: %s", error)
        flash(str(error), "error")
    return redirect(url_for("cart"))


@app.post("/cart/remove")
def cart_remove():
    product_id = str(request.form.get("product_id", "")).strip()
    cart_items = dict(session.get("cart", {}))
    cart_items.pop(product_id, None)
    session["cart"] = cart_items
    flash("Item removed from the simulated cart.", "info")
    return redirect(url_for("cart"))


@app.errorhandler(413)
def request_too_large(_error):
    return "Request is too large.", 413


@app.errorhandler(500)
def internal_error(_error):
    LOGGER.exception("Unhandled Flask error.")
    return render_template(
        "error.html",
        message="Something went wrong. Please return to the home page and try again.",
    ), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").casefold() == "true")
