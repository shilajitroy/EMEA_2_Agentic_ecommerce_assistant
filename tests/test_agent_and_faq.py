from __future__ import annotations

import rag_assistant
import shopping_agent
from llm_config import redact_sensitive_text


def test_workflow_recommends_only_in_stock_and_within_budget(monkeypatch):
    monkeypatch.setattr(shopping_agent, "_llm_preferences", lambda _query: {})
    result = shopping_agent.run_shopping_workflow(
        "I need a laptop under ₹60,000 with 16 GB RAM and 512 GB SSD"
    )
    assert result["recommendations"]
    assert all(product["price"] <= 60000 for product in result["recommendations"])
    assert all(product["stock_quantity"] > 0 for product in result["recommendations"])
    assert result["recommendations"][0]["product_name"]
    assert result["recommendations"][0]["fit_reasons"]
    assert result["recommendations"][0]["advantages"]
    assert result["recommendations"][0]["compromises"]
    assert result["recommendations"][0]["remaining_budget"] >= 0
    assert "budget were kept" not in " ".join(result["messages"]).lower()


def test_workflow_keeps_budget_hard_when_no_match(monkeypatch):
    monkeypatch.setattr(shopping_agent, "_llm_preferences", lambda _query: {})
    result = shopping_agent.run_shopping_workflow(
        "Laptop under ₹100 with at least 64 GB RAM"
    )
    assert result["recommendations"] == []
    assert result["preferences"]["budget"] == 100


def test_out_of_stock_match_offers_in_stock_category_alternatives(monkeypatch):
    monkeypatch.setattr(shopping_agent, "_llm_preferences", lambda _query: {})
    result = shopping_agent.run_shopping_workflow(
        "Laptop with Intel Core i5-1335U under ₹63,000"
    )
    assert result["recommendations"]
    assert all(item["stock_quantity"] > 0 for item in result["recommendations"])
    assert any("closest exact matches are out of stock" in message.casefold() for message in result["messages"])
    assert result["preferences"]["processor"] == "Intel Core i5-1335U"
    assert result["recommendations"][0]["compromises"]


def test_faq_retrieval_answers_directly_without_llm(monkeypatch):
    monkeypatch.setattr(
        rag_assistant,
        "retrieve_faq",
        lambda _question: [
            {
                "question": "Does the cart place a real order?",
                "answer": "No. The cart is simulated and does not create an order or process a payment.",
                "topic": "cart",
                "relevance": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        rag_assistant,
        "get_chat_model",
        lambda: (_ for _ in ()).throw(rag_assistant.LLMUnavailableError("not configured")),
    )
    result = rag_assistant.answer_faq("Does this place an order?")
    assert result["mode"] == "retrieved-faq"
    assert "does not create an order" in result["answer"]


def test_faq_unknown_question_does_not_invent_policy(monkeypatch):
    monkeypatch.setattr(rag_assistant, "retrieve_faq", lambda _question: [])
    result = rag_assistant.answer_faq("What is the exact return window?")
    assert result["answer"] == rag_assistant.NO_FAQ_ANSWER
    assert result["sources"] == []


def test_faq_csv_has_required_sample_questions():
    documents = rag_assistant.load_faq_documents()
    assert len(documents) >= 20
    warranty = [document for document in documents if document.metadata["topic"] == "policy"]
    assert warranty
    assert any("not included" in document.page_content for document in warranty)


def test_llm_input_redacts_contact_details_but_keeps_budget():
    sanitized = redact_sensitive_text(
        "Laptop under ₹60,000; call 9876543210 or email shopper@example.com"
    )
    assert "₹60,000" in sanitized
    assert "9876543210" not in sanitized
    assert "shopper@example.com" not in sanitized
