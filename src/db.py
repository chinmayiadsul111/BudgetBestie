"""
Database layer.

Uses SQLAlchemy so `database.url` in config.yaml can point at SQLite (default,
zero-infra demo) or Postgres (production) without any code changes here.
The Reasoning agent generates SQL text against this schema — this module is
what actually executes it safely (read-only, parameterized where possible).
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config_loader import get_config, PROJECT_ROOT
from src.logging_setup import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    category = Column(String, nullable=False)      # e.g. "rent", "dining", "subscriptions"
    merchant = Column(String, nullable=False)
    amount = Column(Float, nullable=False)          # positive = expense, negative = income
    account = Column(String, nullable=False)        # e.g. "checking", "credit_card"


_engine = None
_SessionLocal = None


def _resolve_sqlite_path_if_relative(url: str) -> str:
    """SQLite URLs in config are relative to project root — resolve them so the
    app works the same whether launched from repo root or elsewhere."""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel_path = url.replace("sqlite:///", "")
        abs_path = PROJECT_ROOT / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{abs_path}"
    return url


def get_engine():
    global _engine
    if _engine is None:
        cfg = get_config()
        url = _resolve_sqlite_path_if_relative(cfg.get("database.url"))
        echo = cfg.get("database.echo_sql", False)
        _engine = create_engine(url, echo=echo, future=True)
        logger.info("db_engine_created", extra={"dialect": _engine.dialect.name})
    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), future=True)
    return _SessionLocal()


def ensure_schema_and_seed() -> None:
    """Creates tables if missing, and seeds realistic demo data if the table is empty."""
    cfg = get_config()
    engine = get_engine()
    Base.metadata.create_all(engine)

    if not cfg.get("database.seed_on_missing", True):
        return

    session = get_session()
    try:
        count = session.query(Transaction).count()
        if count > 0:
            logger.info("db_seed_skipped_already_populated", extra={"row_count": count})
            return

        rows = _generate_demo_transactions()
        session.bulk_save_objects(rows)
        session.commit()
        logger.info("db_seeded", extra={"row_count": len(rows)})
    finally:
        session.close()


def _generate_demo_transactions() -> list[Transaction]:
    """
    Generates a believable ~4 months of Gen-Z entry-level-salary transaction
    history: rent, a chronic food-delivery habit, streaming subscriptions,
    a gym membership nobody uses, one credit card carrying a balance, and a
    paycheck every two weeks. This is what makes the SQL-generation agent's
    answers feel real instead of canned.
    """
    random.seed(42)
    rng = random.Random(42)
    today = datetime.utcnow().date()
    start = today - timedelta(days=120)

    merchants = {
        "rent": [("Maple Court Apartments", 1450.0)],
        "groceries": [("Trader Joe's", (20, 70)), ("Whole Foods", (15, 60))],
        "dining": [("DoorDash", (12, 45)), ("Chipotle", (9, 18)), ("Starbucks", (5, 9))],
        "subscriptions": [
            ("Netflix", 15.49), ("Spotify Premium", 11.99),
            ("Planet Fitness", 24.99), ("iCloud Storage", 2.99),
        ],
        "transport": [("Uber", (8, 30)), ("Shell Gas", (25, 55))],
        "shopping": [("Amazon", (10, 120)), ("Zara", (25, 90))],
        "credit_card_payment": [("Chase Credit Card Payment", (50, 300))],
        "income": [("Payroll Deposit - Employer", 2100.0)],
    }

    rows: list[Transaction] = []
    d = start
    while d <= today:
        # Rent, 1st of month
        if d.day == 1:
            rows.append(Transaction(date=d, category="rent", merchant="Maple Court Apartments",
                                     amount=1450.0, account="checking"))
        # Biweekly paycheck
        if (d - start).days % 14 == 0:
            rows.append(Transaction(date=d, category="income", merchant="Payroll Deposit - Employer",
                                     amount=-2100.0, account="checking"))
        # Subscriptions, once a month around the 5th
        if d.day == 5:
            for name, amt in merchants["subscriptions"]:
                rows.append(Transaction(date=d, category="subscriptions", merchant=name,
                                         amount=amt, account="credit_card"))
        # Random daily spend noise
        if rng.random() < 0.5:
            cat = rng.choice(["groceries", "dining", "transport", "shopping"])
            name, amt_range = rng.choice(merchants[cat])
            amount = round(rng.uniform(*amt_range), 2)
            account = "credit_card" if cat in ("dining", "shopping") else "checking"
            rows.append(Transaction(date=d, category=cat, merchant=name, amount=amount, account=account))
        # Occasional credit card payment
        if d.day == 20:
            name, amt_range = merchants["credit_card_payment"][0]
            amount = round(rng.uniform(*amt_range), 2)
            rows.append(Transaction(date=d, category="credit_card_payment", merchant=name,
                                     amount=amount, account="checking"))
        d += timedelta(days=1)

    return rows


def run_readonly_sql(sql: str) -> list[dict]:
    """
    Executes a SQL SELECT generated by the Reasoning agent and returns rows as
    dicts. Refuses anything that isn't a SELECT — this is a demo finance
    assistant, not a database admin tool, and the agent's output should never
    be able to mutate data.
    """
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        logger.warning("sql_execution_blocked_non_select", extra={"sql": sql})
        raise ValueError("Only SELECT statements are permitted.")

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
    logger.info("sql_executed", extra={"sql": sql, "row_count": len(rows)})
    return rows


if __name__ == "__main__":
    # `python -m src.db` — quick standalone way to (re)seed the demo DB.
    ensure_schema_and_seed()
