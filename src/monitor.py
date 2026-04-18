# src/monitor.py
import sqlite3, json, logging
from datetime import datetime , UTC

DB = "predictions.db"       
logger = logging.getLogger(__name__)

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       TEXT,
            features TEXT,
            pred     INTEGER,
            prob     REAL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised")

def log_prediction(features: dict, pred: int, prob: float):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO predictions(ts, features, pred, prob) VALUES(?,?,?,?)",
        (datetime.now(UTC).isoformat(), json.dumps(features), pred, prob)
    )
    conn.commit()
    conn.close()