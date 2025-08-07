import os
from typing import Optional
import duckdb
import pandas as pd


class DuckDBClient:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.con = duckdb.connect(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                asset TEXT,
                ts TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (asset, ts)
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                asset TEXT,
                ts TIMESTAMP,
                data MAP(TEXT, DOUBLE),
                PRIMARY KEY (asset, ts)
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                asset TEXT,
                ts TIMESTAMP,
                horizon_hours INTEGER,
                predicted_return DOUBLE,
                meta MAP(TEXT, TEXT),
                confidence DOUBLE,
                PRIMARY KEY (asset, ts, horizon_hours)
            );
            """
        )
        # In case existing DB lacks confidence column, try to add it
        try:
            self.con.execute("ALTER TABLE predictions ADD COLUMN confidence DOUBLE;")
        except Exception:
            pass
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                asset TEXT,
                entry_ts TIMESTAMP,
                exit_ts TIMESTAMP,
                side TEXT,
                entry_price DOUBLE,
                exit_price DOUBLE,
                size DOUBLE,
                fee DOUBLE,
                pnl DOUBLE,
                meta MAP(TEXT, TEXT)
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_state (
                asset TEXT,
                ts TIMESTAMP,
                cash DOUBLE,
                position INTEGER,
                entry_price DOUBLE,
                PRIMARY KEY (asset, ts)
            );
            """
        )

    def upsert_ohlcv(self, asset: str, df: pd.DataFrame) -> None:
        df = df.copy()
        df.insert(0, "asset", asset)
        self.con.execute("BEGIN TRANSACTION;")
        self.con.register("tmp_ohlcv", df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO ohlcv
            SELECT asset, ts, open, high, low, close, volume FROM tmp_ohlcv
            """
        )
        self.con.execute("COMMIT;")

    def read_ohlcv(self, asset: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE asset = ? ORDER BY ts",
            [asset],
        ).df()

    def upsert_features(self, asset: str, df: pd.DataFrame) -> None:
        df = df.copy()
        df.insert(0, "asset", asset)
        self.con.execute("BEGIN TRANSACTION;")
        self.con.register("tmp_features", df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO features
            SELECT asset, ts, map_from_entries(list_transform(map_keys, k -> struct_pack(key := k, value := data[k])))
            FROM (
              SELECT asset, ts, map_keys(data) AS map_keys, data FROM tmp_features
            )
            """
        )
        self.con.execute("COMMIT;")

    def read_joined(self, asset: str) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT f.ts, o.open, o.high, o.low, o.close, o.volume, f.data
            FROM features f
            JOIN ohlcv o USING (asset, ts)
            WHERE asset = ?
            ORDER BY ts
            """,
            [asset],
        ).df()

    def insert_predictions(self, asset: str, df: pd.DataFrame, horizon_hours: int, meta: Optional[dict] = None) -> None:
        df = df.copy()
        df.insert(0, "asset", asset)
        df["horizon_hours"] = horizon_hours
        if meta is not None:
            # Convert to map of text->text by casting values to str
            df["meta"] = str({k: str(v) for k, v in meta.items()})
        self.con.execute("BEGIN TRANSACTION;")
        self.con.register("tmp_preds", df)
        # Try insert with confidence; fallback without
        try:
            self.con.execute(
                """
                INSERT OR REPLACE INTO predictions
                SELECT asset, ts, horizon_hours, predicted_return, meta, confidence FROM tmp_preds
                """
            )
        except Exception:
            self.con.execute(
                """
                INSERT OR REPLACE INTO predictions
                SELECT asset, ts, horizon_hours, predicted_return, meta, NULL as confidence FROM tmp_preds
                """
            )
        self.con.execute("COMMIT;")

    def read_predictions(self, asset: str, horizon_hours: int) -> pd.DataFrame:
        return self.con.execute(
            "SELECT ts, predicted_return, confidence FROM predictions WHERE asset = ? AND horizon_hours = ? ORDER BY ts",
            [asset, horizon_hours],
        ).df()

    def insert_trades(self, trades_df: pd.DataFrame) -> None:
        self.con.register("tmp_trades", trades_df)
        self.con.execute(
            """
            INSERT INTO trades
            SELECT asset, entry_ts, exit_ts, side, entry_price, exit_price, size, fee, pnl, meta FROM tmp_trades
            """
        )

    def insert_paper_state(self, asset: str, state_df: pd.DataFrame) -> None:
        state_df = state_df.copy()
        state_df.insert(0, "asset", asset)
        self.con.register("tmp_paper", state_df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO paper_state
            SELECT asset, ts, cash, position, entry_price FROM tmp_paper
            """
        )