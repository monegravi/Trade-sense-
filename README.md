## Trading Bot (Multi-Asset: Crypto + Gold)

Features:
- 1h candlesticks (crypto via CCXT, gold via Yahoo Finance) with continuous refresh
- Technical indicators, sentiment, macro features
- Transformer models (Autoformer, PyTorch) with Optuna auto-tuning
- Feature selection loop with SHAP/permutation importance
- Anomaly detection, regime changes
- Backtest with costs, TP/SL, ROI; paper trading
- Telegram notifications; weekly/monthly prediction summaries

### Quick start
1. Create and activate a virtualenv, then install dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```
2. Set environment variables (or copy `.env.example` to `.env`):
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
FRED_API_KEY=optional_fred_key
```
3. Edit `config/config.yaml` as needed.
4. Run the orchestrator:
```
python -m trading_bot.main --task run_daily
```

### Structure
- `trading_bot/data`: data ingestion and preprocessing
- `trading_bot/features`: indicators and feature selection
- `trading_bot/model`: models and training
- `trading_bot/backtest`: backtesting and paper trading
- `trading_bot/notify`: notifications (Telegram)
- `trading_bot/utils`: logging, config

### Notes
- Autoformer requires sufficient history; use at least 1-2k hourly bars.
- If some data sources or APIs are unavailable, components will fail gracefully and log warnings.