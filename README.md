## Trading Bot (Multi-Asset: Crypto + Gold)

Features:
- 1h candlesticks (crypto via CCXT, gold via Yahoo Finance) with continuous refresh
- Technical indicators, sentiment, macro features
- Sequence models: Autoformer + LightGBM baseline on flattened sequences, ensembling
- Walk-forward CV, threshold optimization, SHAP feature importance
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

### Docker
Build and run:
```
docker build -t trading-bot .
docker run --rm -e TELEGRAM_BOT_TOKEN=xxx -e TELEGRAM_CHAT_ID=yyy trading-bot
```

### CI
- GitHub Actions workflow at `.github/workflows/ci.yml` installs deps and performs a basic import check.

### Notes
- Autoformer uses last-step features from each sequence; LightGBM uses flattened sequences; predictions are ensembled.
- Threshold for trade signals is optimized via walk-forward backtest on recent data.
- Ensure enough history (>= 2000 1h bars) for stable training.