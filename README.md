# Forecast Dashboard (Baba Jina)

Interactive Streamlit dashboard that combines three forecasting models:
- **Costume (Halloween)** — SARIMA, scaled ×25
- **Toys** — Prophet with multiplicative seasonality + Oct/Dec regressors
- **Bicycles** — Prophet, scaled ×10

## Setup
1. Python 3.10 recommended.
2. Clone repo and install deps:
   ```bash
   pip install -r requirements.txt
