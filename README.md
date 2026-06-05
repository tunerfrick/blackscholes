# Black-Scholes-Merton Option Pricer Pro

A Streamlit options pricing app I built for quickly playing around with Black-Scholes-Merton pricing, Greeks, live market data, and now a little LLM-powered market context.

It is basically a quant finance playground: enter a ticker, pull live-ish data from Yahoo Finance, price a European call or put, inspect the Greeks, and get a quick volatility read using recent headlines + IV/RV context through OpenRouter DeepSeek.

## What It Does

- Prices European calls and puts using the Black-Scholes-Merton model
- Calculates Delta, Gamma, Theta, Vega, and Rho
- Pulls ticker data with `yfinance`
- Estimates annualized historical volatility and 30-day realized volatility
- Fetches recent headlines from `yfinance.Ticker(symbol).news`
- Sends headlines + current IV + 30-day realized vol to DeepSeek through OpenRouter
- Classifies the market regime:
  - Low Vol
  - Trending
  - Event Risk
  - Crisis
  - Post-Event Mean Reversion
- Shows payoff diagrams, heatmaps, vol curves, and sensitivity charts
- Lets you export results as CSV or JSON

## LLM Market Context

The LLM panel is meant to answer the question I usually have after seeing an IV number:

> Is this vol actually saying something, or is it just noise?

The app asks DeepSeek to return structured JSON like:

```json
{
  "regime": "Event Risk",
  "driver": "earnings and single-name news flow",
  "explanation": "IV looks elevated because traders are pricing near-term uncertainty around the latest headlines. If realized vol does not pick up, this setup can start to look rich pretty quickly.",
  "iv_assessment": "rich"
}
```

The call is cached for 15 minutes with `st.cache_data`, because burning API credits on every slider move would be tragic and also very avoidable.

Current model:

```python
deepseek/deepseek-v4-flash:free
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a Streamlit secrets file:

```toml
# .streamlit/secrets.toml
OPENROUTER_API_KEY = "sk-or-your-key-here"
```

Run the app:

```bash
streamlit run app.py
```

## Notes

This is not meant to be a trading signal machine. It is a pricing and intuition tool.

The model is still Black-Scholes-Merton, so all the usual assumptions apply: lognormal underlying, constant volatility, constant rates, European exercise, and no magic. The LLM panel is just a market-context layer on top, useful for summarizing headlines and thinking about whether IV looks rich, fair, or cheap versus realized vol.

## Stack

- Python
- Streamlit
- NumPy
- SciPy
- pandas
- Plotly
- yfinance
- OpenRouter API
- DeepSeek

## Project Files

- `app.py` - main Streamlit app
- `requirements.txt` - Python dependencies
- `.streamlit/secrets.toml` - local API key config, do not commit real keys

## Tiny Disclaimer

This is for learning, demos, and quant tinkering. Do your own risk management. The market does not care that the dashboard looks nice.
