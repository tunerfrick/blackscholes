"""
MIT License
Copyright (c) 2026 Vikramadity Shukla (PolyU Physics Quant Projects)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Streamlit Black-Scholes-Merton Pricer with Greeks, live yfinance, Plotly viz.

Install: pip install -r requirements.txt
Run:     streamlit run app.py
Deploy:  Streamlit Cloud / Railway

requirements.txt:
streamlit
numpy
scipy
yfinance
plotly
pandas
requests
"""

import streamlit as st
import html
import json
import numpy as np
import os
import re
import requests
from scipy.stats import norm
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components  # Optional confetti

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BS Pricer Pro",
    page_icon="BS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base / background ──────────────────────────────────────────────────── */
.reportview-container,
.main .block-container,
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 60%, #0d1520 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117 0%, #141824 100%);
    border-right: 1px solid #2a2f3e;
}

/* ── Typography overrides ───────────────────────────────────────────────── */
h1, h2, h3, .stTitle { color: #e8f0fe !important; }
p, label, .stMarkdown { color: #b0bec5 !important; }

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,181,246,0.2);
    border-radius: 12px;
    padding: 14px 18px !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: rgba(100,181,246,0.5); }
[data-testid="stMetricLabel"]  { color: #78909c !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"]  { color: #64b5f6 !important; font-size: 1.6rem !important; font-weight: 700; }
[data-testid="stMetricDelta"]  { color: #81c784 !important; }

/* ── LLM market context card ───────────────────────────────────────────── */
.market-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(129,199,132,0.26);
    border-radius: 12px;
    padding: 18px 20px;
    margin: 8px 0 22px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}
.market-card__top {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: flex-start;
    margin-bottom: 14px;
}
.market-card__title {
    color: #e8f0fe;
    font-size: 1.05rem;
    font-weight: 700;
}
.market-card__subtitle {
    color: #78909c;
    font-size: 0.82rem;
    margin-top: 2px;
}
.market-card__badge {
    color: #0e1117;
    background: #81c784;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    white-space: nowrap;
}
.market-card__grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 14px;
}
.market-card__label {
    color: #78909c;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.market-card__value {
    color: #e8f0fe;
    font-size: 0.95rem;
    font-weight: 650;
    margin-top: 3px;
}
.market-card__explanation {
    color: #cfd8dc;
    font-size: 0.96rem;
    line-height: 1.5;
}
@media (max-width: 760px) {
    .market-card__top { flex-direction: column; }
    .market-card__grid { grid-template-columns: 1fr; }
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(90deg, #1565c0, #0288d1);
    border: none;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    transition: opacity 0.2s;
}
[data-testid="stButton"] button[kind="primary"]:hover { opacity: 0.88; }
[data-testid="stButton"] button:not([kind="primary"]) {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(100,181,246,0.25);
    color: #90caf9;
    border-radius: 8px;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]  { gap: 8px; border-bottom: 1px solid #2a2f3e; }
.stTabs [data-baseweb="tab"]       { background: transparent; border-radius: 8px 8px 0 0; color: #78909c; padding: 8px 18px; }
.stTabs [aria-selected="true"]     { background: rgba(100,181,246,0.12) !important; color: #64b5f6 !important; border-bottom: 2px solid #64b5f6 !important; }

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] { background: rgba(255,255,255,0.03); border: 1px solid #2a2f3e; border-radius: 10px; }

/* ── Dataframe ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Divider ────────────────────────────────────────────────────────────── */
hr { border-color: #2a2f3e; }

/* ── Caption ────────────────────────────────────────────────────────────── */
.stCaption { color: #546e7a !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH (cached 10 min)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def fetch_data(symbol: str):
    """
    Fetch last price, annualised historical volatility, 30-day realised vol,
    and a proxy risk-free rate from yfinance.
    Falls back to sensible defaults on any error.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo", interval="1d")
        if hist.empty:
            raise ValueError("No data returned for symbol")
        S = float(hist["Close"].iloc[-1])
        log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        sigma = float(log_returns.std() * np.sqrt(252))
        realized_vol_30d = float(log_returns.tail(30).std() * np.sqrt(252))
        # Simple proxy: HIBOR-like for HK tickers, Fed funds proxy otherwise
        r_default = 0.045 if ("HSI" in symbol.upper() or ".HK" in symbol.upper()) else 0.04
        return S, max(sigma, 0.01), max(realized_vol_30d, 0.01), r_default
    except Exception:
        return 200.0, 0.30, 0.30, 0.04


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES-MERTON CORE
# ─────────────────────────────────────────────────────────────────────────────
def black_scholes(S: float, K: float, T: float, r: float,
                  sigma: float, q: float, option_type: str):
    """
    Returns (price, d1, d2) for a European call or put.

    Parameters
    ----------
    S           : current underlying price
    K           : strike price
    T           : time to expiry in years  (must be > 0)
    r           : continuously-compounded risk-free rate
    sigma       : annualised implied / historical volatility
    q           : continuous dividend yield
    option_type : 'call' or 'put'
    """
    T = max(T, 1e-8)  # guard against exactly 0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = (S * np.exp(-q * T) * norm.cdf(d1)
                 - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)
                 - S * np.exp(-q * T) * norm.cdf(-d1))
    return float(price), float(d1), float(d2)


# ─────────────────────────────────────────────────────────────────────────────
# GREEKS
# ─────────────────────────────────────────────────────────────────────────────
def greeks(S: float, K: float, T: float, r: float,
           sigma: float, q: float, option_type: str,
           price: float, d1: float, d2: float):
    """
    Returns (delta, gamma, theta_annual, vega_per1pct, rho_per1pct).

    Theta is expressed on an annual basis (divide by 365 for daily).
    Vega  is per 1 percentage-point move in vol.
    Rho   is per 1 percentage-point move in r.
    """
    T = max(T, 1e-8)
    sqrt_T = np.sqrt(T)
    nd1    = norm.pdf(d1)          # standard normal PDF at d1

    # Delta
    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)

    # Gamma (same for call and put)
    gamma = np.exp(-q * T) * nd1 / (S * sigma * sqrt_T)

    # Theta (annual)
    common_theta = (- (S * sigma * np.exp(-q * T) * nd1) / (2 * sqrt_T)
                    - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2)))
    if option_type == "call":
        theta_ann = common_theta + q * S * np.exp(-q * T) * norm.cdf(d1)
    else:
        theta_ann = (- (S * sigma * np.exp(-q * T) * nd1) / (2 * sqrt_T)
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))

    # Vega per 1 % move in sigma
    vega = S * np.exp(-q * T) * nd1 * sqrt_T * 0.01

    # Rho per 1 % move in r
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01

    return (float(delta), float(gamma), float(theta_ann),
            float(vega), float(rho))


def implied_vol_newton_raphson(market_price: float, S: float, K: float, T: float,
                               r: float, q: float, option_type: str,
                               initial_guess: float = 0.30,
                               max_iter: int = 100,
                               tolerance: float = 1e-6):
    """
    Solve Black-Scholes-Merton implied volatility from a market option price.
    Returns None when the price cannot be inverted reliably.
    """
    sigma = float(np.clip(initial_guess, 0.01, 5.0))

    for _ in range(max_iter):
        model_price, d1, _ = black_scholes(S, K, T, r, sigma, q, option_type)
        vega_decimal = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(max(T, 1e-8))

        if vega_decimal < 1e-8:
            return None

        diff = model_price - market_price
        if abs(diff) < tolerance:
            return float(sigma)

        sigma = float(np.clip(sigma - diff / vega_decimal, 0.01, 5.0))

    return None


OPENROUTER_MODEL = "deepseek/deepseek-v4-flash:free"


def get_openrouter_api_key():
    """Read the OpenRouter API key from Streamlit secrets or environment."""
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        api_key = None
    return api_key or os.getenv("OPENROUTER_API_KEY")


@st.cache_data(ttl=900)
def fetch_news_headlines(symbol: str, limit: int = 5):
    """Fetch recent yfinance headlines for a ticker."""
    try:
        news_items = yf.Ticker(symbol).news or []
    except Exception:
        return []

    headlines = []
    for item in news_items:
        content = item.get("content", {}) if isinstance(item, dict) else {}
        title = (
            item.get("title")
            or content.get("title")
            or content.get("summary")
            or ""
        )
        title = " ".join(str(title).split())
        if title and title not in headlines:
            headlines.append(title)
        if len(headlines) >= limit:
            break

    return headlines


def build_market_context_prompt(symbol: str, headlines, current_iv: float,
                                realized_vol_30d: float):
    headline_text = "\n".join(
        f"{idx}. {headline}" for idx, headline in enumerate(headlines, start=1)
    )
    if not headline_text:
        headline_text = "No recent headlines were available from yfinance."

    return f"""
You are an options market analyst. Use only the supplied headlines and volatility
inputs. Be concise, practical, and do not invent unsupported facts.

Ticker: {symbol.upper()}
Current implied volatility: {current_iv * 100:.2f}%
30-day realized volatility: {realized_vol_30d * 100:.2f}%
Recent headlines:
{headline_text}

Tasks:
1. Identify the likely macro or micro driver of current IV levels.
2. Classify the volatility regime as exactly one of:
   Low Vol, Trending, Event Risk, Crisis, Post-Event Mean Reversion.
3. Give a two-sentence plain English explanation a trader would actually find useful.
4. Assess current IV as rich, fair, or cheap versus the realized-vol/news backdrop.

Return only valid JSON with this exact schema:
{{
  "regime": "Low Vol|Trending|Event Risk|Crisis|Post-Event Mean Reversion",
  "driver": "short driver phrase",
  "explanation": "exactly two sentences",
  "iv_assessment": "rich|fair|cheap"
}}
""".strip()


def extract_json_object(text: str):
    """Parse a JSON object, allowing for an accidental markdown wrapper."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@st.cache_data(ttl=900, show_spinner=False)
def get_llm_market_context(symbol: str, headlines, current_iv: float,
                           realized_vol_30d: float):
    """Call OpenRouter DeepSeek for market context. Cached for 15 minutes."""
    api_key = get_openrouter_api_key()
    if not api_key:
        raise RuntimeError(
            "Set `OPENROUTER_API_KEY` in Streamlit secrets or your environment."
        )

    prompt = build_market_context_prompt(
        symbol=symbol,
        headlines=list(headlines),
        current_iv=current_iv,
        realized_vol_30d=realized_vol_30d,
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "BS Pricer Pro",
        },
        json={
            "model": OPENROUTER_MODEL,
            "max_tokens": 500,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You produce compact, valid JSON for an options trader. "
                        "Never include markdown, commentary, or keys outside the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    text = payload["choices"][0]["message"]["content"]
    parsed = extract_json_object(text)

    allowed_regimes = {
        "Low Vol", "Trending", "Event Risk", "Crisis", "Post-Event Mean Reversion"
    }
    allowed_assessments = {"rich", "fair", "cheap"}

    regime = parsed.get("regime", "Trending")
    assessment = str(parsed.get("iv_assessment", "fair")).lower()

    return {
        "regime": regime if regime in allowed_regimes else "Trending",
        "driver": str(parsed.get("driver", "News and volatility backdrop"))[:140],
        "explanation": str(parsed.get("explanation", "No explanation returned."))[:600],
        "iv_assessment": assessment if assessment in allowed_assessments else "fair",
    }


def render_market_context_card(context: dict, current_iv: float,
                               realized_vol_30d: float):
    badge_colors = {
        "rich": "#ef9a9a",
        "fair": "#81c784",
        "cheap": "#64b5f6",
    }
    assessment = context["iv_assessment"]
    badge_color = badge_colors.get(assessment, "#81c784")
    regime = html.escape(context["regime"])
    driver = html.escape(context["driver"])
    explanation = html.escape(context["explanation"])
    assessment_label = html.escape(assessment)

    st.markdown(
        f"""
        <div class="market-card">
            <div class="market-card__top">
                <div>
                    <div class="market-card__title">LLM Market Context</div>
                    <div class="market-card__subtitle">OpenRouter DeepSeek · cached for 15 minutes</div>
                </div>
                <div class="market-card__badge" style="background:{badge_color};">
                    IV {assessment_label}
                </div>
            </div>
            <div class="market-card__grid">
                <div>
                    <div class="market-card__label">Regime</div>
                    <div class="market-card__value">{regime}</div>
                </div>
                <div>
                    <div class="market-card__label">Driver</div>
                    <div class="market-card__value">{driver}</div>
                </div>
                <div>
                    <div class="market-card__label">IV / RV30</div>
                    <div class="market-card__value">{current_iv * 100:.1f}% / {realized_vol_30d * 100:.1f}%</div>
                </div>
            </div>
            <div class="market-card__explanation">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    symbol = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        help="Examples: AAPL, TSLA, ^HSI, 0700.HK",
    )

    if st.button("Fetch Live Data", use_container_width=True):
        with st.spinner(f"Fetching {symbol}…"):
            S_live, sigma_live, realized_vol_30d_live, r_live = fetch_data(symbol)
        st.session_state["S"]     = S_live
        st.session_state["sigma"] = sigma_live
        st.session_state["realized_vol_30d"] = realized_vol_30d_live
        st.session_state["r"]     = r_live
        st.success(
            f"S={S_live:.2f}  σ={sigma_live*100:.1f}%  "
            f"RV30={realized_vol_30d_live*100:.1f}%  r={r_live*100:.1f}%"
        )

    st.divider()

    opt_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)

    K = st.number_input(
        "Strike Price K ($)",
        min_value=1.0,
        max_value=100_000.0,
        value=float(st.session_state.get("S", 200.0)),
        step=1.0,
    )

    T = st.slider(
        "Time to Expiry τ (years)",
        min_value=0.003,   # ~1 day
        max_value=5.0,
        value=0.25,
        step=0.001,
        format="%.3f",
    )

    # Use fetched value as slider default, but always let user override
    r_default   = st.session_state.get("r",     0.04)
    sig_default = st.session_state.get("sigma", 0.30)

    r_slider = st.slider(
        "Risk-Free Rate r (%)",
        min_value=0.0,
        max_value=20.0,
        value=round(r_default * 100, 2),
        step=0.05,
        format="%.2f",
    ) / 100.0

    sigma_slider = st.slider(
        "Volatility σ (%)",
        min_value=1.0,
        max_value=200.0,
        value=round(sig_default * 100, 1),
        step=0.5,
        format="%.1f",
    ) / 100.0

    q = st.slider(
        "Continuous Dividend Yield q (%)",
        min_value=0.0,
        max_value=15.0,
        value=0.0,
        step=0.1,
        format="%.1f",
    ) / 100.0

    st.divider()
    price_btn = st.button("Price & Greeks", type="primary", use_container_width=True)
    if price_btn:
        st.session_state["calculated"] = True


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("Black-Scholes-Merton Option Pricer Pro")
st.markdown(
    "**Live pricing · Greeks · Interactive viz &nbsp;|&nbsp; "
    "AAPL / ^HSI ready &nbsp;|&nbsp; PolyU Quant Hackathon Demo**"
)
st.caption(
    "Built by Vikramadity Shukla &nbsp;|&nbsp; "
    "[GitHub](https://github.com/tunerfrick/blackscholes) &nbsp;|&nbsp; "
    f"Last updated: {datetime.now().strftime('%d %b %Y')}"
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN OUTPUT  (only after user clicks the button)
# ─────────────────────────────────────────────────────────────────────────────
if "calculated" not in st.session_state:
    # Nice splash while waiting for first calculation
    st.info(
        "**Set your parameters in the sidebar**, then click **Price & Greeks** "
        "to see results.\n\n"
        "Tip: hit **Fetch Live Data** first to auto-populate S, σ, and r for any symbol."
    )
else:
    # ── Resolve current spot price ─────────────────────────────────────────
    S_final     = float(st.session_state.get("S", 200.0))
    r_final     = r_slider
    sigma_final = sigma_slider

    # ── Core calculations ──────────────────────────────────────────────────
    price, d1, d2 = black_scholes(
        S_final, K, T, r_final, sigma_final, q, opt_type.lower()
    )
    delta, gamma, theta_ann, vega, rho = greeks(
        S_final, K, T, r_final, sigma_final, q, opt_type.lower(),
        price, d1, d2
    )
    intrinsic = max((S_final - K) if opt_type == "Call" else (K - S_final), 0.0)
    time_val  = price - intrinsic
    moneyness = S_final / K

    # ── KPI row ────────────────────────────────────────────────────────────
    st.subheader("Pricing Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Option Price",   f"${price:.4f}")
    c2.metric("Intrinsic Value", f"${intrinsic:.4f}")
    c3.metric("Time Value",     f"${time_val:.4f}")
    c4.metric("Moneyness (S/K)", f"{moneyness:.4f}",
              delta=("ITM" if moneyness > 1 else ("ATM" if abs(moneyness-1) < 0.005 else "OTM")))

    st.subheader("Greeks")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Delta (Δ)",       f"{delta:.4f}")
    g2.metric("Gamma (Γ)",       f"{gamma:.6f}")
    g3.metric("Theta θ (ann)",   f"{theta_ann:.4f}")
    g4.metric("Vega ν (per 1%)", f"${vega:.4f}")
    g5.metric("Rho ρ (per 1%)",  f"${rho:.4f}")

    # ── d1 / d2 info ───────────────────────────────────────────────────────
    info1, info2 = st.columns(2)
    info1.metric("d₁", f"{d1:.4f}")
    info2.metric("d₂", f"{d2:.4f}")

    # ── LLM market context ─────────────────────────────────────────────────
    solved_iv = implied_vol_newton_raphson(
        market_price=price,
        S=S_final,
        K=K,
        T=T,
        r=r_final,
        q=q,
        option_type=opt_type.lower(),
        initial_guess=sigma_final,
    ) or sigma_final
    current_iv = float(
        st.session_state.get("current_iv", st.session_state.get("iv", solved_iv))
    )

    if "realized_vol_30d" not in st.session_state:
        _, _, realized_vol_30d, _ = fetch_data(symbol)
    else:
        realized_vol_30d = float(st.session_state["realized_vol_30d"])

    st.subheader("Market Context")
    headlines = fetch_news_headlines(symbol)
    if not headlines:
        st.caption("No yfinance headlines found for this ticker; DeepSeek will rely on IV versus realized vol.")

    with st.spinner("Reading headlines and volatility backdrop with DeepSeek..."):
        try:
            market_context = get_llm_market_context(
                symbol=symbol,
                headlines=tuple(headlines),
                current_iv=current_iv,
                realized_vol_30d=realized_vol_30d,
            )
            render_market_context_card(
                context=market_context,
                current_iv=current_iv,
                realized_vol_30d=realized_vol_30d,
            )
        except Exception as exc:
            st.warning(f"Market context unavailable: {exc}")

    st.divider()

    # ── Formula expander ───────────────────────────────────────────────────
    with st.expander("Black-Scholes-Merton Formulae"):
        st.latex(r"""
            d_1 = \frac{\ln(S/K)\;+\;(r - q + \tfrac{1}{2}\sigma^2)\,T}{\sigma\,\sqrt{T}}
        """)
        st.latex(r"d_2 = d_1 - \sigma\,\sqrt{T}")
        st.latex(r"\text{Call} = S\,e^{-qT}\,N(d_1) - K\,e^{-rT}\,N(d_2)")
        st.latex(r"\text{Put}  = K\,e^{-rT}\,N(-d_2) - S\,e^{-qT}\,N(-d_1)")
        st.latex(r"\Delta_{\text{call}} = e^{-qT}\,N(d_1) \qquad \Delta_{\text{put}} = -e^{-qT}\,N(-d_1)")
        st.latex(r"\Gamma = \frac{e^{-qT}\,n(d_1)}{S\,\sigma\,\sqrt{T}}")
        st.latex(r"\nu = S\,e^{-qT}\,n(d_1)\,\sqrt{T}\;\times\;0.01")
        st.latex(r"\rho_{\text{call}} = K\,T\,e^{-rT}\,N(d_2)\;\times\;0.01")
        st.markdown(
            "_N(·) = standard normal CDF &nbsp;|&nbsp; "
            "n(·) = standard normal PDF_"
        )

    # ── Data table + export ────────────────────────────────────────────────
    df = pd.DataFrame({
        "Metric":  ["Spot S", "Strike K", "τ (yrs)", "r", "σ", "q",
                    "d₁", "d₂", "Price", "Delta", "Gamma",
                    "Theta (ann)", "Vega (1%)", "Rho (1%)"],
        "Value":   [S_final, K, T, r_final, sigma_final, q,
                    d1, d2, price, delta, gamma,
                    theta_ann, vega, rho],
        "Unit":    ["$", "$", "yr", "%", "%", "%",
                    "", "", "$", "", "",
                    "$/yr", "$/1%σ", "$/1%r"],
    })
    df["Value"] = df["Value"].apply(lambda x: round(x, 6))

    st.subheader("Full Results Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

    col_dl1, col_dl2, _ = st.columns([1, 1, 3])
    with col_dl1:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"bs_greeks_{symbol}_{opt_type}_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )
    with col_dl2:
        st.download_button(
            "Download JSON",
            data=df.to_json(orient="records", indent=2).encode(),
            file_name=f"bs_greeks_{symbol}_{opt_type}_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json",
        )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────
    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.9)",
        font=dict(color="#b0bec5", family="monospace"),
        title_font=dict(color="#e8f0fe", size=16),
        xaxis=dict(gridcolor="#1e2533", zeroline=False),
        yaxis=dict(gridcolor="#1e2533", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2f3e"),
        margin=dict(l=60, r=30, t=60, b=50),
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Payoff Diagram", "Greeks Heatmap", "Vol Smile", "Sensitivity"]
    )

    # ── Tab 1: Payoff ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Option Payoff at Expiry")
        S_range = np.linspace(max(0.1, 0.4 * K), 1.8 * K, 300)

        if opt_type == "Call":
            intrinsic_curve = np.maximum(S_range - K, 0)
            pnl_curve       = np.maximum(S_range - K, 0) - price
        else:
            intrinsic_curve = np.maximum(K - S_range, 0)
            pnl_curve       = np.maximum(K - S_range, 0) - price

        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(
            x=S_range, y=intrinsic_curve,
            name="Intrinsic Value at Expiry",
            line=dict(color="#64b5f6", width=2),
            fill="tozeroy", fillcolor="rgba(100,181,246,0.08)",
        ))
        fig_pay.add_trace(go.Scatter(
            x=S_range, y=pnl_curve,
            name="P&L (net of premium)",
            line=dict(color="#81c784", width=2, dash="dash"),
        ))
        fig_pay.add_vline(
            x=S_final, line_color="#ffd54f", line_dash="dot",
            annotation_text=f"  S={S_final:.2f}", annotation_font_color="#ffd54f",
        )
        fig_pay.add_vline(
            x=K, line_color="#ef9a9a", line_dash="dot",
            annotation_text=f"  K={K:.2f}", annotation_font_color="#ef9a9a",
        )
        fig_pay.add_hline(y=0, line_color="#546e7a", line_width=0.8)
        fig_pay.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{opt_type} Option Payoff — K={K}  τ={T:.3f}yr",
            xaxis_title="Underlying Price S ($)",
            yaxis_title="Value ($)",
        )
        st.plotly_chart(fig_pay, use_container_width=True)

    # ── Tab 2: Delta Heatmap ───────────────────────────────────────────────
    with tab2:
        st.markdown("#### Delta Heatmap  (S × σ grid)")
        n_grid  = 25
        S_grid  = np.linspace(0.7 * S_final, 1.3 * S_final, n_grid)
        sig_grid = np.linspace(0.05, 0.70, n_grid)
        DELTA_MAT = np.zeros((n_grid, n_grid))

        for i, s_val in enumerate(S_grid):
            for j, sig_val in enumerate(sig_grid):
                _, _d1, _ = black_scholes(
                    s_val, K, T, r_final, sig_val, q, opt_type.lower()
                )
                if opt_type == "Call":
                    DELTA_MAT[i, j] = np.exp(-q * T) * norm.cdf(_d1)
                else:
                    DELTA_MAT[i, j] = -np.exp(-q * T) * norm.cdf(-_d1)

        fig_hm = px.imshow(
            DELTA_MAT,
            x=np.round(sig_grid * 100, 1),
            y=np.round(S_grid, 2),
            labels={"x": "Volatility σ (%)", "y": "Spot S ($)", "color": "Delta"},
            color_continuous_scale="RdYlGn",
            zmin=-1 if opt_type == "Put" else 0,
            zmax=0  if opt_type == "Put" else 1,
            aspect="auto",
            title=f"{opt_type} Delta Heatmap",
        )
        fig_hm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#b0bec5"))
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Tab 3: Volatility Smile ────────────────────────────────────────────
    with tab3:
        st.markdown("#### Price vs. Volatility  (Smile Curve)")
        sig_range   = np.linspace(0.01, 1.50, 200)
        call_prices = [black_scholes(S_final, K, T, r_final, s, q, "call")[0] for s in sig_range]
        put_prices  = [black_scholes(S_final, K, T, r_final, s, q, "put")[0]  for s in sig_range]

        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=sig_range * 100, y=call_prices,
            name="Call Price", line=dict(color="#64b5f6", width=2),
        ))
        fig_smile.add_trace(go.Scatter(
            x=sig_range * 100, y=put_prices,
            name="Put Price", line=dict(color="#ef9a9a", width=2),
        ))
        fig_smile.add_vline(
            x=sigma_final * 100, line_dash="dot", line_color="#ffd54f",
            annotation_text=f"  σ={sigma_final*100:.1f}%",
            annotation_font_color="#ffd54f",
        )
        fig_smile.update_layout(
            **PLOTLY_LAYOUT,
            title="Option Price vs Volatility",
            xaxis_title="Volatility σ (%)",
            yaxis_title="Option Price ($)",
        )
        st.plotly_chart(fig_smile, use_container_width=True)

    # ── Tab 4: Sensitivity (all Greeks vs S) ──────────────────────────────
    with tab4:
        st.markdown("#### Greeks vs. Underlying Price")
        S_sweep = np.linspace(max(0.1, 0.5 * K), 1.5 * K, 200)
        deltas, gammas, vegas_list, thetas = [], [], [], []

        for s_val in S_sweep:
            p_, d1_, d2_ = black_scholes(s_val, K, T, r_final, sigma_final, q, opt_type.lower())
            dl, gm, th, vg, _ = greeks(s_val, K, T, r_final, sigma_final, q, opt_type.lower(), p_, d1_, d2_)
            deltas.append(dl); gammas.append(gm)
            vegas_list.append(vg); thetas.append(th)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=S_sweep, y=deltas,     name="Delta",       line=dict(color="#64b5f6")))
        fig_sens.add_trace(go.Scatter(x=S_sweep, y=[g * 10 for g in gammas],
                                      name="Gamma ×10", line=dict(color="#81c784")))
        fig_sens.add_trace(go.Scatter(x=S_sweep, y=vegas_list, name="Vega (1%)",   line=dict(color="#ffb74d")))
        fig_sens.add_vline(
            x=S_final, line_color="#ffd54f", line_dash="dot",
            annotation_text=f"  S={S_final:.2f}", annotation_font_color="#ffd54f",
        )
        fig_sens.update_layout(
            **PLOTLY_LAYOUT,
            title="Greeks as a Function of Underlying Price",
            xaxis_title="Underlying Price S ($)",
            yaxis_title="Greek Value",
        )
        st.plotly_chart(fig_sens, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#546e7a; font-size:0.82rem;'>"
    "<a href='https://github.com/tunerfrick/blackscholes' style='color:#64b5f6;'>GitHub</a> &nbsp;|&nbsp; "
    "Deployed on <a href='https://streamlit.io/cloud' style='color:#64b5f6;'>Streamlit Cloud</a> &nbsp;|&nbsp; "
    "Black-Scholes-Merton model · Greeks per standard quant finance convention"
    "</div>",
    unsafe_allow_html=True,
)
