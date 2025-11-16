# regime-aware statisticalarbitrage engine
Advanced Statistical Arbitrage with Regime Detection  This system solves the regime-change problem in quantitative finance through: - Hidden Markov Models classifying market states in real-time - Dynamic strategy parameters that adapt to current volatility regimes - Correlation break detection and automated risk reduction - Multi-asset pairs trading with cointegration validation - Professional risk management with VaR, CVaR, and stress testing  The architecture ensures strategies survive when traditional arbitrage approaches fail during market structural breaks. 
  OR 
# 🎯 Jane Street Style Regime-Aware Statistical Arbitrage

A professional quantitative trading system implementing regime-aware statistical arbitrage strategies inspired by Jane Street's approach to surviving market regime changes.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Quantitative Finance](https://img.shields.io/badge/Quantitative-Finance-green)
![Machine Learning](https://img.shields.io/badge/ML-HMM-orange)

## 📊 Overview

Traditional statistical arbitrage strategies often break during market regime shifts. This project implements a **regime-aware** approach that dynamically adapts to changing market conditions using Hidden Markov Models, ensuring robust performance across bull, sideways, and bear markets.

## 🚀 Key Features

- **📈 Regime Detection** - Hidden Markov Models for real-time market regime identification
- **⚖️ Statistical Arbitrage** - Cointegrated pairs trading with dynamic thresholds
- **🛡️ Risk Management** - Regime-aware position sizing and risk controls
- **🎯 Portfolio Optimization** - Modern Portfolio Theory implementation
- **📊 Advanced Analytics** - VaR, CVaR, Sharpe, Alpha/Beta calculations
- **💹 Market Microstructure** - Realistic trading costs and slippage modeling

## 🏗️ Architecture
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Regime │ │ Pairs │ │ Risk │
│ Detection │───▶│ Trading │───▶│ Management │
│ (HMM) │ │ Engine │ │ System │
└─────────────────┘ └──────────────────┘ └─────────────────┘
│ │ │
▼ ▼ ▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Portfolio │ │ Performance │ │ Backtest │
│ Optimizer │ │ Analytics │ │ Engine │
└─────────────────┘ └──────────────────┘ └─────────────────┘

text

## ⚡ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/likithashashishekar/jane-street-arbitrage.git
cd jane-street-arbitrage

# Install dependencies
pip install -r requirements.txt
Run Strategy
bash
python jane_street_arbitrage_advanced.py
The system will:

✅ Auto-install any missing dependencies

✅ Generate synthetic market data

✅ Run complete backtest with regime detection

✅ Generate professional performance charts

✅ Display advanced analytics and metrics

📈 Strategy Details
Regime Detection
Uses Hidden Markov Models (HMM) to identify 3 market regimes:

Bull Market (Low volatility, high leverage)

Sideways Market (Medium volatility, moderate leverage)

Bear Market (High volatility, conservative leverage)

Pairs Trading
Finds cointegrated asset pairs using statistical tests

Implements mean reversion strategies with dynamic z-score thresholds

Adjusts entry/exit points based on current market regime

Risk Management
Dynamic position sizing per regime

Correlation break detection

Value at Risk (VaR) and Conditional VaR monitoring

📊 Sample Output
text
=== BACKTEST RESULTS ===
Total Return: 15.23%
Annualized Volatility: 8.45%
Sharpe Ratio: 1.80
Maximum Drawdown: -4.32%

Regime Distribution:
  Bull: 45 days (35.2%)
  Sideways: 52 days (40.6%) 
  Bear: 31 days (24.2%)
🛠️ Technical Stack
Python 3.8+ - Core programming language

NumPy/SciPy - Scientific computing and optimization

pandas - Data manipulation and analysis

hmmlearn - Hidden Markov Models for regime detection

matplotlib - Professional visualization

scikit-learn - Machine learning utilities

📁 Project Structure
text
jane-street-arbitrage/
├── jane_street_arbitrage_advanced.py  # Main strategy file
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
└── .gitignore                         # Git ignore rules
🎯 Key Innovations
Regime Resilience - Strategy parameters adapt to market conditions

Dynamic Risk Management - Position sizing changes with volatility regimes

Robust Pair Selection - Statistical cointegration with fallback mechanisms

Realistic Modeling - Includes transaction costs and market impact

🤝 Contributing
This project is open for improvements and extensions:

Add live trading capabilities

Incorporate more asset classes

Enhance with deep learning models

Optimize for higher frequency trading

📄 License
MIT License - feel free to use this code for research and educational purposes.

🙏 Acknowledgments
Inspired by Jane Street's approach to quantitative trading and regime-aware strategy design.
