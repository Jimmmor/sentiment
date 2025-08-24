import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import requests
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings
import re
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="CSI-Q Terminal",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Terminal CSS
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background-color: #0a0a0a;
        color: #00ff41;
    }
    
    .main .block-container {
        padding-top: 1rem;
        background-color: #0a0a0a;
    }
    
    /* Terminal-style metric cards */
    .terminal-metric {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid #00ff41;
        padding: 20px;
        border-radius: 4px;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        text-align: center;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    .terminal-metric h3 {
        color: #ffffff;
        margin: 0 0 10px 0;
        font-size: 14px;
        font-weight: normal;
    }
    
    .terminal-metric h2 {
        color: #00ff41;
        margin: 0;
        font-size: 24px;
        font-family: 'Courier New', monospace;
    }
    
    /* Signal indicators */
    .signal-long {
        background: linear-gradient(135deg, #1a3d1a, #2d5a2d);
        border: 1px solid #00ff00;
        padding: 12px;
        border-radius: 4px;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin: 5px 0;
    }
    
    .signal-short {
        background: linear-gradient(135deg, #3d1a1a, #5a2d2d);
        border: 1px solid #ff4444;
        padding: 12px;
        border-radius: 4px;
        color: #ff4444;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin: 5px 0;
    }
    
    .signal-contrarian {
        background: linear-gradient(135deg, #3d3d1a, #5a5a2d);
        border: 1px solid #ffaa00;
        padding: 12px;
        border-radius: 4px;
        color: #ffaa00;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin: 5px 0;
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, #2a2a2a, #3a3a3a);
        border: 1px solid #888888;
        padding: 12px;
        border-radius: 4px;
        color: #888888;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin: 5px 0;
    }
    
    /* Terminal table style */
    .terminal-table {
        background-color: #111111;
        border: 1px solid #00ff41;
        font-family: 'Courier New', monospace;
        color: #00ff41;
    }
    
    /* Status indicators */
    .status-online {
        color: #00ff00;
        font-family: 'Courier New', monospace;
    }
    
    .status-offline {
        color: #ff4444;
        font-family: 'Courier New', monospace;
    }
    
    .status-warning {
        color: #ffaa00;
        font-family: 'Courier New', monospace;
    }
    
    /* Terminal header */
    .terminal-header {
        background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
        border: 1px solid #00ff41;
        padding: 15px;
        border-radius: 4px;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
    
    /* CSI-Q Grid */
    .csiq-grid {
        background-color: #111111;
        border: 1px solid #00ff41;
        border-radius: 4px;
        padding: 10px;
        margin: 5px;
        font-family: 'Courier New', monospace;
    }
    
    .csiq-symbol {
        color: #ffffff;
        font-weight: bold;
        font-size: 14px;
    }
    
    .csiq-score {
        font-size: 20px;
        font-weight: bold;
    }
    
    .csiq-high { color: #00ff00; }
    .csiq-medium { color: #ffaa00; }
    .csiq-low { color: #ff4444; }
    
    /* Hide Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background-color: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #00ff41;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Crypto tickers
TICKERS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
    'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
    'UNIUSDT', 'LTCUSDT', 'BCHUSDT', 'NEARUSDT', 'ALGOUSDT',
    'VETUSDT', 'FILUSDT', 'ETCUSDT', 'AAVEUSDT', 'MKRUSDT',
    'ATOMUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT'
]

class SentimentAnalyzer:
    def __init__(self):
        self.crypto_keywords = {
            'BTCUSDT': ['bitcoin', 'btc', '$btc', 'satoshi'],
            'ETHUSDT': ['ethereum', 'eth', '$eth', 'vitalik', 'gas'],
            'BNBUSDT': ['binance coin', 'bnb', '$bnb', 'binance'],
            'SOLUSDT': ['solana', 'sol', '$sol', 'solana network'],
            'XRPUSDT': ['ripple', 'xrp', '$xrp', 'ripple labs'],
            'ADAUSDT': ['cardano', 'ada', '$ada', 'charles hoskinson'],
            'AVAXUSDT': ['avalanche', 'avax', '$avax'],
            'DOTUSDT': ['polkadot', 'dot', '$dot', 'gavin wood'],
            'LINKUSDT': ['chainlink', 'link', '$link', 'sergey nazarov'],
            'MATICUSDT': ['polygon', 'matic', '$matic', 'polygon network'],
        }
        
        self.bullish_words = [
            'pump', 'moon', 'bullish', 'buy', 'long', 'rally', 'breakout', 'surge', 
            'rocket', 'lambo', 'hodl', 'diamond hands', 'to the moon', 'bull run',
            'massive gains', 'green', 'profitable', 'winner', 'strong', 'adoption',
            'partnership', 'upgrade', 'announcement', 'breakthrough', 'all time high',
            'ath', 'accumulate', 'dip buying', 'support level', 'positive', 'bullish'
        ]
        
        self.bearish_words = [
            'dump', 'crash', 'bearish', 'sell', 'short', 'decline', 'drop', 'fall',
            'red', 'paper hands', 'fear', 'panic', 'liquidation', 'bear market',
            'resistance', 'rejection', 'breakdown', 'correction', 'bubble', 'scam',
            'rugpull', 'dead cat bounce', 'capitulation', 'blood bath', 'massacre',
            'bearish divergence', 'sell off', 'weak hands', 'negative'
        ]
        
    def simple_sentiment_score(self, text):
        if not text:
            return 0.0
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_score = sum(1 for word in words if word in self.bullish_words)
        negative_score = sum(1 for word in words if word in self.bearish_words)
        
        total = positive_score + negative_score
        if total == 0:
            return 0.0
        return (positive_score - negative_score) / total
        
    def analyze_text_sentiment(self, text):
        if not text:
            return {'score': 0, 'magnitude': 0}
            
        sentiment_score = self.simple_sentiment_score(text)
        magnitude = abs(sentiment_score)
        
        return {
            'score': sentiment_score,
            'magnitude': magnitude,
            'bullish_mentions': sum(1 for word in self.bullish_words if word in text.lower()),
            'bearish_mentions': sum(1 for word in self.bearish_words if word in text.lower())
        }
    
    def get_demo_news_sentiment(self, symbol):
        symbol_clean = symbol.replace('USDT', '')
        
        scenarios = [
            {
                'headline': f'{symbol_clean} breaks key resistance level amid institutional buying',
                'sentiment_score': np.random.uniform(0.3, 0.8),
                'mentions': np.random.randint(50, 200),
                'source': 'CoinDesk'
            },
            {
                'headline': f'Whale movements detected in {symbol_clean}',
                'sentiment_score': np.random.uniform(-0.2, 0.4),
                'mentions': np.random.randint(30, 150),
                'source': 'Whale Alert'
            }
        ]
        
        scenario = np.random.choice(scenarios)
        
        social_mentions = {
            'twitter': np.random.randint(100, 1000),
            'reddit': np.random.randint(20, 200),
            'telegram': np.random.randint(50, 300),
            'discord': np.random.randint(10, 100)
        }
        
        total_mentions = sum(social_mentions.values()) + scenario['mentions']
        
        sample_tweets = [
            f"${symbol_clean} looking strong here, might break resistance soon",
            f"Big accumulation happening in ${symbol_clean}. Smart money loading up",
            f"${symbol_clean} RSI cooling down, good entry point",
            f"Whales are dumping ${symbol_clean}, be careful",
            f"${symbol_clean} showing bullish divergence on 4h chart",
        ]
        
        analyzed_tweets = []
        for tweet in sample_tweets:
            sentiment_data = self.analyze_text_sentiment(tweet)
            analyzed_tweets.append({
                'text': tweet,
                'sentiment': sentiment_data['score'],
                'bullish_words': sentiment_data['bullish_mentions'],
                'bearish_words': sentiment_data['bearish_mentions']
            })
        
        tweet_sentiments = [t['sentiment'] for t in analyzed_tweets]
        avg_tweet_sentiment = np.mean(tweet_sentiments) if tweet_sentiments else 0
        combined_sentiment = (scenario['sentiment_score'] * 0.4 + avg_tweet_sentiment * 0.6)
        
        return {
            'symbol': symbol_clean,
            'total_mentions': total_mentions,
            'news_sentiment': scenario['sentiment_score'],
            'social_sentiment': avg_tweet_sentiment,
            'combined_sentiment': combined_sentiment,
            'social_breakdown': social_mentions,
            'top_headline': scenario['headline'],
            'headline_source': scenario['source'],
            'sample_tweets': analyzed_tweets,
            'sentiment_magnitude': abs(combined_sentiment) * (total_mentions / 100)
        }

class MultiSourceDataFetcher:
    def __init__(self):
        self.binance_base = "https://fapi.binance.com"
        self.binance_spot = "https://api.binance.com"
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def generate_demo_data(self):
        np.random.seed(42)
        
        data_list = []
        base_prices = {
            'BTC': 43000, 'ETH': 2600, 'BNB': 310, 'SOL': 100, 'XRP': 0.52,
            'ADA': 0.48, 'AVAX': 38, 'DOT': 7.2, 'LINK': 14.5, 'MATIC': 0.85,
            'UNI': 6.8, 'LTC': 73, 'BCH': 250, 'NEAR': 2.1, 'ALGO': 0.19,
            'VET': 0.025, 'FIL': 5.5, 'ETC': 20, 'AAVE': 95, 'MKR': 1450,
            'ATOM': 9.8, 'FTM': 0.32, 'SAND': 0.42, 'MANA': 0.38, 'AXS': 6.2
        }
        
        for ticker in TICKERS:
            symbol_clean = ticker.replace('USDT', '')
            base_price = base_prices.get(symbol_clean, 1.0)
            
            price_change = np.random.normal(0, 0.05)
            current_price = base_price * (1 + price_change)
            
            change_24h = np.random.normal(0, 8)
            funding_rate = np.random.normal(0.01, 0.05)
            oi_change = np.random.normal(0, 20)
            long_short_ratio = np.random.lognormal(0, 0.5)
            
            if symbol_clean in ['BTC', 'ETH']:
                volume_24h = np.random.uniform(20000000000, 50000000000)
            elif symbol_clean in ['BNB', 'SOL', 'XRP']:
                volume_24h = np.random.uniform(1000000000, 10000000000)
            else:
                volume_24h = np.random.uniform(100000000, 2000000000)
            
            sentiment_data = self.sentiment_analyzer.get_demo_news_sentiment(ticker)
            
            rsi = 50 + np.random.normal(0, 15)
            rsi = max(0, min(100, rsi))
            bb_squeeze = np.random.uniform(0, 1)
            basis = np.random.normal(0, 0.5)
            
            derivatives_score = min(100, max(0,
                (abs(oi_change) * 2) +
                (abs(funding_rate) * 500) +
                (abs(long_short_ratio - 1) * 30) +
                30
            ))
            
            social_score = min(100, max(0,
                ((sentiment_data['combined_sentiment'] + 1) / 2 * 50) +
                (min(sentiment_data['total_mentions'], 1000) / 1000 * 30) +
                (sentiment_data['sentiment_magnitude'] * 20)
            ))
            
            basis_score = min(100, max(0, abs(basis) * 500 + 25))
            tech_score = min(100, max(0,
                (100 - abs(rsi - 50)) * 0.8 +
                ((1 - bb_squeeze) * 40) + 10
            ))
            
            csiq = (
                derivatives_score * 0.35 +
                social_score * 0.35 +
                basis_score * 0.2 +
                tech_score * 0.1
            )
            
            data_list.append({
                'Symbol': symbol_clean,
                'Price': current_price,
                'Change_24h': change_24h,
                'Funding_Rate': funding_rate,
                'OI_Change': oi_change,
                'Long_Short_Ratio': long_short_ratio,
                'Total_Mentions': sentiment_data['total_mentions'],
                'Combined_Sentiment': sentiment_data['combined_sentiment'],
                'CSI_Q': csiq,
                'Derivatives_Score': derivatives_score,
                'Social_Score': social_score,
                'Basis_Score': basis_score,
                'Tech_Score': tech_score,
                'Volume_24h': volume_24h,
                'Last_Updated': datetime.now(),
                'Data_Source': 'demo'
            })
        
        return pd.DataFrame(data_list)

@st.cache_data(ttl=60)
def fetch_crypto_data():
    fetcher = MultiSourceDataFetcher()
    return fetcher.generate_demo_data()

def get_signal_type(csiq, funding_rate, sentiment):
    if csiq > 90 or csiq < 10:
        return "CONTRARIAN"
    elif csiq > 70 and funding_rate < 0.1 and sentiment > 0.2:
        return "LONG"
    elif csiq < 30 and funding_rate > -0.1 and sentiment < -0.2:
        return "SHORT"
    else:
        return "NEUTRAL"

def get_csiq_color_class(score):
    if score >= 70:
        return "csiq-high"
    elif score >= 30:
        return "csiq-medium"
    else:
        return "csiq-low"

def main():
    # Terminal Header
    st.markdown("""
    <div class="terminal-header">
        <h2>CSI-Q TERMINAL v2.1.0</h2>
        <p>COMPOSITE SENTIMENT-QUANT INDEX | PROFESSIONAL TRADING INTERFACE</p>
        <p>STATUS: ONLINE | DATA SOURCE: DEMO MODE | LAST UPDATE: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control panel
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("REFRESH DATA", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        st.markdown('<p class="status-online">SYSTEM: OPERATIONAL</p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="status-warning">MODE: SIMULATION</p>', unsafe_allow_html=True)
    with col4:
        st.markdown('<p class="status-online">LATENCY: <50ms</p>', unsafe_allow_html=True)
    
    # Load data
    df = fetch_crypto_data()
    df['Signal'] = df.apply(lambda row: get_signal_type(
        row['CSI_Q'], 
        row['Funding_Rate'], 
        row['Combined_Sentiment']
    ), axis=1)
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_signals = len(df[df['Signal'] != 'NEUTRAL'])
        st.markdown(f"""
        <div class="terminal-metric">
            <h3>ACTIVE SIGNALS</h3>
            <h2>{active_signals}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_csiq = df['CSI_Q'].mean()
        st.markdown(f"""
        <div class="terminal-metric">
            <h3>AVG CSI-Q</h3>
            <h2>{avg_csiq:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_volume = df['Volume_24h'].sum() / 1e9
        st.markdown(f"""
        <div class="terminal-metric">
            <h3>TOTAL VOL 24H</h3>
            <h2>${total_volume:.1f}B</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_sentiment = df['Combined_Sentiment'].mean()
        st.markdown(f"""
        <div class="terminal-metric">
            <h3>MARKET SENTIMENT</h3>
            <h2>{avg_sentiment:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced CSI-Q Monitor Grid
    st.markdown("### CSI-Q MONITORING GRID")
    
    # Sort by CSI-Q score
    sorted_df = df.sort_values('CSI_Q', ascending=False)
    
    # Create grid layout - 5 columns, multiple rows
    rows = []
    for i in range(0, len(sorted_df), 5):
        rows.append(sorted_df.iloc[i:i+5])
    
    for row_df in rows:
        cols = st.columns(5)
        for idx, (_, asset) in enumerate(row_df.iterrows()):
            if idx < len(cols):
                with cols[idx]:
                    csiq_class = get_csiq_color_class(asset['CSI_Q'])
                    signal_indicator = {
                        'LONG': '[▲ LONG]',
                        'SHORT': '[▼ SHORT]', 
                        'CONTRARIAN': '[◆ CONTR]',
                        'NEUTRAL': '[─ NEUT]'
                    }.get(asset['Signal'], '[─ NEUT]')
                    
                    st.markdown(f"""
                    <div class="csiq-grid">
                        <div class="csiq-symbol">{asset['Symbol']}</div>
                        <div class="csiq-score {csiq_class}">{asset['CSI_Q']:.1f}</div>
                        <div style="font-size: 10px; color: #888888;">{signal_indicator}</div>
                        <div style="font-size: 10px; color: #888888;">${asset['Price']:.4f}</div>
                        <div style="font-size: 10px; color: {'#00ff00' if asset['Change_24h'] > 0 else '#ff4444'};">
                            {asset['Change_24h']:+.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content tabs with terminal styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["CSI-Q MONITOR", "SENTIMENT ANALYSIS", "QUANT ANALYSIS", "TRADING OPPORTUNITIES", "DEEP ANALYSIS"])
    
    with tab1:
        st.markdown("### CSI-Q MONITORING INTERFACE")
        
        # Trading Signals Panel
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### TRADING SIGNALS ANALYSIS")
            
            # Enhanced scatter plot with terminal colors
            fig = go.Figure()
            
            colors = {
                'LONG': '#00ff00',
                'SHORT': '#ff4444', 
                'CONTRARIAN': '#ffaa00',
                'NEUTRAL': '#888888'
            }
            
            for signal in df['Signal'].unique():
                signal_data = df[df['Signal'] == signal]
                fig.add_trace(go.Scatter(
                    x=signal_data['Combined_Sentiment'],
                    y=signal_data['CSI_Q'],
                    mode='markers',
                    name=signal,
                    marker=dict(
                        color=colors[signal],
                        size=10,
                        line=dict(width=1, color='#ffffff')
                    ),
                    text=signal_data['Symbol'],
                    hovertemplate="<b>%{text}</b><br>" +
                                "CSI-Q: %{y:.1f}<br>" +
                                "Sentiment: %{x:.3f}<br>" +
                                "Signal: " + signal + "<br>" +
                                "<extra></extra>"
                ))
            
            fig.add_hline(y=70, line_dash="dash", line_color="#00ff00", line_width=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#ff4444", line_width=1)
            fig.add_vline(x=0, line_dash="dash", line_color="#ffffff", line_width=1)
            
            fig.update_layout(
                title="CSI-Q vs SENTIMENT MATRIX",
                xaxis_title="SENTIMENT SCORE",
                yaxis_title="CSI-Q SCORE",
                plot_bgcolor='#111111',
                paper_bgcolor='#111111',
                font=dict(color='#00ff41', family='Courier New'),
                height=400,
                legend=dict(
                    bgcolor='#1a1a1a',
                    bordercolor='#00ff41',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### PRIORITY ALERTS")
            
            # Generate priority alerts
            priority_assets = df[
                (df['Signal'] != 'NEUTRAL') & 
                ((df['CSI_Q'] > 80) | (df['CSI_Q'] < 20))
            ].sort_values('CSI_Q', key=lambda x: abs(x - 50), ascending=False)
            
            for _, asset in priority_assets.head(8).iterrows():
                signal_class = f"signal-{asset['Signal'].lower()}"
                urgency = "HIGH" if abs(asset['CSI_Q'] - 50) > 35 else "MED"
                
                st.markdown(f"""
                <div class="{signal_class}">
                    <strong>{asset['Symbol']} | {asset['Signal']}</strong><br>
                    CSI-Q: {asset['CSI_Q']:.1f} | URGENCY: {urgency}<br>
                    PRICE: ${asset['Price']:.4f}<br>
                    SENTIMENT: {asset['Combined_Sentiment']:.3f}<br>
                    FUNDING: {asset['Funding_Rate']:.4f}%
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### SENTIMENT ANALYSIS TERMINAL")
        
        if not df.empty:
            # Sentiment overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bullish_count = len(df[df['Combined_Sentiment'] > 0.2])
                st.markdown(f"""
                <div class="terminal-metric">
                    <h3>BULLISH ASSETS</h3>
                    <h2>{bullish_count}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                bearish_count = len(df[df['Combined_Sentiment'] < -0.2])
                st.markdown(f"""
                <div class="terminal-metric">
                    <h3>BEARISH ASSETS</h3>
                    <h2>{bearish_count}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                neutral_count = len(df[abs(df['Combined_Sentiment']) <= 0.2])
                st.markdown(f"""
                <div class="terminal-metric">
                    <h3>NEUTRAL ASSETS</h3>
                    <h2>{neutral_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Sentiment vs Price Performance
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    df,
                    x='Combined_Sentiment',
                    y='Change_24h',
                    size='Total_Mentions',
                    color='CSI_Q',
                    hover_name='Symbol',
                    title="SENTIMENT vs PRICE PERFORMANCE",
                    labels={
                        'Combined_Sentiment': 'Sentiment Score',
                        'Change_24h': '24h Change (%)'
                    },
                    color_continuous_scale='RdYlGn'
                )
                fig.add_vline(x=0, line_dash="dash", line_color="#ffffff")
                fig.add_hline(y=0, line_dash="dash", line_color="#ffffff")
                fig.update_layout(
                    height=400,
                    plot_bgcolor='#111111',
                    paper_bgcolor='#111111',
                    font=dict(color='#00ff41', family='Courier New')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top sentiment movers
                st.markdown("#### TOP SENTIMENT MOVERS")
                
                # Most positive
                st.markdown("**MOST BULLISH:**")
                bullish_assets = df.nlargest(3, 'Combined_Sentiment')
                for _, asset in bullish_assets.iterrows():
                    st.markdown(f"""
                    <div class="signal-long">
                        {asset['Symbol']} | SENT: {asset['Combined_Sentiment']:.3f}<br>
                        MENTIONS: {asset['Total_Mentions']:,}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("**MOST BEARISH:**")
                bearish_assets = df.nsmallest(3, 'Combined_Sentiment')
                for _, asset in bearish_assets.iterrows():
                    st.markdown(f"""
                    <div class="signal-short">
                        {asset['Symbol']} | SENT: {asset['Combined_Sentiment']:.3f}<br>
                        MENTIONS: {asset['Total_Mentions']:,}
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### QUANTITATIVE ANALYSIS TERMINAL")
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Funding Rate vs CSI-Q
                fig = px.scatter(
                    df,
                    x='Funding_Rate',
                    y='CSI_Q',
                    size='Total_Mentions',
                    color='Combined_Sentiment',
                    hover_name='Symbol',
                    title="FUNDING vs CSI-Q MATRIX",
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0
                )
                
                fig.add_vline(x=0, line_dash="dash", line_color="#ffffff")
                fig.add_hline(y=50, line_dash="dash", line_color="#ffffff")
                fig.update_layout(
                    height=400,
                    plot_bgcolor='#111111',
                    paper_bgcolor='#111111',
                    font=dict(color='#00ff41', family='Courier New')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # CSI-Q Component Analysis
                st.markdown("#### CSI-Q COMPONENT BREAKDOWN")
                
                component_df = df[['Symbol', 'CSI_Q', 'Derivatives_Score', 'Social_Score', 'Basis_Score', 'Tech_Score']].head(10)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='DERIVATIVES (35%)',
                    x=component_df['Symbol'],
                    y=component_df['Derivatives_Score'],
                    marker_color='#ff4444'
                ))
                
                fig.add_trace(go.Bar(
                    name='SOCIAL (35%)',
                    x=component_df['Symbol'],
                    y=component_df['Social_Score'],
                    marker_color='#00ff41'
                ))
                
                fig.add_trace(go.Bar(
                    name='BASIS (20%)',
                    x=component_df['Symbol'],
                    y=component_df['Basis_Score'],
                    marker_color='#ffaa00'
                ))
                
                fig.add_trace(go.Bar(
                    name='TECHNICAL (10%)',
                    x=component_df['Symbol'],
                    y=component_df['Tech_Score'],
                    marker_color='#0088ff'
                ))
                
                fig.update_layout(
                    title="TOP 10 CSI-Q COMPONENTS",
                    height=400,
                    barmode='group',
                    plot_bgcolor='#111111',
                    paper_bgcolor='#111111',
                    font=dict(color='#00ff41', family='Courier New'),
                    legend=dict(bgcolor='#1a1a1a', bordercolor='#00ff41', borderwidth=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Opportunity scoring
            st.markdown("#### SENTIMENT-PRICE DIVERGENCES")
            
            divergent_df = df[
                ((df['Combined_Sentiment'] > 0.3) & (df['Change_24h'] < -5)) |
                ((df['Combined_Sentiment'] < -0.3) & (df['Change_24h'] > 5))
            ].sort_values('Total_Mentions', ascending=False)
            
            if len(divergent_df) > 0:
                cols = st.columns(min(4, len(divergent_df)))
                for i, (_, row) in enumerate(divergent_df.head(4).iterrows()):
                    with cols[i]:
                        divergence_type = "UNDERVALUED" if row['Combined_Sentiment'] > 0.3 else "OVERVALUED"
                        
                        st.markdown(f"""
                        <div class="signal-contrarian">
                            <strong>{row['Symbol']}</strong><br>
                            {divergence_type}<br>
                            SENT: {row['Combined_Sentiment']:.2f}<br>
                            PRICE: {row['Change_24h']:.2f}%<br>
                            DIVERGENCE OPPORTUNITY
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-online">NO MAJOR DIVERGENCES DETECTED</p>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### TRADING OPPORTUNITIES TERMINAL")
        
        if not df.empty:
            # Enhanced opportunity scoring
            df['Opportunity_Score'] = (
                (abs(df['CSI_Q'] - 50) / 50 * 0.3) +
                (abs(df['Funding_Rate']) * 10 * 0.25) +
                (abs(df['Long_Short_Ratio'] - 1) * 0.15) +
                (abs(df['Combined_Sentiment']) * 0.2) +
                ((df['Volume_24h'] / df['Volume_24h'].max()) * 0.1)
            ) * 100
            
            # Top opportunities
            opportunities = df.sort_values('Opportunity_Score', ascending=False).head(8)
            
            st.markdown("#### TOP TRADING OPPORTUNITIES")
            
            st.markdown("""
            <div class="terminal-header">
                ENHANCED DEMO MODE: REALISTIC SIMULATED DATA WITH SENTIMENT ANALYSIS
            </div>
            """, unsafe_allow_html=True)
            
            for i, (_, row) in enumerate(opportunities.iterrows()):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="terminal-metric" style="text-align: left;">
                        <h3>{i+1}. {row['Symbol']} | {row['Signal']}</h3>
                        <p>OPPORTUNITY SCORE: {row['Opportunity_Score']:.1f}</p>
                        <p>SENTIMENT: {row['Combined_Sentiment']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("CSI-Q", f"{row['CSI_Q']:.1f}")
                    st.metric("SIGNAL", row['Signal'])
                
                with col3:
                    st.metric("PRICE", f"${row['Price']:.4f}")
                    st.metric("24H CHG", f"{row['Change_24h']:.2f}%")
                
                with col4:
                    # Trade setup
                    if row['Signal'] == 'LONG':
                        setup = "BULLISH SETUP"
                        target = row['Price'] * 1.05
                        stop = row['Price'] * 0.98
                    elif row['Signal'] == 'SHORT':
                        setup = "BEARISH SETUP"
                        target = row['Price'] * 0.95
                        stop = row['Price'] * 1.02
                    else:
                        setup = "CONTRARIAN SETUP"
                        target = row['Price'] * 1.03
                        stop = row['Price'] * 0.99
                    
                    st.markdown(f"""
                    <div class="terminal-metric" style="text-align: left; font-size: 12px;">
                        <strong>{setup}</strong><br>
                        ENTRY: ${row['Price']:.4f}<br>
                        TARGET: ${target:.4f}<br>
                        STOP: ${stop:.4f}<br>
                        FUNDING: {row['Funding_Rate']:.4f}%
                    </div>
                    """, unsafe_allow_html=True)
            
            # Market analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### MARKET OVERVIEW")
                
                avg_csiq = df['CSI_Q'].mean()
                avg_sentiment = df['Combined_Sentiment'].mean()
                
                if avg_sentiment > 0.3 and avg_csiq > 65:
                    market_status = "VERY BULLISH"
                    status_color = "#00ff00"
                elif avg_sentiment > 0.1 and avg_csiq > 55:
                    market_status = "BULLISH"
                    status_color = "#00ff00"
                elif avg_sentiment < -0.3 and avg_csiq < 35:
                    market_status = "VERY BEARISH"
                    status_color = "#ff4444"
                elif avg_sentiment < -0.1 and avg_csiq < 45:
                    market_status = "BEARISH"
                    status_color = "#ff4444"
                else:
                    market_status = "MIXED/NEUTRAL"
                    status_color = "#ffaa00"
                
                st.markdown(f"""
                <div style="background: {status_color}; padding: 15px; border-radius: 4px; color: #000000; text-align: center; font-family: 'Courier New', monospace;">
                    <h3>MARKET STATUS: {market_status}</h3>
                    <p>AVG CSI-Q: {avg_csiq:.1f} | AVG SENTIMENT: {avg_sentiment:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### RISK ANALYSIS")
                
                warnings = []
                
                extreme_csiq = df[(df['CSI_Q'] > 90) | (df['CSI_Q'] < 10)]
                if not extreme_csiq.empty:
                    warnings.append(f"{len(extreme_csiq)} EXTREME CSI-Q LEVELS")
                
                extreme_funding = df[abs(df['Funding_Rate']) > 0.2]
                if not extreme_funding.empty:
                    warnings.append(f"{len(extreme_funding)} EXTREME FUNDING RATES")
                
                if warnings:
                    for warning in warnings:
                        st.markdown(f"""
                        <div style="background: #ff4444; padding: 10px; border-radius: 4px; margin: 5px 0; color: #000000; font-family: 'Courier New', monospace;">
                            WARNING: {warning}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #00ff00; padding: 10px; border-radius: 4px; color: #000000; text-align: center; font-family: 'Courier New', monospace;">
                        SYSTEM STATUS: NORMAL
                    </div>
                    """, unsafe_allow_html=True)

    with tab5:
        st.markdown("### DEEP ANALYSIS TERMINAL")
        
        # Asset selection
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_symbol = st.selectbox("SELECT ASSET FOR DEEP ANALYSIS:", df['Symbol'].tolist())
        
        with col2:
            timeframe = st.selectbox("TIMEFRAME:", ["1H", "4H", "1D", "3D", "1W"])
        
        with col3:
            lookback_days = st.slider("LOOKBACK DAYS:", 1, 30, 7)
        
        if selected_symbol:
            # Get selected asset data
            asset_data = df[df['Symbol'] == selected_symbol].iloc[0]
            
            # Generate historical data for analysis
            np.random.seed(hash(selected_symbol) % 1000)
            periods = lookback_days * 24 if timeframe == "1H" else lookback_days * 6 if timeframe == "4H" else lookback_days
            
            # Create synthetic historical data
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H' if timeframe == "1H" else '4H' if timeframe == "4H" else '1D')
            
            # Price evolution
            price_changes = np.random.normal(0, 0.02, periods)
            prices = [asset_data['Price']]
            for change in price_changes[:-1]:
                prices.append(prices[-1] * (1 + change))
            prices = prices[::-1]  # Reverse to show progression to current price
            
            # CSI-Q evolution
            csiq_base = asset_data['CSI_Q']
            csiq_changes = np.random.normal(0, 5, periods)
            csiq_history = [max(0, min(100, csiq_base + sum(csiq_changes[:i+1]))) for i in range(periods)]
            
            # Sentiment evolution
            sentiment_base = asset_data['Combined_Sentiment']
            sentiment_changes = np.random.normal(0, 0.1, periods)
            sentiment_history = [max(-1, min(1, sentiment_base + sum(sentiment_changes[:i+1]))) for i in range(periods)]
            
            # Volume evolution
            volume_changes = np.random.normal(0, 0.3, periods)
            volumes = [max(0, asset_data['Volume_24h'] * (1 + change)) for change in volume_changes]
            
            # Create historical dataframe
            hist_df = pd.DataFrame({
                'DateTime': dates,
                'Price': prices,
                'CSI_Q': csiq_history,
                'Sentiment': sentiment_history,
                'Volume': volumes
            })
            
            # COMPREHENSIVE ANALYSIS LAYOUT
            st.markdown(f"""
            <div class="terminal-header">
                DEEP ANALYSIS: {selected_symbol} | TIMEFRAME: {timeframe} | LOOKBACK: {lookback_days} DAYS
            </div>
            """, unsafe_allow_html=True)
            
            # Row 1: Current Status & Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_signal = asset_data['Signal']
                signal_color = {'LONG': '#00ff00', 'SHORT': '#ff4444', 'CONTRARIAN': '#ffaa00', 'NEUTRAL': '#888888'}[current_signal]
                
                st.markdown(f"""
                <div class="terminal-metric">
                    <h3>CURRENT SIGNAL</h3>
                    <h2 style="color: {signal_color};">{current_signal}</h2>
                    <p>CSI-Q: {asset_data['CSI_Q']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                price_trend = "UP" if hist_df['Price'].iloc[-1] > hist_df['Price'].iloc[0] else "DOWN"
                trend_color = "#00ff00" if price_trend == "UP" else "#ff4444"
                price_change_pct = ((hist_df['Price'].iloc[-1] / hist_df['Price'].iloc[0]) - 1) * 100
                
                st.markdown(f"""
                <div class="terminal-metric">
                    <h3>PRICE TREND</h3>
                    <h2 style="color: {trend_color};">{price_trend}</h2>
                    <p>{price_change_pct:+.2f}% ({lookback_days}D)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                sentiment_trend = "POSITIVE" if hist_df['Sentiment'].iloc[-1] > hist_df['Sentiment'].iloc[0] else "NEGATIVE"
                sentiment_color = "#00ff00" if sentiment_trend == "POSITIVE" else "#ff4444"
                
                st.markdown(f"""
                <div class="terminal-metric">
                    <h3>SENTIMENT TREND</h3>
                    <h2 style="color: {sentiment_color};">{sentiment_trend}</h2>
                    <p>{asset_data['Combined_Sentiment']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                csiq_trend = "UP" if hist_df['CSI_Q'].iloc[-1] > hist_df['CSI_Q'].iloc[0] else "DOWN"
                csiq_trend_color = "#00ff00" if csiq_trend == "UP" else "#ff4444"
                
                st.markdown(f"""
                <div class="terminal-metric">
                    <h3>CSI-Q TREND</h3>
                    <h2 style="color: {csiq_trend_color};">{csiq_trend}</h2>
                    <p>{asset_data['CSI_Q']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Row 2: Historical Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Price & CSI-Q correlation chart
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    subplot_titles=("PRICE EVOLUTION", "CSI-Q EVOLUTION"),
                    vertical_spacing=0.1
                )
                
                # Price chart
                fig.add_trace(
                    go.Scatter(x=hist_df['DateTime'], y=hist_df['Price'], 
                              name='Price', line=dict(color='#00ff41', width=2)),
                    row=1, col=1
                )
                
                # CSI-Q chart with signal zones
                fig.add_trace(
                    go.Scatter(x=hist_df['DateTime'], y=hist_df['CSI_Q'], 
                              name='CSI-Q', line=dict(color='#ffaa00', width=2)),
                    row=2, col=1
                )
                
                # Add CSI-Q signal zones
                fig.add_hline(y=70, line_dash="dash", line_color="#00ff00", line_width=1, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#ff4444", line_width=1, row=2, col=1)
                fig.add_hline(y=90, line_dash="dash", line_color="#ffaa00", line_width=1, row=2, col=1)
                fig.add_hline(y=10, line_dash="dash", line_color="#ffaa00", line_width=1, row=2, col=1)
                
                fig.update_layout(
                    title="PRICE vs CSI-Q CORRELATION ANALYSIS",
                    height=500,
                    plot_bgcolor='#111111',
                    paper_bgcolor='#111111',
                    font=dict(color='#00ff41', family='Courier New'),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment analysis chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=hist_df['DateTime'], 
                    y=hist_df['Sentiment'],
                    mode='lines+markers',
                    name='Sentiment',
                    line=dict(color='#0088ff', width=2),
                    marker=dict(size=4)
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="#ffffff", line_width=1)
                fig.add_hline(y=0.5, line_dash="dot", line_color="#00ff00", line_width=1)
                fig.add_hline(y=-0.5, line_dash="dot", line_color="#ff4444", line_width=1)
                
                fig.update_layout(
                    title="SENTIMENT EVOLUTION ANALYSIS",
                    xaxis_title="Time",
                    yaxis_title="Sentiment Score",
                    height=500,
                    plot_bgcolor='#111111',
                    paper_bgcolor='#111111',
                    font=dict(color='#00ff41', family='Courier New')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Row 3: Final Analysis and Recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### CORRELATION ANALYSIS")
                
                # Calculate correlations
                price_csiq_corr = hist_df['Price'].corr(hist_df['CSI_Q'])
                price_sentiment_corr = hist_df['Price'].corr(hist_df['Sentiment'])
                csiq_sentiment_corr = hist_df['CSI_Q'].corr(hist_df['Sentiment'])
                
                st.markdown(f"""
                <div class="terminal-metric" style="text-align: left;">
                    <h3>KEY CORRELATIONS</h3>
                    <p>PRICE ↔ CSI-Q: {price_csiq_corr:.3f}</p>
                    <p>PRICE ↔ SENTIMENT: {price_sentiment_corr:.3f}</p>
                    <p>CSI-Q ↔ SENTIMENT: {csiq_sentiment_corr:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk Assessment
                volatility = hist_df['Price'].pct_change().std() * 100
                risk_level = "HIGH" if volatility > 5 else "MEDIUM" if volatility > 2 else "LOW"
                risk_color = "#ff4444" if risk_level == "HIGH" else "#ffaa00" if risk_level == "MEDIUM" else "#00ff00"
                
                st.markdown(f"""
                <div class="terminal-metric" style="text-align: left;">
                    <h3>RISK ASSESSMENT</h3>
                    <p>VOLATILITY: {volatility:.2f}%</p>
                    <p style="color: {risk_color};">RISK LEVEL: <strong>{risk_level}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### TRADE RECOMMENDATION")
                
                # Calculate comprehensive trade score
                technical_score = 1 if asset_data['CSI_Q'] > 70 or asset_data['CSI_Q'] < 30 else 0
                sentiment_score = 1 if abs(asset_data['Combined_Sentiment']) > 0.3 else 0
                correlation_score = 1 if abs(price_csiq_corr) > 0.5 else 0
                
                total_score = technical_score + sentiment_score + correlation_score
                
                if total_score >= 2:
                    recommendation = "STRONG TRADE"
                    rec_color = "#00ff00"
                elif total_score >= 1:
                    recommendation = "MODERATE TRADE"
                    rec_color = "#ffaa00" 
                else:
                    recommendation = "HOLD/WAIT"
                    rec_color = "#ff4444"
                
                st.markdown(f"""
                <div style="background: {rec_color}; padding: 20px; border-radius: 4px; color: #000000; text-align: center; font-family: 'Courier New', monospace;">
                    <h2>RECOMMENDATION</h2>
                    <h3>{recommendation}</h3>
                    <p>SCORE: {total_score}/3</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Trade Setup
                current_price = asset_data['Price']
                if asset_data['Signal'] == 'LONG':
                    target_price = current_price * 1.08
                    stop_loss = current_price * 0.96
                elif asset_data['Signal'] == 'SHORT':
                    target_price = current_price * 0.92
                    stop_loss = current_price * 1.04
                else:
                    target_price = current_price * 1.05
                    stop_loss = current_price * 0.97
                
                risk_reward = abs((target_price - current_price) / (stop_loss - current_price))
                
                st.markdown(f"""
                <div class="terminal-metric" style="text-align: left;">
                    <h3>TRADE SETUP</h3>
                    <p>ENTRY: ${current_price:.4f}</p>
                    <p>TARGET: ${target_price:.4f}</p>
                    <p>STOP: ${stop_loss:.4f}</p>
                    <p>R/R: 1:{risk_reward:.1f}</p>
                    <p>SIGNAL: {asset_data['Signal']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Market Data Table
    st.markdown("### MARKET DATA TERMINAL")
    
    # Prepare display data
    display_df = df[['Symbol', 'CSI_Q', 'Signal', 'Price', 'Change_24h', 
                     'Combined_Sentiment', 'Volume_24h', 'Funding_Rate']].copy()
    
    display_df['Price'] = display_df['Price'].round(6)
    display_df['CSI_Q'] = display_df['CSI_Q'].round(1)
    display_df['Change_24h'] = display_df['Change_24h'].round(2)
    display_df['Combined_Sentiment'] = display_df['Combined_Sentiment'].round(4)
    display_df['Volume_24h'] = (display_df['Volume_24h'] / 1000000).round(1)
    display_df['Funding_Rate'] = display_df['Funding_Rate'].round(4)
    
    display_df = display_df.rename(columns={
        'Volume_24h': 'VOL_24H_M',
        'Change_24h': 'CHG_24H_%',
        'Funding_Rate': 'FUNDING_%',
        'Combined_Sentiment': 'SENTIMENT'
    })
    
    # Sort by CSI-Q
    display_df = display_df.sort_values('CSI_Q', ascending=False)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<p class="status-online">CONNECTION: STABLE</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="status-warning">DATA: SIMULATED</p>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<p class="status-online">UPTIME: {datetime.now().strftime("%H:%M:%S")}</p>', 
                   unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #888888; margin-top: 20px; font-family: "Courier New", monospace;'>
        <p>CSI-Q TERMINAL v2.1.0 | PROFESSIONAL TRADING INTERFACE<br>
        WARNING: SIMULATION MODE ACTIVE - FOR DEMONSTRATION PURPOSES ONLY<br>
        NOT FINANCIAL ADVICE - TRADE AT YOUR OWN RISK</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
