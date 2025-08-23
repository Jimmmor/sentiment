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
import warnings
warnings.filterwarnings('ignore')

# Page config - Dark theme
st.set_page_config(
    page_title="🔥 Crypto CSI-Q | Quant Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Dark Hacker CSS Theme
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #00ff41;
    }
    
    /* Terminal-style containers */
    .terminal-container {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        border: 2px solid #00ff41;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        box-shadow: 0 0 20px rgba(0,255,65,0.3);
    }
    
    /* Metric cards - Cyberpunk style */
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #16537e 100%);
        border: 1px solid #00d4aa;
        padding: 20px;
        border-radius: 12px;
        color: #00ff41;
        text-align: center;
        box-shadow: 0 0 25px rgba(0,212,170,0.4);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 30px rgba(0,255,65,0.6);
    }
    
    /* Signal styles - Enhanced */
    .signal-long {
        background: linear-gradient(135deg, #00ff41, #00cc33);
        border: 2px solid #00ff41;
        padding: 12px;
        border-radius: 8px;
        color: #000000;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 15px rgba(0,255,65,0.5);
        animation: pulse-green 2s infinite;
    }
    
    .signal-short {
        background: linear-gradient(135deg, #ff0040, #cc0033);
        border: 2px solid #ff0040;
        padding: 12px;
        border-radius: 8px;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 15px rgba(255,0,64,0.5);
        animation: pulse-red 2s infinite;
    }
    
    .signal-contrarian {
        background: linear-gradient(135deg, #ffaa00, #ff6600);
        border: 2px solid #ffaa00;
        padding: 12px;
        border-radius: 8px;
        color: #000000;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 15px rgba(255,170,0,0.5);
        animation: pulse-orange 2s infinite;
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, #666666, #444444);
        border: 2px solid #888888;
        padding: 12px;
        border-radius: 8px;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
    }
    
    /* Exit zone styling */
    .exit-zone {
        background: linear-gradient(135deg, #4a0e4e, #2e0536);
        border: 2px solid #ff00ff;
        padding: 10px;
        border-radius: 8px;
        color: #ff00ff;
        text-align: center;
        box-shadow: 0 0 15px rgba(255,0,255,0.4);
    }
    
    /* Pulse animations */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 15px rgba(0,255,65,0.5); }
        50% { box-shadow: 0 0 25px rgba(0,255,65,0.8); }
        100% { box-shadow: 0 0 15px rgba(0,255,65,0.5); }
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 15px rgba(255,0,64,0.5); }
        50% { box-shadow: 0 0 25px rgba(255,0,64,0.8); }
        100% { box-shadow: 0 0 15px rgba(255,0,64,0.5); }
    }
    
    @keyframes pulse-orange {
        0% { box-shadow: 0 0 15px rgba(255,170,0,0.5); }
        50% { box-shadow: 0 0 25px rgba(255,170,0,0.8); }
        100% { box-shadow: 0 0 15px rgba(255,170,0,0.5); }
    }
    
    /* API status */
    .api-status-error {
        background: linear-gradient(135deg, #330000, #660000);
        border: 2px solid #ff0000;
        padding: 15px;
        border-radius: 10px;
        color: #ff4444;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(255,0,0,0.3);
    }
    
    .api-status-demo {
        background: linear-gradient(135deg, #333300, #666600);
        border: 2px solid #ffff00;
        padding: 15px;
        border-radius: 10px;
        color: #ffff44;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(255,255,0,0.3);
    }
    
    /* Terminal header */
    .terminal-header {
        background: #000000;
        color: #00ff41;
        padding: 15px;
        font-family: 'Courier New', monospace;
        border: 2px solid #00ff41;
        border-radius: 8px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 20px rgba(0,255,65,0.3);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: rgba(0,0,0,0.5);
        border-radius: 8px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #1a1a2e, #16537e);
        color: #00ff41;
        border: 1px solid #00d4aa;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Data table styling */
    .stDataFrame {
        background: rgba(0,0,0,0.8);
        border: 1px solid #00ff41;
        border-radius: 8px;
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

class MultiSourceDataFetcher:
    def __init__(self):
        self.binance_base = "https://fapi.binance.com"
        self.binance_spot = "https://api.binance.com"
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        
    def test_api_connectivity(self):
        """Test which APIs are available"""
        apis_status = {
            'binance': False,
            'coingecko': False,
            'demo': True  # Always available fallback
        }
        
        # Test Binance
        try:
            response = requests.get(f"{self.binance_base}/fapi/v1/ping", timeout=5)
            if response.status_code == 200:
                apis_status['binance'] = True
        except Exception as e:
            st.warning(f"⚡ Binance API offline: {str(e)}")
        
        # Test CoinGecko
        try:
            response = requests.get(f"{self.coingecko_base}/ping", timeout=5)
            if response.status_code == 200:
                apis_status['coingecko'] = True
        except Exception as e:
            st.warning(f"🌐 CoinGecko limited: {str(e)}")
        
        return apis_status
    
    def generate_enhanced_sentiment(self, symbol, price_change):
        """Generate more realistic sentiment data"""
        np.random.seed(hash(symbol) % 1000)  # Consistent per symbol
        
        # Base sentiment from price action
        price_sentiment = np.tanh(price_change / 5)
        
        # Add symbol-specific bias (some coins are more "hyped")
        hype_coins = ['BTC', 'ETH', 'SOL', 'DOGE', 'SHIB']
        hype_multiplier = 1.3 if symbol in hype_coins else 1.0
        
        # Add social media cycles (simulate viral trends)
        day_of_week = datetime.now().weekday()
        social_cycle = np.sin(day_of_week * np.pi / 7) * 0.2  # Weekend hype
        
        # Market cap tier influence
        tier_1 = ['BTC', 'ETH', 'BNB']
        tier_2 = ['SOL', 'XRP', 'ADA', 'AVAX']
        
        if symbol in tier_1:
            volatility = 0.3
            base_mentions = np.random.uniform(10000, 50000)
        elif symbol in tier_2:
            volatility = 0.5
            base_mentions = np.random.uniform(2000, 15000)
        else:
            volatility = 0.8
            base_mentions = np.random.uniform(100, 5000)
        
        # Final sentiment calculation
        final_sentiment = (
            price_sentiment * 0.4 +
            social_cycle * 0.3 +
            np.random.normal(0, volatility) * 0.3
        ) * hype_multiplier
        
        # Clamp to reasonable bounds
        final_sentiment = max(-1, min(1, final_sentiment))
        
        # Mentions influenced by sentiment and volatility
        mentions = int(base_mentions * (1 + abs(final_sentiment)) * (1 + abs(price_change)/10))
        
        return final_sentiment, mentions
    
    def calculate_exit_zones(self, csiq_current, signal):
        """Calculate dynamic exit zones based on CSI-Q mean reversion"""
        
        # CSI-Q mean reversion targets
        csiq_mean = 50
        csiq_std = 20
        
        if signal == "LONG":
            # Exit when CSI-Q drops back toward mean
            exit_target = max(45, csiq_mean - (csiq_std * 0.5))  # 40-50 range
            stop_loss_csiq = min(20, csiq_current - 30)  # Hard stop
            
        elif signal == "SHORT":
            # Exit when CSI-Q rises back toward mean  
            exit_target = min(55, csiq_mean + (csiq_std * 0.5))  # 50-60 range
            stop_loss_csiq = max(80, csiq_current + 30)  # Hard stop
            
        elif signal == "CONTRARIAN":
            # Exit when extreme reverts to normal
            if csiq_current > 80:  # Contrarian short
                exit_target = 65
                stop_loss_csiq = 95
            else:  # Contrarian long (CSI-Q < 20)
                exit_target = 35
                stop_loss_csiq = 5
                
        else:  # NEUTRAL
            exit_target = csiq_mean
            stop_loss_csiq = csiq_current
            
        return {
            'exit_target_csiq': exit_target,
            'stop_loss_csiq': stop_loss_csiq,
            'mean_reversion_strength': abs(csiq_current - csiq_mean) / csiq_std
        }
    
    def generate_demo_data(self):
        """Generate enhanced realistic demo data"""
        np.random.seed(42)  # For consistent demo data
        
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
            
            # Add some realistic price movement
            price_change = np.random.normal(0, 0.05)  # 5% volatility
            current_price = base_price * (1 + price_change)
            
            # Generate realistic metrics
            change_24h = np.random.normal(0, 8)
            funding_rate = np.random.normal(0.01, 0.05)
            oi_change = np.random.normal(0, 20)
            long_short_ratio = np.random.lognormal(0, 0.5)
            
            # Enhanced sentiment and mentions
            sentiment, mentions = self.generate_enhanced_sentiment(symbol_clean, change_24h)
            
            # Volume based on market cap tier
            if symbol_clean in ['BTC', 'ETH']:
                volume_24h = np.random.uniform(20000000000, 50000000000)
            elif symbol_clean in ['BNB', 'SOL', 'XRP']:
                volume_24h = np.random.uniform(1000000000, 10000000000)
            else:
                volume_24h = np.random.uniform(100000000, 2000000000)
            
            # Technical indicators
            rsi = 50 + np.random.normal(0, 15)
            rsi = max(0, min(100, rsi))
            bb_squeeze = np.random.uniform(0, 1)
            basis = np.random.normal(0, 0.5)
            
            # Enhanced CSI-Q calculation
            derivatives_score = min(100, max(0,
                (abs(oi_change) * 2) +
                (abs(funding_rate) * 500) +
                (abs(long_short_ratio - 1) * 30) +
                30
            ))
            
            social_score = min(100, max(0,
                ((sentiment + 1) * 25) +
                (min(mentions, 10000) / 100) +
                20
            ))
            
            basis_score = min(100, max(0,
                abs(basis) * 500 + 25
            ))
            
            tech_score = min(100, max(0,
                (100 - abs(rsi - 50)) * 0.8 +
                ((1 - bb_squeeze) * 40) +
                10
            ))
            
            csiq = (
                derivatives_score * 0.4 +
                social_score * 0.3 +
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
                'Mentions': mentions,
                'Sentiment': sentiment,
                'Spot_Futures_Basis': basis,
                'RSI': rsi,
                'BB_Squeeze': bb_squeeze,
                'CSI_Q': csiq,
                'Derivatives_Score': derivatives_score,
                'Social_Score': social_score,
                'Basis_Score': basis_score,
                'Tech_Score': tech_score,
                'ATR': abs(current_price * 0.05),
                'Volume_24h': volume_24h,
                'Open_Interest': volume_24h * np.random.uniform(0.1, 2.0),
                'Last_Updated': datetime.now(),
                'Data_Source': 'demo'
            })
        
        return pd.DataFrame(data_list)

def get_signal_type(csiq, funding_rate):
    """Determine signal type based on CSI-Q and funding rate"""
    if csiq > 90 or csiq < 10:
        return "CONTRARIAN"
    elif csiq > 70 and funding_rate < 0.1:
        return "LONG"
    elif csiq < 30 and funding_rate > -0.1:
        return "SHORT"
    else:
        return "NEUTRAL"

def get_signal_color(signal):
    """Get emoji for signal type"""
    colors = {
        "LONG": "🚀",
        "SHORT": "📉", 
        "CONTRARIAN": "⚡",
        "NEUTRAL": "⚪"
    }
    return colors.get(signal, "⚪")

@st.cache_data(ttl=60)
def fetch_crypto_data_with_fallback():
    """Fetch crypto data with fallback options"""
    fetcher = MultiSourceDataFetcher()
    
    # For demo purposes, we'll use enhanced demo data
    st.markdown("""
    <div class="api-status-demo">
        ⚡ <b>ENHANCED DEMO MODE ACTIVE</b><br>
        Advanced sentiment simulation + CSI-Q mean reversion exits<br>
        All quant features fully operational
    </div>
    """, unsafe_allow_html=True)
    
    return fetcher.generate_demo_data()

# Main App
def main():
    # Terminal-style header
    st.markdown("""
    <div class="terminal-header">
        <h1>⚡ CRYPTO CSI-Q QUANT TERMINAL ⚡</h1>
        <p>>>> ADVANCED SENTIMENT & DERIVATIVES ANALYSIS <<<</p>
        <p>[ENHANCED] Multi-Source Data + Mean Reversion Exits + Dynamic Sentiment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="terminal-container">📡 <b>STATUS:</b> ONLINE</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="terminal-container">⏰ <b>TIME:</b> {datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="terminal-container">🔥 <b>MODE:</b> QUANT</div>', unsafe_allow_html=True)
    with col4:
        if st.button("🔄 REFRESH DATA", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    df = fetch_crypto_data_with_fallback()
    
    if df.empty:
        st.error("❌ TERMINAL ERROR: No data stream available")
        st.stop()
    
    # Add signals and exit zones
    df['Signal'] = df.apply(lambda row: get_signal_type(row['CSI_Q'], row['Funding_Rate']), axis=1)
    
    # Calculate exit zones for each position
    fetcher = MultiSourceDataFetcher()
    exit_data = []
    for _, row in df.iterrows():
        exit_info = fetcher.calculate_exit_zones(row['CSI_Q'], row['Signal'])
        exit_data.append(exit_info)
    
    exit_df = pd.DataFrame(exit_data)
    df = pd.concat([df, exit_df], axis=1)
    
    # Sidebar - Terminal style
    st.sidebar.markdown("## ⚡ TERMINAL CONTROLS")
    
    min_csiq = st.sidebar.slider("⚡ Min CSI-Q Score", 0, 100, 0)
    max_csiq = st.sidebar.slider("⚡ Max CSI-Q Score", 0, 100, 100)
    
    signal_filter = st.sidebar.multiselect(
        "🎯 Signal Types",
        ["LONG", "SHORT", "CONTRARIAN", "NEUTRAL"],
        default=["LONG", "SHORT", "CONTRARIAN"]
    )
    
    min_volume = st.sidebar.number_input("💰 Min Volume ($M)", 0, 1000, 0)
    min_mentions = st.sidebar.number_input("📱 Min Social Mentions", 0, 10000, 0)
    
    # Apply filters
    filtered_df = df[
        (df['CSI_Q'] >= min_csiq) & 
        (df['CSI_Q'] <= max_csiq) &
        (df['Signal'].isin(signal_filter)) &
        (df['Volume_24h'] >= min_volume * 1000000) &
        (df['Mentions'] >= min_mentions)
    ].copy()
    
    # Top metrics - Cyberpunk cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_signals = len(filtered_df[filtered_df['Signal'] != 'NEUTRAL'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 ACTIVE SIGNALS</h3>
            <h1>{active_signals}</h1>
            <p>Ready for execution</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_csiq = filtered_df['CSI_Q'].mean() if not filtered_df.empty else 0
        status = "OVERHEATED" if avg_csiq > 70 else "OVERSOLD" if avg_csiq < 30 else "BALANCED"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 MARKET CSI-Q</h3>
            <h1>{avg_csiq:.1f}</h1>
            <p>{status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        exit_opportunities = len(filtered_df[abs(filtered_df['CSI_Q'] - filtered_df['exit_target_csiq']) <= 5])
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 EXIT ZONES</h3>
            <h1>{exit_opportunities}</h1>
            <p>Near reversion targets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_mentions = filtered_df['Mentions'].sum() if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📱 SOCIAL BUZZ</h3>
            <h1>{total_mentions:,}</h1>
            <p>Total mentions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main tabs - Enhanced styling
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 LIVE SIGNALS", "🎯 QUANT ANALYSIS", "💰 TRADING TERMINAL", "📊 EXIT MONITOR"])
    
    with tab1:
        st.markdown("## ⚡ LIVE SIGNAL MONITOR")
        
        if not filtered_df.empty:
            col1, col2 = st.columns([2.5, 1])
            
            with col1:
                # Enhanced CSI-Q visualization
                display_df = filtered_df.sort_values('CSI_Q', ascending=False)
                
                fig = go.Figure()
                
                # Add scatter points for each signal type
                for signal_type in ['LONG', 'SHORT', 'CONTRARIAN', 'NEUTRAL']:
                    signal_data = display_df[display_df['Signal'] == signal_type]
                    if not signal_data.empty:
                        colors = {
                            'LONG': '#00ff41',
                            'SHORT': '#ff0040', 
                            'CONTRARIAN': '#ffaa00',
                            'NEUTRAL': '#666666'
                        }
                        
                        fig.add_trace(go.Scatter(
                            x=signal_data['Symbol'],
                            y=signal_data['CSI_Q'],
                            mode='markers+text',
                            name=signal_type,
                            marker=dict(
                                size=np.sqrt(signal_data['Volume_24h']) / 80000,
                                color=colors[signal_type],
                                line=dict(width=2, color='white')
                            ),
                            text=signal_data['Symbol'],
                            textposition="middle center",
                            hovertemplate="<b>%{text}</b><br>" +
                                        "CSI-Q: %{y:.1f}<br>" +
                                        "Price: $" + signal_data['Price'].round(4).astype(str) + "<br>" +
                                        "Sentiment: " + signal_data['Sentiment'].round(3).astype(str) + "<br>" +
                                        "Mentions: " + signal_data['Mentions'].astype(str) + "<br>" +
                                        "Exit Target: " + signal_data['exit_target_csiq'].round(1).astype(str) + "<br>" +
                                        "<extra></extra>"
                        ))
                
                fig.update_layout(
                    title="🔥 CSI-Q SIGNAL MATRIX - Enhanced Social Sentiment",
                    xaxis_title="SYMBOLS",
                    yaxis_title="CSI-Q SCORE",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#00ff41'),
                    showlegend=True
                )
                
                # Enhanced signal zones
                fig.add_hline(y=90, line_dash="dash", line_color="#ffaa00", line_width=3,
                             annotation_text="⚡ CONTRARIAN ZONE", annotation_position="right")
                fig.add_hline(y=70, line_dash="dash", line_color="#00ff41", line_width=3,
                             annotation_text="🚀 LONG ZONE", annotation_position="right")
                fig.add_hline(y=50, line_dash="solid", line_color="#ffffff", line_width=2,
                             annotation_text="🎯 MEAN REVERSION", annotation_position="right")
                fig.add_hline(y=30, line_dash="dash", line_color="#ff0040", line_width=3,
                             annotation_text="📉 SHORT ZONE", annotation_position="right")
                fig.add_hline(y=10, line_dash="dash", line_color="#ffaa00", line_width=3,
                             annotation_text="⚡ CONTRARIAN ZONE", annotation_position="right")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🚨 PRIORITY ALERTS")
                
                # Enhanced alerts with exit zones
                alerts = []
                for _, row in filtered_df.iterrows():
                    signal = row['Signal']
                    if signal != 'NEUTRAL':
                        # Calculate urgency based on multiple factors
                        csiq_extremity = abs(row['CSI_Q'] - 50) / 50
                        sentiment_strength = abs(row['Sentiment'])
                        mention_momentum = min(row['Mentions'] / 1000, 10) / 10
                        
                        urgency_score = (csiq_extremity * 0.4 + 
                                       sentiment_strength * 0.3 + 
                                       mention_momentum * 0.3)
                        
                        if urgency_score > 0.7:
                            strength = "🔥 EXTREME"
                            priority = 1
                        elif urgency_score > 0.5:
                            strength = "⚡ HIGH"
                            priority = 2
                        else:
                            strength = "⚠️ MEDIUM"
                            priority = 3
                        
                        # Calculate distance to exit
                        exit_distance = abs(row['CSI_Q'] - row['exit_target_csiq'])
                        exit_status = "🎯 NEAR EXIT" if exit_distance <= 10 else "🔄 ACTIVE"
                        
                        alerts.append({
                            'Symbol': row['Symbol'],
                            'Signal': signal,
                            'CSI_Q': row['CSI_Q'],
                            'Strength': strength,
                            'Priority': priority,
                            'Price': row['Price'],
                            'Sentiment': row['Sentiment'],
                            'Mentions': row['Mentions'],
                            'Exit_Target': row['exit_target_csiq'],
                            'Exit_Status': exit_status,
                            'Mean_Reversion_Strength': row['mean_reversion_strength']
                        })
                
                # Sort by priority and CSI-Q extremity
                alerts = sorted(alerts, key=lambda x: (x['Priority'], -abs(x['CSI_Q'] - 50)))
                
                for alert in alerts[:8]:
                    signal_emoji = get_signal_color(alert['Signal'])
                    
                    st.markdown(f"""
                    <div class="signal-{alert['Signal'].lower()}">
                        {signal_emoji} <b>{alert['Symbol']}</b> | {alert['Strength']}<br>
                        <b>{alert['Signal']} SIGNAL</b><br>
                        CSI-Q: {alert['CSI_Q']:.1f} → Exit: {alert['Exit_Target']:.1f}<br>
                        Sentiment: {alert['Sentiment']:.3f} | Mentions: {alert['Mentions']:,}<br>
                        ${alert['Price']:.4f} | {alert['Exit_Status']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
        
        # Enhanced data table with exit zones
        st.markdown("### 📊 TERMINAL DATA STREAM")
        
        if not filtered_df.empty:
            display_cols = ['Symbol', 'Signal', 'CSI_Q', 'exit_target_csiq', 'Price', 'Change_24h', 
                           'Sentiment', 'Mentions', 'Funding_Rate', 'Long_Short_Ratio', 'Volume_24h']
            
            styled_df = filtered_df[display_cols].copy()
            styled_df = styled_df.rename(columns={
                'exit_target_csiq': 'Exit_Target',
                'Change_24h': 'Change_24h_(%)',
                'Volume_24h': 'Volume_($M)',
                'Funding_Rate': 'Funding_(%)'
            })
            
            # Format columns
            styled_df['Price'] = styled_df['Price'].round(4)
            styled_df['CSI_Q'] = styled_df['CSI_Q'].round(1)
            styled_df['Exit_Target'] = styled_df['Exit_Target'].round(1)
            styled_df['Change_24h_(%)'] = styled_df['Change_24h_(%)'].round(2)
            styled_df['Sentiment'] = styled_df['Sentiment'].round(3)
            styled_df['Funding_(%)'] = styled_df['Funding_(%)'].round(4)
            styled_df['Long_Short_Ratio'] = styled_df['Long_Short_Ratio'].round(2)
            styled_df['Volume_($M)'] = (styled_df['Volume_($M)'] / 1000000).round(1)
            
            st.dataframe(styled_df, use_container_width=True, height=400)
    
    with tab2:
        st.markdown("## 🎯 ADVANCED QUANT ANALYSIS")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment vs CSI-Q correlation
                fig = px.scatter(
                    filtered_df,
                    x='Sentiment',
                    y='CSI_Q',
                    size='Mentions',
                    color='Signal',
                    hover_name='Symbol',
                    title="🧠 ENHANCED SENTIMENT vs CSI-Q CORRELATION",
                    color_discrete_map={
                        'LONG': '#00ff41',
                        'SHORT': '#ff0040',
                        'CONTRARIAN': '#ffaa00',
                        'NEUTRAL': '#666666'
                    }
                )
                
                fig.add_vline(x=0, line_dash="dash", line_color="white")
                fig.add_hline(y=50, line_dash="dash", line_color="white")
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#00ff41')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Mean reversion analysis
                fig = px.scatter(
                    filtered_df,
                    x='CSI_Q',
                    y='mean_reversion_strength',
                    size='Volume_24h',
                    color='Signal',
                    hover_name='Symbol',
                    title="🔄 MEAN REVERSION POTENTIAL",
                    color_discrete_map={
                        'LONG': '#00ff41',
                        'SHORT': '#ff0040',
                        'CONTRARIAN': '#ffaa00',
                        'NEUTRAL': '#666666'
                    }
                )
                
                fig.add_hline(y=2, line_dash="dash", line_color="orange", 
                             annotation_text="High Reversion Potential")
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#00ff41')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Social sentiment heatmap
            st.markdown("### 📱 SOCIAL SENTIMENT MATRIX")
            
            # Create sentiment bins
            filtered_df['Sentiment_Bin'] = pd.cut(filtered_df['Sentiment'], 
                                                 bins=[-1, -0.5, -0.1, 0.1, 0.5, 1], 
                                                 labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
            
            sentiment_matrix = filtered_df.groupby(['Sentiment_Bin', 'Signal']).size().unstack(fill_value=0)
            
            fig = px.imshow(
                sentiment_matrix.values,
                x=sentiment_matrix.columns,
                y=sentiment_matrix.index,
                color_continuous_scale='Viridis',
                title="🔥 SENTIMENT vs SIGNAL TYPE HEATMAP",
                aspect="auto"
            )
            
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0.8)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#00ff41')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Advanced metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📊 SENTIMENT ANALYSIS")
                avg_sentiment = filtered_df['Sentiment'].mean()
                sentiment_std = filtered_df['Sentiment'].std()
                
                sentiment_status = "BULLISH" if avg_sentiment > 0.2 else "BEARISH" if avg_sentiment < -0.2 else "NEUTRAL"
                volatility_level = "HIGH" if sentiment_std > 0.4 else "MEDIUM" if sentiment_std > 0.2 else "LOW"
                
                st.markdown(f"""
                <div class="terminal-container">
                    <b>Market Sentiment:</b> {sentiment_status}<br>
                    <b>Average:</b> {avg_sentiment:.3f}<br>
                    <b>Volatility:</b> {volatility_level}<br>
                    <b>Std Dev:</b> {sentiment_std:.3f}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🔄 REVERSION STATS")
                high_reversion = len(filtered_df[filtered_df['mean_reversion_strength'] > 2])
                avg_reversion = filtered_df['mean_reversion_strength'].mean()
                
                st.markdown(f"""
                <div class="terminal-container">
                    <b>High Reversion Coins:</b> {high_reversion}<br>
                    <b>Avg Reversion Strength:</b> {avg_reversion:.2f}<br>
                    <b>Market Extreme Level:</b> {avg_reversion * 50:.1f}%<br>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("### 📱 SOCIAL METRICS")
                total_mentions = filtered_df['Mentions'].sum()
                avg_mentions = filtered_df['Mentions'].mean()
                top_buzzer = filtered_df.loc[filtered_df['Mentions'].idxmax(), 'Symbol']
                
                st.markdown(f"""
                <div class="terminal-container">
                    <b>Total Mentions:</b> {total_mentions:,}<br>
                    <b>Avg per Coin:</b> {avg_mentions:.0f}<br>
                    <b>Top Buzzer:</b> {top_buzzer}<br>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## 💰 ADVANCED TRADING TERMINAL")
        
        if not filtered_df.empty:
            # Enhanced opportunity scoring
            filtered_df['Risk_Reward_Score'] = (
                (abs(filtered_df['CSI_Q'] - 50) / 50 * 0.3) +  # CSI-Q extremity
                (abs(filtered_df['Sentiment']) * 0.2) +         # Sentiment strength
                (filtered_df['mean_reversion_strength'] * 0.2) +  # Reversion potential
                (abs(filtered_df['Funding_Rate']) * 10 * 0.15) +  # Funding opportunity
                ((filtered_df['Volume_24h'] / filtered_df['Volume_24h'].max()) * 0.15)  # Liquidity
            ) * 100
            
            # Top opportunities with enhanced exit logic
            opportunities = filtered_df.sort_values('Risk_Reward_Score', ascending=False).head(10)
            
            st.markdown("### 🚀 TOP 10 QUANT OPPORTUNITIES")
            
            st.info("⚡ Enhanced with CSI-Q mean reversion exits & realistic sentiment analysis")
            
            for i, (_, row) in enumerate(opportunities.iterrows()):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2.5])
                
                with col1:
                    signal_color = get_signal_color(row['Signal'])
                    reversion_strength = "🔥 EXTREME" if row['mean_reversion_strength'] > 2 else "⚡ HIGH" if row['mean_reversion_strength'] > 1 else "⚠️ MODERATE"
                    
                    st.markdown(f"**{i+1}. {signal_color} {row['Symbol']}**")
                    st.markdown(f"Risk/Reward Score: **{row['Risk_Reward_Score']:.1f}**")
                    st.markdown(f"Reversion Potential: **{reversion_strength}**")
                
                with col2:
                    st.metric("CSI-Q", f"{row['CSI_Q']:.1f}")
                    st.metric("Signal", row['Signal'])
                
                with col3:
                    st.metric("Price", f"${row['Price']:.4f}")
                    st.metric("Sentiment", f"{row['Sentiment']:.3f}")
                
                with col4:
                    # Enhanced trade setup with CSI-Q exits
                    entry_price = row['Price']
                    csiq_current = row['CSI_Q']
                    exit_target_csiq = row['exit_target_csiq']
                    stop_loss_csiq = row['stop_loss_csiq']
                    
                    # Estimate price targets based on typical CSI-Q/price correlations
                    csiq_change_to_exit = abs(csiq_current - exit_target_csiq)
                    estimated_price_change = csiq_change_to_exit * 0.5  # Rough correlation
                    
                    if row['Signal'] == 'LONG':
                        target_price = entry_price * (1 + estimated_price_change/100)
                        stop_price = entry_price * (1 - estimated_price_change/200)
                    elif row['Signal'] == 'SHORT':
                        target_price = entry_price * (1 - estimated_price_change/100)  
                        stop_price = entry_price * (1 + estimated_price_change/200)
                    else:  # CONTRARIAN
                        if csiq_current > 80:
                            target_price = entry_price * (1 - estimated_price_change/100)
                            stop_price = entry_price * (1 + estimated_price_change/300)
                        else:
                            target_price = entry_price * (1 + estimated_price_change/100)
                            stop_price = entry_price * (1 - estimated_price_change/300)
                    
                    risk_reward = abs((target_price - entry_price) / (stop_price - entry_price)) if stop_price != entry_price else 1
                    
                    # Position sizing based on volatility and sentiment
                    volatility_adj = 1 / max(0.1, abs(row['Sentiment']))
                    position_size = min(5, max(0.5, volatility_adj))
                    
                    st.markdown(f"""
                    **🎯 Enhanced Setup:**
                    - Entry: ${entry_price:.4f}
                    - Target: ${target_price:.4f}
                    - Stop: ${stop_price:.4f}
                    - **CSI-Q Exit: {exit_target_csiq:.1f}**
                    - R/R: 1:{risk_reward:.1f}
                    - Position: {position_size:.1f}% portfolio
                    - Mentions: {row['Mentions']:,}
                    - Social Sentiment: {row['Sentiment']:.3f}
                    """)
                
                st.markdown("---")
            
            # Market overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 MARKET REGIME ANALYSIS")
                
                avg_csiq = filtered_df['CSI_Q'].mean()
                avg_sentiment = filtered_df['Sentiment'].mean()
                avg_reversion = filtered_df['mean_reversion_strength'].mean()
                
                if avg_csiq > 70 and avg_sentiment > 0.3:
                    regime = "🔥 EUPHORIC BULL"
                    regime_color = "#ff6600"
                    advice = "Look for contrarian shorts & reversion plays"
                elif avg_csiq < 30 and avg_sentiment < -0.3:
                    regime = "🩸 PANIC BEAR"
                    regime_color = "#ff0040"
                    advice = "Look for contrarian longs & bounce plays"
                elif avg_reversion > 1.5:
                    regime = "⚡ HIGH VOLATILITY"
                    regime_color = "#ffaa00"
                    advice = "Focus on mean reversion strategies"
                else:
                    regime = "⚪ BALANCED MARKET"
                    regime_color = "#00ff41"
                    advice = "Follow momentum signals"
                
                st.markdown(f"""
                <div style="background: {regime_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
                    <h3>Market Regime: {regime}</h3>
                    <p><b>Strategy:</b> {advice}</p>
                    <p>Avg CSI-Q: {avg_csiq:.1f} | Sentiment: {avg_sentiment:.3f} | Reversion: {avg_reversion:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### ⚠️ RISK DASHBOARD")
                
                # Enhanced risk warnings
                warnings = []
                
                extreme_csiq = len(filtered_df[(filtered_df['CSI_Q'] > 95) | (filtered_df['CSI_Q'] < 5)])
                if extreme_csiq > 0:
                    warnings.append(f"🔥 {extreme_csiq} coins at MAXIMUM hysteria")
                
                extreme_sentiment = len(filtered_df[abs(filtered_df['Sentiment']) > 0.8])
                if extreme_sentiment > 0:
                    warnings.append(f"📱 {extreme_sentiment} coins with EXTREME sentiment")
                
                high_reversion = len(filtered_df[filtered_df['mean_reversion_strength'] > 2.5])
                if high_reversion > 0:
                    warnings.append(f"🔄 {high_reversion} coins primed for MAJOR reversion")
                
                low_liquidity = len(filtered_df[filtered_df['Volume_24h'] < 50000000])
                if low_liquidity > 0:
                    warnings.append(f"💧 {low_liquidity} coins with LOW liquidity")
                
                if warnings:
                    for warning in warnings:
                        st.markdown(f"""
                        <div style="background: #ff3333; padding: 8px; border-radius: 5px; margin: 3px 0; color: white; font-size: 0.9em;">
                            {warning}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #00ff41; padding: 10px; border-radius: 5px; color: black; text-align: center;">
                        ✅ Risk levels within normal parameters
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("## 📊 CSI-Q EXIT MONITOR")
        
        if not filtered_df.empty:
            # Filter for positions near exit zones
            near_exit_df = filtered_df[
                abs(filtered_df['CSI_Q'] - filtered_df['exit_target_csiq']) <= 15
            ].copy()
            
            near_exit_df['Distance_to_Exit'] = abs(near_exit_df['CSI_Q'] - near_exit_df['exit_target_csiq'])
            near_exit_df = near_exit_df.sort_values('Distance_to_Exit')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if not near_exit_df.empty:
                    # Exit zone visualization
                    fig = go.Figure()
                    
                    for _, row in near_exit_df.head(15).iterrows():
                        # Current CSI-Q
                        fig.add_trace(go.Scatter(
                            x=[row['Symbol']],
                            y=[row['CSI_Q']],
                            mode='markers',
                            name=f"{row['Symbol']} Current",
                            marker=dict(
                                size=15,
                                color='#00ff41' if row['Signal'] == 'LONG' else '#ff0040' if row['Signal'] == 'SHORT' else '#ffaa00',
                                symbol='circle'
                            ),
                            showlegend=False
                        ))
                        
                        # Exit target
                        fig.add_trace(go.Scatter(
                            x=[row['Symbol']],
                            y=[row['exit_target_csiq']],
                            mode='markers',
                            name=f"{row['Symbol']} Exit",
                            marker=dict(
                                size=12,
                                color='white',
                                symbol='x',
                                line=dict(width=3)
                            ),
                            showlegend=False
                        ))
                        
                        # Connection line
                        fig.add_trace(go.Scatter(
                            x=[row['Symbol'], row['Symbol']],
                            y=[row['CSI_Q'], row['exit_target_csiq']],
                            mode='lines',
                            line=dict(
                                color='rgba(255,255,255,0.5)',
                                width=2,
                                dash='dot'
                            ),
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title="🎯 CSI-Q EXIT ZONE MONITOR",
                        xaxis_title="SYMBOLS",
                        yaxis_title="CSI-Q LEVELS",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#00ff41')
                    )
                    
                    # Add mean reversion line
                    fig.add_hline(y=50, line_dash="solid", line_color="white", line_width=2,
                                 annotation_text="🎯 MEAN REVERSION TARGET")
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🎯 No positions currently near exit zones")
            
            with col2:
                st.markdown("### 🚨 EXIT ALERTS")
                
                if not near_exit_df.empty:
                    for _, row in near_exit_df.head(6).iterrows():
                        distance = row['Distance_to_Exit']
                        urgency = "🔥 IMMEDIATE" if distance <= 5 else "⚡ SOON" if distance <= 10 else "⚠️ WATCH"
                        
                        st.markdown(f"""
                        <div class="exit-zone">
                            <b>{row['Symbol']}</b> | {urgency}<br>
                            Current: {row['CSI_Q']:.1f}<br>
                            Target: {row['exit_target_csiq']:.1f}<br>
                            Distance: {distance:.1f}<br>
                            Signal: {row['Signal']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="terminal-container">
                        <b>🎯 ALL CLEAR</b><br>
                        No imminent exits detected<br>
                        Monitor for mean reversion
                    </div>
                    """, unsafe_allow_html=True)
            
            # Exit statistics
            st.markdown("### 📊 EXIT ZONE STATISTICS")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                immediate_exits = len(filtered_df[abs(filtered_df['CSI_Q'] - filtered_df['exit_target_csiq']) <= 5])
                st.markdown(f"""
                <div class="terminal-container">
                    <b>🔥 IMMEDIATE EXITS</b><br>
                    {immediate_exits} positions<br>
                    ≤ 5 CSI-Q points away
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                soon_exits = len(filtered_df[abs(filtered_df['CSI_Q'] - filtered_df['exit_target_csiq']) <= 10])
                st.markdown(f"""
                <div class="terminal-container">
                    <b>⚡ NEAR EXITS</b><br>
                    {soon_exits} positions<br>
                    ≤ 10 CSI-Q points away
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_distance = filtered_df['Distance_to_Exit'].mean() if 'Distance_to_Exit' in filtered_df.columns else abs(filtered_df['CSI_Q'] - filtered_df['exit_target_csiq']).mean()
                st.markdown(f"""
                <div class="terminal-container">
                    <b>📊 AVG DISTANCE</b><br>
                    {avg_distance:.1f} points<br>
                    To exit targets
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                strong_reversion = len(filtered_df[filtered_df['mean_reversion_strength'] > 2])
                st.markdown(f"""
                <div class="terminal-container">
                    <b>🔄 HIGH REVERSION</b><br>
                    {strong_reversion} positions<br>
                    Strong mean reversion
                </div>
                """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div class="terminal-container" style="text-align: center;">
        <h3>⚡ CRYPTO CSI-Q QUANT TERMINAL ⚡</h3>
        <p><b>🔥 ENHANCED FEATURES:</b> Advanced Sentiment Analysis + CSI-Q Mean Reversion Exits + Social Buzz Integration</p>
        <p><b>🎯 TRADING EDGE:</b> Multi-factor signal generation with dynamic exit zones</p>
        <p><b>⚠️ DISCLAIMER:</b> This is advanced quantitative analysis for educational purposes. Always implement proper risk management!</p>
        <p><b>📡 DATA:</b> Enhanced demo mode with realistic social sentiment simulation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
