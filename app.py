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

# Page config - Terminal theme
st.set_page_config(
    page_title="CSI-Q Quant Terminal",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Terminal CSS
st.markdown("""
<style>
    /* Global terminal theme */
    .stApp {
        background-color: #000000;
        color: #00ff00;
        font-family: 'Courier New', monospace;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Terminal containers */
    .terminal-box {
        background-color: #0a0a0a;
        border: 1px solid #00ff00;
        border-radius: 4px;
        padding: 15px;
        margin: 10px 0;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    
    /* Clickable metric cards */
    .metric-terminal {
        background-color: #111111;
        border: 1px solid #00ff00;
        padding: 15px;
        border-radius: 4px;
        color: #00ff00;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        font-family: 'Courier New', monospace;
    }
    
    .metric-terminal:hover {
        background-color: #1a1a1a;
        border-color: #00ff88;
        transform: translateY(-2px);
    }
    
    /* Signal status indicators */
    .signal-long {
        background-color: #002200;
        border: 1px solid #00ff00;
        padding: 8px;
        border-radius: 3px;
        color: #00ff00;
        font-weight: bold;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    
    .signal-short {
        background-color: #220000;
        border: 1px solid #ff0000;
        padding: 8px;
        border-radius: 3px;
        color: #ff0000;
        font-weight: bold;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    
    .signal-contrarian {
        background-color: #221100;
        border: 1px solid #ffff00;
        padding: 8px;
        border-radius: 3px;
        color: #ffff00;
        font-weight: bold;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    
    .signal-neutral {
        background-color: #111111;
        border: 1px solid #666666;
        padding: 8px;
        border-radius: 3px;
        color: #666666;
        font-weight: bold;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    
    /* Risk levels */
    .risk-extreme {
        background-color: #330000;
        border: 1px solid #ff0000;
        padding: 5px;
        border-radius: 3px;
        color: #ff0000;
        font-size: 11px;
        margin: 2px 0;
    }
    
    .risk-high {
        background-color: #332200;
        border: 1px solid #ff8800;
        padding: 5px;
        border-radius: 3px;
        color: #ff8800;
        font-size: 11px;
        margin: 2px 0;
    }
    
    .risk-medium {
        background-color: #222200;
        border: 1px solid #ffff00;
        padding: 5px;
        border-radius: 3px;
        color: #ffff00;
        font-size: 11px;
        margin: 2px 0;
    }
    
    /* Terminal header */
    .terminal-header {
        background-color: #000000;
        color: #00ff00;
        padding: 20px;
        font-family: 'Courier New', monospace;
        border: 2px solid #00ff00;
        border-radius: 4px;
        text-align: center;
        margin: 15px 0;
        font-size: 14px;
    }
    
    /* Data tables */
    .stDataFrame {
        background-color: #000000;
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0a0a0a;
        border-radius: 4px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #111111;
        color: #00ff00;
        border: 1px solid #333333;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #002200;
        border-color: #00ff00;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #111111;
        color: #00ff00;
        border: 1px solid #00ff00;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    
    .stButton > button:hover {
        background-color: #002200;
        border-color: #00ff88;
    }
    
    /* Clickable coin analysis */
    .coin-detail {
        background-color: #0a0a0a;
        border: 1px solid #00ff00;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }
    
    .clickable-coin {
        cursor: pointer;
        padding: 5px;
        border: 1px solid #333333;
        margin: 2px;
        border-radius: 3px;
        display: inline-block;
        background-color: #111111;
    }
    
    .clickable-coin:hover {
        border-color: #00ff00;
        background-color: #002200;
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

class AdvancedDataFetcher:
    def __init__(self):
        self.binance_base = "https://fapi.binance.com"
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        
    def generate_enhanced_sentiment(self, symbol, price_change):
        """Generate realistic sentiment data with multiple factors"""
        np.random.seed(hash(symbol) % 1000)
        
        # Base sentiment from price action
        price_sentiment = np.tanh(price_change / 5)
        
        # Symbol-specific characteristics
        hype_multipliers = {
            'BTC': 1.2, 'ETH': 1.2, 'SOL': 1.4, 'DOGE': 2.0, 'SHIB': 2.5,
            'ADA': 1.1, 'DOT': 1.0, 'LINK': 1.1, 'AVAX': 1.3
        }
        hype_factor = hype_multipliers.get(symbol, 1.0)
        
        # Market timing effects
        hour = datetime.now().hour
        weekend_factor = 1.3 if datetime.now().weekday() >= 5 else 1.0
        
        # Volatility clustering
        volatility_regimes = np.random.choice([0.2, 0.5, 0.8, 1.2], p=[0.4, 0.3, 0.2, 0.1])
        
        # Final sentiment
        sentiment = (
            price_sentiment * 0.4 +
            np.random.normal(0, volatility_regimes) * 0.4 +
            (np.sin(hour * np.pi / 12) * 0.1 * weekend_factor) * 0.2
        ) * hype_factor
        
        sentiment = max(-1, min(1, sentiment))
        
        # Mentions based on multiple factors
        base_mentions = {
            'BTC': 50000, 'ETH': 30000, 'SOL': 15000, 'XRP': 10000,
            'ADA': 8000, 'AVAX': 5000, 'DOT': 4000
        }.get(symbol, 2000)
        
        mention_multiplier = (1 + abs(sentiment)) * (1 + abs(price_change)/10) * weekend_factor
        mentions = int(base_mentions * mention_multiplier * np.random.uniform(0.5, 1.5))
        
        return sentiment, mentions
    
    def calculate_advanced_metrics(self, row):
        """Calculate advanced risk and opportunity metrics"""
        symbol = row['Symbol']
        csiq = row['CSI_Q']
        sentiment = row['Sentiment']
        funding = row['Funding_Rate']
        
        # Risk categorization
        risk_factors = []
        risk_score = 0
        
        if abs(sentiment) > 0.8:
            risk_factors.append("EXTREME_SENTIMENT")
            risk_score += 30
            
        if csiq > 95 or csiq < 5:
            risk_factors.append("MAX_HYSTERIA")
            risk_score += 40
            
        if abs(funding) > 0.2:
            risk_factors.append("EXTREME_FUNDING")
            risk_score += 25
            
        if row['Long_Short_Ratio'] > 4 or row['Long_Short_Ratio'] < 0.25:
            risk_factors.append("EXTREME_POSITIONING")
            risk_score += 20
            
        if row['mean_reversion_strength'] > 2.5:
            risk_factors.append("HIGH_REVERSION_RISK")
            risk_score += 15
            
        # Opportunity scoring
        opportunity_factors = []
        opportunity_score = 0
        
        if 70 < csiq < 85 or 15 < csiq < 30:
            opportunity_factors.append("SIGNAL_ZONE")
            opportunity_score += 25
            
        if 0.3 < abs(sentiment) < 0.7:
            opportunity_factors.append("MODERATE_SENTIMENT")
            opportunity_score += 20
            
        if row['Volume_24h'] > 100000000:
            opportunity_factors.append("HIGH_LIQUIDITY")
            opportunity_score += 15
            
        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'opportunity_factors': opportunity_factors,
            'opportunity_score': opportunity_score
        }
    
    def calculate_exit_zones(self, csiq_current, signal):
        """Calculate dynamic exit zones"""
        if signal == "LONG":
            exit_target = max(40, 50 - (csiq_current - 50) * 0.3)
            stop_loss_csiq = min(15, csiq_current - 25)
        elif signal == "SHORT":
            exit_target = min(60, 50 + (50 - csiq_current) * 0.3)
            stop_loss_csiq = max(85, csiq_current + 25)
        elif signal == "CONTRARIAN":
            if csiq_current > 80:
                exit_target = 60
                stop_loss_csiq = 95
            else:
                exit_target = 40
                stop_loss_csiq = 5
        else:
            exit_target = 50
            stop_loss_csiq = csiq_current
            
        return {
            'exit_target_csiq': exit_target,
            'stop_loss_csiq': stop_loss_csiq,
            'mean_reversion_strength': abs(csiq_current - 50) / 20
        }
    
    def generate_terminal_data(self):
        """Generate comprehensive terminal data"""
        np.random.seed(42)
        
        base_prices = {
            'BTC': 43000, 'ETH': 2600, 'BNB': 310, 'SOL': 100, 'XRP': 0.52,
            'ADA': 0.48, 'AVAX': 38, 'DOT': 7.2, 'LINK': 14.5, 'MATIC': 0.85,
            'UNI': 6.8, 'LTC': 73, 'BCH': 250, 'NEAR': 2.1, 'ALGO': 0.19,
            'VET': 0.025, 'FIL': 5.5, 'ETC': 20, 'AAVE': 95, 'MKR': 1450,
            'ATOM': 9.8, 'FTM': 0.32, 'SAND': 0.42, 'MANA': 0.38, 'AXS': 6.2
        }
        
        data_list = []
        
        for ticker in TICKERS:
            symbol_clean = ticker.replace('USDT', '')
            base_price = base_prices.get(symbol_clean, 1.0)
            
            # Market dynamics
            price_change = np.random.normal(0, 0.06)
            current_price = base_price * (1 + price_change)
            change_24h = np.random.normal(0, 8)
            funding_rate = np.random.normal(0.01, 0.06)
            oi_change = np.random.normal(0, 25)
            long_short_ratio = np.random.lognormal(0, 0.6)
            
            # Enhanced sentiment and social
            sentiment, mentions = self.generate_enhanced_sentiment(symbol_clean, change_24h)
            
            # Volume tiers
            volume_tiers = {
                'T1': ['BTC', 'ETH'],
                'T2': ['BNB', 'SOL', 'XRP', 'ADA'],
                'T3': ['AVAX', 'DOT', 'LINK', 'MATIC', 'UNI']
            }
            
            if symbol_clean in volume_tiers['T1']:
                volume_24h = np.random.uniform(15000000000, 40000000000)
            elif symbol_clean in volume_tiers['T2']:
                volume_24h = np.random.uniform(800000000, 8000000000)
            elif symbol_clean in volume_tiers['T3']:
                volume_24h = np.random.uniform(200000000, 2000000000)
            else:
                volume_24h = np.random.uniform(50000000, 800000000)
            
            # Technical indicators
            rsi = 50 + change_24h * 1.5 + np.random.normal(0, 12)
            rsi = max(5, min(95, rsi))
            bb_squeeze = np.random.beta(2, 5)
            basis = sentiment * 0.3 + np.random.normal(0, 0.4)
            
            # CSI-Q calculation
            derivatives_score = min(100, max(0,
                (abs(oi_change) * 1.8) +
                (abs(funding_rate) * 400) +
                (abs(long_short_ratio - 1) * 25) +
                35
            ))
            
            social_score = min(100, max(0,
                ((sentiment + 1) * 22) +
                (min(mentions, 15000) / 150) +
                25
            ))
            
            basis_score = min(100, max(0,
                abs(basis) * 400 + 30
            ))
            
            tech_score = min(100, max(0,
                (100 - abs(rsi - 50)) * 0.75 +
                ((1 - bb_squeeze) * 35) +
                15
            ))
            
            csiq = (
                derivatives_score * 0.4 +
                social_score * 0.3 +
                basis_score * 0.2 +
                tech_score * 0.1
            )
            
            # Calculate exit zones and signal
            signal = self.get_signal_type(csiq, funding_rate)
            exit_data = self.calculate_exit_zones(csiq, signal)
            
            row_data = {
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
                'ATR': abs(current_price * 0.045),
                'Volume_24h': volume_24h,
                'Open_Interest': volume_24h * np.random.uniform(0.15, 1.8),
                'Signal': signal,
                **exit_data
            }
            
            # Add advanced metrics
            advanced_metrics = self.calculate_advanced_metrics(row_data)
            row_data.update(advanced_metrics)
            
            data_list.append(row_data)
        
        return pd.DataFrame(data_list)
    
    def get_signal_type(self, csiq, funding_rate):
        """Determine signal type"""
        if csiq > 90 or csiq < 10:
            return "CONTRARIAN"
        elif csiq > 70 and funding_rate < 0.1:
            return "LONG"
        elif csiq < 30 and funding_rate > -0.1:
            return "SHORT"
        else:
            return "NEUTRAL"

@st.cache_data(ttl=300)
def load_terminal_data():
    """Load data for terminal"""
    fetcher = AdvancedDataFetcher()
    return fetcher.generate_terminal_data()

def render_coin_analysis(coin_data):
    """Render detailed coin analysis"""
    symbol = coin_data['Symbol']
    
    st.markdown(f"""
    <div class="coin-detail">
    <h3>>>> DETAILED ANALYSIS: {symbol} <<<</h3>
    
    <b>PRICE DATA:</b>
    • Current Price: ${coin_data['Price']:.4f}
    • 24h Change: {coin_data['Change_24h']:.2f}%
    • ATR: ${coin_data['ATR']:.4f}
    • Volume 24h: ${coin_data['Volume_24h']/1000000:.1f}M
    
    <b>CSI-Q BREAKDOWN:</b>
    • Overall CSI-Q: {coin_data['CSI_Q']:.1f}/100
    • Derivatives Score: {coin_data['Derivatives_Score']:.1f} (40% weight)
    • Social Score: {coin_data['Social_Score']:.1f} (30% weight)
    • Basis Score: {coin_data['Basis_Score']:.1f} (20% weight)
    • Technical Score: {coin_data['Tech_Score']:.1f} (10% weight)
    
    <b>DERIVATIVES METRICS:</b>
    • Funding Rate: {coin_data['Funding_Rate']:.4f}%
    • OI Change 24h: {coin_data['OI_Change']:.1f}%
    • Long/Short Ratio: {coin_data['Long_Short_Ratio']:.2f}
    • Open Interest: ${coin_data['Open_Interest']/1000000:.1f}M
    
    <b>SOCIAL SENTIMENT:</b>
    • Sentiment Score: {coin_data['Sentiment']:.3f} (-1 to +1)
    • Social Mentions: {coin_data['Mentions']:,}
    • Spot-Futures Basis: {coin_data['Spot_Futures_Basis']:.4f}
    
    <b>TECHNICAL INDICATORS:</b>
    • RSI: {coin_data['RSI']:.1f}
    • Bollinger Squeeze: {coin_data['BB_Squeeze']:.3f}
    
    <b>SIGNAL ANALYSIS:</b>
    • Current Signal: {coin_data['Signal']}
    • Exit Target CSI-Q: {coin_data['exit_target_csiq']:.1f}
    • Stop Loss CSI-Q: {coin_data['stop_loss_csiq']:.1f}
    • Mean Reversion Strength: {coin_data['mean_reversion_strength']:.2f}
    
    <b>RISK ASSESSMENT:</b>
    • Risk Score: {coin_data['risk_score']}/100
    • Risk Factors: {', '.join(coin_data['risk_factors']) if coin_data['risk_factors'] else 'None'}
    • Opportunity Score: {coin_data['opportunity_score']}/100
    • Opportunity Factors: {', '.join(coin_data['opportunity_factors']) if coin_data['opportunity_factors'] else 'None'}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Terminal header
    st.markdown("""
    <div class="terminal-header">
    <h1>CSI-Q QUANTITATIVE TERMINAL</h1>
    <p>ADVANCED DERIVATIVES & SENTIMENT ANALYSIS SYSTEM</p>
    <p>STATUS: ACTIVE | MODE: PROFESSIONAL | DATA: ENHANCED SIMULATION</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_terminal_data()
    
    # Initialize session state for coin selection
    if 'selected_coin' not in st.session_state:
        st.session_state.selected_coin = None
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="terminal-box">SYS: ONLINE</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="terminal-box">TIME: {datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="terminal-box">CONN: STABLE</div>', unsafe_allow_html=True)
    with col4:
        if st.button("REFRESH", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Metrics overview with clickable elements
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_signals = len(df[df['Signal'] != 'NEUTRAL'])
        st.markdown(f"""
        <div class="metric-terminal" onclick="document.getElementById('signals-tab').click()">
            <h4>ACTIVE SIGNALS</h4>
            <h2>{active_signals}</h2>
            <small>Click for details</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_csiq = df['CSI_Q'].mean()
        market_state = "EXTREME" if avg_csiq > 70 or avg_csiq < 30 else "NORMAL"
        st.markdown(f"""
        <div class="metric-terminal" onclick="document.getElementById('analysis-tab').click()">
            <h4>MARKET CSI-Q</h4>
            <h2>{avg_csiq:.1f}</h2>
            <small>{market_state} | Click for analysis</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        extreme_risk = len(df[df['risk_score'] > 60])
        st.markdown(f"""
        <div class="metric-terminal" onclick="document.getElementById('risk-tab').click()">
            <h4>HIGH RISK ASSETS</h4>
            <h2>{extreme_risk}</h2>
            <small>Click for risk dashboard</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        near_exit = len(df[abs(df['CSI_Q'] - df['exit_target_csiq']) <= 10])
        st.markdown(f"""
        <div class="metric-terminal" onclick="document.getElementById('exit-tab').click()">
            <h4>NEAR EXIT ZONES</h4>
            <h2>{near_exit}</h2>
            <small>Click for exit monitor</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("### TERMINAL CONTROLS")
    min_csiq = st.sidebar.slider("MIN CSI-Q", 0, 100, 0)
    max_csiq = st.sidebar.slider("MAX CSI-Q", 0, 100, 100)
    signal_filter = st.sidebar.multiselect("SIGNALS", ["LONG", "SHORT", "CONTRARIAN", "NEUTRAL"], default=["LONG", "SHORT", "CONTRARIAN"])
    min_volume = st.sidebar.number_input("MIN VOLUME ($M)", 0, 1000, 0)
    min_risk = st.sidebar.slider("MIN RISK SCORE", 0, 100, 0)
    
    # Apply filters
    filtered_df = df[
        (df['CSI_Q'] >= min_csiq) & 
        (df['CSI_Q'] <= max_csiq) &
        (df['Signal'].isin(signal_filter)) &
        (df['Volume_24h'] >= min_volume * 1000000) &
        (df['risk_score'] >= min_risk)
    ].copy()
    
    # Main terminal tabs
    tab1, tab2, tab3, tab4 = st.tabs(["SIGNALS", "ANALYSIS", "RISK DASHBOARD", "EXIT MONITOR"])
    
    with tab1:
        st.markdown("### SIGNAL MATRIX")
        
        if not filtered_df.empty:
            # Signal visualization
            fig = go.Figure()
            
            signal_colors = {'LONG': '#00ff00', 'SHORT': '#ff0000', 'CONTRARIAN': '#ffff00', 'NEUTRAL': '#666666'}
            
            for signal_type in signal_colors.keys():
                signal_data = filtered_df[filtered_df['Signal'] == signal_type]
                if not signal_data.empty:
                    fig.add_trace(go.Scatter(
                        x=signal_data['Symbol'],
                        y=signal_data['CSI_Q'],
                        mode='markers+text',
                        name=signal_type,
                        marker=dict(
                            size=np.sqrt(signal_data['Volume_24h']) / 100000,
                            color=signal_colors[signal_type],
                            line=dict(width=1, color='white')
                        ),
                        text=signal_data['Symbol'],
                        textposition="middle center",
                        hovertemplate="<b>%{text}</b><br>CSI-Q: %{y:.1f}<br>Risk: " + 
                                    signal_data['risk_score'].astype(str) + "<br><extra></extra>"
                    ))
            
            fig.update_layout(
                title="CSI-Q SIGNAL DISTRIBUTION",
                xaxis_title="ASSETS",
                yaxis_title="CSI-Q SCORE",
                height=500,
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#00ff00', family='Courier New'),
                showlegend=True,
                legend=dict(font=dict(color='#00ff00'))
            )
            
            # Add signal zones
            fig.add_hline(y=90, line_dash="dash", line_color="#ffff00", annotation_text="CONTRARIAN")
            fig.add_hline(y=70, line_dash="dash", line_color="#00ff00", annotation_text="LONG ZONE")
            fig.add_hline(y=50, line_dash="solid", line_color="#ffffff", annotation_text="MEAN")
            fig.add_hline(y=30, line_dash="dash", line_color="#ff0000", annotation_text="SHORT ZONE") 
            fig.add_hline(y=10, line_dash="dash", line_color="#ffff00", annotation_text="CONTRARIAN")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal table
            st.markdown("### ACTIVE SIGNALS TABLE")
            display_df = filtered_df[['Symbol', 'Signal', 'CSI_Q', 'Price', 'Change_24h', 'risk_score', 'opportunity_score']].copy()
            display_df = display_df.round({'CSI_Q': 1, 'Price': 4, 'Change_24h': 2})
            st.dataframe(display_df, use_container_width=True, height=400)
    
    with tab2:
        st.markdown("### QUANTITATIVE ANALYSIS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSI-Q distribution
            fig = px.histogram(
                filtered_df, 
                x='CSI_Q', 
                nbins=20,
                title="CSI-Q DISTRIBUTION",
                color_discrete_sequence=['#00ff00']
            )
            fig.update_layout(
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#00ff00', family='Courier New')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment vs CSI-Q
            fig = px.scatter(
                filtered_df,
                x='Sentiment',
                y='CSI_Q',
                size='Volume_24h',
                color='risk_score',
                hover_name='Symbol',
                title="SENTIMENT vs CSI-Q",
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#00ff00', family='Courier New')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Component analysis
        st.markdown("### CSI-Q COMPONENT BREAKDOWN")
        
        component_df = filtered_df[['Symbol', 'CSI_Q', 'Derivatives_Score', 'Social_Score', 'Basis_Score', 'Tech_Score']].head(12)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Derivatives (40%)', x=component_df['Symbol'], y=component_df['Derivatives_Score'], marker_color='#ff4444'))
        fig.add_trace(go.Bar(name='Social (30%)', x=component_df['Symbol'], y=component_df['Social_Score'], marker_color='#44ff44'))
        fig.add_trace(go.Bar(name='Basis (20%)', x=component_df['Symbol'], y=component_df['Basis_Score'], marker_color='#ffff44'))
        fig.add_trace(go.Bar(name='Technical (10%)', x=component_df['Symbol'], y=component_df['Tech_Score'], marker_color='#4444ff'))
        
        fig.update_layout(
            title="CSI-Q COMPONENT SCORES",
            barmode='group',
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#00ff00', family='Courier New'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### COMPREHENSIVE RISK DASHBOARD")
        
        # Risk categorization
        extreme_sentiment = filtered_df[filtered_df['risk_factors'].apply(lambda x: 'EXTREME_SENTIMENT' in x)]
        max_hysteria = filtered_df[filtered_df['risk_factors'].apply(lambda x: 'MAX_HYSTERIA' in x)]
        extreme_funding = filtered_df[filtered_df['risk_factors'].apply(lambda x: 'EXTREME_FUNDING' in x)]
        extreme_positioning = filtered_df[filtered_df['risk_factors'].apply(lambda x: 'EXTREME_POSITIONING' in x)]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### EXTREME SENTIMENT ASSETS")
            if not extreme_sentiment.empty:
                for _, row in extreme_sentiment.iterrows():
                    st.markdown(f"""
                    <div class="risk-extreme clickable-coin" onclick="selectCoin('{row['Symbol']}')">
                    {row['Symbol']}: Sentiment {row['Sentiment']:.3f} | CSI-Q {row['CSI_Q']:.1f}
                    Risk Score: {row['risk_score']}/100
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="terminal-box">No extreme sentiment detected</div>', unsafe_allow_html=True)
            
            st.markdown("#### MAX HYSTERIA LEVELS")
            if not max_hysteria.empty:
                for _, row in max_hysteria.iterrows():
                    st.markdown(f"""
                    <div class="risk-extreme clickable-coin" onclick="selectCoin('{row['Symbol']}')">
                    {row['Symbol']}: CSI-Q {row['CSI_Q']:.1f} | Risk {row['risk_score']}/100
                    Mean Reversion: {row['mean_reversion_strength']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="terminal-box">No max hysteria levels</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### EXTREME FUNDING RATES")
            if not extreme_funding.empty:
                for _, row in extreme_funding.iterrows():
                    st.markdown(f"""
                    <div class="risk-high clickable-coin" onclick="selectCoin('{row['Symbol']}')">
                    {row['Symbol']}: Funding {row['Funding_Rate']:.4f}% | CSI-Q {row['CSI_Q']:.1f}
                    Risk Score: {row['risk_score']}/100
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="terminal-box">Normal funding rates</div>', unsafe_allow_html=True)
            
            st.markdown("#### EXTREME POSITIONING")
            if not extreme_positioning.empty:
                for _, row in extreme_positioning.iterrows():
                    st.markdown(f"""
                    <div class="risk-medium clickable-coin" onclick="selectCoin('{row['Symbol']}')">
                    {row['Symbol']}: L/S Ratio {row['Long_Short_Ratio']:.2f} | CSI-Q {row['CSI_Q']:.1f}
                    Risk Score: {row['risk_score']}/100
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="terminal-box">Balanced positioning</div>', unsafe_allow_html=True)
        
        # Overall risk metrics
        st.markdown("#### SYSTEM RISK OVERVIEW")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_risk_count = len(filtered_df[filtered_df['risk_score'] > 70])
            st.markdown(f"""
            <div class="terminal-box">
            <b>HIGH RISK ASSETS:</b><br>
            {high_risk_count} / {len(filtered_df)}<br>
            <small>{high_risk_count/len(filtered_df)*100:.1f}% of portfolio</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_risk = filtered_df['risk_score'].mean()
            risk_level = "EXTREME" if avg_risk > 60 else "HIGH" if avg_risk > 40 else "MODERATE" if avg_risk > 20 else "LOW"
            st.markdown(f"""
            <div class="terminal-box">
            <b>AVERAGE RISK:</b><br>
            {avg_risk:.1f} / 100<br>
            <small>Level: {risk_level}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            extreme_csiq_count = len(filtered_df[(filtered_df['CSI_Q'] > 90) | (filtered_df['CSI_Q'] < 10)])
            st.markdown(f"""
            <div class="terminal-box">
            <b>EXTREME CSI-Q:</b><br>
            {extreme_csiq_count} assets<br>
            <small>Contrarian candidates</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            high_reversion_count = len(filtered_df[filtered_df['mean_reversion_strength'] > 2])
            st.markdown(f"""
            <div class="terminal-box">
            <b>HIGH REVERSION:</b><br>
            {high_reversion_count} assets<br>
            <small>Mean reversion plays</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk heatmap
        st.markdown("#### RISK CORRELATION MATRIX")
        risk_data = filtered_df[['CSI_Q', 'Sentiment', 'Funding_Rate', 'Long_Short_Ratio', 'risk_score']].corr()
        
        fig = px.imshow(
            risk_data,
            title="RISK FACTOR CORRELATIONS",
            color_continuous_scale='RdYlGn_r',
            aspect="auto"
        )
        fig.update_layout(
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#00ff00', family='Courier New')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Clickable coin list for detailed analysis
        st.markdown("#### CLICK FOR DETAILED ANALYSIS")
        
        # Create clickable coin buttons
        coin_cols = st.columns(6)
        for i, symbol in enumerate(filtered_df['Symbol'].head(18)):
            with coin_cols[i % 6]:
                if st.button(symbol, key=f"coin_{symbol}"):
                    st.session_state.selected_coin = symbol
    
    with tab4:
        st.markdown("### EXIT ZONE MONITOR")
        
        # Calculate distance to exit for all positions
        filtered_df['exit_distance'] = abs(filtered_df['CSI_Q'] - filtered_df['exit_target_csiq'])
        near_exit = filtered_df[filtered_df['exit_distance'] <= 15].sort_values('exit_distance')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not near_exit.empty:
                # Exit zone visualization
                fig = go.Figure()
                
                for _, row in near_exit.head(15).iterrows():
                    # Current position
                    fig.add_trace(go.Scatter(
                        x=[row['Symbol']],
                        y=[row['CSI_Q']],
                        mode='markers',
                        name=f"{row['Symbol']} Current",
                        marker=dict(size=15, color='#00ff00', symbol='circle'),
                        showlegend=False
                    ))
                    
                    # Exit target
                    fig.add_trace(go.Scatter(
                        x=[row['Symbol']],
                        y=[row['exit_target_csiq']],
                        mode='markers',
                        name=f"{row['Symbol']} Exit",
                        marker=dict(size=12, color='#ffffff', symbol='x', line=dict(width=3)),
                        showlegend=False
                    ))
                    
                    # Connection line
                    fig.add_trace(go.Scatter(
                        x=[row['Symbol'], row['Symbol']],
                        y=[row['CSI_Q'], row['exit_target_csiq']],
                        mode='lines',
                        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="EXIT ZONE PROXIMITY MONITOR",
                    xaxis_title="ASSETS",
                    yaxis_title="CSI-Q LEVELS",
                    height=400,
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='#00ff00', family='Courier New')
                )
                
                fig.add_hline(y=50, line_dash="solid", line_color="#ffffff", annotation_text="MEAN REVERSION")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('<div class="terminal-box">No positions near exit zones</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### EXIT PRIORITIES")
            
            if not near_exit.empty:
                for _, row in near_exit.head(8).iterrows():
                    distance = row['exit_distance']
                    if distance <= 5:
                        urgency_class = "risk-extreme"
                        urgency_text = "IMMEDIATE"
                    elif distance <= 10:
                        urgency_class = "risk-high"
                        urgency_text = "SOON"
                    else:
                        urgency_class = "risk-medium"
                        urgency_text = "WATCH"
                    
                    st.markdown(f"""
                    <div class="{urgency_class} clickable-coin" onclick="selectCoin('{row['Symbol']}')">
                    {row['Symbol']} | {urgency_text}<br>
                    Current: {row['CSI_Q']:.1f} Target: {row['exit_target_csiq']:.1f}<br>
                    Distance: {distance:.1f} points
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="terminal-box">No immediate exit alerts</div>', unsafe_allow_html=True)
        
        # Exit statistics
        st.markdown("#### EXIT ZONE STATISTICS")
        
        col1, col2, col3, col4 = st.columns(4)
        
        immediate_exits = len(filtered_df[filtered_df['exit_distance'] <= 5]) if 'exit_distance' in filtered_df.columns else 0
        near_exits = len(filtered_df[filtered_df['exit_distance'] <= 10]) if 'exit_distance' in filtered_df.columns else 0
        avg_distance = filtered_df['exit_distance'].mean() if 'exit_distance' in filtered_df.columns else 0
        reversion_candidates = len(filtered_df[filtered_df['mean_reversion_strength'] > 2])
        
        with col1:
            st.markdown(f"""
            <div class="terminal-box">
            <b>IMMEDIATE EXITS:</b><br>
            {immediate_exits}<br>
            <small>≤ 5 points</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="terminal-box">
            <b>NEAR EXITS:</b><br>
            {near_exits}<br>
            <small>≤ 10 points</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="terminal-box">
            <b>AVG DISTANCE:</b><br>
            {avg_distance:.1f}<br>
            <small>To exit targets</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="terminal-box">
            <b>REVERSION PLAYS:</b><br>
            {reversion_candidates}<br>
            <small>High probability</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed coin analysis section
    if st.session_state.selected_coin:
        st.markdown("---")
        st.markdown("### DETAILED ASSET ANALYSIS")
        
        coin_data = df[df['Symbol'] == st.session_state.selected_coin].iloc[0]
        render_coin_analysis(coin_data)
        
        if st.button("CLEAR SELECTION"):
            st.session_state.selected_coin = None
            st.rerun()
    
    # Terminal footer
    st.markdown("---")
    st.markdown("""
    <div class="terminal-box" style="text-align: center;">
    <b>CSI-Q QUANTITATIVE TERMINAL</b><br>
    Advanced Multi-Factor Analysis | Mean Reversion Exit System | Professional Risk Management<br>
    STATUS: OPERATIONAL | MODE: ENHANCED SIMULATION | RISK MANAGEMENT: ACTIVE<br>
    <small>Disclaimer: Advanced quantitative analysis for professional traders. Implement proper risk controls.</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
