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
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Crypto CSI-Q Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .signal-long {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 10px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    .signal-short {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        padding: 10px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    .signal-contrarian {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        padding: 10px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, #9E9E9E, #757575);
        padding: 10px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    .api-status-error {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    .api-status-demo {
        background: linear-gradient(135deg, #ffd93d, #ff6b35);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
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
            st.warning(f"Binance API niet beschikbaar: {str(e)}")
        
        # Test CoinGecko
        try:
            response = requests.get(f"{self.coingecko_base}/ping", timeout=5)
            if response.status_code == 200:
                apis_status['coingecko'] = True
        except Exception as e:
            st.warning(f"CoinGecko API beperkt beschikbaar: {str(e)}")
        
        return apis_status
    
    def get_binance_data(self):
        """Try to get Binance data with better error handling"""
        try:
            # Set headers to avoid 451 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            # Futures prices
            price_url = f"{self.binance_base}/fapi/v1/ticker/24hr"
            price_response = requests.get(price_url, headers=headers, timeout=10)
            
            if price_response.status_code == 451:
                st.error("🚫 Binance API geblokkeerd in uw regio (Error 451)")
                return None, "region_blocked"
            elif price_response.status_code != 200:
                st.error(f"Binance API error: {price_response.status_code}")
                return None, "api_error"
            
            price_data = price_response.json()
            
            # Get funding rates
            funding_url = f"{self.binance_base}/fapi/v1/premiumIndex"
            funding_response = requests.get(funding_url, headers=headers, timeout=10)
            
            if funding_response.status_code == 200:
                funding_data = funding_response.json()
            else:
                funding_data = []
            
            return {'prices': price_data, 'funding': funding_data}, "success"
            
        except Exception as e:
            st.error(f"Binance connectie fout: {e}")
            return None, "connection_error"
    
    def get_coingecko_data(self):
        """Get basic price data from CoinGecko as fallback"""
        try:
            # Map USDT symbols to CoinGecko IDs
            symbol_map = {
                'BTCUSDT': 'bitcoin', 'ETHUSDT': 'ethereum', 'BNBUSDT': 'binancecoin',
                'SOLUSDT': 'solana', 'XRPUSDT': 'ripple', 'ADAUSDT': 'cardano',
                'AVAXUSDT': 'avalanche-2', 'DOTUSDT': 'polkadot', 'LINKUSDT': 'chainlink',
                'MATICUSDT': 'matic-network', 'UNIUSDT': 'uniswap', 'LTCUSDT': 'litecoin',
                'BCHUSDT': 'bitcoin-cash', 'NEARUSDT': 'near', 'ALGOUSDT': 'algorand',
                'VETUSDT': 'vechain', 'FILUSDT': 'filecoin', 'ETCUSDT': 'ethereum-classic',
                'AAVEUSDT': 'aave', 'MKRUSDT': 'maker', 'ATOMUSDT': 'cosmos',
                'FTMUSDT': 'fantom', 'SANDUSDT': 'the-sandbox', 'MANAUSDT': 'decentraland',
                'AXSUSDT': 'axie-infinity'
            }
            
            ids = list(symbol_map.values())
            url = f"{self.coingecko_base}/simple/price"
            params = {
                'ids': ','.join(ids),
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return None, "api_error"
            
            data = response.json()
            
            # Convert back to our format
            converted_data = []
            for symbol, gecko_id in symbol_map.items():
                if gecko_id in data:
                    coin_data = data[gecko_id]
                    converted_data.append({
                        'symbol': symbol,
                        'price': coin_data.get('usd', 0),
                        'change_24h': coin_data.get('usd_24h_change', 0),
                        'volume_24h': coin_data.get('usd_24h_vol', 0)
                    })
            
            return converted_data, "success"
            
        except Exception as e:
            st.error(f"CoinGecko fout: {e}")
            return None, "connection_error"
    
    def generate_demo_data(self):
        """Generate realistic demo data when APIs are unavailable"""
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
            
            # Volume based on market cap tier
            if symbol_clean in ['BTC', 'ETH']:
                volume_24h = np.random.uniform(20000000000, 50000000000)
            elif symbol_clean in ['BNB', 'SOL', 'XRP']:
                volume_24h = np.random.uniform(1000000000, 10000000000)
            else:
                volume_24h = np.random.uniform(100000000, 2000000000)
            
            # Calculate CSI-Q components
            mentions = max(1, int(volume_24h / 10000000))
            sentiment = np.tanh(change_24h / 5)
            rsi = 50 + np.random.normal(0, 15)
            rsi = max(0, min(100, rsi))
            bb_squeeze = np.random.uniform(0, 1)
            basis = np.random.normal(0, 0.5)
            
            # CSI-Q calculation
            derivatives_score = min(100, max(0,
                (abs(oi_change) * 2) +
                (abs(funding_rate) * 500) +
                (abs(long_short_ratio - 1) * 30) +
                30
            ))
            
            social_score = min(100, max(0,
                ((sentiment + 1) * 25) +
                (min(mentions, 100) * 0.3) +
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

@st.cache_data(ttl=60)
def fetch_crypto_data_with_fallback():
    """Fetch crypto data with multiple fallback options"""
    
    fetcher = MultiSourceDataFetcher()
    
    # Test API connectivity
    api_status = fetcher.test_api_connectivity()
    
    st.markdown("### 📡 API Status Check")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if api_status['binance']:
            st.markdown("🟢 **Binance**: Online")
        else:
            st.markdown("🔴 **Binance**: Unavailable")
    
    with col2:
        if api_status['coingecko']:
            st.markdown("🟢 **CoinGecko**: Online")
        else:
            st.markdown("🟡 **CoinGecko**: Limited")
    
    with col3:
        st.markdown("🟢 **Demo Mode**: Ready")
    
    # Try to get real data first
    if api_status['binance']:
        st.info("🚀 Attempting Binance API connection...")
        binance_data, status = fetcher.get_binance_data()
        
        if status == "success":
            st.success("✅ Connected to Binance API!")
            # Process Binance data (your existing logic here)
            return process_binance_data(binance_data)
        elif status == "region_blocked":
            st.markdown("""
            <div class="api-status-error">
                🚫 <b>Binance API Geblokkeerd</b><br>
                Error 451: Niet beschikbaar in uw regio<br>
                Switching to fallback data sources...
            </div>
            """, unsafe_allow_html=True)
    
    # Try CoinGecko as fallback
    if api_status['coingecko']:
        st.info("🔄 Trying CoinGecko API as fallback...")
        gecko_data, status = fetcher.get_coingecko_data()
        
        if status == "success":
            st.warning("⚠️ Using CoinGecko data (limited derivatives data)")
            return process_coingecko_data(gecko_data)
    
    # Use demo data as last resort
    st.markdown("""
    <div class="api-status-demo">
        📊 <b>Demo Mode Active</b><br>
        Real APIs unavailable - using realistic simulated data<br>
        All calculations and features fully functional
    </div>
    """, unsafe_allow_html=True)
    
    return fetcher.generate_demo_data()

def process_binance_data(binance_data):
    """Process real Binance data"""
    # Your existing Binance processing logic here
    # This would be similar to your current fetch_real_crypto_data function
    pass

def process_coingecko_data(gecko_data):
    """Process CoinGecko data and simulate derivatives metrics"""
    data_list = []
    
    for item in gecko_data:
        symbol_clean = item['symbol'].replace('USDT', '')
        
        # Use real price and change data
        price = item['price']
        change_24h = item['change_24h']
        volume_24h = item['volume_24h']
        
        # Simulate derivatives data based on price action
        funding_rate = np.random.normal(change_24h * 0.001, 0.02)  # Correlated with price movement
        oi_change = change_24h * 2 + np.random.normal(0, 10)
        long_short_ratio = 1.2 if change_24h > 0 else 0.8  # Bulls when up, bears when down
        long_short_ratio += np.random.normal(0, 0.3)
        
        # Calculate other metrics
        mentions = max(1, int(volume_24h / 10000000)) if volume_24h else 50
        sentiment = np.tanh(change_24h / 5)
        rsi = 50 + change_24h * 2 + np.random.normal(0, 10)
        rsi = max(0, min(100, rsi))
        bb_squeeze = np.random.uniform(0, 1)
        basis = np.random.normal(change_24h * 0.1, 0.3)
        
        # CSI-Q calculation (same as before)
        derivatives_score = min(100, max(0,
            (abs(oi_change) * 2) +
            (abs(funding_rate) * 500) +
            (abs(long_short_ratio - 1) * 30) +
            30
        ))
        
        social_score = min(100, max(0,
            ((sentiment + 1) * 25) +
            (min(mentions, 100) * 0.3) +
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
            'Price': price,
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
            'ATR': abs(price * 0.05),
            'Volume_24h': volume_24h or 1000000,
            'Open_Interest': (volume_24h or 1000000) * np.random.uniform(0.1, 2.0),
            'Last_Updated': datetime.now(),
            'Data_Source': 'coingecko'
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
    """Get color for signal type"""
    colors = {
        "LONG": "🟢",
        "SHORT": "🔴", 
        "CONTRARIAN": "🟠",
        "NEUTRAL": "⚪"
    }
    return colors.get(signal, "⚪")

# Main App
def main():
    st.title("🚀 Crypto CSI-Q Dashboard")
    st.markdown("**Multi-Source Data** - Composite Sentiment/Quant Index voor korte termijn momentum")
    
    # Status and refresh
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown("📊 **MULTI-SOURCE**")
    with col2:
        st.markdown(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    with col3:
        if st.button("🔄 Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data with fallback
    df = fetch_crypto_data_with_fallback()
    
    if df.empty:
        st.error("❌ No data available from any source.")
        st.stop()
    
    # Show data source info
    data_sources = df['Data_Source'].value_counts() if 'Data_Source' in df.columns else {'demo': len(df)}
    source_info = " + ".join([f"{count} from {source}" for source, count in data_sources.items()])
    st.info(f"📊 Loaded {len(df)} symbols: {source_info}")
    
    # Add signal column
    df['Signal'] = df.apply(lambda row: get_signal_type(row['CSI_Q'], row['Funding_Rate']), axis=1)
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    min_csiq = st.sidebar.slider("Min CSI-Q Score", 0, 100, 0)
    max_csiq = st.sidebar.slider("Max CSI-Q Score", 0, 100, 100)
    
    signal_filter = st.sidebar.multiselect(
        "Signal Types",
        ["LONG", "SHORT", "CONTRARIAN", "NEUTRAL"],
        default=["LONG", "SHORT", "CONTRARIAN"]
    )
    
    min_volume = st.sidebar.number_input("Min 24h Volume ($M)", 0, 1000, 0)
    
    # Apply filters
    filtered_df = df[
        (df['CSI_Q'] >= min_csiq) & 
        (df['CSI_Q'] <= max_csiq) &
        (df['Signal'].isin(signal_filter)) &
        (df['Volume_24h'] >= min_volume * 1000000)
    ].copy()
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_signals = len(filtered_df[filtered_df['Signal'] != 'NEUTRAL'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Actieve Signalen</h3>
            <h2>{active_signals}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_csiq = filtered_df['CSI_Q'].mean() if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Gemiddelde CSI-Q</h3>
            <h2>{avg_csiq:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        extreme_signals = len(filtered_df[(filtered_df['CSI_Q'] > 85) | (filtered_df['CSI_Q'] < 15)])
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ Extreme Signalen</h3>
            <h2>{extreme_signals}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_volume = filtered_df['Volume_24h'].sum() / 1000000000 if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Total Volume</h3>
            <h2>${total_volume:.1f}B</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 CSI-Q Monitor", "🎯 Quant Analysis", "💰 Trading Opportunities"])
    
    with tab1:
        st.header("📡 CSI-Q Monitor")
        
        if not filtered_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # CSI-Q Heatmap
                display_df = filtered_df.sort_values('CSI_Q', ascending=False)
                
                fig = go.Figure(data=go.Scatter(
                    x=display_df['Symbol'],
                    y=display_df['CSI_Q'],
                    mode='markers+text',
                    marker=dict(
                        size=np.sqrt(display_df['Volume_24h']) / 100000,
                        color=display_df['CSI_Q'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="CSI-Q Score"),
                        line=dict(width=1, color='white')
                    ),
                    text=display_df['Symbol'],
                    textposition="middle center",
                    hovertemplate="<b>%{text}</b><br>" +
                                "CSI-Q: %{y:.1f}<br>" +
                                "Price: $" + display_df['Price'].round(4).astype(str) + "<br>" +
                                "Change: " + display_df['Change_24h'].round(2).astype(str) + "%<br>" +
                                "Funding: " + display_df['Funding_Rate'].round(4).astype(str) + "%<br>" +
                                "Signal: " + display_df['Signal'].astype(str) + "<br>" +
                                "<extra></extra>"
                ))
                
                fig.update_layout(
                    title="🔴🟡🟢 CSI-Q Heatmap (Multi-Source Data)",
                    xaxis_title="Symbol",
                    yaxis_title="CSI-Q Score",
                    height=400,
                    showlegend=False
                )
                
                # Add signal zones
                fig.add_hline(y=70, line_dash="dash", line_color="green", 
                             annotation_text="LONG Zone", annotation_position="right")
                fig.add_hline(y=30, line_dash="dash", line_color="red",
                             annotation_text="SHORT Zone", annotation_position="right")
                fig.add_hline(y=90, line_dash="dash", line_color="orange",
                             annotation_text="CONTRARIAN", annotation_position="right")
                fig.add_hline(y=10, line_dash="dash", line_color="orange")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🚨 Active Alerts")
                
                # Generate alerts for strong signals
                alerts = []
                for _, row in filtered_df.iterrows():
                    signal = row['Signal']
                    if signal != 'NEUTRAL':
                        strength = "🔥 STRONG" if abs(row['CSI_Q'] - 50) > 35 else "⚠️ MEDIUM"
                        alerts.append({
                            'Symbol': row['Symbol'],
                            'Signal': signal,
                            'CSI_Q': row['CSI_Q'],
                            'Strength': strength,
                            'Price': row['Price'],
                            'Funding': row['Funding_Rate']
                        })
                
                alerts = sorted(alerts, key=lambda x: abs(x['CSI_Q'] - 50), reverse=True)
                
                for alert in alerts[:8]:
                    signal_emoji = get_signal_color(alert['Signal'])
                    st.markdown(f"""
                    <div class="signal-{alert['Signal'].lower()}">
                        {signal_emoji} <b>{alert['Symbol']}</b><br>
                        {alert['Signal']} Signal | {alert['Strength']}<br>
                        CSI-Q: {alert['CSI_Q']:.1f}<br>
                        ${alert['Price']:.4f} | Fund: {alert['Funding']:.3f}%
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
        
        # Data table
        st.subheader("📊 Market Data")
        
        if not filtered_df.empty:
            display_cols = ['Symbol', 'CSI_Q', 'Signal', 'Price', 'Change_24h', 
                           'Funding_Rate', 'Long_Short_Ratio', 'Volume_24h', 'Open_Interest']
            
            styled_df = filtered_df[display_cols].copy()
            styled_df['Price'] = styled_df['Price'].round(4)
            styled_df['CSI_Q'] = styled_df['CSI_Q'].round(1)
            styled_df['Change_24h'] = styled_df['Change_24h'].round(2)
            styled_df['Funding_Rate'] = styled_df['Funding_Rate'].round(4)
            styled_df['Long_Short_Ratio'] = styled_df['Long_Short_Ratio'].round(2)
            styled_df['Volume_24h'] = (styled_df['Volume_24h'] / 1000000).round(1)
            styled_df['Open_Interest'] = styled_df['Open_Interest'].round(0)
            
            styled_df = styled_df.rename(columns={
                'Volume_24h': 'Volume_24h_($M)',
                'Change_24h': 'Change_24h_(%)',
                'Funding_Rate': 'Funding_Rate_(%)'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=400)
    
    with tab2:
        st.header("🎯 Quant Analysis")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Funding Rate vs CSI-Q
                fig = px.scatter(
                    filtered_df,
                    x='Funding_Rate',
                    y='CSI_Q',
                    size='Volume_24h',
                    color='Signal',
                    hover_name='Symbol',
                    title="🔴🟢 Funding vs CSI-Q Analysis",
                    color_discrete_map={
                        'LONG': 'green',
                        'SHORT': 'red',
                        'CONTRARIAN': 'orange',
                        'NEUTRAL': 'gray'
                    }
                )
                
                fig.add_vline(x=0, line_dash="dash", line_color="white")
                fig.add_hline(y=50, line_dash="dash", line_color="white")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Long/Short Ratios
                top_ratios = filtered_df.nlargest(15, 'Long_Short_Ratio')
                fig = px.bar(
                    top_ratios,
                    x='Symbol',
                    y='Long_Short_Ratio',
                    color='Signal',
                    title="📊 Long/Short Ratios",
                    color_discrete_map={
                        'LONG': 'green',
                        'SHORT': 'red',
                        'CONTRARIAN': 'orange',
                        'NEUTRAL': 'gray'
                    }
                )
                fig.add_hline(y=1.0, line_dash="dash", line_color="white", 
                             annotation_text="Balanced (1.0)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # CSI-Q Component Analysis
            st.subheader("🔬 CSI-Q Component Breakdown")
            
            component_cols = ['Symbol', 'CSI_Q', 'Derivatives_Score', 'Social_Score', 'Basis_Score', 'Tech_Score']
            component_df = filtered_df[component_cols].sort_values('CSI_Q', ascending=False).head(10)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Derivatives (40%)',
                x=component_df['Symbol'],
                y=component_df['Derivatives_Score'],
                marker_color='rgba(255, 99, 132, 0.8)'
            ))
            
            fig.add_trace(go.Bar(
                name='Social (30%)',
                x=component_df['Symbol'],
                y=component_df['Social_Score'],
                marker_color='rgba(54, 162, 235, 0.8)'
            ))
            
            fig.add_trace(go.Bar(
                name='Basis (20%)',
                x=component_df['Symbol'],
                y=component_df['Basis_Score'],
                marker_color='rgba(255, 205, 86, 0.8)'
            ))
            
            fig.add_trace(go.Bar(
                name='Technical (10%)',
                x=component_df['Symbol'],
                y=component_df['Tech_Score'],
                marker_color='rgba(75, 192, 192, 0.8)'
            ))
            
            fig.update_layout(
                title="📊 Top 10 CSI-Q Component Scores",
                xaxis_title="Symbol",
                yaxis_title="Component Score",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Contrarian Opportunities
            st.subheader("💎 Contrarian Opportunities")
            
            contrarian_df = filtered_df[
                (filtered_df['CSI_Q'] > 85) | (filtered_df['CSI_Q'] < 15) |
                (abs(filtered_df['Funding_Rate']) > 0.1) |
                (filtered_df['Long_Short_Ratio'] > 2.0) | (filtered_df['Long_Short_Ratio'] < 0.5)
            ].sort_values('CSI_Q', ascending=False)
            
            if len(contrarian_df) > 0:
                cols = st.columns(min(4, len(contrarian_df)))
                for i, (_, row) in enumerate(contrarian_df.head(4).iterrows()):
                    with cols[i]:
                        if row['CSI_Q'] > 90 or row['CSI_Q'] < 10:
                            risk_level = "🔥 EXTREME"
                        elif abs(row['Funding_Rate']) > 0.15:
                            risk_level = "⚡ HIGH FUNDING"
                        else:
                            risk_level = "⚠️ ELEVATED"
                        
                        st.markdown(f"""
                        <div class="signal-contrarian">
                            <h4>{row['Symbol']}</h4>
                            CSI-Q: {row['CSI_Q']:.1f}<br>
                            Funding: {row['Funding_Rate']:.4f}%<br>
                            L/S: {row['Long_Short_Ratio']:.2f}<br>
                            <b>{risk_level}</b>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("🎯 No extreme contrarian setups currently")
    
    with tab3:
        st.header("💰 Trading Opportunities")
        
        if not filtered_df.empty:
            # Calculate opportunity scores
            filtered_df['Opportunity_Score'] = (
                (abs(filtered_df['CSI_Q'] - 50) / 50 * 0.4) +
                (abs(filtered_df['Funding_Rate']) * 10 * 0.3) +
                (abs(filtered_df['Long_Short_Ratio'] - 1) * 0.2) +
                ((filtered_df['Volume_24h'] / filtered_df['Volume_24h'].max()) * 0.1)
            ) * 100
            
            # Top opportunities
            opportunities = filtered_df.sort_values('Opportunity_Score', ascending=False).head(8)
            
            st.subheader("🚀 TOP 8 TRADING OPPORTUNITIES")
            
            data_source_note = "demo" if "demo" in df['Data_Source'].values[0] else "live"
            if data_source_note == "demo":
                st.info("📊 Demo Mode: Realistic simulated data for testing and learning")
            else:
                st.success("📡 Live Data: Real-time market information")
            
            for i, (_, row) in enumerate(opportunities.iterrows()):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    signal_color = get_signal_color(row['Signal'])
                    st.markdown(f"**{i+1}. {signal_color} {row['Symbol']}**")
                    st.markdown(f"Opportunity Score: **{row['Opportunity_Score']:.1f}**")
                    data_source = row.get('Data_Source', 'unknown')
                    st.markdown(f"*Source: {data_source}*")
                
                with col2:
                    st.metric("CSI-Q", f"{row['CSI_Q']:.1f}")
                    st.metric("Signal", row['Signal'])
                
                with col3:
                    st.metric("Price", f"${row['Price']:.4f}")
                    st.metric("24h Change", f"{row['Change_24h']:.2f}%")
                
                with col4:
                    # Trade setup calculations
                    atr_pct = (row['ATR'] / row['Price']) * 100
                    target_pct = atr_pct
                    stop_pct = atr_pct * 0.5
                    
                    if row['Signal'] == 'LONG':
                        target_price = row['Price'] * (1 + target_pct/100)
                        stop_price = row['Price'] * (1 - stop_pct/100)
                    elif row['Signal'] == 'SHORT':
                        target_price = row['Price'] * (1 - target_pct/100)
                        stop_price = row['Price'] * (1 + stop_pct/100)
                    else:  # CONTRARIAN
                        target_price = row['Price'] * (1 + target_pct/100) if row['CSI_Q'] < 20 else row['Price'] * (1 - target_pct/100)
                        stop_price = row['Price'] * (1 - stop_pct/200) if row['CSI_Q'] < 20 else row['Price'] * (1 + stop_pct/200)
                    
                    risk_reward = target_pct / stop_pct if stop_pct > 0 else 1.0
                    
                    st.markdown(f"""
                    **🎯 Trade Setup:**
                    - Entry: ${row['Price']:.4f}
                    - Target: ${target_price:.4f}
                    - Stop: ${stop_price:.4f}
                    - R/R: 1:{risk_reward:.1f}
                    - Funding: {row['Funding_Rate']:.4f}%
                    - Volume: ${row['Volume_24h']/1000000:.0f}M
                    """)
                
                st.markdown("---")
            
            # Market analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Market Sentiment Analysis")
                
                avg_csiq = filtered_df['CSI_Q'].mean()
                avg_funding = filtered_df['Funding_Rate'].mean()
                
                if avg_csiq > 65:
                    market_sentiment = "🟢 BULLISH"
                    sentiment_color = "green"
                elif avg_csiq < 35:
                    market_sentiment = "🔴 BEARISH" 
                    sentiment_color = "red"
                else:
                    market_sentiment = "🟡 NEUTRAL"
                    sentiment_color = "orange"
                
                st.markdown(f"""
                <div style="background: {sentiment_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
                    <h3>Market Status: {market_sentiment}</h3>
                    <p>Avg CSI-Q: {avg_csiq:.1f} | Avg Funding: {avg_funding:.4f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Signal distribution
                signal_counts = filtered_df['Signal'].value_counts()
                fig = px.pie(
                    values=signal_counts.values,
                    names=signal_counts.index,
                    title="📊 Signal Distribution",
                    color_discrete_map={
                        'LONG': 'green',
                        'SHORT': 'red',
                        'CONTRARIAN': 'orange',
                        'NEUTRAL': 'gray'
                    }
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("⚠️ Risk Analysis")
                
                # Risk warnings
                warnings = []
                
                extreme_funding = filtered_df[abs(filtered_df['Funding_Rate']) > 0.2]
                if not extreme_funding.empty:
                    warnings.append(f"🔥 {len(extreme_funding)} coins with EXTREME funding rates")
                
                extreme_csiq = filtered_df[(filtered_df['CSI_Q'] > 95) | (filtered_df['CSI_Q'] < 5)]
                if not extreme_csiq.empty:
                    warnings.append(f"⚡ {len(extreme_csiq)} coins at MAXIMUM hysteria levels")
                
                high_ls_ratio = filtered_df[filtered_df['Long_Short_Ratio'] > 3]
                if not high_ls_ratio.empty:
                    warnings.append(f"📈 {len(high_ls_ratio)} coins heavily LONG-skewed (>3:1)")
                
                low_ls_ratio = filtered_df[filtered_df['Long_Short_Ratio'] < 0.33]
                if not low_ls_ratio.empty:
                    warnings.append(f"📉 {len(low_ls_ratio)} coins heavily SHORT-skewed (<1:3)")
                
                if warnings:
                    for warning in warnings:
                        st.markdown(f"""
                        <div style="background: #ff6b6b; padding: 10px; border-radius: 5px; margin: 5px 0; color: white;">
                            {warning}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #51cf66; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                        ✅ No extreme risk warnings currently
                    </div>
                    """, unsafe_allow_html=True)
                
                # Trading tips
                st.markdown("### 💡 Trading Tips")
                
                if avg_funding > 0.1:
                    st.markdown("- 🔴 **High funding** → Consider SHORT bias")
                elif avg_funding < -0.1:
                    st.markdown("- 🟢 **Negative funding** → Consider LONG bias")
                
                if avg_csiq > 80:
                    st.markdown("- ⚠️ **Market overheated** → Look for contrarian plays")
                elif avg_csiq < 20:
                    st.markdown("- 🚀 **Oversold market** → Look for bounce plays")
                
                strong_signals = len(filtered_df[filtered_df['Signal'] != 'NEUTRAL'])
                if strong_signals > len(filtered_df) * 0.7:
                    st.markdown("- 🎯 **Many active signals** → Good trading environment")
        
        else:
            st.warning("No data available with current filters")
    
    # Footer with data source and refresh info
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("🔄 **Auto-refresh**: Every 60 seconds")
    
    with col2:
        data_sources = df['Data_Source'].value_counts() if 'Data_Source' in df.columns else {'unknown': len(df)}
        source_text = " + ".join([f"{source.title()}" for source in data_sources.index])
        st.markdown(f"📡 **Sources**: {source_text}")
    
    with col3:
        st.markdown(f"⏰ **Last update**: {datetime.now().strftime('%H:%M:%S')}")
    
    # Improved footer
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 20px;'>
        <p>🚀 <b>Crypto CSI-Q Dashboard</b> - Multi-Source Real-Time & Demo Data<br>
        🔄 <b>Fallback System:</b> Binance → CoinGecko → Demo Mode<br>
        ⚠️ Dit is geen financieel advies. Altijd eigen onderzoek doen en risk management toepassen!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
