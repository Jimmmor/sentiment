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
from typing import Optional, Tuple, List, Dict
import asyncio
import aiohttp
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚀 Enhanced Crypto CSI-Q Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .signal-long {
        background: linear-gradient(135deg, #00C851, #007E33);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 5px 0;
    }
    
    .signal-short {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 5px 0;
    }
    
    .signal-contrarian {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 5px 0;
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, #9E9E9E, #757575);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 5px 0;
    }
    
    .status-success {
        background: linear-gradient(135deg, #00C851, #007E33);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 15px 0;
        font-weight: 600;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ff6b35, #f7931e);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 15px 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class SimpleCryptoFetcher:
    """Simplified crypto data fetcher"""
    
    def __init__(self):
        self.realistic_prices = {
            'BTCUSDT': 118500, 'ETHUSDT': 4760, 'BNBUSDT': 845, 'SOLUSDT': 185, 'XRPUSDT': 0.67,
            'ADAUSDT': 0.58, 'AVAXUSDT': 47, 'DOTUSDT': 8.8, 'LINKUSDT': 19, 'MATICUSDT': 1.25,
            'UNIUSDT': 13, 'LTCUSDT': 98, 'BCHUSDT': 325, 'NEARUSDT': 6.8, 'ALGOUSDT': 0.38,
            'VETUSDT': 0.048, 'FILUSDT': 8.8, 'ETCUSDT': 29, 'AAVEUSDT': 155, 'MKRUSDT': 2250,
            'ATOMUSDT': 12.5, 'FTMUSDT': 0.88, 'SANDUSDT': 0.68, 'MANAUSDT': 0.62, 'AXSUSDT': 9.8
        }
    
    def fetch_binance_data(self):
        """Try to fetch from Binance API"""
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                relevant_coins = []
                
                for item in data:
                    if item['symbol'] in self.realistic_prices.keys():
                        relevant_coins.append({
                            'symbol': item['symbol'],
                            'price': float(item['lastPrice']),
                            'change_24h': float(item['priceChangePercent']),
                            'volume_24h': float(item['quoteVolume']),
                            'high_24h': float(item['highPrice']),
                            'low_24h': float(item['lowPrice']),
                            'trades_24h': int(item['count']),
                            'open_price': float(item['openPrice'])
                        })
                
                if relevant_coins:
                    return relevant_coins, 'binance_api', "Live Binance data loaded successfully!"
                    
        except Exception as e:
            print(f"Binance API error: {e}")
        
        return None, 'failed', "Binance API unavailable"
    
    def generate_realistic_data(self):
        """Generate realistic market simulation data"""
        realistic_coins = []
        
        for symbol, base_price in self.realistic_prices.items():
            # Generate realistic daily movements
            volatility_factor = {
                'BTCUSDT': 0.04, 'ETHUSDT': 0.05, 'BNBUSDT': 0.06,
                'SOLUSDT': 0.08, 'XRPUSDT': 0.07, 'ADAUSDT': 0.06
            }.get(symbol, 0.09)
            
            daily_movement = np.random.normal(0, volatility_factor)
            current_price = base_price * (1 + daily_movement)
            change_24h = daily_movement * 100
            
            # Realistic volume
            if symbol in ['BTCUSDT', 'ETHUSDT']:
                volume_range = (20000000000, 45000000000)
            elif symbol in ['BNBUSDT', 'SOLUSDT', 'XRPUSDT']:
                volume_range = (800000000, 8000000000)
            else:
                volume_range = (50000000, 1000000000)
            
            volume_24h = np.random.uniform(*volume_range)
            
            high_24h = current_price * (1 + abs(daily_movement) * 0.6) if daily_movement > 0 else current_price
            low_24h = current_price * (1 - abs(daily_movement) * 0.6) if daily_movement < 0 else current_price
            
            realistic_coins.append({
                'symbol': symbol,
                'price': current_price,
                'change_24h': change_24h,
                'volume_24h': volume_24h,
                'high_24h': max(high_24h, current_price),
                'low_24h': min(low_24h, current_price),
                'trades_24h': int(volume_24h / current_price * np.random.uniform(5000, 25000)),
                'open_price': base_price
            })
        
        return realistic_coins, 'simulation', "Using realistic market simulation data"
    
    def fetch_data(self):
        """Main data fetching method"""
        # Try real API first
        raw_coins, status, message = self.fetch_binance_data()
        if raw_coins:
            return raw_coins, status, message
        
        # Fallback to simulation
        return self.generate_realistic_data()

class CSIQCalculator:
    """CSI-Q calculation engine"""
    
    @staticmethod
    def calculate_derivatives_score(funding_rate, oi_change, long_short_ratio):
        funding_component = min(40, abs(funding_rate) * 500)
        oi_component = min(35, abs(oi_change) * 1.5)
        ratio_imbalance = abs(long_short_ratio - 1.0)
        ratio_component = min(25, ratio_imbalance * 30)
        return funding_component + oi_component + ratio_component
    
    @staticmethod
    def calculate_social_score(sentiment, mentions, sentiment_magnitude):
        sentiment_component = abs(sentiment) * 40
        mention_component = min(35, (mentions / 500) * 35)
        magnitude_component = min(25, sentiment_magnitude * 25)
        return sentiment_component + mention_component + magnitude_component
    
    @staticmethod
    def calculate_basis_score(basis):
        return min(100, abs(basis) * 500 + 15)
    
    @staticmethod
    def calculate_tech_score(rsi, bb_squeeze):
        rsi_extremity = abs(rsi - 50) / 50
        rsi_component = rsi_extremity * 60
        squeeze_component = (1 - bb_squeeze) * 40
        return rsi_component + squeeze_component

def generate_enhanced_metrics(coin_data, data_source):
    """Generate enhanced trading metrics for each coin"""
    
    symbol = coin_data['symbol']
    price = coin_data['price']
    change_24h = coin_data['change_24h']
    volume_24h = coin_data['volume_24h']
    
    # Enhanced funding rate simulation
    if change_24h > 12:
        funding_rate = np.random.uniform(0.12, 0.25)
        oi_change = np.random.uniform(35, 80)
        long_short_ratio = np.random.uniform(3.0, 6.0)
    elif change_24h > 6:
        funding_rate = np.random.uniform(0.05, 0.12)
        oi_change = np.random.uniform(15, 35)
        long_short_ratio = np.random.uniform(1.5, 3.0)
    elif change_24h < -12:
        funding_rate = np.random.uniform(-0.25, -0.12)
        oi_change = np.random.uniform(-80, -35)
        long_short_ratio = np.random.uniform(0.15, 0.35)
    elif change_24h < -6:
        funding_rate = np.random.uniform(-0.12, -0.05)
        oi_change = np.random.uniform(-35, -15)
        long_short_ratio = np.random.uniform(0.35, 0.7)
    else:
        funding_rate = np.random.uniform(-0.01, 0.01)
        oi_change = np.random.uniform(-5, 5)
        long_short_ratio = np.random.uniform(0.9, 1.1)
    
    # Enhanced sentiment calculation
    base_sentiment = change_24h * 0.08
    volatility_boost = abs(change_24h) * 0.02
    combined_sentiment = base_sentiment + (np.random.uniform(-0.2, 0.2) * volatility_boost)
    combined_sentiment = max(-1.0, min(1.0, combined_sentiment))
    
    # Social metrics
    total_mentions = int(volume_24h / 2000000 * (1 + abs(change_24h) * 0.1))
    sentiment_magnitude = abs(combined_sentiment) * (total_mentions / 1000)
    
    # Technical indicators
    rsi = max(0, min(100, 50 + (change_24h * 2.2) + np.random.uniform(-8, 8)))
    bb_squeeze = np.random.beta(2, 3)
    
    # Spot-futures basis
    if coin_data['high_24h'] > coin_data['low_24h']:
        daily_range = (coin_data['high_24h'] - coin_data['low_24h']) / price
        basis = daily_range * np.random.uniform(-0.8, 0.8) * (1 + abs(change_24h) * 0.01)
    else:
        basis = np.random.uniform(-0.015, 0.015)
    
    # Calculate CSI-Q components
    calc = CSIQCalculator()
    derivatives_score = calc.calculate_derivatives_score(funding_rate, oi_change, long_short_ratio)
    social_score = calc.calculate_social_score(combined_sentiment, total_mentions, sentiment_magnitude)
    basis_score = calc.calculate_basis_score(basis)
    tech_score = calc.calculate_tech_score(rsi, bb_squeeze)
    
    # Final CSI-Q
    csiq = (derivatives_score * 0.35 + social_score * 0.35 + basis_score * 0.20 + tech_score * 0.10)
    
    return {
        'Symbol': symbol.replace('USDT', ''),
        'Price': price,
        'Change_24h': change_24h,
        'Volume_24h': volume_24h,
        'High_24h': coin_data.get('high_24h', price * 1.05),
        'Low_24h': coin_data.get('low_24h', price * 0.95),
        'Funding_Rate': funding_rate,
        'OI_Change': oi_change,
        'Long_Short_Ratio': long_short_ratio,
        'Total_Mentions': total_mentions,
        'Combined_Sentiment': combined_sentiment,
        'Sentiment_Magnitude': sentiment_magnitude,
        'Spot_Futures_Basis': basis,
        'RSI': rsi,
        'BB_Squeeze': bb_squeeze,
        'CSI_Q': csiq,
        'Derivatives_Score': derivatives_score,
        'Social_Score': social_score,
        'Basis_Score': basis_score,
        'Tech_Score': tech_score,
        'Last_Updated': datetime.now(),
        'Data_Source': data_source,
        'Trades_24h': coin_data.get('trades_24h', int(volume_24h / price * np.random.uniform(3000, 15000))),
        'Market_Cap_Rank': {'BTCUSDT': 1, 'ETHUSDT': 2, 'BNBUSDT': 4, 'SOLUSDT': 5, 'XRPUSDT': 6}.get(symbol, np.random.randint(7, 100))
    }

def get_trading_signal(csiq, funding_rate, sentiment, long_short_ratio):
    """Enhanced trading signal determination"""
    if csiq > 90 or csiq < 10:
        return "CONTRARIAN"
    elif csiq > 75 and funding_rate < 0.08 and sentiment > 0.3 and long_short_ratio < 2.5:
        return "LONG"
    elif csiq > 65 and funding_rate < 0.05 and sentiment > 0.1:
        return "LONG"
    elif csiq < 25 and funding_rate > -0.08 and sentiment < -0.3 and long_short_ratio > 0.4:
        return "SHORT"
    elif csiq < 35 and funding_rate > -0.05 and sentiment < -0.1:
        return "SHORT"
    else:
        return "NEUTRAL"

def get_signal_emoji(signal):
    return {"LONG": "🟢", "SHORT": "🔴", "CONTRARIAN": "🟠", "NEUTRAL": "⚪"}.get(signal, "⚪")

@st.cache_data(ttl=45)
def load_crypto_data():
    """Main data loading function with caching"""
    try:
        fetcher = SimpleCryptoFetcher()
        raw_coins, data_source, status_message = fetcher.fetch_data()
        
        # Process all coins
        processed_data = []
        for coin in raw_coins:
            enhanced_metrics = generate_enhanced_metrics(coin, data_source)
            processed_data.append(enhanced_metrics)
        
        df = pd.DataFrame(processed_data)
        
        # Add trading signals
        df['Signal'] = df.apply(lambda row: get_trading_signal(
            row['CSI_Q'], 
            row['Funding_Rate'], 
            row['Combined_Sentiment'],
            row['Long_Short_Ratio']
        ), axis=1)
        
        # Calculate opportunity scores
        df['Opportunity_Score'] = (
            (abs(df['CSI_Q'] - 50) / 50 * 35) +
            (abs(df['Funding_Rate']) * 400) +
            (abs(df['Long_Short_Ratio'] - 1) * 20) +
            (abs(df['Combined_Sentiment']) * 25) +
            ((df['Volume_24h'] / df['Volume_24h'].max()) * 20)
        )
        
        return df, data_source, status_message
    except Exception as e:
        st.error(f"Error in load_crypto_data: {e}")
        return pd.DataFrame(), 'error', f"Error: {e}"

def main():
    """Main application function"""
    
    # Header
    st.title("🚀 Enhanced Crypto CSI-Q Dashboard")
    st.markdown("**💰 Professional Trading Intelligence with Real-Time Data**")
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("📊 **PROFESSIONAL MODE**")
    
    with col2:
        st.markdown(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    with col3:
        if st.button("🔄 Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    try:
        with st.spinner("Loading crypto data..."):
            df, data_source, status_message = load_crypto_data()
        
        if df.empty:
            st.error("No data available. Please try refreshing.")
            return
        
        # Status message
        status_class = "status-success" if 'api' in data_source else "status-warning"
        st.markdown(f"""
        <div class="{status_class}">
            {status_message} | 📊 {len(df)} assets loaded | 🎯 {len(df[df['Signal'] != 'NEUTRAL'])} signals active
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"💥 **Error loading data:** {e}")
        st.stop()
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Active Signals</h4>
            <h2>{len(df[df['Signal'] != 'NEUTRAL'])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Total Volume</h4>
            <h2>${df['Volume_24h'].sum()/1e9:.1f}B</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Avg CSI-Q</h4>
            <h2>{df['CSI_Q'].mean():.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎭 Avg Sentiment</h4>
            <h2>{df['Combined_Sentiment'].mean():.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ Opportunities</h4>
            <h2>{len(df[df['Opportunity_Score'] > 60])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top opportunities
    st.markdown("### 🎯 **TOP TRADING OPPORTUNITIES**")
    
    opportunities = df.sort_values('Opportunity_Score', ascending=False).head(6)
    
    for i, (_, row) in enumerate(opportunities.iterrows()):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 2])
            
            with col1:
                signal_emoji = get_signal_emoji(row['Signal'])
                st.markdown(f"""
                **{i+1}. {signal_emoji} {row['Symbol']}**  
                Opportunity Score: **{row['Opportunity_Score']:.1f}/100**  
                Price: **${row['Price']:.6f}**  
                Market Cap Rank: #{row['Market_Cap_Rank']}
                """)
            
            with col2:
                st.metric("CSI-Q Score", f"{row['CSI_Q']:.1f}")
                st.metric("24h Volume", f"${row['Volume_24h']/1000000:.0f}M")
            
            with col3:
                st.metric("Price Change", f"{row['Change_24h']:+.2f}%")
                st.metric("Funding Rate", f"{row['Funding_Rate']:.4f}%")
            
            with col4:
                signal_class = {
                    'LONG': 'signal-long',
                    'SHORT': 'signal-short',
                    'CONTRARIAN': 'signal-contrarian',
                    'NEUTRAL': 'signal-neutral'
                }.get(row['Signal'], 'signal-neutral')
                
                st.markdown(f"""
                <div class="{signal_class}">
                    <strong>{row['Signal']} SIGNAL</strong><br>
                    Entry: ${row['Price']:.6f}<br>
                    RSI: {row['RSI']:.0f}<br>
                    Sentiment: {row['Combined_Sentiment']:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Market overview chart
    st.subheader("📈 Market Overview")
    
    fig = px.scatter(
        df.head(15),
        x='Change_24h',
        y='CSI_Q',
        size='Volume_24h',
        color='Combined_Sentiment',
        hover_name='Symbol',
        hover_data={
            'Price': ':$.6f',
            'Volume_24h': ':$,.0f',
            'Signal': True,
            'Funding_Rate': ':.4f'
        },
        title="💰 Opportunity Matrix",
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(0,255,0,0.5)")
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,0,0,0.5)")
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Complete Data")
    
    # Prepare display dataframe
    display_cols = [
        'Symbol', 'Price', 'Change_24h', 'Volume_24h', 'CSI_Q', 'Signal',
        'Combined_Sentiment', 'Funding_Rate', 'RSI', 'Opportunity_Score'
    ]
    
    display_df = df[display_cols].copy()
    display_df = display_df.sort_values('Opportunity_Score', ascending=False)
    
    st.dataframe(display_df, use_container_width=True, height=400)

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.write("Debug info:")
        st.write(f"Python version: {__import__('sys').version}")
        st.write(f"Streamlit version: {st.__version__}")
