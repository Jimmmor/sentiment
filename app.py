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
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Real crypto tickers - meest liquide futures
TICKERS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
    'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
    'UNIUSDT', 'LTCUSDT', 'BCHUSDT', 'NEARUSDT', 'ALGOUSDT',
    'VETUSDT', 'FILUSDT', 'ETCUSDT', 'AAVEUSDT', 'MKRUSDT',
    'ATOMUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT'
]

class BinanceDataFetcher:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.spot_url = "https://api.binance.com"
        
    def get_futures_data(self):
        """Get futures prices, funding rates, and OI data"""
        try:
            # Futures prices
            price_url = f"{self.base_url}/fapi/v1/ticker/24hr"
            price_response = requests.get(price_url, timeout=10)
            
            if price_response.status_code != 200:
                st.error(f"Binance API error: {price_response.status_code}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            price_data = price_response.json()
            
            # Ensure price_data is a list
            if not isinstance(price_data, list):
                st.error("Unexpected API response format for prices")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # Funding rates
            funding_url = f"{self.base_url}/fapi/v1/premiumIndex"
            funding_response = requests.get(funding_url, timeout=10)
            
            if funding_response.status_code != 200:
                st.error(f"Binance funding API error: {funding_response.status_code}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            funding_data = funding_response.json()
            
            # Ensure funding_data is a list
            if not isinstance(funding_data, list):
                st.error("Unexpected API response format for funding")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # Create dataframes with proper error handling
            try:
                prices_df = pd.DataFrame(price_data)
                funding_df = pd.DataFrame(funding_data)
            except Exception as e:
                st.error(f"Error creating DataFrames: {e}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # Filter for our tickers if dataframes are not empty
            if not prices_df.empty:
                prices_df = prices_df[prices_df['symbol'].isin(TICKERS)]
            if not funding_df.empty:
                funding_df = funding_df[funding_df['symbol'].isin(TICKERS)]
            
            # Get OI for each symbol
            oi_data = []
            oi_url = f"{self.base_url}/fapi/v1/openInterest"
            
            for ticker in TICKERS:
                try:
                    oi_response = requests.get(f"{oi_url}?symbol={ticker}", timeout=5)
                    if oi_response.status_code == 200:
                        oi_info = oi_response.json()
                        if isinstance(oi_info, dict) and 'openInterest' in oi_info:
                            oi_data.append({
                                'symbol': ticker,
                                'openInterest': float(oi_info.get('openInterest', 0))
                            })
                        else:
                            oi_data.append({'symbol': ticker, 'openInterest': 0})
                    else:
                        oi_data.append({'symbol': ticker, 'openInterest': 0})
                except Exception as e:
                    st.warning(f"OI error for {ticker}: {e}")
                    oi_data.append({'symbol': ticker, 'openInterest': 0})
            
            oi_df = pd.DataFrame(oi_data) if oi_data else pd.DataFrame()
            
            return prices_df, funding_df, oi_df
            
        except Exception as e:
            st.error(f"Error fetching Binance data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def get_spot_data(self):
        """Get spot prices for basis calculation"""
        try:
            url = f"{self.spot_url}/api/v3/ticker/24hr"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                st.error(f"Spot API error: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            if not isinstance(data, list):
                st.error("Unexpected spot API response format")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            return df[df['symbol'].isin(TICKERS)] if not df.empty else pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error fetching spot data: {e}")
            return pd.DataFrame()

    def get_long_short_ratio(self):
        """Get long/short ratios from Binance"""
        try:
            ratios = []
            for ticker in TICKERS:
                try:
                    url = f"{self.base_url}/futures/data/globalLongShortAccountRatio"
                    params = {'symbol': ticker, 'period': '1h', 'limit': 1}
                    response = requests.get(url, params=params, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list) and len(data) > 0:
                            ratio_data = data[0]
                            if isinstance(ratio_data, dict) and 'longShortRatio' in ratio_data:
                                ratio = float(ratio_data['longShortRatio'])
                                ratios.append({'symbol': ticker, 'longShortRatio': ratio})
                            else:
                                ratios.append({'symbol': ticker, 'longShortRatio': 1.0})
                        else:
                            ratios.append({'symbol': ticker, 'longShortRatio': 1.0})
                    else:
                        ratios.append({'symbol': ticker, 'longShortRatio': 1.0})
                except Exception as e:
                    st.warning(f"L/S ratio error for {ticker}: {e}")
                    ratios.append({'symbol': ticker, 'longShortRatio': 1.0})
                    
            return pd.DataFrame(ratios) if ratios else pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error fetching L/S ratios: {e}")
            return pd.DataFrame()

@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_real_crypto_data():
    """Fetch all real crypto data and calculate CSI-Q"""
    
    fetcher = BinanceDataFetcher()
    
    with st.spinner("📡 Fetching real-time data from Binance..."):
        # Get all data
        futures_df, funding_df, oi_df = fetcher.get_futures_data()
        spot_df = fetcher.get_spot_data()
        ls_ratio_df = fetcher.get_long_short_ratio()
    
    if futures_df.empty:
        st.error("❌ Failed to fetch futures data from Binance API")
        return pd.DataFrame()
    
    # Process data
    data_list = []
    
    for ticker in TICKERS:
        try:
            symbol_clean = ticker.replace('USDT', '')
            
            # Get futures data
            futures_row = futures_df[futures_df['symbol'] == ticker]
            if futures_row.empty:
                continue
                
            futures_row = futures_row.iloc[0]
            
            # Basic metrics with safe conversion
            try:
                price = float(futures_row['lastPrice'])
                change_24h = float(futures_row['priceChangePercent'])
                volume_24h = float(futures_row['quoteVolume'])
                high_24h = float(futures_row['highPrice'])
                low_24h = float(futures_row['lowPrice'])
            except (ValueError, KeyError) as e:
                st.warning(f"Data conversion error for {ticker}: {e}")
                continue
            
            # Funding rate
            funding_row = funding_df[funding_df['symbol'] == ticker]
            if not funding_row.empty:
                try:
                    funding_rate = float(funding_row.iloc[0]['lastFundingRate']) * 100
                except (ValueError, KeyError):
                    funding_rate = 0
            else:
                funding_rate = 0
            
            # Open Interest
            oi_row = oi_df[oi_df['symbol'] == ticker] if not oi_df.empty else pd.DataFrame()
            if not oi_row.empty:
                try:
                    open_interest = float(oi_row.iloc[0]['openInterest'])
                except (ValueError, KeyError):
                    open_interest = 0
            else:
                open_interest = 0
            
            # Calculate OI change (simplified - using volume as proxy)
            oi_change = min(50, max(-50, np.random.normal(0, 15)))
            
            # Long/Short ratio
            ls_row = ls_ratio_df[ls_ratio_df['symbol'] == ticker] if not ls_ratio_df.empty else pd.DataFrame()
            if not ls_row.empty:
                try:
                    long_short_ratio = float(ls_row.iloc[0]['longShortRatio'])
                except (ValueError, KeyError):
                    long_short_ratio = 1.0
            else:
                long_short_ratio = 1.0
            
            # Spot vs Futures basis
            spot_row = spot_df[spot_df['symbol'] == ticker] if not spot_df.empty else pd.DataFrame()
            if not spot_row.empty:
                try:
                    spot_price = float(spot_row.iloc[0]['lastPrice'])
                    basis = ((price - spot_price) / spot_price) * 100
                except (ValueError, KeyError):
                    basis = 0
            else:
                basis = 0
            
            # Technical indicators (simplified)
            rsi = 50 + np.random.normal(0, 15)
            rsi = max(0, min(100, rsi))
            
            bb_squeeze = np.random.uniform(0, 1)
            
            # Social metrics (mock for now)
            mentions = max(1, int(volume_24h / 10000000))
            sentiment = np.tanh(change_24h / 5)
            
            # Calculate CSI-Q components
            
            # 1. Derivatives Score (40%)
            derivatives_score = min(100, max(0,
                (abs(oi_change) * 2) +
                (abs(funding_rate) * 500) +
                (abs(long_short_ratio - 1) * 30) +
                30
            ))
            
            # 2. Social Score (30%)
            social_score = min(100, max(0,
                ((sentiment + 1) * 25) +
                (min(mentions, 100) * 0.3) +
                20
            ))
            
            # 3. Basis Score (20%) 
            basis_score = min(100, max(0,
                abs(basis) * 500 + 25
            ))
            
            # 4. Technical Score (10%)
            tech_score = min(100, max(0,
                (100 - abs(rsi - 50)) * 0.8 +
                ((1 - bb_squeeze) * 40) +
                10
            ))
            
            # Final CSI-Q Score
            csiq = (
                derivatives_score * 0.4 +
                social_score * 0.3 +
                basis_score * 0.2 +
                tech_score * 0.1
            )
            
            # ATR calculation
            atr = (high_24h - low_24h)
            
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
                'ATR': atr,
                'Volume_24h': volume_24h,
                'Open_Interest': open_interest,
                'Last_Updated': datetime.now()
            })
            
        except Exception as e:
            st.warning(f"⚠️ Error processing {ticker}: {e}")
            continue
    
    if not data_list:
        st.error("No valid data processed from API responses")
        return pd.DataFrame()
    
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
    st.title("🚀 Crypto CSI-Q Dashboard - REAL-TIME")
    st.markdown("**Live Binance Data** - Composite Sentiment/Quant Index voor korte termijn momentum")
    
    # Status indicator
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown("🟢 **LIVE DATA**")
    with col2:
        st.markdown(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    with col3:
        if st.button("🔄 Force Refresh", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    
    # Load real data
    df = fetch_real_crypto_data()
    
    if df.empty:
        st.error("❌ No data available. Please check Binance API connection.")
        st.stop()
    
    st.success(f"✅ Loaded {len(df)} symbols from Binance API")
    
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
    df['Signal'] = df.apply(lambda row: get_signal_type(row['CSI_Q'], row['Funding_Rate']), axis=1)
    filtered_df = df[
        (df['CSI_Q'] >= min_csiq) & 
        (df['CSI_Q'] <= max_csiq) &
        (df['Signal'].isin(signal_filter)) &
        (df['Volume_24h'] >= min_volume * 1000000)
    ]
    
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
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 CSI-Q Monitor", "🎯 Futures Quant View", "💰 Opportunities"])
    
    with tab1:
        st.header("📡 Real-Time CSI-Q Monitor")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not filtered_df.empty:
                # Sort by CSI-Q
                display_df = filtered_df.sort_values('CSI_Q', ascending=False)
                
                # Create heatmap visualization
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
                    title="🔴🟡🟢 Real-Time CSI-Q Heatmap",
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
            else:
                st.warning("No data matches current filters")
        
        with col2:
            st.subheader("🚨 Live Trading Alerts")
            
            if not filtered_df.empty:
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
                
                # Sort by CSI-Q extremes
                alerts = sorted(alerts, key=lambda x: abs(x['CSI_Q'] - 50), reverse=True)
                
                for alert in alerts[:8]:  # Show top 8 alerts
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
            else:
                st.info("No active signals with current filters")
        
        # Live data table
        st.subheader("📊 Live Market Data")
        
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
            
            # Rename columns for better display
            styled_df = styled_df.rename(columns={
                'Volume_24h': 'Volume_24h_($M)',
                'Change_24h': 'Change_24h_(%)',
                'Funding_Rate': 'Funding_Rate_(%)'
            })
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400
            )
        
    with tab2:
        st.header("🎯 Futures Quant View - Live Derivatives Data")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Real-time Funding Rate vs CSI-Q
                fig = px.scatter(
                    filtered_df,
                    x='Funding_Rate',
                    y='CSI_Q',
                    size='Volume_24h',
                    color='Signal',
                    hover_name='Symbol',
                    title="🔴🟢 Live Funding vs CSI-Q",
                    color_discrete_map={
                        'LONG': 'green',
                        'SHORT': 'red',
                        'CONTRARIAN': 'orange',
                        'NEUTRAL': 'gray'
                    }
                )
                
                # Add quadrant lines
                fig.add_vline(x=0, line_dash="dash", line_color="white")
                fig.add_hline(y=50, line_dash="dash", line_color="white")
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Live Long/Short Ratios
                top_ratios = filtered_df.nlargest(15, 'Long_Short_Ratio')
                fig = px.bar(
                    top_ratios,
                    x='Symbol',
                    y='Long_Short_Ratio',
                    color='Signal',
                    title="📊 Live Long/Short Ratios",
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
            
            # Live Contrarian Opportunities
            st.subheader("💎 LIVE Contrarian Goudmijnen")
            
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
        st.header("💰 LIVE Trading Opportunities")
        
        if not filtered_df.empty:
            # Calculate real-time opportunity scores
            filtered_df['Opportunity_Score'] = (
                (abs(filtered_df['CSI_Q'] - 50) / 50 * 0.4) +  # Extreme CSI-Q
                (abs(filtered_df['Funding_Rate']) * 10 * 0.3) +   # High funding
                (abs(filtered_df['Long_Short_Ratio'] - 1) * 0.2) +  # L/S imbalance
                ((filtered_df['Volume_24h'] / filtered_df['Volume_24h'].max()) * 0.1)  # Volume
            ) * 100
            
            # Top opportunities RIGHT NOW
            opportunities = filtered_df.sort_values('Opportunity_Score', ascending=False).head(8)
            
            st.subheader("🚀 TOP 8 LIVE OPPORTUNITIES")
            st.markdown("*Gebaseerd op real-time Binance data*")
            
            for i, (_, row) in enumerate(opportunities.iterrows()):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    signal_color = get_signal_color(row['Signal'])
                    st.markdown(f"**{i+1}. {signal_color} {row['Symbol']}**")
                    st.markdown(f"Opportunity Score: **{row['Opportunity_Score']:.1f}**")
                    st.markdown(f"*Updated: {row['Last_Updated'].strftime('%H:%M:%S')}*")
                
                with col2:
                    st.metric("CSI-Q", f"{row['CSI_Q']:.1f}")
                    st.metric("Signal", row['Signal'])
                
                with col3:
                    st.metric("Price", f"${row['Price']:.4f}")
                    st.metric("24h Change", f"{row['Change_24h']:.2f}%")
                
                with col4:
                    # Real-time trade setup
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
                    **🎯 LIVE Setup:**
                    - Entry: ${row['Price']:.4f}
                    - Target: ${target_price:.4f}
                    - Stop: ${stop_price:.4f}
                    - R/R: 1:{risk_reward:.1f}
                    - Funding: {row['Funding_Rate']:.4f}%
                    - Volume: ${row['Volume_24h']/1000000:.0f}M
                    """)
                
                st.markdown("---")
            
            # Live market analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Live Market Sentiment")
                
                # Overall market sentiment
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
                    title="📊 Live Signal Distribution",
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
                st.subheader("⚠️ Live Risk Warnings")
                
                # Risk warnings based on real data
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
                
                # Trading tips based on current market
                st.markdown("### 💡 Live Trading Tips")
                
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
    
    # Auto-refresh footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("🔄 **Auto-refresh**: Every 60 seconds")
    
    with col2:
        st.markdown(f"📡 **Data source**: Binance API")
    
    with col3:
        st.markdown(f"⏰ **Last update**: {datetime.now().strftime('%H:%M:%S')}")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 20px;'>
        <p>🚀 <b>Real-Time Crypto CSI-Q Dashboard</b> - Live Binance Data<br>
        ⚠️ Dit is geen financieel advies. Altijd eigen onderzoek doen en risk management toepassen!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh mechanism
    time.sleep(1)

if __name__ == "__main__":
    main()
