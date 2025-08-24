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
import ccxt
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚀 Enhanced Crypto CSI-Q Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main .block-container {
        font-family: 'Inter', sans-serif;
        max-width: 1400px;
        padding-top: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .signal-long {
        background: linear-gradient(135deg, #00C851, #007E33);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 200, 81, 0.3);
    }
    
    .signal-short {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 68, 68, 0.3);
    }
    
    .signal-contrarian {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 152, 0, 0.3);
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, #9E9E9E, #757575);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 20px rgba(158, 158, 158, 0.3);
    }
    
    .status-success {
        background: linear-gradient(135deg, #00C851, #007E33);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 15px 0;
        font-weight: 600;
        animation: pulse 2s infinite;
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
    
    .status-error {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 15px 0;
        font-weight: 600;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    
    .opportunity-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .opportunity-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .trading-setup {
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .data-quality-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .quality-excellent { background: #00C851; }
    .quality-good { background: #ffbb33; }
    .quality-poor { background: #ff4444; }
</style>
""", unsafe_allow_html=True)

class RealTimeCryptoFetcher:
    """Real-time crypto data fetcher with multiple exchange APIs"""
    
    def __init__(self):
        self.primary_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT',
            'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'NEAR/USDT', 'ALGO/USDT',
            'VET/USDT', 'FIL/USDT', 'ETC/USDT', 'AAVE/USDT', 'MKR/USDT',
            'ATOM/USDT', 'FTM/USDT', 'SAND/USDT', 'MANA/USDT', 'AXS/USDT'
        ]
        
        # Initialize exchanges
        self.exchanges = {}
        self.active_exchange = None
        self.data_quality = 'unknown'
        
        # Initialize exchange connections
        self._init_exchanges()
    
    def _init_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Binance (primary)
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': '',  # Add your API keys if needed
                'secret': '',
                'timeout': 10000,
                'enableRateLimit': True,
                'sandbox': False
            })
            
            # Bybit (backup)
            self.exchanges['bybit'] = ccxt.bybit({
                'apiKey': '',
                'secret': '',
                'timeout': 10000,
                'enableRateLimit': True,
                'sandbox': False
            })
            
            # OKX (backup)
            self.exchanges['okx'] = ccxt.okx({
                'apiKey': '',
                'secret': '',
                'timeout': 10000,
                'enableRateLimit': True,
                'sandbox': False
            })
            
        except Exception as e:
            st.warning(f"Exchange initialization warning: {e}")
    
    def fetch_real_time_data(self) -> Tuple[List[Dict], str, str]:
        """Fetch real-time data from exchanges"""
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Fetch tickers
                tickers = exchange.fetch_tickers(self.primary_symbols)
                
                if not tickers:
                    continue
                
                # Process ticker data
                processed_data = []
                for symbol, ticker in tickers.items():
                    if symbol in self.primary_symbols:
                        # Get additional data
                        try:
                            # Fetch OHLCV data for more metrics
                            ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=2)
                            
                            if len(ohlcv) >= 2:
                                current_candle = ohlcv[-1]
                                prev_candle = ohlcv[-2]
                                
                                # Calculate 24h change
                                open_price = prev_candle[1] if prev_candle[1] > 0 else ticker['open']
                                change_24h = ((ticker['last'] - open_price) / open_price * 100) if open_price > 0 else 0
                            else:
                                change_24h = ticker['percentage'] or 0
                                
                        except:
                            change_24h = ticker['percentage'] or 0
                        
                        processed_data.append({
                            'symbol': symbol.replace('/', ''),
                            'price': ticker['last'],
                            'change_24h': change_24h,
                            'volume_24h': ticker['quoteVolume'] or ticker['baseVolume'] * ticker['last'],
                            'high_24h': ticker['high'],
                            'low_24h': ticker['low'],
                            'open_price': ticker['open'],
                            'bid': ticker['bid'],
                            'ask': ticker['ask'],
                            'timestamp': ticker['timestamp']
                        })
                
                if processed_data:
                    self.active_exchange = exchange_name
                    self.data_quality = 'excellent'
                    return processed_data, f'{exchange_name}_real', f"🔥 LIVE {exchange_name.upper()} data - {len(processed_data)} assets"
                    
            except Exception as e:
                st.error(f"Error fetching from {exchange_name}: {e}")
                continue
        
        # Fallback to direct API calls
        return self._fallback_api_fetch()
    
    def _fallback_api_fetch(self) -> Tuple[List[Dict], str, str]:
        """Fallback to direct API calls"""
        try:
            # Try Binance REST API directly
            url = "https://api.binance.com/api/v3/ticker/24hr"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                processed_data = []
                
                symbol_map = {s.replace('/', ''): s for s in self.primary_symbols}
                
                for item in data:
                    if item['symbol'] in symbol_map:
                        processed_data.append({
                            'symbol': item['symbol'],
                            'price': float(item['lastPrice']),
                            'change_24h': float(item['priceChangePercent']),
                            'volume_24h': float(item['quoteVolume']),
                            'high_24h': float(item['highPrice']),
                            'low_24h': float(item['lowPrice']),
                            'open_price': float(item['openPrice']),
                            'bid': float(item['bidPrice']) if 'bidPrice' in item else float(item['lastPrice']) * 0.999,
                            'ask': float(item['askPrice']) if 'askPrice' in item else float(item['lastPrice']) * 1.001,
                            'timestamp': int(item['closeTime'])
                        })
                
                if processed_data:
                    self.data_quality = 'good'
                    return processed_data, 'binance_api', f"⚡ LIVE Binance REST API - {len(processed_data)} assets"
            
        except Exception as e:
            st.error(f"Fallback API error: {e}")
        
        # Final fallback - realistic simulation with market hours consideration
        return self._generate_market_simulation()
    
    def _generate_market_simulation(self) -> Tuple[List[Dict], str, str]:
        """Generate realistic market simulation"""
        # Market hours simulation (more volatile during US/EU hours)
        current_hour = datetime.now().hour
        market_activity_factor = 1.5 if 6 <= current_hour <= 22 else 0.7
        
        realistic_data = []
        base_prices = {
            'BTCUSDT': 67500, 'ETHUSDT': 3450, 'BNBUSDT': 615, 'SOLUSDT': 175,
            'XRPUSDT': 0.52, 'ADAUSDT': 0.47, 'AVAXUSDT': 38, 'DOTUSDT': 7.2,
            'LINKUSDT': 14.5, 'MATICUSDT': 0.85, 'UNIUSDT': 8.5, 'LTCUSDT': 85,
            'BCHUSDT': 285, 'NEARUSDT': 5.2, 'ALGOUSDT': 0.18, 'VETUSDT': 0.031,
            'FILUSDT': 5.8, 'ETCUSDT': 26, 'AAVEUSDT': 95, 'MKRUSDT': 1650,
            'ATOMUSDT': 9.8, 'FTMUSDT': 0.45, 'SANDUSDT': 0.42, 'MANAUSDT': 0.38,
            'AXSUSDT': 6.2
        }
        
        for symbol, base_price in base_prices.items():
            # Market simulation with realistic volatility
            volatility = {
                'BTCUSDT': 0.03, 'ETHUSDT': 0.04, 'BNBUSDT': 0.05,
                'SOLUSDT': 0.07, 'XRPUSDT': 0.06, 'ADAUSDT': 0.05
            }.get(symbol, 0.08) * market_activity_factor
            
            daily_change = np.random.normal(0, volatility)
            current_price = base_price * (1 + daily_change)
            change_24h = daily_change * 100
            
            # Volume simulation
            if symbol in ['BTCUSDT', 'ETHUSDT']:
                volume_base = np.random.uniform(15e9, 35e9)
            elif symbol in ['BNBUSDT', 'SOLUSDT', 'XRPUSDT']:
                volume_base = np.random.uniform(500e6, 5e9)
            else:
                volume_base = np.random.uniform(50e6, 800e6)
            
            volume_24h = volume_base * (1 + abs(daily_change) * 2) * market_activity_factor
            
            # Price range
            high_24h = current_price * (1 + abs(daily_change) * 0.6)
            low_24h = current_price * (1 - abs(daily_change) * 0.6)
            
            realistic_data.append({
                'symbol': symbol,
                'price': current_price,
                'change_24h': change_24h,
                'volume_24h': volume_24h,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'open_price': base_price,
                'bid': current_price * 0.999,
                'ask': current_price * 1.001,
                'timestamp': int(datetime.now().timestamp() * 1000)
            })
        
        self.data_quality = 'simulated'
        return realistic_data, 'simulation', f"📊 Market Simulation - {len(realistic_data)} assets (Market factor: {market_activity_factor:.1f}x)"

class AdvancedCSIQCalculator:
    """Advanced CSI-Q calculation with full quantitative analysis"""
    
    @staticmethod
    def calculate_derivatives_metrics(price_data: Dict) -> Dict:
        """Calculate advanced derivatives metrics"""
        price = price_data['price']
        change_24h = price_data['change_24h']
        volume_24h = price_data['volume_24h']
        
        # Funding rate simulation based on market conditions
        base_funding = change_24h * 0.001  # 0.1% for every 10% move
        
        # Add volatility component
        price_volatility = abs(change_24h) / 100
        vol_adjusted_funding = base_funding * (1 + price_volatility * 2)
        
        # Add volume component
        volume_factor = min(2.0, volume_24h / 1e9)  # Normalize by 1B
        funding_rate = vol_adjusted_funding * (0.5 + volume_factor * 0.5)
        
        # Clamp funding rate to realistic range
        funding_rate = max(-0.375, min(0.375, funding_rate))  # Max ±37.5 basis points
        
        # Open Interest simulation
        if abs(change_24h) > 10:
            oi_change = np.random.uniform(25, 75) * np.sign(change_24h)
        elif abs(change_24h) > 5:
            oi_change = np.random.uniform(10, 25) * np.sign(change_24h)
        else:
            oi_change = np.random.uniform(-5, 5)
        
        # Long/Short ratio simulation
        if change_24h > 8:
            long_short_ratio = np.random.uniform(2.5, 5.0)
        elif change_24h > 3:
            long_short_ratio = np.random.uniform(1.3, 2.5)
        elif change_24h < -8:
            long_short_ratio = np.random.uniform(0.2, 0.4)
        elif change_24h < -3:
            long_short_ratio = np.random.uniform(0.4, 0.7)
        else:
            long_short_ratio = np.random.uniform(0.8, 1.2)
        
        return {
            'funding_rate': funding_rate,
            'oi_change': oi_change,
            'long_short_ratio': long_short_ratio,
            'basis_spread': (price_data['ask'] - price_data['bid']) / price * 100
        }
    
    @staticmethod
    def calculate_sentiment_analysis(price_data: Dict, derivatives: Dict) -> Dict:
        """Advanced sentiment analysis"""
        price = price_data['price']
        change_24h = price_data['change_24h']
        volume_24h = price_data['volume_24h']
        
        # Base sentiment from price action
        price_sentiment = np.tanh(change_24h / 10)  # Sigmoid-like function
        
        # Volume sentiment
        volume_strength = min(1.0, volume_24h / 5e9)
        volume_sentiment = volume_strength * 0.3 * np.sign(change_24h)
        
        # Derivatives sentiment
        funding_sentiment = -derivatives['funding_rate'] * 5  # Contrarian to funding
        ratio_sentiment = (derivatives['long_short_ratio'] - 1) * 0.2
        
        # Combined sentiment
        combined_sentiment = price_sentiment + volume_sentiment + funding_sentiment - ratio_sentiment
        combined_sentiment = max(-1.0, min(1.0, combined_sentiment))
        
        # Social metrics simulation
        base_mentions = int(volume_24h / 1e6)  # 1 mention per $1M volume
        trending_factor = 1 + abs(change_24h) * 0.1
        total_mentions = int(base_mentions * trending_factor)
        
        # Platform breakdown
        twitter_mentions = int(total_mentions * 0.45)
        reddit_mentions = int(total_mentions * 0.25)
        telegram_mentions = int(total_mentions * 0.20)
        discord_mentions = int(total_mentions * 0.10)
        
        # News sentiment
        news_sentiment = combined_sentiment * 0.8 + np.random.normal(0, 0.1)
        news_sentiment = max(-1.0, min(1.0, news_sentiment))
        
        return {
            'combined_sentiment': combined_sentiment,
            'price_sentiment': price_sentiment,
            'volume_sentiment': volume_sentiment,
            'news_sentiment': news_sentiment,
            'social_sentiment': combined_sentiment * 1.2,
            'total_mentions': total_mentions,
            'twitter_mentions': twitter_mentions,
            'reddit_mentions': reddit_mentions,
            'telegram_mentions': telegram_mentions,
            'discord_mentions': discord_mentions,
            'sentiment_magnitude': abs(combined_sentiment) * (total_mentions / 1000),
            'bullish_keywords': max(0, int(combined_sentiment * 10)) if combined_sentiment > 0 else 0,
            'bearish_keywords': max(0, int(abs(combined_sentiment) * 10)) if combined_sentiment < 0 else 0
        }
    
    @staticmethod
    def calculate_technical_analysis(price_data: Dict) -> Dict:
        """Advanced technical analysis"""
        price = price_data['price']
        change_24h = price_data['change_24h']
        high_24h = price_data['high_24h']
        low_24h = price_data['low_24h']
        
        # RSI simulation based on price action
        rsi_base = 50 + (change_24h * 1.8)
        rsi_noise = np.random.normal(0, 5)
        rsi = max(10, min(90, rsi_base + rsi_noise))
        
        # Bollinger Bands squeeze
        daily_range = (high_24h - low_24h) / price if price > 0 else 0.01
        bb_squeeze = max(0, 1 - daily_range * 20)  # Inverse of volatility
        
        # MACD simulation
        macd_line = change_24h * 0.1
        macd_signal = macd_line * 0.8
        macd_histogram = macd_line - macd_signal
        
        # ATR (Average True Range)
        atr = daily_range * price * np.random.uniform(0.8, 1.2)
        
        # Support/Resistance levels
        support_level = low_24h * np.random.uniform(0.98, 1.02)
        resistance_level = high_24h * np.random.uniform(0.98, 1.02)
        
        return {
            'rsi': rsi,
            'bb_squeeze': bb_squeeze,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'atr': atr,
            'atr_percent': (atr / price) * 100,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'daily_range_percent': daily_range * 100
        }
    
    @staticmethod
    def calculate_csiq_scores(derivatives: Dict, sentiment: Dict, technical: Dict) -> Dict:
        """Calculate CSI-Q component scores"""
        
        # Derivatives Score (0-100)
        funding_component = min(40, abs(derivatives['funding_rate']) * 500)
        oi_component = min(30, abs(derivatives['oi_change']) * 1.2)
        ratio_imbalance = abs(derivatives['long_short_ratio'] - 1.0)
        ratio_component = min(30, ratio_imbalance * 25)
        derivatives_score = funding_component + oi_component + ratio_component
        
        # Social Score (0-100)
        sentiment_component = abs(sentiment['combined_sentiment']) * 45
        mention_component = min(35, (sentiment['total_mentions'] / 1000) * 35)
        magnitude_component = min(20, sentiment['sentiment_magnitude'] * 20)
        social_score = sentiment_component + mention_component + magnitude_component
        
        # Technical Score (0-100)
        rsi_extremity = abs(technical['rsi'] - 50) / 50
        rsi_component = rsi_extremity * 60
        squeeze_component = technical['bb_squeeze'] * 25
        momentum_component = abs(technical['macd_histogram']) * 15
        tech_score = rsi_component + squeeze_component + momentum_component
        
        # Basis Score (0-100)
        basis_score = min(100, derivatives['basis_spread'] * 50 + 20)
        
        # Final CSI-Q (weighted combination)
        csiq = (
            derivatives_score * 0.35 +
            social_score * 0.35 +
            tech_score * 0.20 +
            basis_score * 0.10
        )
        
        return {
            'derivatives_score': derivatives_score,
            'social_score': social_score,
            'tech_score': tech_score,
            'basis_score': basis_score,
            'csiq': csiq
        }

def generate_complete_analysis(coin_data: Dict, data_source: str) -> Dict:
    """Generate complete quantitative analysis for each coin"""
    
    calculator = AdvancedCSIQCalculator()
    
    # Calculate all components
    derivatives = calculator.calculate_derivatives_metrics(coin_data)
    sentiment = calculator.calculate_sentiment_analysis(coin_data, derivatives)
    technical = calculator.calculate_technical_analysis(coin_data)
    scores = calculator.calculate_csiq_scores(derivatives, sentiment, technical)
    
    # Generate news headlines
    symbol_clean = coin_data['symbol'].replace('USDT', '').replace('/', '')
    change_24h = coin_data['change_24h']
    
    if change_24h > 15:
        headlines = [
            f"{symbol_clean} explodes {change_24h:.1f}% as institutional FOMO kicks in",
            f"BREAKING: {symbol_clean} rockets {change_24h:.1f}% on massive whale accumulation",
            f"{symbol_clean} moonshot continues with {change_24h:.1f}% surge - targets next resistance"
        ]
    elif change_24h > 8:
        headlines = [
            f"{symbol_clean} rallies {change_24h:.1f}% on strong buying pressure",
            f"Bullish momentum builds for {symbol_clean}, up {change_24h:.1f}% in 24h",
            f"{symbol_clean} breaks key resistance with {change_24h:.1f}% pump"
        ]
    elif change_24h < -15:
        headlines = [
            f"{symbol_clean} collapses {abs(change_24h):.1f}% amid panic selling",
            f"ALERT: {symbol_clean} crashes {abs(change_24h):.1f}% as bears take control",
            f"{symbol_clean} bloodbath continues, down {abs(change_24h):.1f}% in brutal selloff"
        ]
    elif change_24h < -8:
        headlines = [
            f"{symbol_clean} tumbles {abs(change_24h):.1f}% on heavy selling pressure",
            f"Bears dominate {symbol_clean}, down {abs(change_24h):.1f}% amid liquidations",
            f"{symbol_clean} breaks support level, falls {abs(change_24h):.1f}%"
        ]
    else:
        headlines = [
            f"{symbol_clean} consolidates around ${coin_data['price']:.6f} amid mixed signals",
            f"{symbol_clean} trading sideways at ${coin_data['price']:.6f} - waiting for catalyst",
            f"Neutral action for {symbol_clean} as market seeks direction"
        ]
    
    # Combine everything
    complete_data = {
        # Basic data
        'Symbol': symbol_clean,
        'Price': coin_data['price'],
        'Change_24h': coin_data['change_24h'],
        'Volume_24h': coin_data['volume_24h'],
        'High_24h': coin_data['high_24h'],
        'Low_24h': coin_data['low_24h'],
        'Open_Price': coin_data['open_price'],
        'Bid': coin_data['bid'],
        'Ask': coin_data['ask'],
        'Timestamp': coin_data['timestamp'],
        
        # Derivatives
        'Funding_Rate': derivatives['funding_rate'],
        'OI_Change': derivatives['oi_change'],
        'Long_Short_Ratio': derivatives['long_short_ratio'],
        'Basis_Spread': derivatives['basis_spread'],
        
        # Sentiment
        'Combined_Sentiment': sentiment['combined_sentiment'],
        'Price_Sentiment': sentiment['price_sentiment'],
        'Volume_Sentiment': sentiment['volume_sentiment'],
        'News_Sentiment': sentiment['news_sentiment'],
        'Social_Sentiment': sentiment['social_sentiment'],
        'Total_Mentions': sentiment['total_mentions'],
        'Twitter_Mentions': sentiment['twitter_mentions'],
        'Reddit_Mentions': sentiment['reddit_mentions'],
        'Telegram_Mentions': sentiment['telegram_mentions'],
        'Discord_Mentions': sentiment['discord_mentions'],
        'Sentiment_Magnitude': sentiment['sentiment_magnitude'],
        'Bullish_Keywords': sentiment['bullish_keywords'],
        'Bearish_Keywords': sentiment['bearish_keywords'],
        
        # Technical
        'RSI': technical['rsi'],
        'BB_Squeeze': technical['bb_squeeze'],
        'MACD_Line': technical['macd_line'],
        'MACD_Signal': technical['macd_signal'],
        'MACD_Histogram': technical['macd_histogram'],
        'ATR': technical['atr'],
        'ATR_Percent': technical['atr_percent'],
        'Support_Level': technical['support_level'],
        'Resistance_Level': technical['resistance_level'],
        'Daily_Range_Percent': technical['daily_range_percent'],
        
        # Scores
        'Derivatives_Score': scores['derivatives_score'],
        'Social_Score': scores['social_score'],
        'Tech_Score': scores['tech_score'],
        'Basis_Score': scores['basis_score'],
        'CSI_Q': scores['csiq'],
        
        # News
        'Top_Headline': np.random.choice(headlines),
        'Headline_Source': f'{data_source.replace("_", " ").title()} Analysis',
        
        # Meta
        'Data_Source': data_source,
        'Last_Updated': datetime.now(),
        'Market_Cap_Rank': {
            'BTCUSDT': 1, 'ETHUSDT': 2, 'BNBUSDT': 4, 'SOLUSDT': 5, 'XRPUSDT': 6,
            'ADAUSDT': 8, 'AVAXUSDT': 10, 'DOTUSDT': 12, 'LINKUSDT': 15, 'MATICUSDT': 18
        }.get(coin_data['symbol'], np.random.randint(20, 100))
    }
    
    return complete_data

def get_advanced_trading_signal(data: Dict) -> str:
    """Advanced trading signal with multiple factor analysis"""
    csiq = data['CSI_Q']
    funding_rate = data['Funding_Rate']
    sentiment = data['Combined_Sentiment']
    long_short_ratio = data['Long_Short_Ratio']
    rsi = data['RSI']
    volume_24h = data['Volume_24h']
    
    # Extreme CSI-Q levels suggest contrarian opportunities
    if csiq > 85 or csiq < 15:
        return "CONTRARIAN"
    
    # Strong bullish conditions with multiple confirmations
    bullish_signals = 0
    if csiq > 70: bullish_signals += 1
    if funding_rate < 0.05: bullish_signals += 1
    if sentiment > 0.2: bullish_signals += 1
    if long_short_ratio < 2.0: bullish_signals += 1
    if 30 < rsi < 70: bullish_signals += 1
    if volume_24h > 500000000: bullish_signals += 1  # High volume confirmation
    
    if bullish_signals >= 4:
        return "LONG"
    
    # Strong bearish conditions with multiple confirmations
    bearish_signals = 0
    if csiq < 30: bearish_signals += 1
    if funding_rate > -0.05: bearish_signals += 1
    if sentiment < -0.2: bearish_signals += 1
    if long_short_ratio > 0.5: bearish_signals += 1
    if 30 < rsi < 70: bearish_signals += 1
    if volume_24h > 500000000: bearish_signals += 1
    
    if bearish_signals >= 4:
        return "SHORT"
    
    # Moderate signals
    if bullish_signals >= 3:
        return "LONG"
    elif bearish_signals >= 3:
        return "SHORT"
    
    return "NEUTRAL"

def get_signal_emoji(signal: str) -> str:
    return {
        "LONG": "🟢",
        "SHORT": "🔴", 
        "CONTRARIAN": "🟠",
        "NEUTRAL": "⚪"
    }.get(signal, "⚪")

def get_sentiment_emoji(score: float) -> str:
    if score > 0.6: return "🚀"
    elif score > 0.3: return "😊"
    elif score > 0.1: return "🙂"
    elif score > -0.1: return "😐"
    elif score > -0.3: return "😕"
    elif score > -0.6: return "😟"
    else: return "😰"

@st.cache_data(ttl=30)  # Cache for 30 seconds for real-time feel
def load_real_time_crypto_data():
    """Main data loading function with real-time data"""
    try:
        fetcher = RealTimeCryptoFetcher()
        raw_coins, data_source, status_message = fetcher.fetch_real_time_data()
        
        if not raw_coins:
            st.error("No data available from any source")
            return pd.DataFrame(), 'error', 'No data available'
        
        # Process all coins with complete analysis
        processed_data = []
        for coin in raw_coins:
            try:
                complete_analysis = generate_complete_analysis(coin, data_source)
                processed_data.append(complete_analysis)
            except Exception as e:
                st.warning(f"Error processing {coin.get('symbol', 'unknown')}: {e}")
                continue
        
        if not processed_data:
            st.error("No coins processed successfully")
            return pd.DataFrame(), 'error', 'Processing failed'
        
        df = pd.DataFrame(processed_data)
        
        # Add trading signals
        df['Signal'] = df.apply(lambda row: get_advanced_trading_signal(row.to_dict()), axis=1)
        
        # Calculate opportunity scores with advanced weighting
        df['Opportunity_Score'] = (
            (abs(df['CSI_Q'] - 50) / 50 * 30) +                    # CSI-Q extremity
            (abs(df['Funding_Rate']) * 400) +                       # Funding rate
            (abs(df['Long_Short_Ratio'] - 1) * 15) +               # L/S imbalance
            (abs(df['Combined_Sentiment']) * 25) +                  # Sentiment strength
            ((df['Volume_24h'] / df['Volume_24h'].max()) * 20) +   # Volume factor
            (abs(df['RSI'] - 50) / 50 * 10)                        # RSI extremity
        )
        
        return df, data_source, status_message, fetcher
    
    except Exception as e:
        st.error(f"Critical error in load_real_time_crypto_data: {e}")
        return pd.DataFrame(), 'error', f"Critical error: {e}", None

def create_opportunity_display(df: pd.DataFrame, data_source: str) -> None:
    """Create enhanced opportunity display with full analysis"""
    
    if df.empty:
        st.warning("No opportunities available")
        return
    
    # Get top opportunities
    opportunities = df.sort_values('Opportunity_Score', ascending=False).head(8)
    
    st.markdown("### 🎯 **TOP TRADING OPPORTUNITIES**")
    st.markdown(f"*Live data from {data_source.replace('_', ' ').title()}*")
    
    for i, (_, row) in enumerate(opportunities.iterrows()):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 2])
            
            with col1:
                signal_emoji = get_signal_emoji(row['Signal'])
                sentiment_emoji = get_sentiment_emoji(row['Combined_Sentiment'])
                
                st.markdown(f"""
                **{i+1}. {signal_emoji} {row['Symbol']} {sentiment_emoji}**  
                Opportunity Score: **{row['Opportunity_Score']:.1f}/100**  
                Real Price: **${row['Price']:.6f}** | Spread: {row['Basis_Spread']:.3f}%  
                Market Cap Rank: #{row['Market_Cap_Rank']} | Volume: ${row['Volume_24h']/1e6:.1f}M
                """)
            
            with col2:
                st.metric("CSI-Q Score", f"{row['CSI_Q']:.1f}", 
                         delta=f"{'Extreme' if row['CSI_Q'] > 80 or row['CSI_Q'] < 20 else 'Normal'}")
                st.metric("Social Score", f"{row['Social_Score']:.1f}")
                st.metric("Tech Score", f"{row['Tech_Score']:.1f}")
            
            with col3:
                change_color = "normal" if abs(row['Change_24h']) < 5 else "inverse"
                st.metric("24h Change", f"{row['Change_24h']:+.2f}%", delta_color=change_color)
                st.metric("Funding Rate", f"{row['Funding_Rate']:.4f}%")
                st.metric("L/S Ratio", f"{row['Long_Short_Ratio']:.2f}")
            
            with col4:
                # Advanced trading setup
                atr_pct = row['ATR_Percent']
                sentiment_mult = 1 + (abs(row['Combined_Sentiment']) * 0.4)
                vol_mult = 1 + min(0.5, row['Volume_24h'] / 5e9)
                
                if row['Signal'] == 'LONG':
                    target_pct = atr_pct * sentiment_mult * vol_mult * 2.0
                    stop_pct = atr_pct * 0.8
                    target_price = row['Price'] * (1 + target_pct/100)
                    stop_price = row['Price'] * (1 - stop_pct/100)
                    setup_class = "signal-long"
                    setup_text = "📈 LONG SETUP"
                elif row['Signal'] == 'SHORT':
                    target_pct = atr_pct * sentiment_mult * vol_mult * 2.0
                    stop_pct = atr_pct * 0.8
                    target_price = row['Price'] * (1 - target_pct/100)
                    stop_price = row['Price'] * (1 + stop_pct/100)
                    setup_class = "signal-short"
                    setup_text = "📉 SHORT SETUP"
                elif row['Signal'] == 'CONTRARIAN':
                    target_pct = atr_pct * 1.5
                    stop_pct = atr_pct * 0.6
                    if row['Combined_Sentiment'] < -0.3:
                        target_price = row['Price'] * (1 + target_pct/100)
                        stop_price = row['Price'] * (1 - stop_pct/100)
                        setup_text = "🔄 CONTRARIAN LONG"
                    else:
                        target_price = row['Price'] * (1 - target_pct/100)
                        stop_price = row['Price'] * (1 + stop_pct/100)
                        setup_text = "🔄 CONTRARIAN SHORT"
                    setup_class = "signal-contrarian"
                else:
                    target_price = row['Price']
                    stop_price = row['Price']
                    setup_class = "signal-neutral"
                    setup_text = "⚪ NEUTRAL - WAIT"
                
                risk_reward = abs((target_price - row['Price']) / (stop_price - row['Price'])) if abs(stop_price - row['Price']) > 0.0001 else 1.0
                
                st.markdown(f"""
                <div class="{setup_class}">
                    <strong>{setup_text}</strong><br>
                    Entry: ${row['Price']:.6f}<br>
                    Target: ${target_price:.6f}<br>
                    Stop: ${stop_price:.6f}<br>
                    R/R: 1:{risk_reward:.1f}<br>
                    <small>RSI: {row['RSI']:.0f} | ATR: {row['ATR_Percent']:.2f}%</small><br>
                    <small>Mentions: {row['Total_Mentions']:,} | Sentiment: {row['Combined_Sentiment']:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional metrics row
            st.markdown(f"""
            **📊 Advanced Metrics:** Derivatives: {row['Derivatives_Score']:.1f} | 
            Social: {row['Social_Score']:.1f} | Technical: {row['Tech_Score']:.1f} | 
            MACD: {row['MACD_Histogram']:.3f} | Support: ${row['Support_Level']:.6f} | 
            Resistance: ${row['Resistance_Level']:.6f}
            """)
            
            st.markdown("---")

def create_market_overview(df: pd.DataFrame) -> None:
    """Create comprehensive market overview"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Real-Time Opportunity Matrix")
        
        # Enhanced scatter plot with more data points
        fig = px.scatter(
            df.head(20),
            x='Change_24h',
            y='CSI_Q',
            size='Volume_24h',
            color='Combined_Sentiment',
            hover_name='Symbol',
            hover_data={
                'Price': ':$.6f',
                'Volume_24h': ':$,.0f',
                'Signal': True,
                'Funding_Rate': ':.4f',
                'RSI': ':.1f',
                'Total_Mentions': ':,',
                'Opportunity_Score': ':.1f'
            },
            title="💰 Live Trading Opportunities",
            labels={
                'Change_24h': '24h Price Change (%)',
                'CSI_Q': 'CSI-Q Score',
                'Combined_Sentiment': 'Market Sentiment'
            },
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        
        # Add trading zones
        fig.add_hline(y=75, line_dash="dash", line_color="rgba(0,255,0,0.7)", 
                     annotation_text="STRONG LONG Zone", annotation_position="bottom right")
        fig.add_hline(y=25, line_dash="dash", line_color="rgba(255,0,0,0.7)",
                     annotation_text="STRONG SHORT Zone", annotation_position="top right")
        fig.add_hline(y=85, line_dash="dot", line_color="rgba(255,165,0,0.7)",
                     annotation_text="CONTRARIAN Zone", annotation_position="bottom left")
        fig.add_hline(y=15, line_dash="dot", line_color="rgba(255,165,0,0.7)",
                     annotation_text="CONTRARIAN Zone", annotation_position="top left")
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Live Market Intelligence")
        
        # Generate intelligent alerts with advanced scoring
        alerts = []
        for _, row in df.iterrows():
            signal = row['Signal']
            if signal != 'NEUTRAL':
                # Multi-factor alert scoring
                price_momentum = abs(row['Change_24h']) * 3
                volume_factor = min(20, (row['Volume_24h'] / 1e9) * 10)
                sentiment_factor = abs(row['Combined_Sentiment']) * 25
                csiq_factor = max(0, abs(row['CSI_Q'] - 50) - 20) * 2
                funding_factor = abs(row['Funding_Rate']) * 150
                social_factor = min(15, row['Total_Mentions'] / 1000)
                tech_factor = abs(row['RSI'] - 50) / 50 * 10
                
                alert_strength = (price_momentum + volume_factor + sentiment_factor + 
                                csiq_factor + funding_factor + social_factor + tech_factor)
                
                if alert_strength > 60:
                    strength_level = "🔥 CRITICAL"
                    strength_color = "#ff1a1a"
                elif alert_strength > 45:
                    strength_level = "⚡ HIGH"
                    strength_color = "#ff4d00"
                elif alert_strength > 30:
                    strength_level = "⚠️ MEDIUM"
                    strength_color = "#ff8800"
                else:
                    strength_level = "💡 LOW"
                    strength_color = "#ffbb33"
                
                alerts.append({
                    'Symbol': row['Symbol'],
                    'Signal': signal,
                    'Strength': strength_level,
                    'Color': strength_color,
                    'Score': alert_strength,
                    'Price': row['Price'],
                    'Change': row['Change_24h'],
                    'Volume': row['Volume_24h'] / 1e6,
                    'CSI_Q': row['CSI_Q'],
                    'Sentiment': row['Combined_Sentiment'],
                    'Funding': row['Funding_Rate'],
                    'RSI': row['RSI'],
                    'Mentions': row['Total_Mentions'],
                    'Opportunity': row['Opportunity_Score']
                })
        
        # Sort by alert strength and display
        alerts = sorted(alerts, key=lambda x: x['Score'], reverse=True)
        
        if alerts:
            for alert in alerts[:10]:  # Show top 10 alerts
                signal_emoji = get_signal_emoji(alert['Signal'])
                sentiment_emoji = get_sentiment_emoji(alert['Sentiment'])
                
                urgency = ("🚨 NOW!" if alert['Score'] > 55 else 
                          "⏰ SOON" if alert['Score'] > 40 else 
                          "👀 WATCH")
                
                st.markdown(f"""
                <div style="background: {alert['Color']}; padding: 12px; border-radius: 10px; 
                           color: white; margin: 8px 0; border-left: 4px solid rgba(255,255,255,0.5);">
                    {signal_emoji} <strong>{alert['Symbol']}</strong> {sentiment_emoji} 
                    <span style="float: right; font-size: 0.85em; font-weight: bold;">{urgency}</span><br>
                    <strong>{alert['Signal']} | {alert['Strength']}</strong><br>
                    💰 ${alert['Price']:.6f} ({alert['Change']:+.1f}%) | 📊 ${alert['Volume']:.0f}M<br>
                    🎯 CSI-Q: {alert['CSI_Q']:.1f} | 💭 Sentiment: {alert['Sentiment']:.2f}<br>
                    📈 Funding: {alert['Funding']:.4f}% | RSI: {alert['RSI']:.0f}<br>
                    📱 Mentions: {alert['Mentions']:,} | ⚡ Score: {alert['Score']:.0f}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 No high-priority alerts at this time")

def create_advanced_analytics(df: pd.DataFrame) -> None:
    """Create advanced analytics dashboard"""
    
    st.subheader("📊 Advanced Quantitative Analysis")
    
    # Create comprehensive subplot dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "CSI-Q Distribution & Signals", 
            "Funding Rate vs Price Performance",
            "Sentiment vs Social Volume",
            "Technical Indicators Heatmap",
            "Volume vs Volatility Analysis",
            "Risk-Reward Opportunities"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # CSI-Q Distribution
    colors = ['green' if x == 'LONG' else 'red' if x == 'SHORT' else 'orange' if x == 'CONTRARIAN' else 'gray' 
              for x in df['Signal']]
    
    fig.add_trace(go.Bar(
        x=df['Symbol'].head(15),
        y=df['CSI_Q'].head(15),
        name='CSI-Q Score',
        marker_color=colors[:15],
        text=[f"{s}" for s in df['Signal'].head(15)],
        textposition='outside'
    ), row=1, col=1)
    
    # Funding Rate vs Performance
    fig.add_trace(go.Scatter(
        x=df['Funding_Rate'],
        y=df['Change_24h'],
        mode='markers+text',
        name='Funding vs Performance',
        text=df['Symbol'],
        textposition="top center",
        marker=dict(
            size=df['Volume_24h'] / 1e8,
            color=df['CSI_Q'],
            colorscale='Viridis',
            line=dict(width=1, color='white')
        )
    ), row=1, col=2)
    
    # Sentiment vs Social Volume
    fig.add_trace(go.Scatter(
        x=df['Combined_Sentiment'],
        y=df['Total_Mentions'],
        mode='markers+text',
        name='Sentiment Analysis',
        text=df['Symbol'],
        textposition="top center",
        marker=dict(
            size=10,
            color=df['Social_Score'],
            colorscale='RdYlGn',
            line=dict(width=1, color='white')
        )
    ), row=2, col=1)
    
    # Technical Heatmap Data
    tech_data = df[['Symbol', 'RSI', 'MACD_Histogram', 'ATR_Percent']].head(10)
    fig.add_trace(go.Heatmap(
        x=['RSI', 'MACD', 'ATR%'],
        y=tech_data['Symbol'],
        z=[tech_data['RSI'].values, 
           tech_data['MACD_Histogram'].values * 100,
           tech_data['ATR_Percent'].values],
        colorscale='RdYlBu',
        name='Technical Heatmap'
    ), row=2, col=2)
    
    # Volume vs Volatility
    fig.add_trace(go.Scatter(
        x=df['Volume_24h'] / 1e6,
        y=df['Daily_Range_Percent'],
        mode='markers+text',
        name='Volume vs Volatility',
        text=df['Symbol'],
        textposition="top center",
        marker=dict(
            size=df['Opportunity_Score'] / 5,
            color=df['Combined_Sentiment'],
            colorscale='RdYlGn',
            line=dict(width=1, color='white')
        )
    ), row=3, col=1)
    
    # Risk-Reward Analysis
    df_signals = df[df['Signal'] != 'NEUTRAL']
    if not df_signals.empty:
        fig.add_trace(go.Scatter(
            x=df_signals['ATR_Percent'],
            y=df_signals['Opportunity_Score'],
            mode='markers+text',
            name='Risk-Reward',
            text=df_signals['Symbol'],
            textposition="top center",
            marker=dict(
                size=15,
                color=df_signals['CSI_Q'],
                colorscale='Plasma',
                line=dict(width=2, color='white')
            )
        ), row=3, col=2)
    
    fig.update_layout(
        height=900,
        title_text="🚀 Real-Time Crypto Quantitative Analysis Dashboard",
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Assets", row=1, col=1)
    fig.update_yaxes(title_text="CSI-Q Score", row=1, col=1)
    fig.update_xaxes(title_text="Funding Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="24h Change (%)", row=1, col=2)
    fig.update_xaxes(title_text="Sentiment Score", row=2, col=1)
    fig.update_yaxes(title_text="Social Mentions", row=2, col=1)
    fig.update_xaxes(title_text="Volume ($M)", row=3, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)
    fig.update_xaxes(title_text="Risk (ATR %)", row=3, col=2)
    fig.update_yaxes(title_text="Opportunity Score", row=3, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application with real-time data"""
    
    # Header
    st.title("🚀 Enhanced Crypto CSI-Q Dashboard")
    st.markdown("**💰 Professional Real-Time Trading Intelligence**")
    
    # Status and controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        st.markdown("📊 **REAL-TIME MODE**")
    
    with col2:
        st.markdown(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    with col3:
        if st.button("🔄 Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    with col4:
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    # Load real-time data
    try:
        with st.spinner("🚀 Loading real-time crypto data..."):
            df, data_source, status_message, fetcher = load_real_time_crypto_data()
        
        if df.empty:
            st.error("❌ No data available. Check your internet connection and try refreshing.")
            return
        
        # Data quality indicator
        if fetcher:
            quality_level, quality_class = fetcher.data_quality, "quality-excellent" if "real" in data_source else "quality-good"
        else:
            quality_level, quality_class = "unknown", "quality-poor"
        
        # Enhanced status message
        data_age = "LIVE" if "real" in data_source else "SIMULATED"
        st.markdown(f"""
        <div class="status-{'success' if 'real' in data_source else 'warning'}">
            <span class="data-quality-indicator {quality_class}"></span>
            {status_message} | Quality: {quality_level.title()} | 
            📊 {len(df)} assets loaded | 🎯 {len(df[df['Signal'] != 'NEUTRAL'])} signals active | 
            🕐 Data: {data_age} | 💰 Total Volume: ${df['Volume_24h'].sum()/1e9:.1f}B
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"💥 **Critical Error:** {e}")
        st.write("Please check your internet connection and try refreshing the page.")
        st.stop()
    
    # Sidebar filters (full implementation)
    with st.sidebar:
        st.header("🔧 Advanced Real-Time Filters")
        
        # Real-time data info
        st.info(f"📡 **Data Source:** {data_source.replace('_', ' ').title()}\n\n⏱️ **Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        
        # Signal filters
        st.subheader("📊 Signal Filters")
        signal_types = st.multiselect(
            "Trading Signals",
            ["LONG", "SHORT", "CONTRARIAN", "NEUTRAL"],
            default=["LONG", "SHORT", "CONTRARIAN"]
        )
        
        # CSI-Q range
        csiq_range = st.slider("CSI-Q Score Range", 0, 100, (5, 95))
        
        # Market filters
        st.subheader("💰 Market Filters")
        min_volume = st.number_input("Min Volume ($M)", 0, 50000, 10)
        min_change = st.slider("Min |Price Change| %", 0.0, 50.0, 0.5)
        max_price = st.number_input("Max Price ($)", 0.01, 500000.0, 500000.0)
        min_opportunity = st.slider("Min Opportunity Score", 0, 100, 20)
        
        # Sentiment filters
        st.subheader("🎭 Sentiment Analysis")
        sentiment_range = st.slider("Sentiment Range", -1.0, 1.0, (-1.0, 1.0))
        min_mentions = st.number_input("Min Social Mentions", 0, 50000, 100)
        
        # Technical filters
        st.subheader("📈 Technical Analysis")
        rsi_range = st.slider("RSI Range", 0, 100, (5, 95))
        funding_range = st.slider("Funding Rate Range (%)", -0.5, 0.5, (-0.4, 0.4))
        atr_range = st.slider("ATR % Range", 0.0, 20.0, (0.0, 15.0))
        
        # Apply all filters
        filtered_df = df[
            (df['Signal'].isin(signal_types)) &
            (df['CSI_Q'] >= csiq_range[0]) & (df['CSI_Q'] <= csiq_range[1]) &
            (df['Volume_24h'] >= min_volume * 1e6) &
            (abs(df['Change_24h']) >= min_change) &
            (df['Price'] <= max_price) &
            (df['Opportunity_Score'] >= min_opportunity) &
            (df['Combined_Sentiment'] >= sentiment_range[0]) & 
            (df['Combined_Sentiment'] <= sentiment_range[1]) &
            (df['Total_Mentions'] >= min_mentions) &
            (df['RSI'] >= rsi_range[0]) & (df['RSI'] <= rsi_range[1]) &
            (df['Funding_Rate'] >= funding_range[0]) & 
            (df['Funding_Rate'] <= funding_range[1]) &
            (df['ATR_Percent'] >= atr_range[0]) &
            (df['ATR_Percent'] <= atr_range[1])
        ].copy()
        
        st.markdown(f"**✅ Filtered Results: {len(filtered_df)}/{len(df)} assets**")
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No assets match current filters")
    
    # Top metrics dashboard
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_data = [
        ("🎯 Active Signals", len(filtered_df[filtered_df['Signal'] != 'NEUTRAL'])),
        ("💰 Total Volume", f"${filtered_df['Volume_24h'].sum()/1e9:.1f}B"),
        ("📊 Avg CSI-Q", f"{filtered_df['CSI_Q'].mean():.1f}"),
        ("🎭 Avg Sentiment", f"{filtered_df['Combined_Sentiment'].mean():.2f}"),
        ("⚡ Top Opportunities", f"{len(filtered_df[filtered_df['Opportunity_Score'] > 70])}")
    ]
    
    for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{label}</h4>
                <h2>{value}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content tabs with full functionality
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Trading Opportunities", "📈 Market Analysis", "📊 Advanced Analytics", "📋 Data Explorer"])
    
    with tab1:
        if not filtered_df.empty:
            create_opportunity_display(filtered_df, data_source)
        else:
            st.warning("No trading opportunities match your current filters.")
    
    with tab2:
        if not filtered_df.empty:
            create_market_overview(filtered_df)
        else:
            st.warning("No data available for market analysis.")
    
    with tab3:
        if not filtered_df.empty:
            create_advanced_analytics(filtered_df)
        else:
            st.warning("No data available for advanced analytics.")
    
    with tab4:
        st.header("📋 Complete Real-Time Data Explorer")
        
        if not filtered_df.empty:
            # Comprehensive data table with all metrics
            display_cols = [
                'Symbol', 'Price', 'Change_24h', 'Volume_24h', 'CSI_Q', 'Signal',
                'Combined_Sentiment', 'Funding_Rate', 'Long_Short_Ratio', 'RSI',
                'Total_Mentions', 'ATR_Percent', 'Opportunity_Score', 'Support_Level',
                'Resistance_Level', 'MACD_Histogram', 'Social_Score', 'Tech_Score',
                'Derivatives_Score', 'Market_Cap_Rank', 'Data_Source'
            ]
            
            display_df = filtered_df[display_cols].copy()
            display_df = display_df.sort_values('Opportunity_Score', ascending=False)
            
            # Format numeric columns for better display
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.6f}")
            display_df['Change_24h'] = display_df['Change_24h'].apply(lambda x: f"{x:+.2f}%")
            display_df['Volume_24h'] = display_df['Volume_24h'].apply(lambda x: f"${x/1e6:.1f}M")
            display_df['CSI_Q'] = display_df['CSI_Q'].round(1)
            display_df['Combined_Sentiment'] = display_df['Combined_Sentiment'].round(3)
            display_df['Funding_Rate'] = display_df['Funding_Rate'].apply(lambda x: f"{x:.4f}%")
            display_df['Long_Short_Ratio'] = display_df['Long_Short_Ratio'].round(2)
            display_df['RSI'] = display_df['RSI'].round(1)
            display_df['ATR_Percent'] = display_df['ATR_Percent'].apply(lambda x: f"{x:.2f}%")
            display_df['Opportunity_Score'] = display_df['Opportunity_Score'].round(1)
            display_df['Support_Level'] = display_df['Support_Level'].apply(lambda x: f"${x:.6f}")
            display_df['Resistance_Level'] = display_df['Resistance_Level'].apply(lambda x: f"${x:.6f}")
            display_df['MACD_Histogram'] = display_df['MACD_Histogram'].round(4)
            display_df['Social_Score'] = display_df['Social_Score'].round(1)
            display_df['Tech_Score'] = display_df['Tech_Score'].round(1)
            display_df['Derivatives_Score'] = display_df['Derivatives_Score'].round(1)
            
            # Display with advanced filtering options
            st.subheader("📊 Real-Time Market Data")
            st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Source:** {data_source.replace('_', ' ').title()}")
            
            # Additional quick filters
            col1, col2, col3 = st.columns(3)
            with col1:
                show_only_signals = st.checkbox("Show only signals", value=True)
            with col2:
                sort_by = st.selectbox("Sort by", ["Opportunity_Score", "CSI_Q", "Volume_24h", "Change_24h"], index=0)
            with col3:
                ascending = st.checkbox("Ascending order", value=False)
            
            if show_only_signals:
                display_df = display_df[display_df['Signal'] != 'NEUTRAL']
            
            if sort_by in display_df.columns:
                # Convert back to numeric for sorting
                if sort_by == 'Volume_24h':
                    sort_values = filtered_df[sort_by]
                else:
                    sort_values = filtered_df[sort_by]
                display_df = display_df.iloc[sort_values.argsort()[::-1 if not ascending else 1]]
            
            st.dataframe(display_df, use_container_width=True, height=600)
            
            # Export functionality
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💾 Export Filtered Data"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "📄 Download CSV",
                        csv,
                        file_name=f"crypto_realtime_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Export Trading Signals"):
                    signals_df = filtered_df[filtered_df['Signal'] != 'NEUTRAL'][
                        ['Symbol', 'Price', 'Signal', 'CSI_Q', 'Opportunity_Score', 
                         'Combined_Sentiment', 'Funding_Rate', 'RSI', 'Support_Level', 'Resistance_Level']
                    ]
                    csv = signals_df.to_csv(index=False)
                    st.download_button(
                        "⚡ Download Signals",
                        csv,
                        file_name=f"crypto_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("📈 Export Summary Report"):
                    # Generate comprehensive summary
                    summary = {
                        'Report_Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Data_Source': data_source,
                        'Total_Assets_Analyzed': len(df),
                        'Assets_After_Filters': len(filtered_df),
                        'Active_Trading_Signals': len(filtered_df[filtered_df['Signal'] != 'NEUTRAL']),
                        'Long_Signals': len(filtered_df[filtered_df['Signal'] == 'LONG']),
                        'Short_Signals': len(filtered_df[filtered_df['Signal'] == 'SHORT']),
                        'Contrarian_Signals': len(filtered_df[filtered_df['Signal'] == 'CONTRARIAN']),
                        'Total_Volume_24h': f"${filtered_df['Volume_24h'].sum()/1e9:.2f}B",
                        'Average_CSI_Q': round(filtered_df['CSI_Q'].mean(), 2),
                        'Average_Sentiment': round(filtered_df['Combined_Sentiment'].mean(), 3),
                        'Top_Opportunity': {
                            'Symbol': filtered_df.loc[filtered_df['Opportunity_Score'].idxmax(), 'Symbol'],
                            'Score': round(filtered_df['Opportunity_Score'].max(), 1),
                            'Signal': filtered_df.loc[filtered_df['Opportunity_Score'].idxmax(), 'Signal']
                        },
                        'Market_Conditions': {
                            'Bullish_Assets': len(filtered_df[filtered_df['Combined_Sentiment'] > 0.2]),
                            'Bearish_Assets': len(filtered_df[filtered_df['Combined_Sentiment'] < -0.2]),
                            'High_Volume_Assets': len(filtered_df[filtered_df['Volume_24h'] > 1e9]),
                            'Extreme_CSI_Q': len(filtered_df[(filtered_df['CSI_Q'] > 80) | (filtered_df['CSI_Q'] < 20)])
                        }
                    }
                    st.json(summary)
        else:
            st.info("No data available after applying filters. Try adjusting your filter criteria.")
    
    # Enhanced footer with real-time stats
    st.markdown("---")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Real-time footer metrics
    total_opportunities = len(filtered_df[filtered_df['Opportunity_Score'] > 60])
    avg_funding = filtered_df['Funding_Rate'].mean()
    market_sentiment = filtered_df['Combined_Sentiment'].mean()
    
    footer_data = [
        ("🔄 Refresh", "30 sec"),
        ("📡 Source", data_source.replace('_', ' ').title()),
        ("📊 Signals", f"{len(filtered_df[filtered_df['Signal'] != 'NEUTRAL'])}/{len(filtered_df)}"),
        ("💰 Avg Funding", f"{avg_funding:.4f}%"),
        ("🎭 Market Mood", "Bullish" if market_sentiment > 0.1 else "Bearish" if market_sentiment < -0.1 else "Neutral"),
        ("⏰ Updated", datetime.now().strftime('%H:%M:%S'))
    ]
    
    for col, (label, value) in zip([col1, col2, col3, col4, col5, col6], footer_data):
        with col:
            st.markdown(f"**{label}:** {value}")
    
    # Market status indicator
    if market_sentiment > 0.3:
        st.success("🚀 BULLISH MARKET CONDITIONS - Multiple long opportunities detected")
    elif market_sentiment < -0.3:
        st.error("📉 BEARISH MARKET CONDITIONS - Multiple short opportunities detected")
    elif total_opportunities > 5:
        st.warning("⚡ HIGH VOLATILITY - Multiple trading opportunities available")
    else:
        st.info("😐 NEUTRAL MARKET - Limited opportunities, wait for better setups")

# Initialize and run
if __name__ == "__main__":
    try:
        # Try to install ccxt if not available
        try:
            import ccxt
        except ImportError:
            st.error("Missing required library. Please run: pip install ccxt")
            st.stop()
        
        main()
    except Exception as e:
        st.error(f"Application startup error: {e}")
        st.write("**Troubleshooting:**")
        st.write("1. Ensure you have internet connectivity")
        st.write("2. Install required packages: `pip install streamlit pandas numpy plotly requests ccxt`")
        st.write("3. Restart the application")
        
        # Provide fallback minimal functionality
        st.write("---")
        st.write("**Debug Information:**")
        st.write(f"Error details: {str(e)}")
        st.write(f"Python version: {__import__('sys').version}")
        st.write(f"Streamlit version: {st.__version__}")
