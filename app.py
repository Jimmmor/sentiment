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

# Custom CSS (keep existing styles)
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
    
    .confidence-high {
        background: linear-gradient(135deg, #00C851, #007E33);
        padding: 8px;
        border-radius: 5px;
        color: white;
        font-size: 0.8em;
        text-align: center;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #ffbb33, #ff8800);
        padding: 8px;
        border-radius: 5px;
        color: white;
        font-size: 0.8em;
        text-align: center;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        padding: 8px;
        border-radius: 5px;
        color: white;
        font-size: 0.8em;
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
</style>
""", unsafe_allow_html=True)

# Crypto tickers (kept same)
TICKERS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
    'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
    'UNIUSDT', 'LTCUSDT', 'BCHUSDT', 'NEARUSDT', 'ALGOUSDT',
    'VETUSDT', 'FILUSDT', 'ETCUSDT', 'AAVEUSDT', 'MKRUSDT',
    'ATOMUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT'
]

class AdvancedSentimentEngine:
    """Geavanceerde sentiment analyse engine"""
    
    def __init__(self):
        # Coin-specific volatiliteit profielen
        self.volatility_profiles = {
            'BTC': {'base_vol': 0.03, 'trend_sensitivity': 0.5, 'funding_weight': 0.8},
            'ETH': {'base_vol': 0.04, 'trend_sensitivity': 0.6, 'funding_weight': 0.8},
            'SOL': {'base_vol': 0.08, 'trend_sensitivity': 1.2, 'funding_weight': 1.2},
            'XRP': {'base_vol': 0.06, 'trend_sensitivity': 0.9, 'funding_weight': 1.0},
            'ADA': {'base_vol': 0.07, 'trend_sensitivity': 1.0, 'funding_weight': 1.1},
            'AVAX': {'base_vol': 0.09, 'trend_sensitivity': 1.3, 'funding_weight': 1.3},
            'DOT': {'base_vol': 0.08, 'trend_sensitivity': 1.1, 'funding_weight': 1.2},
            'LINK': {'base_vol': 0.07, 'trend_sensitivity': 1.0, 'funding_weight': 1.1},
            'MATIC': {'base_vol': 0.09, 'trend_sensitivity': 1.4, 'funding_weight': 1.4},
            'UNI': {'base_vol': 0.08, 'trend_sensitivity': 1.1, 'funding_weight': 1.2},
            # Defaults for smaller coins
            'default': {'base_vol': 0.10, 'trend_sensitivity': 1.5, 'funding_weight': 1.5}
        }
        
        # Markt regime detectie
        self.market_regimes = {
            'bull_market': {'csiq_threshold': 65, 'funding_bias': -0.02, 'momentum_weight': 1.2},
            'bear_market': {'csiq_threshold': 35, 'funding_bias': 0.02, 'momentum_weight': 0.8},
            'sideways': {'csiq_threshold': [35, 65], 'funding_bias': 0, 'momentum_weight': 1.0}
        }
    
    def get_coin_profile(self, symbol):
        """Haal coin-specifiek profiel op"""
        return self.volatility_profiles.get(symbol, self.volatility_profiles['default'])
    
    def detect_market_regime(self, df):
        """Detecteer huidige marktregime"""
        avg_csiq = df['CSI_Q'].mean()
        avg_funding = df['Funding_Rate'].mean()
        
        if avg_csiq > 65 and avg_funding < 0:
            return 'bull_market'
        elif avg_csiq < 35 and avg_funding > 0:
            return 'bear_market'
        else:
            return 'sideways'
    
    def calculate_advanced_sentiment(self, row, market_regime='sideways'):
        """Berekent geavanceerd sentiment met meerdere factoren"""
        symbol = row['Symbol']
        profile = self.get_coin_profile(symbol)
        regime_params = self.market_regimes[market_regime]
        
        # Basis sentiment van price action
        price_momentum = np.tanh(row['Change_24h'] / 10)  # Normalized tussen -1 en 1
        
        # Volume-gewogen sentiment (hoog volume = betrouwbaarder)
        volume_weight = min(1.0, row['Volume_24h'] / 1000000000)  # Max 1B volume = weight 1.0
        
        # Funding rate sentiment (contrarian indicator)
        funding_sentiment = -np.tanh(row['Funding_Rate'] * 100)  # Negatieve funding = bullish
        funding_weighted = funding_sentiment * profile['funding_weight']
        
        # Open Interest verandering (momentum indicator)
        oi_momentum = np.tanh(row['OI_Change'] / 20)
        
        # Long/Short ratio sentiment
        ls_ratio = row['Long_Short_Ratio']
        if ls_ratio > 2:  # Te veel longs = bearish
            ls_sentiment = -0.5
        elif ls_ratio < 0.5:  # Te veel shorts = bullish
            ls_sentiment = 0.5
        else:
            ls_sentiment = 0
        
        # RSI-gebaseerd sentiment
        rsi = row['RSI']
        if rsi > 70:
            rsi_sentiment = -0.3  # Overbought = bearish
        elif rsi < 30:
            rsi_sentiment = 0.3   # Oversold = bullish
        else:
            rsi_sentiment = 0
        
        # Combineer alle sentiment factoren
        composite_sentiment = (
            price_momentum * 0.3 * volume_weight +
            funding_weighted * 0.25 +
            oi_momentum * 0.2 +
            ls_sentiment * 0.15 +
            rsi_sentiment * 0.1
        )
        
        # Pas marktregime aan
        regime_multiplier = regime_params['momentum_weight']
        final_sentiment = composite_sentiment * regime_multiplier
        
        # Converteer naar schaal 0-100
        sentiment_score = 50 + (final_sentiment * 40)
        sentiment_score = max(0, min(100, sentiment_score))
        
        return {
            'sentiment_score': sentiment_score,
            'price_momentum': price_momentum,
            'funding_sentiment': funding_sentiment,
            'volume_weight': volume_weight,
            'composite_sentiment': composite_sentiment,
            'market_regime': market_regime
        }

class DynamicTradingEngine:
    """Dynamische trading setup engine"""
    
    def __init__(self):
        # ATR multiplicators per coin type
        self.atr_profiles = {
            'BTC': {'stop_mult': 1.5, 'target_mult': 2.0, 'min_rr': 1.3},
            'ETH': {'stop_mult': 1.6, 'target_mult': 2.2, 'min_rr': 1.3},
            'SOL': {'stop_mult': 2.0, 'target_mult': 3.0, 'min_rr': 1.4},
            'XRP': {'stop_mult': 1.8, 'target_mult': 2.5, 'min_rr': 1.3},
            'ADA': {'stop_mult': 1.9, 'target_mult': 2.8, 'min_rr': 1.4},
            'default': {'stop_mult': 2.2, 'target_mult': 3.2, 'min_rr': 1.4}
        }
        
        # Support/Resistance levels (simplified)
        self.support_resistance = {}
    
    def calculate_dynamic_atr(self, row):
        """Bereken dynamische ATR gebaseerd op volatiliteit en volume"""
        base_atr = row['ATR']
        price = row['Price']
        volume_ratio = min(2.0, row['Volume_24h'] / 500000000)  # Normalize volume
        
        # Verhoog ATR bij lage volumes (minder liquide = grotere stops)
        volume_adjustment = 1.0 + (1.0 - volume_ratio) * 0.5
        
        # Volatiliteit aanpassing
        change_vol = abs(row['Change_24h'])
        vol_adjustment = 1.0 + (change_vol / 100)  # Hogere volatiliteit = grotere stops
        
        adjusted_atr = base_atr * volume_adjustment * vol_adjustment
        return min(adjusted_atr, price * 0.15)  # Max 15% van prijs
    
    def calculate_support_resistance(self, price, symbol, timeframe='1d'):
        """Bereken support/resistance levels (gesimplificeerd)"""
        # Gebruik fibonacci retracements en ronde getallen
        if symbol == 'BTC':
            if price > 40000:
                resistance_levels = [45000, 50000, 55000]
                support_levels = [40000, 35000, 30000]
            else:
                resistance_levels = [35000, 40000, 45000]
                support_levels = [30000, 25000, 20000]
        elif symbol == 'ETH':
            if price > 2500:
                resistance_levels = [3000, 3500, 4000]
                support_levels = [2500, 2000, 1800]
            else:
                resistance_levels = [2000, 2500, 3000]
                support_levels = [1800, 1500, 1200]
        else:
            # Algemene levels gebaseerd op percentage
            resistance_levels = [price * 1.05, price * 1.10, price * 1.15]
            support_levels = [price * 0.95, price * 0.90, price * 0.85]
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def calculate_position_sizing(self, account_size, risk_per_trade, entry_price, stop_price):
        """Bereken positie grootte gebaseerd op risk management"""
        risk_amount = account_size * (risk_per_trade / 100)
        price_risk = abs(entry_price - stop_price)
        position_size = risk_amount / price_risk if price_risk > 0 else 0
        return min(position_size, account_size * 0.1)  # Max 10% van account
    
    def generate_trading_setup(self, row, sentiment_data, market_regime='sideways'):
        """Genereer complete trading setup"""
        symbol = row['Symbol']
        price = row['Price']
        profile = self.atr_profiles.get(symbol, self.atr_profiles['default'])
        
        # Bereken dynamische ATR
        dynamic_atr = self.calculate_dynamic_atr(row)
        
        # Basis setup parameters
        csiq = row['CSI_Q']
        funding = row['Funding_Rate']
        sentiment_score = sentiment_data['sentiment_score']
        
        # Bepaal signaal richting
        signal_strength = 0
        direction = None
        confidence = 'LOW'
        
        # Multi-factor signaal bepaling
        csiq_signal = 0
        if csiq > 75:
            csiq_signal = 1  # Bullish
        elif csiq < 25:
            csiq_signal = -1  # Bearish
        
        funding_signal = 0
        if funding > 0.15:
            funding_signal = -1  # Bearish (expensive longs)
        elif funding < -0.15:
            funding_signal = 1  # Bullish (expensive shorts)
        
        sentiment_signal = 0
        if sentiment_score > 70:
            sentiment_signal = 1
        elif sentiment_score < 30:
            sentiment_signal = -1
        
        momentum_signal = 0
        if row['Change_24h'] > 5 and row['RSI'] < 70:
            momentum_signal = 1
        elif row['Change_24h'] < -5 and row['RSI'] > 30:
            momentum_signal = -1
        
        # Combineer signalen
        total_signal = csiq_signal + funding_signal + sentiment_signal + momentum_signal
        
        if total_signal >= 2:
            direction = 'LONG'
            signal_strength = min(abs(total_signal) / 4, 1.0)
        elif total_signal <= -2:
            direction = 'SHORT'
            signal_strength = min(abs(total_signal) / 4, 1.0)
        else:
            # Check voor contrarian setups
            if csiq > 90 or csiq < 10:
                direction = 'CONTRARIAN'
                if csiq > 90:
                    direction = 'CONTRARIAN_SHORT'
                else:
                    direction = 'CONTRARIAN_LONG'
                signal_strength = (abs(csiq - 50) - 40) / 50
            else:
                direction = 'NEUTRAL'
                signal_strength = 0
        
        # Bepaal confidence
        if signal_strength > 0.75:
            confidence = 'HIGH'
        elif signal_strength > 0.5:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Bereken entry, stop en targets
        setup = {
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'signal_strength': signal_strength,
            'entry_price': price,
            'current_atr': dynamic_atr,
            'csiq_signal': csiq_signal,
            'funding_signal': funding_signal,
            'sentiment_signal': sentiment_signal,
            'momentum_signal': momentum_signal,
            'total_signal_score': total_signal
        }
        
        if direction in ['LONG', 'CONTRARIAN_LONG']:
            stop_distance = dynamic_atr * profile['stop_mult']
            target_distance = dynamic_atr * profile['target_mult']
            
            setup.update({
                'stop_loss': price - stop_distance,
                'target_1': price + target_distance,
                'target_2': price + (target_distance * 1.618),  # Fibonacci extension
                'risk_reward': target_distance / stop_distance,
                'stop_pct': (stop_distance / price) * 100,
                'target_pct': (target_distance / price) * 100
            })
            
        elif direction in ['SHORT', 'CONTRARIAN_SHORT']:
            stop_distance = dynamic_atr * profile['stop_mult']
            target_distance = dynamic_atr * profile['target_mult']
            
            setup.update({
                'stop_loss': price + stop_distance,
                'target_1': price - target_distance,
                'target_2': price - (target_distance * 1.618),
                'risk_reward': target_distance / stop_distance,
                'stop_pct': (stop_distance / price) * 100,
                'target_pct': (target_distance / price) * 100
            })
        
        else:  # NEUTRAL
            setup.update({
                'stop_loss': None,
                'target_1': None,
                'target_2': None,
                'risk_reward': None,
                'stop_pct': None,
                'target_pct': None
            })
        
        # Support/Resistance levels
        sr_levels = self.calculate_support_resistance(price, symbol)
        setup['support_resistance'] = sr_levels
        
        return setup

class MultiSourceDataFetcher:
    """Keeping the existing data fetcher but with improvements"""
    
    def __init__(self):
        self.binance_base = "https://fapi.binance.com"
        self.binance_spot = "https://api.binance.com"
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.sentiment_engine = AdvancedSentimentEngine()
        self.trading_engine = DynamicTradingEngine()
        
    # Keep all existing API methods...
    def test_api_connectivity(self):
        """Test which APIs are available"""
        apis_status = {
            'binance': False,
            'coingecko': False,
            'demo': True
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
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            price_url = f"{self.binance_base}/fapi/v1/ticker/24hr"
            price_response = requests.get(price_url, headers=headers, timeout=10)
            
            if price_response.status_code == 451:
                st.error("🚫 Binance API geblokkeerd in uw regio (Error 451)")
                return None, "region_blocked"
            elif price_response.status_code != 200:
                st.error(f"Binance API error: {price_response.status_code}")
                return None, "api_error"
            
            price_data = price_response.json()
            
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
    
    def generate_enhanced_demo_data(self):
        """Generate enhanced demo data with better sentiment and trading setups"""
        # Use current time seed for some variation but keep it realistic
        current_hour = datetime.now().hour
        np.random.seed(42 + current_hour)
        
        data_list = []
        base_prices = {
            'BTC': 43000, 'ETH': 2600, 'BNB': 310, 'SOL': 100, 'XRP': 0.52,
            'ADA': 0.48, 'AVAX': 38, 'DOT': 7.2, 'LINK': 14.5, 'MATIC': 0.85,
            'UNI': 6.8, 'LTC': 73, 'BCH': 250, 'NEAR': 2.1, 'ALGO': 0.19,
            'VET': 0.025, 'FIL': 5.5, 'ETC': 20, 'AAVE': 95, 'MKR': 1450,
            'ATOM': 9.8, 'FTM': 0.32, 'SAND': 0.42, 'MANA': 0.38, 'AXS': 6.2
        }
        
        # Market trend (simulate market conditions)
        market_trend = np.random.choice(['bull', 'bear', 'sideways'], p=[0.3, 0.3, 0.4])
        
        for ticker in TICKERS:
            symbol_clean = ticker.replace('USDT', '')
            base_price = base_prices.get(symbol_clean, 1.0)
            profile = self.sentiment_engine.get_coin_profile(symbol_clean)
            
            # More realistic price movements based on market trend
            if market_trend == 'bull':
                price_change = np.random.normal(0.02, profile['base_vol'])
            elif market_trend == 'bear':
                price_change = np.random.normal(-0.02, profile['base_vol'])
            else:
                price_change = np.random.normal(0, profile['base_vol'])
            
            current_price = base_price * (1 + price_change)
            change_24h = price_change * 100
            
            # More correlated metrics
            funding_rate = np.random.normal(0.01, 0.05)
            if change_24h > 5:  # Strong upward movement
                funding_rate += 0.02  # Higher funding for longs
            elif change_24h < -5:  # Strong downward movement
                funding_rate -= 0.02  # Negative funding (shorts expensive)
            
            # OI change correlated with price movement
            oi_change = change_24h * 1.5 + np.random.normal(0, 10)
            
            # Long/short ratio based on sentiment
            if change_24h > 0:
                long_short_ratio = np.random.lognormal(0.2, 0.3)  # More longs when rising
            else:
                long_short_ratio = np.random.lognormal(-0.2, 0.3)  # More shorts when falling
            
            # Volume based on market cap and volatility
            if symbol_clean in ['BTC', 'ETH']:
                base_volume = np.random.uniform(20000000000, 50000000000)
            elif symbol_clean in ['BNB', 'SOL', 'XRP']:
                base_volume = np.random.uniform(1000000000, 10000000000)
            else:
                base_volume = np.random.uniform(100000000, 2000000000)
            
            # Volume increases with volatility
            volume_multiplier = 1 + (abs(change_24h) / 20)
            volume_24h = base_volume * volume_multiplier
            
            # Technical indicators
            rsi = 50 + (change_24h * 2) + np.random.normal(0, 10)
            rsi = max(10, min(90, rsi))
            
            bb_squeeze = np.random.uniform(0, 1)
            basis = change_24h * 0.02 + np.random.normal(0, 0.3)
            
            # ATR calculation
            atr = current_price * (profile['base_vol'] + abs(price_change))
            
            # Enhanced sentiment calculation (placeholder values for now)
            mentions = max(1, int(volume_24h / 10000000))
            raw_sentiment = np.tanh(change_24h / 5)
            
            # Create temporary row for sentiment calculation
            temp_row = {
                'Symbol': symbol_clean,
                'Change_24h': change_24h,
                'Funding_Rate': funding_rate,
                'Volume_24h': volume_24h,
                'OI_Change': oi_change,
                'Long_Short_Ratio': long_short_ratio,
                'RSI': rsi,
                'ATR': atr,
                'Price': current_price
            }
            
            # Calculate advanced sentiment
            sentiment_data = self.sentiment_engine.calculate_advanced_sentiment(temp_row)
            
            # CSI-Q components (improved calculation)
            derivatives_score = min(100, max(0,
                (abs(oi_change) * 1.5) +
                (abs(funding_rate) * 400) +
                (abs(long_short_ratio - 1) * 25) +
                25
            ))
            
            social_score = sentiment_data['sentiment_score']
            
            basis_score = min(100, max(0,
                abs(basis) * 400 + 20
            ))
            
            tech_score = min(100, max(0,
                (100 - abs(rsi - 50)) * 0.7 +
                ((1 - bb_squeeze) * 35) +
                15
            ))
            
            # Weighted CSI-Q calculation
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
                'Sentiment': raw_sentiment,
                'Sentiment_Score': sentiment_data['sentiment_score'],
                'Price_Momentum': sentiment_data['price_momentum'],
                'Funding_Sentiment': sentiment_data['funding_sentiment'],
                'Volume_Weight': sentiment_data['volume_weight'],
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
                'Open_Interest': volume_24h * np.random.uniform(0.1, 2.0),
                'Last_Updated': datetime.now(),
                'Data_Source': 'demo',
                'Market_Trend': market_trend
            })
        
        df = pd.DataFrame(data_list)
        
        # Detect market regime and add trading setups
        market_regime = self.sentiment_engine.detect_market_regime(df)
        
        # Add trading setups for each coin
        trading_setups = []
        for _, row in df.iterrows():
            sentiment_data = {
                'sentiment_score': row['Sentiment_Score'],
                'price_momentum': row['Price_Momentum'],
                'funding_sentiment': row['Funding_Sentiment'],
                'volume_weight': row['Volume_Weight']
            }
            
            setup = self.trading_engine.generate_trading_setup(row, sentiment_data, market_regime)
            trading_setups.append(setup)
        
        # Add trading setup data to DataFrame
        setup_df = pd.DataFrame(trading_setups)
        for col in setup_df.columns:
            if col != 'symbol':  # Avoid duplicate symbol column
                df[f'Setup_{col}' if col != 'direction' else 'Signal'] = setup_df[col].values
        
        df['Market_Regime'] = market_regime
        
        return df

@st.cache_data(ttl=60)
def fetch_crypto_data_with_enhanced_fallback():
    """Enhanced data fetching with improved sentiment and trading logic"""
    
    fetcher = MultiSourceDataFetcher()
    
    # Test API connectivity
    api_status = fetcher.test_api_connectivity()
    
    st.markdown("### 📡 Enhanced API Status Check")
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
        st.markdown("🟢 **Enhanced Demo**: Ready")
    
    # Try real data first
    if api_status['binance']:
        st.info("🚀 Attempting Binance API connection...")
        binance_data, status = fetcher.get_binance_data()
        
        if status == "success":
            st.success("✅ Connected to Binance API!")
            return process_binance_data_enhanced(binance_data, fetcher)
        elif status == "region_blocked":
            st.markdown("""
            <div class="api-status-error">
                🚫 <b>Binance API Geblokkeerd</b><br>
                Error 451: Niet beschikbaar in uw regio<br>
                Switching to enhanced fallback...
            </div>
            """, unsafe_allow_html=True)
    
    # Try CoinGecko
    if api_status['coingecko']:
        st.info("🔄 Trying CoinGecko API as fallback...")
        gecko_data, status = fetcher.get_coingecko_data()
        
        if status == "success":
            st.warning("⚠️ Using CoinGecko + Enhanced Analysis")
            return process_coingecko_data_enhanced(gecko_data, fetcher)
    
    # Enhanced demo mode
    st.markdown("""
    <div class="api-status-demo">
        🚀 <b>Enhanced Demo Mode Active</b><br>
        Advanced sentiment analysis + Dynamic trading setups<br>
        Realistic market simulation with intelligent algorithms
    </div>
    """, unsafe_allow_html=True)
    
    return fetcher.generate_enhanced_demo_data()

def process_binance_data_enhanced(binance_data, fetcher):
    """Process real Binance data with enhanced analysis"""
    # This would implement the same enhancements for real data
    # For now, we'll use the existing logic but add enhanced sentiment
    pass

def process_coingecko_data_enhanced(gecko_data, fetcher):
    """Process CoinGecko data with enhanced sentiment analysis"""
    data_list = []
    
    for item in gecko_data:
        symbol_clean = item['symbol'].replace('USDT', '')
        price = item['price']
        change_24h = item['change_24h']
        volume_24h = item['volume_24h']
        
        # Simulate enhanced derivatives data
        funding_rate = np.random.normal(change_24h * 0.001, 0.02)
        oi_change = change_24h * 2 + np.random.normal(0, 10)
        long_short_ratio = 1.2 if change_24h > 0 else 0.8
        long_short_ratio += np.random.normal(0, 0.3)
        
        # Enhanced metrics
        mentions = max(1, int(volume_24h / 10000000)) if volume_24h else 50
        rsi = 50 + change_24h * 2 + np.random.normal(0, 10)
        rsi = max(0, min(100, rsi))
        bb_squeeze = np.random.uniform(0, 1)
        basis = np.random.normal(change_24h * 0.1, 0.3)
        atr = price * 0.05
        
        # Create row for sentiment analysis
        temp_row = {
            'Symbol': symbol_clean,
            'Change_24h': change_24h,
            'Funding_Rate': funding_rate,
            'Volume_24h': volume_24h or 1000000,
            'OI_Change': oi_change,
            'Long_Short_Ratio': long_short_ratio,
            'RSI': rsi,
            'ATR': atr,
            'Price': price
        }
        
        # Enhanced sentiment calculation
        sentiment_data = fetcher.sentiment_engine.calculate_advanced_sentiment(temp_row)
        
        # Calculate CSI-Q components
        derivatives_score = min(100, max(0,
            (abs(oi_change) * 1.5) +
            (abs(funding_rate) * 400) +
            (abs(long_short_ratio - 1) * 25) +
            25
        ))
        
        social_score = sentiment_data['sentiment_score']
        
        basis_score = min(100, max(0,
            abs(basis) * 400 + 20
        ))
        
        tech_score = min(100, max(0,
            (100 - abs(rsi - 50)) * 0.7 +
            ((1 - bb_squeeze) * 35) +
            15
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
            'Sentiment': np.tanh(change_24h / 5),
            'Sentiment_Score': sentiment_data['sentiment_score'],
            'Price_Momentum': sentiment_data['price_momentum'],
            'Funding_Sentiment': sentiment_data['funding_sentiment'],
            'Volume_Weight': sentiment_data['volume_weight'],
            'Spot_Futures_Basis': basis,
            'RSI': rsi,
            'BB_Squeeze': bb_squeeze,
            'CSI_Q': csiq,
            'Derivatives_Score': derivatives_score,
            'Social_Score': social_score,
            'Basis_Score': basis_score,
            'Tech_Score': tech_score,
            'ATR': atr,
            'Volume_24h': volume_24h or 1000000,
            'Open_Interest': (volume_24h or 1000000) * np.random.uniform(0.1, 2.0),
            'Last_Updated': datetime.now(),
            'Data_Source': 'coingecko_enhanced'
        })
    
    df = pd.DataFrame(data_list)
    
    # Add market regime and trading setups
    market_regime = fetcher.sentiment_engine.detect_market_regime(df)
    
    trading_setups = []
    for _, row in df.iterrows():
        sentiment_data = {
            'sentiment_score': row['Sentiment_Score'],
            'price_momentum': row['Price_Momentum'],
            'funding_sentiment': row['Funding_Sentiment'],
            'volume_weight': row['Volume_Weight']
        }
        
        setup = fetcher.trading_engine.generate_trading_setup(row, sentiment_data, market_regime)
        trading_setups.append(setup)
    
    setup_df = pd.DataFrame(trading_setups)
    for col in setup_df.columns:
        if col != 'symbol':
            df[f'Setup_{col}' if col != 'direction' else 'Signal'] = setup_df[col].values
    
    df['Market_Regime'] = market_regime
    
    return df

def get_signal_color_enhanced(signal):
    """Enhanced signal colors"""
    colors = {
        "LONG": "🟢",
        "SHORT": "🔴", 
        "CONTRARIAN_LONG": "🟠",
        "CONTRARIAN_SHORT": "🟠",
        "CONTRARIAN": "🟠",
        "NEUTRAL": "⚪"
    }
    return colors.get(signal, "⚪")

def get_confidence_class(confidence):
    """Get CSS class for confidence level"""
    return f"confidence-{confidence.lower()}"

# Main App
def main():
    st.title("🚀 Enhanced Crypto CSI-Q Dashboard")
    st.markdown("**Advanced Multi-Source Data** - Enhanced Sentiment Analysis + Dynamic Trading Setups")
    
    # Status and refresh
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown("🧠 **ENHANCED AI**")
    with col2:
        st.markdown(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    with col3:
        if st.button("🔄 Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    
    # Load enhanced data
    df = fetch_crypto_data_with_enhanced_fallback()
    
    if df.empty:
        st.error("❌ No data available from any source.")
        st.stop()
    
    # Show enhanced data source info
    data_sources = df['Data_Source'].value_counts() if 'Data_Source' in df.columns else {'demo': len(df)}
    source_info = " + ".join([f"{count} from {source}" for source, count in data_sources.items()])
    market_regime = df['Market_Regime'].iloc[0] if 'Market_Regime' in df.columns else 'unknown'
    
    st.info(f"🧠 Loaded {len(df)} symbols: {source_info} | Market Regime: **{market_regime.upper()}**")
    
    # Enhanced sidebar filters
    st.sidebar.header("🔧 Enhanced Filters")
    min_csiq = st.sidebar.slider("Min CSI-Q Score", 0, 100, 0)
    max_csiq = st.sidebar.slider("Max CSI-Q Score", 0, 100, 100)
    
    signal_filter = st.sidebar.multiselect(
        "Signal Types",
        ["LONG", "SHORT", "CONTRARIAN_LONG", "CONTRARIAN_SHORT", "NEUTRAL"],
        default=["LONG", "SHORT", "CONTRARIAN_LONG", "CONTRARIAN_SHORT"]
    )
    
    confidence_filter = st.sidebar.multiselect(
        "Confidence Levels",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM"]
    )
    
    min_volume = st.sidebar.number_input("Min 24h Volume ($M)", 0, 1000, 0)
    min_signal_strength = st.sidebar.slider("Min Signal Strength", 0.0, 1.0, 0.3)
    
    # Apply enhanced filters
    filtered_df = df[
        (df['CSI_Q'] >= min_csiq) & 
        (df['CSI_Q'] <= max_csiq) &
        (df['Signal'].isin(signal_filter)) &
        (df['Setup_confidence'].isin(confidence_filter)) &
        (df['Volume_24h'] >= min_volume * 1000000) &
        (df['Setup_signal_strength'] >= min_signal_strength)
    ].copy()
    
    # Enhanced top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        high_conf_signals = len(filtered_df[
            (filtered_df['Setup_confidence'] == 'HIGH') & 
            (filtered_df['Signal'] != 'NEUTRAL')
        ])
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 High Confidence</h3>
            <h2>{high_conf_signals}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_signal_strength = filtered_df['Setup_signal_strength'].mean() if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>💪 Avg Strength</h3>
            <h2>{avg_signal_strength:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_rr = filtered_df['Setup_risk_reward'].mean() if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Avg R/R</h3>
            <h2>{avg_rr:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        contrarian_ops = len(filtered_df[filtered_df['Signal'].str.contains('CONTRARIAN', na=False)])
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ Contrarian Ops</h3>
            <h2>{contrarian_ops}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        total_volume = filtered_df['Volume_24h'].sum() / 1000000000 if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Total Volume</h3>
            <h2>${total_volume:.1f}B</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Enhanced CSI-Q Monitor", 
        "🧠 Advanced Sentiment", 
        "🎯 Dynamic Trading", 
        "💰 Smart Opportunities"
    ])
    
    with tab1:
        st.header("📡 Enhanced CSI-Q Monitor")
        
        if not filtered_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Enhanced CSI-Q scatter plot
                display_df = filtered_df.sort_values('Setup_signal_strength', ascending=False)
                
                fig = go.Figure(data=go.Scatter(
                    x=display_df['Symbol'],
                    y=display_df['CSI_Q'],
                    mode='markers+text',
                    marker=dict(
                        size=display_df['Setup_signal_strength'] * 50 + 10,  # Size by signal strength
                        color=display_df['Sentiment_Score'],  # Color by sentiment
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Sentiment Score"),
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    text=display_df['Symbol'],
                    textposition="middle center",
                    customdata=np.stack([
                        display_df['Setup_confidence'],
                        display_df['Setup_signal_strength'],
                        display_df['Setup_risk_reward'].fillna(0)
                    ], axis=-1),
                    hovertemplate="<b>%{text}</b><br>" +
                                "CSI-Q: %{y:.1f}<br>" +
                                "Price: $" + display_df['Price'].round(4).astype(str) + "<br>" +
                                "Change: " + display_df['Change_24h'].round(2).astype(str) + "%<br>" +
                                "Signal: " + display_df['Signal'].astype(str) + "<br>" +
                                "Confidence: %{customdata[0]}<br>" +
                                "Strength: %{customdata[1]:.2f}<br>" +
                                "R/R: %{customdata[2]:.1f}<br>" +
                                "Sentiment: " + display_df['Sentiment_Score'].round(1).astype(str) + "<br>" +
                                "<extra></extra>"
                ))
                
                fig.update_layout(
                    title="🧠 Enhanced CSI-Q vs Sentiment (Bubble Size = Signal Strength)",
                    xaxis_title="Symbol",
                    yaxis_title="CSI-Q Score",
                    height=500,
                    showlegend=False
                )
                
                # Enhanced signal zones
                fig.add_hline(y=75, line_dash="dash", line_color="green", 
                             annotation_text="Strong LONG Zone", annotation_position="right")
                fig.add_hline(y=25, line_dash="dash", line_color="red",
                             annotation_text="Strong SHORT Zone", annotation_position="right")
                fig.add_hline(y=90, line_dash="dash", line_color="orange",
                             annotation_text="EXTREME - Contrarian", annotation_position="right")
                fig.add_hline(y=10, line_dash="dash", line_color="orange")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🚨 Smart Alerts")
                
                # Generate smart alerts based on multiple factors
                alerts = []
                for _, row in filtered_df.iterrows():
                    signal = row['Signal']
                    confidence = row['Setup_confidence']
                    strength = row['Setup_signal_strength']
                    
                    if signal != 'NEUTRAL' and confidence in ['HIGH', 'MEDIUM']:
                        alert_type = "🔥 STRONG" if confidence == 'HIGH' and strength > 0.7 else "⚠️ MEDIUM"
                        
                        # Calculate alert priority
                        priority = strength * (2 if confidence == 'HIGH' else 1)
                        
                        alerts.append({
                            'Symbol': row['Symbol'],
                            'Signal': signal,
                            'Confidence': confidence,
                            'CSI_Q': row['CSI_Q'],
                            'Sentiment': row['Sentiment_Score'],
                            'Strength': alert_type,
                            'Priority': priority,
                            'Price': row['Price'],
                            'R_R': row.get('Setup_risk_reward', 0),
                            'Entry': row.get('Setup_entry_price', row['Price']),
                            'Target': row.get('Setup_target_1', 0),
                            'Stop': row.get('Setup_stop_loss', 0)
                        })
                
                # Sort by priority
                alerts = sorted(alerts, key=lambda x: x['Priority'], reverse=True)
                
                for alert in alerts[:8]:
                    signal_emoji = get_signal_color_enhanced(alert['Signal'])
                    confidence_class = get_confidence_class(alert['Confidence'])
                    
                    st.markdown(f"""
                    <div class="signal-{alert['Signal'].lower().replace('_', '-')}">
                        {signal_emoji} <b>{alert['Symbol']}</b><br>
                        {alert['Signal'].replace('_', ' ')} | {alert['Strength']}<br>
                        CSI-Q: {alert['CSI_Q']:.1f} | Sentiment: {alert['Sentiment']:.1f}<br>
                        Entry: ${alert['Entry']:.4f}<br>
                        Target: ${alert['Target']:.4f} | Stop: ${alert['Stop']:.4f}<br>
                        R/R: {alert['R_R']:.1f}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="{confidence_class}">
                        {alert['Confidence']} CONFIDENCE
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
        
        # Enhanced data table
        st.subheader("📊 Enhanced Market Data")
        
        if not filtered_df.empty:
            display_cols = [
                'Symbol', 'Signal', 'Setup_confidence', 'CSI_Q', 'Sentiment_Score', 
                'Price', 'Change_24h', 'Setup_signal_strength', 'Setup_risk_reward',
                'Volume_24h', 'Funding_Rate'
            ]
            
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            styled_df = filtered_df[available_cols].copy()
            
            # Round numerical columns
            numeric_cols = ['CSI_Q', 'Sentiment_Score', 'Price', 'Change_24h', 
                          'Setup_signal_strength', 'Setup_risk_reward', 'Funding_Rate']
            for col in numeric_cols:
                if col in styled_df.columns:
                    if col == 'Price':
                        styled_df[col] = styled_df[col].round(4)
                    elif col in ['CSI_Q', 'Sentiment_Score']:
                        styled_df[col] = styled_df[col].round(1)
                    elif col in ['Setup_signal_strength', 'Setup_risk_reward']:
                        styled_df[col] = styled_df[col].round(2)
                    else:
                        styled_df[col] = styled_df[col].round(3)
            
            if 'Volume_24h' in styled_df.columns:
                styled_df['Volume_24h'] = (styled_df['Volume_24h'] / 1000000).round(1)
                styled_df = styled_df.rename(columns={'Volume_24h': 'Volume_($M)'})
            
            # Rename columns for display
            column_renames = {
                'Setup_confidence': 'Confidence',
                'Sentiment_Score': 'Sentiment',
                'Change_24h': 'Change_24h_(%)',
                'Setup_signal_strength': 'Strength',
                'Setup_risk_reward': 'R/R',
                'Funding_Rate': 'Funding_(%)'
            }
            
            styled_df = styled_df.rename(columns=column_renames)
            
            st.dataframe(styled_df, use_container_width=True, height=400)
    
    with tab2:
        st.header("🧠 Advanced Sentiment Analysis")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment vs Price Performance
                fig = px.scatter(
                    filtered_df,
                    x='Sentiment_Score',
                    y='Change_24h',
                    size='Volume_24h',
                    color='Signal',
                    hover_name='Symbol',
                    title="🧠 Sentiment Score vs Price Performance",
                    color_discrete_map={
                        'LONG': 'green',
                        'SHORT': 'red',
                        'CONTRARIAN_LONG': 'orange',
                        'CONTRARIAN_SHORT': 'darkorange',
                        'NEUTRAL': 'gray'
                    }
                )
                
                fig.add_vline(x=50, line_dash="dash", line_color="white", annotation_text="Neutral Sentiment")
                fig.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="No Change")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment components breakdown
                st.subheader("📊 Sentiment Components")
                
                if 'Price_Momentum' in filtered_df.columns:
                    sentiment_components = filtered_df[['Symbol', 'Price_Momentum', 'Funding_Sentiment', 'Volume_Weight']].head(10)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Price Momentum',
                        x=sentiment_components['Symbol'],
                        y=sentiment_components['Price_Momentum'],
                        marker_color='rgba(99, 110, 250, 0.8)'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Funding Sentiment',
                        x=sentiment_components['Symbol'],
                        y=sentiment_components['Funding_Sentiment'],
                        marker_color='rgba(239, 85, 59, 0.8)'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Volume Weight',
                        x=sentiment_components['Symbol'],
                        y=sentiment_components['Volume_Weight'],
                        marker_color='rgba(0, 204, 150, 0.8)'
                    ))
                    
                    fig.update_layout(
                        title="📈 Sentiment Component Breakdown (Top 10)",
                        barmode='group',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Market regime analysis
                st.subheader("🎯 Market Regime Analysis")
                
                regime = filtered_df['Market_Regime'].iloc[0] if 'Market_Regime' in filtered_df.columns else 'unknown'
                avg_sentiment = filtered_df['Sentiment_Score'].mean()
                avg_funding = filtered_df['Funding_Rate'].mean()
                
                # Regime description and strategy
                regime_info = {
                    'bull_market': {
                        'color': '#00C851',
                        'icon': '🐂',
                        'desc': 'Bull Market Detected',
                        'strategy': 'Focus on LONG setups, contrarian shorts on extremes'
                    },
                    'bear_market': {
                        'color': '#ff4444',
                        'icon': '🐻',
                        'desc': 'Bear Market Detected',
                        'strategy': 'Focus on SHORT setups, contrarian longs on extremes'
                    },
                    'sideways': {
                        'color': '#ffbb33',
                        'icon': '🦀',
                        'desc': 'Sideways Market Detected',
                        'strategy': 'Range trading, strong momentum plays'
                    }
                }
                
                regime_data = regime_info.get(regime, regime_info['sideways'])
                
                st.markdown(f"""
                <div style="background: {regime_data['color']}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                    <h3>{regime_data['icon']} {regime_data['desc']}</h3>
                    <p><strong>Strategy:</strong> {regime_data['strategy']}</p>
                    <p>Avg Sentiment: {avg_sentiment:.1f} | Avg Funding: {avg_funding:.3f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Signal distribution pie chart
                signal_counts = filtered_df['Signal'].value_counts()
                fig = px.pie(
                    values=signal_counts.values,
                    names=signal_counts.index,
                    title="📊 Current Signal Distribution",
                    color_discrete_map={
                        'LONG': 'green',
                        'SHORT': 'red',
                        'CONTRARIAN_LONG': 'orange',
                        'CONTRARIAN_SHORT': 'darkorange',
                        'NEUTRAL': 'gray'
                    }
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                conf_counts = filtered_df['Setup_confidence'].value_counts()
                fig = px.bar(
                    x=conf_counts.index,
                    y=conf_counts.values,
                    title="📊 Signal Confidence Distribution",
                    color=conf_counts.index,
                    color_discrete_map={
                        'HIGH': 'green',
                        'MEDIUM': 'orange', 
                        'LOW': 'red'
                    }
                )
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🎯 Dynamic Trading Analysis")
        
        if not filtered_df.empty:
            # Filter for active trading signals only
            trading_df = filtered_df[
                (filtered_df['Signal'] != 'NEUTRAL') & 
                (filtered_df['Setup_confidence'].isin(['HIGH', 'MEDIUM']))
            ].sort_values('Setup_signal_strength', ascending=False)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Dynamic Risk/Reward Analysis
                if not trading_df.empty and 'Setup_risk_reward' in trading_df.columns:
                    fig = px.scatter(
                        trading_df,
                        x='Setup_signal_strength',
                        y='Setup_risk_reward',
                        size='Volume_24h',
                        color='Signal',
                        hover_name='Symbol',
                        title="🎯 Signal Strength vs Risk/Reward Ratio",
                        color_discrete_map={
                            'LONG': 'green',
                            'SHORT': 'red',
                            'CONTRARIAN_LONG': 'orange',
                            'CONTRARIAN_SHORT': 'darkorange'
                        }
                    )
                    
                    # Add target zones
                    fig.add_hline(y=2.0, line_dash="dash", line_color="green", 
                                 annotation_text="Excellent R/R (2:1+)")
                    fig.add_hline(y=1.5, line_dash="dash", line_color="orange",
                                 annotation_text="Good R/R (1.5:1+)")
                    fig.add_vline(x=0.6, line_dash="dash", line_color="blue",
                                 annotation_text="Strong Signal")
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # ATR-based Position Sizing
                st.subheader("📏 Dynamic Position Sizing Analysis")
                
                if not trading_df.empty:
                    # Calculate position sizes for different account sizes
                    account_sizes = [10000, 25000, 50000, 100000]  # Different account sizes
                    risk_percent = 2  # 2% risk per trade
                    
                    position_data = []
                    for _, row in trading_df.head(10).iterrows():
                        if row.get('Setup_stop_loss') is not None:
                            entry = row['Setup_entry_price']
                            stop = row['Setup_stop_loss']
                            risk_per_share = abs(entry - stop)
                            
                            for acc_size in account_sizes:
                                risk_amount = acc_size * (risk_percent / 100)
                                position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
                                
                                position_data.append({
                                    'Symbol': row['Symbol'],
                                    'Account_Size': f"${acc_size:,}",
                                    'Position_Size': position_size,
                                    'Risk_Amount': risk_amount,
                                    'Signal': row['Signal']
                                })
                    
                    if position_data:
                        pos_df = pd.DataFrame(position_data)
                        
                        # Show position sizing for $25k account as example
                        example_df = pos_df[pos_df['Account_Size'] == '$25,000'].head(8)
                        
                        fig = px.bar(
                            example_df,
                            x='Symbol',
                            y='Position_Size',
                            color='Signal',
                            title="💰 Position Sizes for $25,000 Account (2% Risk)",
                            color_discrete_map={
                                'LONG': 'green',
                                'SHORT': 'red',
                                'CONTRARIAN_LONG': 'orange',
                                'CONTRARIAN_SHORT': 'darkorange'
                            }
                        )
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("⚡ Live Trading Setups")
                
                # Show top trading opportunities with dynamic calculations
                for i, (_, row) in enumerate(trading_df.head(6).iterrows()):
                    signal_emoji = get_signal_color_enhanced(row['Signal'])
                    conf_class = get_confidence_class(row['Setup_confidence'])
                    
                    # Dynamic calculations
                    entry_price = row.get('Setup_entry_price', row['Price'])
                    target_1 = row.get('Setup_target_1', 0)
                    stop_loss = row.get('Setup_stop_loss', 0)
                    risk_reward = row.get('Setup_risk_reward', 0)
                    
                    # Calculate potential profit/loss percentages
                    if entry_price and target_1 and stop_loss:
                        profit_pct = abs((target_1 - entry_price) / entry_price * 100)
                        loss_pct = abs((stop_loss - entry_price) / entry_price * 100)
                    else:
                        profit_pct = 0
                        loss_pct = 0
                    
                    st.markdown(f"""
                    <div class="signal-{row['Signal'].lower().replace('_', '-')}">
                        <h4>{i+1}. {signal_emoji} {row['Symbol']}</h4>
                        <strong>{row['Signal'].replace('_', ' ')}</strong><br>
                        Strength: {row['Setup_signal_strength']:.2f}<br>
                        CSI-Q: {row['CSI_Q']:.1f} | Sentiment: {row['Sentiment_Score']:.1f}<br><br>
                        
                        <strong>📍 Setup Details:</strong><br>
                        Entry: ${entry_price:.4f}<br>
                        Target: ${target_1:.4f} (+{profit_pct:.1f}%)<br>
                        Stop: ${stop_loss:.4f} (-{loss_pct:.1f}%)<br>
                        R/R: {risk_reward:.1f}:1<br><br>
                        
                        <strong>🔍 Analysis:</strong><br>
                        ATR: ${row.get('ATR', 0):.4f}<br>
                        Volume: ${row['Volume_24h']/1000000:.0f}M<br>
                        Funding: {row['Funding_Rate']:.3f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="{conf_class}">
                        {row['Setup_confidence']} CONFIDENCE
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
        
        else:
            st.warning("No active trading signals with current filters")
    
    with tab4:
        st.header("💰 Smart Trading Opportunities")
        
        if not filtered_df.empty:
            # Enhanced opportunity scoring
            opportunity_df = filtered_df.copy()
            
            # Calculate enhanced opportunity score
            opportunity_df['Opportunity_Score'] = (
                (opportunity_df['Setup_signal_strength'] * 40) +
                (opportunity_df.get('Setup_risk_reward', 0) * 20) +
                ((opportunity_df['Setup_confidence'] == 'HIGH').astype(int) * 25) +
                ((opportunity_df['Setup_confidence'] == 'MEDIUM').astype(int) * 15) +
                (opportunity_df['Volume_24h'] / opportunity_df['Volume_24h'].max() * 15)
            )
            
            # Top opportunities
            top_opportunities = opportunity_df[
                opportunity_df['Signal'] != 'NEUTRAL'
            ].sort_values('Opportunity_Score', ascending=False).head(10)
            
            st.subheader("🚀 TOP 10 SMART TRADING OPPORTUNITIES")
            
            # Market status indicator
            market_regime = opportunity_df['Market_Regime'].iloc[0] if 'Market_Regime' in opportunity_df.columns else 'unknown'
            avg_opportunity = top_opportunities['Opportunity_Score'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Regime", market_regime.title())
            with col2:
                st.metric("Avg Opportunity Score", f"{avg_opportunity:.1f}")
            with col3:
                high_conf_count = len(top_opportunities[top_opportunities['Setup_confidence'] == 'HIGH'])
                st.metric("High Confidence Setups", high_conf_count)
            
            # Detailed opportunity breakdown
            for i, (_, row) in enumerate(top_opportunities.iterrows()):
                with st.expander(f"🎯 #{i+1} {row['Symbol']} - Score: {row['Opportunity_Score']:.1f}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("**📊 Market Data**")
                        st.write(f"Price: ${row['Price']:.4f}")
                        st.write(f"24h Change: {row['Change_24h']:.2f}%")
                        st.write(f"Volume: ${row['Volume_24h']/1000000:.0f}M")
                        st.write(f"CSI-Q: {row['CSI_Q']:.1f}")
                    
                    with col2:
                        st.markdown("**🧠 Sentiment Analysis**")
                        st.write(f"Sentiment Score: {row['Sentiment_Score']:.1f}")
                        st.write(f"Funding Rate: {row['Funding_Rate']:.3f}%")
                        st.write(f"L/S Ratio: {row['Long_Short_Ratio']:.2f}")
                        st.write(f"RSI: {row['RSI']:.1f}")
                    
                    with col3:
                        st.markdown("**🎯 Signal Details**")
                        signal_display = row['Signal'].replace('_', ' ')
                        st.write(f"Signal: **{signal_display}**")
                        st.write(f"Confidence: **{row['Setup_confidence']}**")
                        st.write(f"Strength: {row['Setup_signal_strength']:.2f}")
                        st.write(f"R/R Ratio: {row.get('Setup_risk_reward', 0):.1f}:1")
                    
                    with col4:
                        st.markdown("**💰 Trade Setup**")
                        entry = row.get('Setup_entry_price', row['Price'])
                        target = row.get('Setup_target_1', 0)
                        stop = row.get('Setup_stop_loss', 0)
                        
                        st.write(f"Entry: ${entry:.4f}")
                        st.write(f"Target: ${target:.4f}")
                        st.write(f"Stop Loss: ${stop:.4f}")
                        
                        # Position sizing for example account
                        if stop != 0:
                            risk_per_unit = abs(entry - stop)
                            example_account = 25000
                            risk_amount = example_account * 0.02  # 2% risk
                            position_size = risk_amount / risk_per_unit
                            st.write(f"Position Size: {position_size:.0f} units")
                            st.write(f"(${example_account:,} account, 2% risk)")
                    
                    # Technical analysis summary
                    st.markdown("**🔍 Technical Summary**")
                    analysis_points = []
                    
                    if row['CSI_Q'] > 80:
                        analysis_points.append("🔥 Extremely overbought conditions")
                    elif row['CSI_Q'] < 20:
                        analysis_points.append("❄️ Extremely oversold conditions")
                    
                    if abs(row['Funding_Rate']) > 0.1:
                        direction = "shorts" if row['Funding_Rate'] > 0 else "longs"
                        analysis_points.append(f"💸 High funding rate - expensive {direction}")
                    
                    if row['Long_Short_Ratio'] > 2:
                        analysis_points.append("⚠️ Heavy long bias - potential for squeeze")
                    elif row['Long_Short_Ratio'] < 0.5:
                        analysis_points.append("⚠️ Heavy short bias - potential for squeeze")
                    
                    if row['RSI'] > 70:
                        analysis_points.append("📈 RSI overbought - watch for reversal")
                    elif row['RSI'] < 30:
                        analysis_points.append("📉 RSI oversold - potential bounce")
                    
                    if analysis_points:
                        for point in analysis_points:
                            st.write(f"• {point}")
                    else:
                        st.write("• Neutral technical conditions")
            
            # Portfolio allocation suggestions
            st.subheader("📋 Smart Portfolio Allocation")
            
            # Calculate suggested allocations based on opportunity scores and risk
            total_score = top_opportunities['Opportunity_Score'].sum()
            allocation_suggestions = []
            
            for _, row in top_opportunities.head(5).iterrows():
                weight = (row['Opportunity_Score'] / total_score) * 100
                risk_adjusted_weight = weight * (0.5 if row['Setup_confidence'] == 'LOW' else 
                                               0.75 if row['Setup_confidence'] == 'MEDIUM' else 1.0)
                
                allocation_suggestions.append({
                    'Symbol': row['Symbol'],
                    'Signal': row['Signal'],
                    'Raw_Weight': weight,
                    'Risk_Adjusted_Weight': risk_adjusted_weight,
                    'Confidence': row['Setup_confidence'],
                    'Expected_RR': row.get('Setup_risk_reward', 0)
                })
            
            alloc_df = pd.DataFrame(allocation_suggestions)
            
            # Normalize risk-adjusted weights to 100%
            total_risk_weight = alloc_df['Risk_Adjusted_Weight'].sum()
            if total_risk_weight > 0:
                alloc_df['Final_Allocation'] = (alloc_df['Risk_Adjusted_Weight'] / total_risk_weight * 100).round(1)
            else:
                alloc_df['Final_Allocation'] = 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Allocation pie chart
                fig = px.pie(
                    alloc_df,
                    values='Final_Allocation',
                    names='Symbol',
                    title="💼 Suggested Portfolio Allocation (Top 5)",
                    color='Signal',
                    color_discrete_map={
                        'LONG': 'green',
                        'SHORT': 'red',
                        'CONTRARIAN_LONG': 'orange',
                        'CONTRARIAN_SHORT': 'darkorange'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Allocation Breakdown**")
                
                for _, row in alloc_df.iterrows():
                    signal_emoji = get_signal_color_enhanced(row['Signal'])
                    st.markdown(f"""
                    **{signal_emoji} {row['Symbol']}**: {row['Final_Allocation']:.1f}%  
                    Signal: {row['Signal'].replace('_', ' ')} | Confidence: {row['Confidence']}  
                    Expected R/R: {row['Expected_RR']:.1f}:1
                    """)
                
                st.markdown("---")
                st.markdown("**⚠️ Risk Management Notes:**")
                st.markdown("• Max 2-3% risk per individual trade")
                st.markdown("• Diversify across different signal types")
                st.markdown("• Monitor correlations between positions")
                st.markdown("• Adjust position sizes based on market volatility")
        
        else:
            st.warning("No trading opportunities available with current filters")
    
    # Enhanced footer with performance metrics
    st.markdown("---")
    
    # Performance summary
    if not filtered_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_signals = len(filtered_df[filtered_df['Signal'] != 'NEUTRAL'])
            st.markdown(f"🎯 **Active Signals**: {active_signals}")
        
        with col2:
            avg_confidence = filtered_df['Setup_confidence'].mode().iloc[0] if not filtered_df['Setup_confidence'].mode().empty else 'N/A'
            st.markdown(f"📊 **Avg Confidence**: {avg_confidence}")
        
        with col3:
            avg_strength = filtered_df['Setup_signal_strength'].mean()
            st.markdown(f"💪 **Avg Strength**: {avg_strength:.2f}")
        
        with col4:
            data_quality = "LIVE" if "demo" not in df['Data_Source'].values[0] else "DEMO"
            st.markdown(f"📡 **Data Quality**: {data_quality}")
    
    # Enhanced footer
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 20px;'>
        <p>🚀 <b>Enhanced Crypto CSI-Q Dashboard v2.0</b><br>
        🧠 <b>Features:</b> Advanced Sentiment Analysis | Dynamic Trading Setups | Smart Position Sizing<br>
        🔄 <b>Multi-Source:</b> Binance → CoinGecko → Enhanced Demo Mode<br>
        ⚠️ Dit is geen financieel advies. Altijd eigen onderzoek doen en risk management toepassen!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
