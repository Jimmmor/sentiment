import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config voor backtest module
st.set_page_config(
    page_title="CSI-Q Backtest",
    page_icon="📈",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .backtest-card {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .performance-positive {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 5px 0;
    }
    
    .performance-negative {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 5px 0;
    }
    
    .strategy-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 10px 0;
    }
    
    .debug-info {
        background: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class CSIQBacktester:
    def __init__(self, start_date, end_date, initial_capital=10000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.daily_returns = []
        self.positions = {}
        
    def generate_historical_data(self, symbols, days):
        """Genereer realistische historische data voor backtesting"""
        np.random.seed(42)  # Voor consistente resultaten
        
        historical_data = []
        
        # Base prijzen
        base_prices = {
            'BTC': 43000, 'ETH': 2600, 'BNB': 310, 'SOL': 100, 'XRP': 0.52,
            'ADA': 0.48, 'AVAX': 38, 'DOT': 7.2, 'LINK': 14.5, 'MATIC': 0.85,
            'UNI': 6.8, 'LTC': 73, 'BCH': 250, 'NEAR': 2.1, 'ALGO': 0.19,
            'VET': 0.025, 'FIL': 5.5, 'ETC': 20, 'AAVE': 95, 'MKR': 1450,
            'ATOM': 9.8, 'FTM': 0.32, 'SAND': 0.42, 'MANA': 0.38, 'AXS': 6.2
        }
        
        # Gebruik verschillende seeds per symbol voor meer variatie
        for symbol in symbols:
            np.random.seed(42 + hash(symbol) % 1000)
            base_price = base_prices.get(symbol, 1.0)
            
            for day in range(days):
                date = self.start_date + timedelta(days=day)
                
                # Meer volatiele prijsbeweging
                trend_cycle = np.sin(day * 0.1) * 0.03
                noise = np.random.normal(0, 0.08)  # Verhoogde volatiliteit
                momentum = np.random.choice([-0.02, 0, 0.02], p=[0.3, 0.4, 0.3])  # Momentum bursts
                
                price_change = trend_cycle + noise + momentum
                price = base_price * (1 + price_change) * (1 + day * 0.0005)
                
                # Zorg dat prijzen positief blijven
                price = max(price, base_price * 0.1)
                
                # Meer extreme CSI-Q waarden genereren
                funding_base = np.random.normal(0.01, 0.05)  # Meer variatie
                oi_change = np.random.normal(0, 25)  # Meer volatiel
                long_short_ratio = np.random.lognormal(0, 0.6)
                
                # Volume gecorreleerd met volatiliteit
                volatility_factor = abs(noise) * 5
                volume = base_price * 1000000 * (1 + volatility_factor)
                
                # RSI met meer spreiding
                rsi_base = 50 + np.random.normal(0, 30)
                rsi = max(0, min(100, rsi_base))
                
                # Social metrics met meer variatie
                social_sentiment = np.random.uniform(-1, 1)
                mentions = max(1, int(np.random.exponential(100)))
                
                # Bereken CSI-Q componenten met meer extreme waarden
                derivatives_score = np.clip(
                    (abs(oi_change) * 1.5) +
                    (abs(funding_base) * 800) +
                    (abs(long_short_ratio - 1) * 40) +
                    np.random.uniform(10, 40),
                    0, 100
                )
                
                social_score = np.clip(
                    ((social_sentiment + 1) * 30) +
                    (min(mentions, 200) * 0.2) +
                    np.random.uniform(5, 25),
                    0, 100
                )
                
                basis_score = np.clip(
                    abs(funding_base) * 1200 + np.random.uniform(15, 35),
                    0, 100
                )
                
                tech_score = np.clip(
                    (100 - abs(rsi - 50)) * 1.2 + np.random.uniform(5, 20),
                    0, 100
                )
                
                # CSI-Q berekening met meer gewicht op extremen
                csiq_raw = (
                    derivatives_score * 0.35 +
                    social_score * 0.25 +
                    basis_score * 0.25 +
                    tech_score * 0.15
                )
                
                # Add extreme values
                random_val = np.random.random()
                if random_val < 0.1:  # 10% chance of extreme positive
                    extreme_factor = 30
                elif random_val < 0.2:  # 10% chance of extreme negative
                    extreme_factor = -30
                else:  # 80% chance of no extreme factor
                    extreme_factor = 0
                    
                csiq = np.clip(csiq_raw + extreme_factor, 0, 100)
                
                historical_data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Price': round(price, 6),
                    'CSI_Q': round(csiq, 2),
                    'Funding_Rate': round(funding_base, 6),
                    'OI_Change': round(oi_change, 2),
                    'Long_Short_Ratio': round(long_short_ratio, 3),
                    'Volume': round(volume, 0),
                    'RSI': round(rsi, 1),
                    'Sentiment': round(social_sentiment, 3),
                    'Derivatives_Score': round(derivatives_score, 1),
                    'Social_Score': round(social_score, 1),
                    'Basis_Score': round(basis_score, 1),
                    'Tech_Score': round(tech_score, 1)
                })
        
        return pd.DataFrame(historical_data)
    
    def get_signal(self, csiq, funding_rate, rsi=50):
        """Verbeterde signal logic met meer condities"""
        # Extreme contrarian signals
        if csiq >= 85:
            return "SHORT"  # Extreme greed, short
        elif csiq <= 15:
            return "LONG"   # Extreme fear, long
        
        # Regular trend following
        elif csiq >= 70 and funding_rate > -0.05 and rsi > 45:
            return "LONG"   # Strong bullish sentiment
        elif csiq <= 30 and funding_rate < 0.05 and rsi < 55:
            return "SHORT"  # Strong bearish sentiment
        
        # Mean reversion in middle ranges
        elif 55 < csiq < 70 and rsi > 70:
            return "SHORT"  # Overbought
        elif 30 < csiq < 45 and rsi < 30:
            return "LONG"   # Oversold
            
        else:
            return "NEUTRAL"
    
    def execute_strategy(self, data, strategy_params):
        """Verbeterde strategy execution met debugging"""
        trades = []
        portfolio_value = []
        current_positions = {}
        cash = self.initial_capital
        
        debug_info = {
            'signals_generated': 0,
            'signals_by_type': {},
            'filtered_by_volume': 0,
            'filtered_by_csiq': 0,
            'filtered_by_cash': 0,
            'positions_opened': 0,
            'positions_closed': 0
        }
        
        # Groepeer data per datum
        daily_data = data.groupby('Date')
        
        for date, day_data in daily_data:
            day_portfolio_value = cash
            
            # Controleer bestaande posities
            for symbol in list(current_positions.keys()):
                symbol_data = day_data[day_data['Symbol'] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]['Price']
                    position = current_positions[symbol]
                    
                    # Update positie waarde
                    position_value = abs(position['quantity']) * current_price
                    if position['type'] == 'LONG':
                        day_portfolio_value += position_value
                    else:  # SHORT - simplified voor demo
                        day_portfolio_value += position_value
                    
                    # Check exit condities met relaxed thresholds
                    should_exit = False
                    exit_reason = ""
                    
                    # Stop loss (relaxed)
                    stop_loss_pct = 0.08  # 8% stop loss
                    if position['type'] == 'LONG' and current_price < position['entry_price'] * (1 - stop_loss_pct):
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif position['type'] == 'SHORT' and current_price > position['entry_price'] * (1 + stop_loss_pct):
                        should_exit = True
                        exit_reason = "Stop Loss"
                    
                    # Take profit (relaxed)
                    take_profit_pct = 0.06  # 6% take profit
                    if position['type'] == 'LONG' and current_price > position['entry_price'] * (1 + take_profit_pct):
                        should_exit = True
                        exit_reason = "Take Profit"
                    elif position['type'] == 'SHORT' and current_price < position['entry_price'] * (1 - take_profit_pct):
                        should_exit = True
                        exit_reason = "Take Profit"
                    
                    # Time-based exit
                    holding_days = (date - position['entry_date']).days
                    max_holding = strategy_params.get('max_holding_days', 5)
                    if holding_days >= max_holding:
                        should_exit = True
                        exit_reason = "Time Exit"
                    
                    if should_exit:
                        # Bereken PnL
                        if position['type'] == 'LONG':
                            pnl = (current_price - position['entry_price']) * position['quantity']
                        else:  # SHORT
                            pnl = (position['entry_price'] - current_price) * position['quantity']
                        
                        cash += abs(position['quantity']) * current_price
                        
                        trades.append({
                            'Date': date,
                            'Symbol': symbol,
                            'Type': 'EXIT',
                            'Signal': position['signal'],
                            'Price': current_price,
                            'Quantity': position['quantity'],
                            'PnL': pnl,
                            'Reason': exit_reason,
                            'Hold_Days': holding_days,
                            'Entry_Price': position['entry_price']
                        })
                        
                        debug_info['positions_closed'] += 1
                        del current_positions[symbol]
            
            # Zoek nieuwe trading kansen
            for _, row in day_data.iterrows():
                symbol = row['Symbol']
                signal = self.get_signal(row['CSI_Q'], row['Funding_Rate'], row['RSI'])
                
                debug_info['signals_generated'] += 1
                debug_info['signals_by_type'][signal] = debug_info['signals_by_type'].get(signal, 0) + 1
                
                # Skip als we al een positie hebben of neutral signal
                if symbol in current_positions or signal == 'NEUTRAL':
                    continue
                
                # Relaxed entry filters
                min_csiq_strength = strategy_params.get('min_csiq_strength', 65)
                min_volume = strategy_params.get('min_volume', 500000)  # Verlaagd
                
                # Volume filter
                if row['Volume'] < min_volume:
                    debug_info['filtered_by_volume'] += 1
                    continue
                
                # CSI-Q strength filter - aangepast voor nieuwe logic
                if signal in ['LONG', 'SHORT']:
                    if signal == 'LONG' and row['CSI_Q'] > 50:  # Long signals bij lagere CSI-Q
                        strength_ok = row['CSI_Q'] <= 15 or (row['CSI_Q'] >= 30 and row['RSI'] < 30)
                    elif signal == 'SHORT' and row['CSI_Q'] < 50:  # Short signals bij hogere CSI-Q  
                        strength_ok = row['CSI_Q'] >= 85 or (row['CSI_Q'] <= 70 and row['RSI'] > 70)
                    else:
                        strength_ok = True  # Allow all other combinations
                        
                    if not strength_ok:
                        debug_info['filtered_by_csiq'] += 1
                        continue
                
                # Bereken positie grootte - meer conservatief
                risk_per_trade = strategy_params.get('risk_per_trade', 0.05)  # 5% risk
                current_portfolio_value = cash
                for pos in current_positions.values():
                    pos_value = abs(pos['quantity']) * row['Price']  # Simplified
                    current_portfolio_value += pos_value
                
                position_value = current_portfolio_value * risk_per_trade
                position_size = position_value / row['Price']
                
                # Check cash en max position limits
                max_position_pct = strategy_params.get('max_position_size', 0.15)
                max_position_value = current_portfolio_value * max_position_pct
                
                if position_value > cash * 0.9 or position_value > max_position_value:
                    debug_info['filtered_by_cash'] += 1
                    continue
                
                # Limit aantal gelijktijdige posities
                max_positions = 5
                if len(current_positions) >= max_positions:
                    continue
                
                # Open nieuwe positie
                current_positions[symbol] = {
                    'entry_date': date,
                    'entry_price': row['Price'],
                    'quantity': position_size,
                    'type': signal,
                    'signal': signal,
                    'csiq': row['CSI_Q'],
                    'rsi': row['RSI']
                }
                
                cash -= position_value
                debug_info['positions_opened'] += 1
                
                trades.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Type': 'ENTRY',
                    'Signal': signal,
                    'Price': row['Price'],
                    'Quantity': position_size,
                    'CSI_Q': row['CSI_Q'],
                    'RSI': row['RSI'],
                    'Volume': row['Volume'],
                    'Position_Value': position_value
                })
            
            # Bereken totale portfolio waarde
            total_value = cash
            for symbol, position in current_positions.items():
                symbol_data = day_data[day_data['Symbol'] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]['Price']
                    position_value = abs(position['quantity']) * current_price
                    total_value += position_value
            
            portfolio_value.append({
                'Date': date,
                'Portfolio_Value': total_value,
                'Cash': cash,
                'Positions': len(current_positions)
            })
        
        # Debug info opslaan
        self.debug_info = debug_info
        
        return pd.DataFrame(trades), pd.DataFrame(portfolio_value)
    
    def calculate_metrics(self, portfolio_df, trades_df):
        """Bereken backtest performance metrics"""
        if portfolio_df.empty:
            return {
                'Total_Return': 0, 'Sharpe_Ratio': 0, 'Max_Drawdown': 0,
                'Win_Rate': 0, 'Total_Trades': 0, 'Avg_Win': 0, 'Avg_Loss': 0,
                'Profit_Factor': 0, 'Final_Portfolio_Value': self.initial_capital
            }
        
        # Basis metrics
        final_value = portfolio_df.iloc[-1]['Portfolio_Value']
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Dagelijkse returns
        portfolio_df['Daily_Return'] = portfolio_df['Portfolio_Value'].pct_change().fillna(0)
        avg_daily_return = portfolio_df['Daily_Return'].mean()
        daily_std = portfolio_df['Daily_Return'].std()
        
        # Sharpe ratio
        sharpe_ratio = (avg_daily_return / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        
        # Max drawdown
        peak = portfolio_df['Portfolio_Value'].expanding().max()
        drawdown = (portfolio_df['Portfolio_Value'] - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Trade analytics
        if not trades_df.empty:
            exit_trades = trades_df[trades_df['Type'] == 'EXIT']
            
            if not exit_trades.empty and 'PnL' in exit_trades.columns:
                win_trades = exit_trades[exit_trades['PnL'] > 0]
                win_rate = len(win_trades) / len(exit_trades) * 100
                avg_win = win_trades['PnL'].mean() if len(win_trades) > 0 else 0
                
                loss_trades = exit_trades[exit_trades['PnL'] < 0]
                avg_loss = loss_trades['PnL'].mean() if len(loss_trades) > 0 else 0
                
                total_wins = win_trades['PnL'].sum() if len(win_trades) > 0 else 0
                total_losses = abs(loss_trades['PnL'].sum()) if len(loss_trades) > 0 else 1
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'Total_Return': total_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Total_Trades': len(trades_df[trades_df['Type'] == 'EXIT']) if not trades_df.empty else 0,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss,
            'Profit_Factor': profit_factor,
            'Final_Portfolio_Value': final_value
        }

def main():
    st.title("📈 CSI-Q Strategy Backtester")
    st.markdown("**Historische Performance Analyse van CSI-Q Trading Strategy**")
    
    # Sidebar parameters
    st.sidebar.header("🔧 Backtest Parameters")
    
    # Datum range
    end_date = st.sidebar.date_input("End Date", datetime.now().date())
    start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=60))
    
    if start_date >= end_date:
        st.error("Start date moet voor end date liggen!")
        return
    
    # Portfolio parameters
    initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 100000, 10000)
    
    # Strategy parameters
    st.sidebar.subheader("📊 Strategy Settings")
    min_csiq_strength = st.sidebar.slider("Min CSI-Q Strength", 50, 80, 60)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 15, 5) / 100
    max_position_size = st.sidebar.slider("Max Position Size (%)", 5, 30, 15) / 100
    max_holding_days = st.sidebar.slider("Max Holding Days", 1, 14, 3)
    min_volume = st.sidebar.number_input("Min Volume ($)", 10000, 5000000, 500000)
    
    # Symbol selection
    all_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 
                   'LINK', 'MATIC', 'UNI', 'LTC', 'BCH', 'NEAR', 'ALGO']
    selected_symbols = st.sidebar.multiselect("Select Symbols", all_symbols, default=all_symbols[:8])
    
    if not selected_symbols:
        st.error("Selecteer minimaal één symbol!")
        return
    
    # Strategy parameters dict
    strategy_params = {
        'min_csiq_strength': min_csiq_strength,
        'risk_per_trade': risk_per_trade,
        'max_position_size': max_position_size,
        'max_holding_days': max_holding_days,
        'min_volume': min_volume
    }
    
    # Run backtest button
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        
        with st.spinner("🔄 Generating data and running backtest..."):
            # Initialize backtester
            backtester = CSIQBacktester(start_date, end_date, initial_capital)
            
            # Generate historical data
            days = (end_date - start_date).days + 1
            historical_data = backtester.generate_historical_data(selected_symbols, days)
            
            # Run strategy
            trades_df, portfolio_df = backtester.execute_strategy(historical_data, strategy_params)
            
            # Calculate metrics
            metrics = backtester.calculate_metrics(portfolio_df, trades_df)
        
        # Display results
        st.success(f"✅ Backtest completed! Analyzed {days} days with {len(selected_symbols)} symbols")
        
        # Debug information
        if hasattr(backtester, 'debug_info'):
            debug = backtester.debug_info
            st.markdown(f"""
            <div class="debug-info">
                <h4>🔍 Debug Information</h4>
                <p><b>Signals Generated:</b> {debug['signals_generated']}</p>
                <p><b>Signal Types:</b> {dict(debug['signals_by_type'])}</p>
                <p><b>Positions Opened:</b> {debug['positions_opened']}</p>
                <p><b>Positions Closed:</b> {debug['positions_closed']}</p>
                <p><b>Filtered by Volume:</b> {debug['filtered_by_volume']}</p>
                <p><b>Filtered by CSI-Q:</b> {debug['filtered_by_csiq']}</p>
                <p><b>Filtered by Cash:</b> {debug['filtered_by_cash']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            return_color = "performance-positive" if metrics['Total_Return'] > 0 else "performance-negative"
            st.markdown(f"""
            <div class="{return_color}">
                <h3>💰 Total Return</h3>
                <h2>{metrics['Total_Return']:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="backtest-card">
                <h3>🎯 Total Trades</h3>
                <h2>{metrics['Total_Trades']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="backtest-card">
                <h3>📉 Max Drawdown</h3>
                <h2>{metrics['Max_Drawdown']:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="backtest-card">
                <h3>🏆 Win Rate</h3>
                <h2>{metrics['Win_Rate']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Show sample of data
        if not historical_data.empty:
            st.subheader("📊 Sample Generated Data")
            st.write("First few rows of generated historical data:")
            st.dataframe(historical_data.head(10))
            
            # CSI-Q distribution
            fig_csiq = px.histogram(
                historical_data, 
                x='CSI_Q', 
                title="CSI-Q Distribution",
                nbins=30
            )
            st.plotly_chart(fig_csiq, use_container_width=True)
        
        # Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["📈 Portfolio Performance", "📋 Trade Analysis", "🔍 Detailed Metrics"])
        
        with tab1:
            if not portfolio_df.empty:
                # Portfolio value chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=portfolio_df['Date'],
                    y=portfolio_df['Portfolio_Value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#2E86AB', width=2)
                ))
                
                fig.update_layout(
                    title="💼 Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Portfolio metrics over time
                if 'Positions' in portfolio_df.columns:
                    fig_positions = px.line(
                        portfolio_df,
                        x='Date',
                        y='Positions',
                        title="📊 Active Positions Over Time"
                    )
                    fig_positions.update_layout(height=300)
                    st.plotly_chart(fig_positions, use_container_width=True)
            else:
                st.warning("No portfolio data generated - check debug information above")
        
        with tab2:
            if not trades_df.empty:
                st.subheader("🔍 All Trades")
                st.dataframe(trades_df, use_container_width=True)
                
                # Trade analysis
                entry_trades = trades_df[trades_df['Type'] == 'ENTRY']
                exit_trades = trades_df[trades_df['Type'] == 'EXIT']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if not entry_trades.empty:
                        signal_counts = entry_trades['Signal'].value_counts()
                        fig_signals = px.pie(
                            values=signal_counts.values,
                            names=signal_counts.index,
                            title="🎯 Entry Signal Distribution"
                        )
                        st.plotly_chart(fig_signals, use_container_width=True)
                
                with col2:
                    if not exit_trades.empty and 'PnL' in exit_trades.columns:
                        fig_pnl = px.histogram(
                            exit_trades,
                            x='PnL',
                            title="💰 P&L Distribution",
                            nbins=15
                        )
                        st.plotly_chart(fig_pnl, use_container_width=True)
                
                # Performance by symbol
                if not exit_trades.empty and 'PnL' in exit_trades.columns:
                    symbol_performance = exit_trades.groupby('Symbol')['PnL'].agg(['sum', 'count', 'mean']).round(2)
                    symbol_performance.columns = ['Total_PnL', 'Trade_Count', 'Avg_PnL']
                    symbol_performance['Win_Rate'] = exit_trades.groupby('Symbol').apply(
                        lambda x: (x['PnL'] > 0).mean() * 100
                    ).round(1)
                    
                    st.subheader("📊 Performance by Symbol")
                    st.dataframe(symbol_performance.sort_values('Total_PnL', ascending=False))
            else:
                st.warning("No trades executed during this period")
        
        with tab3:
            st.subheader("🔍 Detailed Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="strategy-box">
                    <h4>📈 Return Metrics</h4>
                    <p><b>Total Return:</b> {metrics['Total_Return']:.2f}%</p>
                    <p><b>Sharpe Ratio:</b> {metrics['Sharpe_Ratio']:.3f}</p>
                    <p><b>Max Drawdown:</b> {metrics['Max_Drawdown']:.2f}%</p>
                    <p><b>Final Portfolio Value:</b> ${metrics['Final_Portfolio_Value']:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="strategy-box">
                    <h4>🎯 Trade Metrics</h4>
                    <p><b>Total Trades:</b> {metrics['Total_Trades']}</p>
                    <p><b>Win Rate:</b> {metrics['Win_Rate']:.1f}%</p>
                    <p><b>Average Win:</b> ${metrics['Avg_Win']:.2f}</p>
                    <p><b>Average Loss:</b> ${metrics['Avg_Loss']:.2f}</p>
                    <p><b>Profit Factor:</b> {metrics['Profit_Factor']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Strategy explanation
            st.markdown("""
            <div class="strategy-box">
                <h4>🧠 Strategy Logic</h4>
                <p><b>Extreme Contrarian:</b> Short bij CSI-Q ≥ 85 (extreme greed), Long bij CSI-Q ≤ 15 (extreme fear)</p>
                <p><b>Trend Following:</b> Long bij CSI-Q ≥ 70 + positive funding + RSI > 45</p>
                <p><b>Mean Reversion:</b> Short bij overbought (CSI-Q 55-70 + RSI > 70)</p>
                <p><b>Risk Management:</b> 8% stop loss, 6% take profit, max 3-5 days holding period</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Show strategy explanation when not running
        st.markdown("""
        ## 🎯 CSI-Q Strategy Overview
        
        Deze backtester test een geavanceerde cryptocurrency trading strategie gebaseerd op de **Crypto Sentiment Intelligence Quotient (CSI-Q)**.
        
        ### 📊 Key Features:
        - **Multi-timeframe analyse** met extreme sentiment detection
        - **Risk management** met stop-loss en take-profit orders
        - **Portfolio diversificatie** over meerdere crypto assets
        - **Real-time debugging** en performance tracking
        
        ### ⚡ Trading Signals:
        1. **Extreme Contrarian**: Short bij extreme greed (CSI-Q ≥ 85), Long bij extreme fear (CSI-Q ≤ 15)
        2. **Trend Following**: Long bij sterke bullish sentiment (CSI-Q ≥ 70)
        3. **Mean Reversion**: Short bij overbought condities in mid-range
        
        ### 📈 Performance Metrics:
        - Total Return & Sharpe Ratio
        - Maximum Drawdown
        - Win Rate & Profit Factor
        - Trade-by-trade analysis
        
        **👈 Configureer je parameters in de sidebar en klik "Run Backtest" om te beginnen!**
        """)
        
        # Show example CSI-Q distribution
        st.subheader("📊 Example CSI-Q Data Distribution")
        
        # Generate sample data for demo
        np.random.seed(42)
        sample_csiq = []
        
        for _ in range(1000):
            base = np.random.normal(50, 20)
            if np.random.random() < 0.1:
                base += np.random.choice([-30, 30])
            sample_csiq.append(np.clip(base, 0, 100))
        
        fig_demo = px.histogram(
            x=sample_csiq,
            title="Sample CSI-Q Distribution (Demo Data)",
            nbins=20,
            labels={'x': 'CSI-Q Value', 'y': 'Frequency'}
        )
        fig_demo.add_vline(x=15, line_dash="dash", line_color="green", 
                          annotation_text="Extreme Fear (LONG)")
        fig_demo.add_vline(x=85, line_dash="dash", line_color="red", 
                          annotation_text="Extreme Greed (SHORT)")
        fig_demo.add_vrect(x0=30, x1=70, fillcolor="yellow", opacity=0.2, 
                          annotation_text="Normal Range")
        
        st.plotly_chart(fig_demo, use_container_width=True)

if __name__ == "__main__":
    main()
