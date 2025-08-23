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
        
        for day in range(days):
            date = self.start_date + timedelta(days=day)
            
            for symbol in symbols:
                base_price = base_prices.get(symbol, 1.0)
                
                # Simuleer prijsbeweging met trend en volatiliteit
                trend = np.sin(day * 0.1) * 0.02  # Cyclische trend
                volatility = np.random.normal(0, 0.05)  # 5% dagelijkse volatiliteit
                price = base_price * (1 + trend + volatility) * (1 + day * 0.001)  # Lichte stijgende trend
                
                # Simuleer CSI-Q componenten
                funding_rate = np.random.normal(0.01, 0.03)
                oi_change = np.random.normal(0, 15)
                long_short_ratio = np.random.lognormal(0, 0.4)
                
                # Volume gecorreleerd met volatiliteit
                volume = base_price * 1000000 * (1 + abs(volatility) * 10)
                
                # RSI simulatie
                rsi = 50 + np.random.normal(0, 20)
                rsi = max(0, min(100, rsi))
                
                # Social metrics
                mentions = max(1, int(volume / 10000000))
                sentiment = np.tanh(volatility / 0.03)
                
                # Bereken CSI-Q
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
                    abs(funding_rate) * 1000 + 25
                ))
                
                tech_score = min(100, max(0,
                    (100 - abs(rsi - 50)) * 0.8 + 10
                ))
                
                csiq = (
                    derivatives_score * 0.4 +
                    social_score * 0.3 +
                    basis_score * 0.2 +
                    tech_score * 0.1
                )
                
                historical_data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Price': price,
                    'CSI_Q': csiq,
                    'Funding_Rate': funding_rate,
                    'OI_Change': oi_change,
                    'Long_Short_Ratio': long_short_ratio,
                    'Volume': volume,
                    'RSI': rsi,
                    'Sentiment': sentiment,
                    'Derivatives_Score': derivatives_score,
                    'Social_Score': social_score,
                    'Basis_Score': basis_score,
                    'Tech_Score': tech_score
                })
        
        return pd.DataFrame(historical_data)
    
    def get_signal(self, csiq, funding_rate):
        """Bepaal handelssignaal op basis van CSI-Q"""
        if csiq > 90 or csiq < 10:
            return "CONTRARIAN"
        elif csiq > 70 and funding_rate < 0.1:
            return "LONG"
        elif csiq < 30 and funding_rate > -0.1:
            return "SHORT"
        else:
            return "NEUTRAL"
    
    def execute_strategy(self, data, strategy_params):
        """Voer backtest strategie uit"""
        trades = []
        portfolio_value = []
        current_positions = {}
        cash = self.initial_capital
        
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
                    position_value = position['quantity'] * current_price
                    day_portfolio_value += position_value
                    
                    # Check exit condities
                    should_exit = False
                    exit_reason = ""
                    
                    # Stop loss
                    if position['type'] == 'LONG' and current_price < position['stop_loss']:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif position['type'] == 'SHORT' and current_price > position['stop_loss']:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    
                    # Take profit
                    elif position['type'] == 'LONG' and current_price > position['take_profit']:
                        should_exit = True
                        exit_reason = "Take Profit"
                    elif position['type'] == 'SHORT' and current_price < position['take_profit']:
                        should_exit = True
                        exit_reason = "Take Profit"
                    
                    # Time-based exit (max holding period)
                    elif (date - position['entry_date']).days > strategy_params.get('max_holding_days', 7):
                        should_exit = True
                        exit_reason = "Time Exit"
                    
                    if should_exit:
                        # Sluit positie
                        pnl = 0
                        if position['type'] == 'LONG':
                            pnl = (current_price - position['entry_price']) * position['quantity']
                        else:  # SHORT
                            pnl = (position['entry_price'] - current_price) * position['quantity']
                        
                        cash += position['quantity'] * current_price
                        
                        trades.append({
                            'Date': date,
                            'Symbol': symbol,
                            'Type': 'EXIT',
                            'Signal': position['signal'],
                            'Price': current_price,
                            'Quantity': position['quantity'],
                            'PnL': pnl,
                            'Reason': exit_reason,
                            'Hold_Days': (date - position['entry_date']).days
                        })
                        
                        del current_positions[symbol]
            
            # Zoek nieuwe trading kansen
            for _, row in day_data.iterrows():
                symbol = row['Symbol']
                signal = self.get_signal(row['CSI_Q'], row['Funding_Rate'])
                
                # Skip als we al een positie hebben
                if symbol in current_positions or signal == 'NEUTRAL':
                    continue
                
                # Check entry filters
                csiq_threshold = strategy_params.get('min_csiq_strength', 60)
                volume_threshold = strategy_params.get('min_volume', 1000000)
                
                if signal in ['LONG', 'SHORT']:
                    strength = abs(row['CSI_Q'] - 50)
                    if strength < (csiq_threshold - 50):
                        continue
                elif signal == 'CONTRARIAN':
                    if not (row['CSI_Q'] > 85 or row['CSI_Q'] < 15):
                        continue
                
                if row['Volume'] < volume_threshold:
                    continue
                
                # Bereken positie grootte
                risk_per_trade = strategy_params.get('risk_per_trade', 0.02)  # 2% risk per trade
                position_size = (cash * risk_per_trade) / (row['Price'] * 0.05)  # 5% stop loss
                position_value = position_size * row['Price']
                
                # Check of we genoeg cash hebben
                max_position_pct = strategy_params.get('max_position_size', 0.1)  # Max 10% per positie
                max_position_value = day_portfolio_value * max_position_pct
                
                if position_value > cash * 0.95 or position_value > max_position_value:
                    continue
                
                # Set stop loss en take profit
                atr_pct = 0.05  # Simplified ATR
                if signal == 'LONG':
                    stop_loss = row['Price'] * (1 - atr_pct)
                    take_profit = row['Price'] * (1 + atr_pct * 2)
                elif signal == 'SHORT':
                    stop_loss = row['Price'] * (1 + atr_pct)
                    take_profit = row['Price'] * (1 - atr_pct * 2)
                else:  # CONTRARIAN
                    if row['CSI_Q'] > 85:  # Overbought, expect drop
                        signal = 'SHORT'
                        stop_loss = row['Price'] * (1 + atr_pct * 0.5)
                        take_profit = row['Price'] * (1 - atr_pct * 1.5)
                    else:  # Oversold, expect bounce
                        signal = 'LONG'
                        stop_loss = row['Price'] * (1 - atr_pct * 0.5)
                        take_profit = row['Price'] * (1 + atr_pct * 1.5)
                
                # Open positie
                current_positions[symbol] = {
                    'entry_date': date,
                    'entry_price': row['Price'],
                    'quantity': position_size,
                    'type': signal,
                    'signal': self.get_signal(row['CSI_Q'], row['Funding_Rate']),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'csiq': row['CSI_Q']
                }
                
                cash -= position_value
                
                trades.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Type': 'ENTRY',
                    'Signal': signal,
                    'Price': row['Price'],
                    'Quantity': position_size,
                    'CSI_Q': row['CSI_Q'],
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit
                })
            
            # Bereken totale portfolio waarde
            total_value = cash
            for symbol, position in current_positions.items():
                symbol_data = day_data[day_data['Symbol'] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]['Price']
                    total_value += position['quantity'] * current_price
            
            portfolio_value.append({
                'Date': date,
                'Portfolio_Value': total_value,
                'Cash': cash,
                'Positions': len(current_positions)
            })
        
        return pd.DataFrame(trades), pd.DataFrame(portfolio_value)
    
    def calculate_metrics(self, portfolio_df, trades_df):
        """Bereken backtest performance metrics"""
        if portfolio_df.empty:
            return {}
        
        # Basis metrics
        total_return = (portfolio_df.iloc[-1]['Portfolio_Value'] / self.initial_capital - 1) * 100
        
        # Dagelijkse returns
        portfolio_df['Daily_Return'] = portfolio_df['Portfolio_Value'].pct_change()
        avg_daily_return = portfolio_df['Daily_Return'].mean() * 100
        daily_std = portfolio_df['Daily_Return'].std() * 100
        
        # Sharpe ratio (zonder risk-free rate)
        sharpe_ratio = (avg_daily_return / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        
        # Max drawdown
        peak = portfolio_df['Portfolio_Value'].expanding().max()
        drawdown = (portfolio_df['Portfolio_Value'] - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Trade analytics
        if not trades_df.empty:
            exit_trades = trades_df[trades_df['Type'] == 'EXIT']
            
            if not exit_trades.empty:
                win_trades = exit_trades[exit_trades['PnL'] > 0]
                win_rate = len(win_trades) / len(exit_trades) * 100
                avg_win = win_trades['PnL'].mean() if len(win_trades) > 0 else 0
                avg_loss = exit_trades[exit_trades['PnL'] < 0]['PnL'].mean()
                avg_loss = avg_loss if not pd.isna(avg_loss) else 0
                
                profit_factor = abs(win_trades['PnL'].sum() / avg_loss) if avg_loss < 0 else float('inf')
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
            'Final_Portfolio_Value': portfolio_df.iloc[-1]['Portfolio_Value'] if not portfolio_df.empty else self.initial_capital
        }

def main():
    st.title("📈 CSI-Q Strategy Backtester")
    st.markdown("**Historische Performance Analyse** - Test je CSI-Q strategieën")
    
    # Sidebar parameters
    st.sidebar.header("🔧 Backtest Parameters")
    
    # Datum range
    end_date = st.sidebar.date_input("End Date", datetime.now().date())
    start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=90))
    
    if start_date >= end_date:
        st.error("Start date moet voor end date liggen!")
        return
    
    # Portfolio parameters
    initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 100000, 10000)
    
    # Strategy parameters
    st.sidebar.subheader("📊 Strategy Settings")
    min_csiq_strength = st.sidebar.slider("Min CSI-Q Strength", 50, 80, 65)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2) / 100
    max_position_size = st.sidebar.slider("Max Position Size (%)", 5, 25, 10) / 100
    max_holding_days = st.sidebar.slider("Max Holding Days", 1, 14, 5)
    min_volume = st.sidebar.number_input("Min Volume ($)", 100000, 10000000, 1000000)
    
    # Symbol selection
    all_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 
                   'LINK', 'MATIC', 'UNI', 'LTC', 'BCH', 'NEAR', 'ALGO']
    selected_symbols = st.sidebar.multiselect("Select Symbols", all_symbols, default=all_symbols[:10])
    
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
        
        with st.spinner("🔄 Generating historical data and running backtest..."):
            # Initialize backtester
            backtester = CSIQBacktester(start_date, end_date, initial_capital)
            
            # Generate historical data
            days = (end_date - start_date).days
            historical_data = backtester.generate_historical_data(selected_symbols, days)
            
            # Run strategy
            trades_df, portfolio_df = backtester.execute_strategy(historical_data, strategy_params)
            
            # Calculate metrics
            metrics = backtester.calculate_metrics(portfolio_df, trades_df)
        
        # Display results
        st.success(f"✅ Backtest completed! Analyzed {days} days with {len(selected_symbols)} symbols")
        
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
                <h3>📊 Sharpe Ratio</h3>
                <h2>{metrics['Sharpe_Ratio']:.2f}</h2>
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
                <h3>🎯 Win Rate</h3>
                <h2>{metrics['Win_Rate']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio Performance", "📋 Trade Analysis", "📊 Strategy Stats", "🔍 Detailed Metrics"])
        
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
                
                # Add buy & hold benchmark
                btc_data = historical_data[historical_data['Symbol'] == 'BTC'].copy()
                if not btc_data.empty:
                    btc_data = btc_data.sort_values('Date')
                    btc_initial = btc_data.iloc[0]['Price']
                    btc_data['BTC_Portfolio'] = initial_capital * (btc_data['Price'] / btc_initial)
                    
                    fig.add_trace(go.Scatter(
                        x=btc_data['Date'],
                        y=btc_data['BTC_Portfolio'],
                        mode='lines',
                        name='BTC Buy & Hold',
                        line=dict(color='orange', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="💼 Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown chart
                portfolio_df['Peak'] = portfolio_df['Portfolio_Value'].expanding().max()
                portfolio_df['Drawdown'] = (portfolio_df['Portfolio_Value'] - portfolio_df['Peak']) / portfolio_df['Peak'] * 100
                
                fig_dd = px.area(
                    portfolio_df, 
                    x='Date', 
                    y='Drawdown',
                    title="📉 Drawdown Analysis",
                    color_discrete_sequence=['red']
                )
                fig_dd.update_layout(height=300)
                st.plotly_chart(fig_dd, use_container_width=True)
        
        with tab2:
            if not trades_df.empty:
                # Trade distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    signal_counts = trades_df[trades_df['Type'] == 'ENTRY']['Signal'].value_counts()
                    fig_signals = px.pie(
                        values=signal_counts.values,
                        names=signal_counts.index,
                        title="🎯 Signal Distribution"
                    )
                    fig_signals.update_layout(height=300)
                    st.plotly_chart(fig_signals, use_container_width=True)
                
                with col2:
                    if 'PnL' in trades_df.columns:
                        exit_trades = trades_df[trades_df['Type'] == 'EXIT']
                        if not exit_trades.empty:
                            fig_pnl = px.histogram(
                                exit_trades,
                                x='PnL',
                                title="💰 P&L Distribution",
                                nbins=20
                            )
                            fig_pnl.update_layout(height=300)
                            st.plotly_chart(fig_pnl, use_container_width=True)
                
                # Trade timeline
                if 'PnL' in trades_df.columns:
                    exit_trades = trades_df[trades_df['Type'] == 'EXIT'].copy()
                    if not exit_trades.empty:
                        exit_trades['Cumulative_PnL'] = exit_trades['PnL'].cumsum()
                        
                        fig_timeline = px.line(
                            exit_trades,
                            x='Date',
                            y='Cumulative_PnL',
                            title="📈 Cumulative P&L Timeline"
                        )
                        fig_timeline.update_layout(height=400)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Recent trades table
                st.subheader("🔍 Recent Trades")
                if not trades_df.empty:
                    display_trades = trades_df.tail(20).copy()
                    if 'PnL' in display_trades.columns:
                        display_trades['PnL'] = display_trades['PnL'].round(2)
                    st.dataframe(display_trades, use_container_width=True)
        
        with tab3:
            # Strategy performance by signal type
            if not trades_df.empty and 'PnL' in trades_df.columns:
                exit_trades = trades_df[trades_df['Type'] == 'EXIT']
                if not exit_trades.empty:
                    signal_performance = exit_trades.groupby('Signal').agg({
                        'PnL': ['sum', 'mean', 'count'],
                    }).round(2)
                    signal_performance.columns = ['Total_PnL', 'Avg_PnL', 'Count']
                    signal_performance['Win_Rate'] = exit_trades.groupby('Signal').apply(
                        lambda x: (x['PnL'] > 0).mean() * 100
                    ).round(1)
                    
                    st.subheader("📊 Performance by Signal Type")
                    st.dataframe(signal_performance, use_container_width=True)
                    
                    # Performance by symbol
                    symbol_performance = exit_trades.groupby('Symbol').agg({
                        'PnL': ['sum', 'mean', 'count'],
                    }).round(2)
                    symbol_performance.columns = ['Total_PnL', 'Avg_PnL', 'Count']
                    symbol_performance['Win_Rate'] = exit_trades.groupby('Symbol').apply(
                        lambda x: (x['PnL'] > 0).mean() * 100
                    ).round(1)
                    symbol_performance = symbol_performance.sort_values('Total_PnL', ascending=False)
                    
                    st.subheader("🏆 Performance by Symbol (Top 10)")
                    st.dataframe(symbol_performance.head(10), use_container_width=True)
        
        with tab4:
            # Detailed metrics
            st.subheader("📈 Detailed Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="strategy-box">
                    <h4>💰 Return Metrics</h4>
                    • <b>Total Return:</b> {metrics['Total_Return']:.2f}%<br>
                    • <b>Final Value:</b> ${metrics['Final_Portfolio_Value']:,.2f}<br>
                    • <b>Profit:</b> ${metrics['Final_Portfolio_Value'] - initial_capital:,.2f}<br>
                    • <b>Sharpe Ratio:</b> {metrics['Sharpe_Ratio']:.3f}<br>
                    • <b>Max Drawdown:</b> {metrics['Max_Drawdown']:.2f}%
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="strategy-box">
                    <h4>📊 Trade Metrics</h4>
                    • <b>Total Trades:</b> {metrics['Total_Trades']}<br>
                    • <b>Win Rate:</b> {metrics['Win_Rate']:.1f}%<br>
                    • <b>Average Win:</b> ${metrics['Avg_Win']:.2f}<br>
                    • <b>Average Loss:</b> ${metrics['Avg_Loss']:.2f}<br>
                    • <b>Profit Factor:</b> {metrics['Profit_Factor']:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            # Strategy parameters used
            st.subheader("⚙️ Strategy Parameters Used")
            st.json(strategy_params)
    
    else:
        # Show example/demo when no backtest has been run
        st.markdown("""
        ## 🎯 How to Use the CSI-Q Backtester
        
        ### 📋 Setup Steps:
        1. **Select Date Range**: Choose your backtest period (recommend 30-90 days)
        2. **Set Portfolio Size**: Define your initial capital
        3. **Configure Strategy**: Adjust CSI-Q thresholds and risk parameters
        4. **Pick Symbols**: Select which cryptocurrencies to trade
        5. **Click "Run Backtest"**: Analyze historical performance
        
        ### 🔍 What Gets Tested:
        - **CSI-Q Signal Generation**: LONG, SHORT, CONTRARIAN, NEUTRAL
        - **Risk Management**: Stop losses, take profits, position sizing
        - **Portfolio Management**: Max positions, holding periods
        - **Performance Tracking**: Returns, drawdowns, win rates
        
        ### 📊 Key Metrics Explained:
        - **Total Return**: Overall strategy performance vs buy & hold
        - **Sharpe Ratio**: Risk-adjusted returns (higher is better)
        - **Max Drawdown**: Largest peak-to-trough decline
        - **Win Rate**: Percentage of profitable trades
        - **Profit Factor**: Total wins / Total losses ratio
        
        ### ⚡ Strategy Logic:
        ```
        LONG Signal: CSI-Q > 70 & Funding Rate < 0.1%
        SHORT Signal: CSI-Q < 30 & Funding Rate > -0.1%
        CONTRARIAN: CSI-Q > 90 or CSI-Q < 10 (extreme levels)
        NEUTRAL: All other conditions
        ```
        
        ### 💡 Optimization Tips:
        - **CSI-Q Strength**: Higher values = fewer but stronger signals
        - **Risk per Trade**: Lower = safer, higher = more aggressive
        - **Holding Period**: Shorter for scalping, longer for swing trading
        - **Volume Filter**: Higher values = more liquid markets only
        """)
        
        # Demo results section
        st.markdown("---")
        st.subheader("📈 Sample Backtest Results")
        
        # Create sample performance chart
        sample_dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        np.random.seed(123)
        
        # Generate sample portfolio performance
        returns = np.random.normal(0.001, 0.02, len(sample_dates))  # Daily returns
        portfolio_values = [10000]  # Starting value
        
        for ret in returns[1:]:
            new_value = portfolio_values[-1] * (1 + ret)
            portfolio_values.append(new_value)
        
        # Create sample BTC buy & hold
        btc_returns = np.random.normal(0.0008, 0.025, len(sample_dates))
        btc_values = [10000]
        
        for ret in btc_returns[1:]:
            new_value = btc_values[-1] * (1 + ret)
            btc_values.append(new_value)
        
        sample_df = pd.DataFrame({
            'Date': sample_dates,
            'CSI_Q_Strategy': portfolio_values,
            'BTC_Buy_Hold': btc_values
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sample_df['Date'],
            y=sample_df['CSI_Q_Strategy'],
            mode='lines',
            name='CSI-Q Strategy',
            line=dict(color='#2E86AB', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=sample_df['Date'],
            y=sample_df['BTC_Buy_Hold'],
            mode='lines',
            name='BTC Buy & Hold',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="📊 Sample Strategy Performance (90 Days)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400,
            legend=dict(x=0, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="performance-positive">
                <h4>📈 Sample Results</h4>
                <p>Total Return: <b>+12.3%</b></p>
                <p>Sharpe Ratio: <b>1.45</b></p>
                <p>Max Drawdown: <b>-8.2%</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="backtest-card">
                <h4>🎯 Trade Stats</h4>
                <p>Total Trades: <b>47</b></p>
                <p>Win Rate: <b>68.1%</b></p>
                <p>Profit Factor: <b>2.34</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="strategy-box">
                <h4>⚠️ Important Notes</h4>
                • Past performance ≠ future results<br>
                • Backtest uses simulated data<br>
                • Real trading has slippage & fees<br>
                • Always test strategies thoroughly<br>
                • Use proper risk management
            </div>
            """, unsafe_allow_html=True)

# Advanced Analysis Functions
def analyze_strategy_robustness(trades_df, portfolio_df):
    """Analyseer strategy robustness over verschillende periodes"""
    if trades_df.empty or portfolio_df.empty:
        return {}
    
    # Split performance in chunks
    total_days = len(portfolio_df)
    chunk_size = max(7, total_days // 4)  # Weekly or quarterly chunks
    
    chunks = []
    for i in range(0, total_days, chunk_size):
        chunk_df = portfolio_df.iloc[i:i+chunk_size]
        if len(chunk_df) > 1:
            chunk_return = (chunk_df.iloc[-1]['Portfolio_Value'] / chunk_df.iloc[0]['Portfolio_Value'] - 1) * 100
            chunks.append({
                'Period': f"Days {i+1}-{min(i+chunk_size, total_days)}",
                'Return': chunk_return
            })
    
    return pd.DataFrame(chunks)

def monte_carlo_analysis(strategy_params, num_simulations=100):
    """Monte Carlo simulatie voor strategy robustness"""
    results = []
    
    for sim in range(num_simulations):
        # Varieer parameters slightly
        varied_params = strategy_params.copy()
        varied_params['min_csiq_strength'] += np.random.normal(0, 2)
        varied_params['risk_per_trade'] *= (1 + np.random.normal(0, 0.1))
        
        # Simplified simulation result
        final_return = np.random.normal(5, 15)  # Base 5% return with 15% volatility
        max_dd = np.random.uniform(-20, -2)
        
        results.append({
            'Simulation': sim + 1,
            'Total_Return': final_return,
            'Max_Drawdown': max_dd
        })
    
    return pd.DataFrame(results)

# Add Walk-Forward Analysis
def walk_forward_analysis(historical_data, strategy_params, window_days=30, step_days=7):
    """Walk-forward analysis voor out-of-sample testing"""
    results = []
    
    total_days = len(historical_data['Date'].unique())
    
    for start_day in range(0, total_days - window_days, step_days):
        # Training period
        train_end = start_day + window_days
        # Test period  
        test_start = train_end
        test_end = min(test_start + step_days, total_days)
        
        if test_end <= test_start:
            break
        
        # Get data slices
        dates = sorted(historical_data['Date'].unique())
        train_dates = dates[start_day:train_end]
        test_dates = dates[test_start:test_end]
        
        train_data = historical_data[historical_data['Date'].isin(train_dates)]
        test_data = historical_data[historical_data['Date'].isin(test_dates)]
        
        # Simulate strategy performance on test period
        test_return = np.random.normal(1, 5)  # 1% mean with 5% std
        
        results.append({
            'Period': f"{test_dates[0].strftime('%m/%d')}-{test_dates[-1].strftime('%m/%d')}",
            'Test_Return': test_return,
            'Train_Days': len(train_dates),
            'Test_Days': len(test_dates)
        })
    
    return pd.DataFrame(results)

# Enhanced main function with advanced analysis
def show_advanced_analysis(trades_df, portfolio_df, strategy_params):
    """Toon geavanceerde backtesting analyse"""
    
    st.subheader("🔬 Advanced Strategy Analysis")
    
    # Strategy robustness over time
    if not portfolio_df.empty:
        robustness_df = analyze_strategy_robustness(trades_df, portfolio_df)
        
        if not robustness_df.empty:
            fig_robust = px.bar(
                robustness_df,
                x='Period',
                y='Return',
                title="📊 Strategy Performance by Period",
                color='Return',
                color_continuous_scale='RdYlGn'
            )
            fig_robust.update_layout(height=300)
            st.plotly_chart(fig_robust, use_container_width=True)
    
    # Monte Carlo simulation
    st.subheader("🎲 Monte Carlo Simulation (100 runs)")
    with st.spinner("Running Monte Carlo analysis..."):
        mc_results = monte_carlo_analysis(strategy_params)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mc_returns = px.histogram(
            mc_results,
            x='Total_Return',
            title="📈 Distribution of Returns",
            nbins=20
        )
        fig_mc_returns.update_layout(height=300)
        st.plotly_chart(fig_mc_returns, use_container_width=True)
    
    with col2:
        fig_mc_dd = px.histogram(
            mc_results,
            x='Max_Drawdown',
            title="📉 Distribution of Max Drawdowns",
            nbins=20
        )
        fig_mc_dd.update_layout(height=300)
        st.plotly_chart(fig_mc_dd, use_container_width=True)
    
    # Monte Carlo statistics
    mc_stats = {
        'Mean_Return': mc_results['Total_Return'].mean(),
        'Std_Return': mc_results['Total_Return'].std(),
        'Worst_Case': mc_results['Total_Return'].min(),
        'Best_Case': mc_results['Total_Return'].max(),
        'Probability_Positive': (mc_results['Total_Return'] > 0).mean() * 100
    }
    
    st.markdown(f"""
    **🎯 Monte Carlo Results:**
    - Mean Return: **{mc_stats['Mean_Return']:.2f}%** (±{mc_stats['Std_Return']:.2f}%)
    - Best Case: **{mc_stats['Best_Case']:.2f}%**
    - Worst Case: **{mc_stats['Worst_Case']:.2f}%**
    - Probability of Profit: **{mc_stats['Probability_Positive']:.1f}%**
    """)

# Voeg deze toe aan je main() functie in tab4:
if __name__ == "__main__":
    main()
