import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import urllib3
import torch
from datetime import datetime, timedelta
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- SETUP & CONFIG ---
st.set_page_config(page_title="Multi-Model Stock Battleground", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- DATA ACQUISITION ---
def get_dse_data(symbol, start_date, end_date):
    base_url = "https://dsebd.org/day_end_archive.php"
    params = {
        'inst': symbol,
        'startDate': start_date,
        'endDate': end_date,
        'archive': 'data'
    }
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        session = requests.Session()
        session.get(base_url, headers=headers, verify=False, timeout=10)
        try:
            response = session.get(base_url, params=params, headers=headers, verify=False, timeout=20)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            return None, "Internet connection lost. Please check your network and try again."
        except requests.exceptions.Timeout:
            return None, "DSE Server timed out. Try again later."

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'table table-bordered background-white shares-table fixedHeader'})

        if not table:
            return None, "No data found. Check ticker or date range."

        data = []
        for row in table.find_all('tr')[1:]:
            cols = [td.text.strip() for td in row.find_all('td')]
            if len(cols) >= 10:
                try:
                    close_val = float(cols[7].replace(',', '') if cols[7] != '--' else 0)
                    if close_val > 0:
                        data.append({'Date': cols[1], 'Close': close_val})
                except: continue

        if not data: return None, "No valid historical records found."

        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Floor Price Detection (5+ consecutive days unchanged)
        df['Is_Floor'] = False
        count = 0
        for i in range(1, len(df)):
            if df.loc[i, 'Close'] == df.loc[i-1, 'Close']:
                count += 1
            else: count = 0
            if count >= 4:
                for j in range(i - count, i + 1): df.loc[j, 'Is_Floor'] = True
        return df, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# --- INFERENCE ENGINE ---
class LocalInference:
    @staticmethod
    @st.cache_resource
    def load_chronos(local_path=None):
        try:
            from chronos import ChronosPipeline
            # Bolt small sometimes has config issues in some envs, fallback to T5-small if needed
            model_id = local_path if local_path and os.path.exists(local_path) else "amazon/chronos-bolt-small"
            try:
                return ChronosPipeline.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float32,
                )
            except:
                return ChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-small",
                    device_map="auto",
                    torch_dtype=torch.float32,
                )
        except Exception as e:
            st.error(f"Chronos Load Error: {e}")
            return None

    @staticmethod
    @st.cache_resource
    def load_timesfm(local_path=None):
        try:
            from timesfm import TimesFM_2p5_200M_torch
            model_id = local_path if local_path and os.path.exists(local_path) else "google/timesfm-2.5-200m-pytorch"
            model = TimesFM_2p5_200M_torch.from_pretrained(model_id)
            return model
        except Exception as e:
            return None

def run_single_inference(model, history, horizon, model_type):
    if model_type == 'chronos':
        context = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
        forecast = model.predict(context, horizon)
        if len(forecast.shape) == 3:
            return forecast[0].mean(axis=0)
        return forecast[0]
    elif model_type == 'timesfm':
        # Padding history to multiple of 32 as required by TimesFM 2.5
        p = 32
        pad_len = p - (len(history) % p)
        if pad_len < p:
            history = np.concatenate([np.zeros(pad_len), history])

        from timesfm import ForecastConfig
        # Re-compiling for specific horizon if needed
        if not hasattr(model, 'forecast_config') or model.forecast_config is None or model.forecast_config.max_horizon < horizon:
            model.compile(ForecastConfig(max_context=max(512, len(history)), max_horizon=max(128, horizon)))

        timesfm_pred, _ = model.compiled_decode(horizon, [history], [np.zeros_like(history, dtype=bool)])
        return timesfm_pred[0]
    return None

def backtest_models(df, horizon):
    if len(df) < horizon * 2:
        return None, None

    test_size = horizon
    train_data = df['Close'].iloc[:-test_size].values
    actual_data = df['Close'].iloc[-test_size:].values

    chronos_model = LocalInference.load_chronos()
    timesfm_model = LocalInference.load_timesfm()

    results = {}

    if chronos_model:
        try:
            pred = run_single_inference(chronos_model, train_data, test_size, 'chronos')
            results['chronos_mae'] = mean_absolute_error(actual_data, pred)
        except: results['chronos_mae'] = None

    if timesfm_model:
        try:
            pred = run_single_inference(timesfm_model, train_data, test_size, 'timesfm')
            results['timesfm_mae'] = mean_absolute_error(actual_data, pred)
        except: results['timesfm_mae'] = None

    return results, actual_data

def run_predictions(df, horizon):
    history = df['Close'].values
    chronos_model = LocalInference.load_chronos()
    timesfm_model = LocalInference.load_timesfm()

    chronos_pred = None
    if chronos_model:
        try:
            chronos_pred = run_single_inference(chronos_model, history, horizon, 'chronos')
        except Exception as e:
            st.warning(f"Chronos Inference failed: {e}")

    timesfm_pred = None
    if timesfm_model:
        try:
            timesfm_pred = run_single_inference(timesfm_model, history, horizon, 'timesfm')
        except Exception as e:
            st.warning(f"TimesFM Inference failed: {e}")

    return chronos_pred, timesfm_pred

# --- UI LAYOUT ---
st.title("🛡️ Multi-Model Stock Battleground (DSE)")
st.markdown("### 100% Local Inference: Google TimesFM 2.5 vs Amazon Chronos-Bolt")

with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("DSE Ticker", value="GP").upper()
    days_back = st.slider("Historical Lookback (Days)", 30, 1000, 365)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 15)

    st.subheader("Local Model Paths (Optional)")
    chronos_path = st.text_input("Chronos Local Path", placeholder="e.g. /models/chronos-bolt")
    timesfm_path = st.text_input("TimesFM Local Path", placeholder="e.g. /models/timesfm-2.5")

    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    run_btn = st.button("🚀 Launch Battle", width='stretch')

if run_btn:
    with st.spinner(f"Scraping DSE and running local models for {ticker}..."):
        df, err = get_dse_data(ticker, start_date, end_date)

        if err:
            st.error(err)
        else:
            # Backtesting
            backtest_res, actual_backtest = backtest_models(df, horizon)

            # Future Predictions
            chronos_p, timesfm_p = run_predictions(df, horizon)

            # Prepare Data for Plotly
            last_date = df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]

            # Main Chart
            fig = go.Figure()

            # Historical
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Historical', line=dict(color='white', width=2)))

            # Floor Prices
            floor_df = df[df['Is_Floor']]
            if not floor_df.empty:
                fig.add_trace(go.Scatter(x=floor_df['Date'], y=floor_df['Close'], mode='markers',
                                         name='DSE Floor Price', marker=dict(color='red', size=6)))

            # Chronos
            if chronos_p is not None:
                fig.add_trace(go.Scatter(x=future_dates, y=chronos_p, name='Amazon Chronos-Bolt',
                                         line=dict(color='#00f2ff', width=3, dash='dot')))

            # TimesFM
            if timesfm_p is not None:
                fig.add_trace(go.Scatter(x=future_dates, y=timesfm_p, name='Google TimesFM 2.5',
                                         line=dict(color='#ff007f', width=3, dash='dash')))

            fig.update_layout(template='plotly_dark', height=600,
                              xaxis_title="Timeline", yaxis_title="Price (BDT)",
                              legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

            st.plotly_chart(fig, width='stretch')

            # Metrics Panel
            col1, col2, col3 = st.columns(3)

            chronos_dir = "N/A"
            if chronos_p is not None:
                chronos_dir = "UP 📈" if chronos_p[-1] > df['Close'].iloc[-1] else "DOWN 📉"

            timesfm_dir = "N/A"
            if timesfm_p is not None:
                timesfm_dir = "UP 📈" if timesfm_p[-1] > df['Close'].iloc[-1] else "DOWN 📉"

            consensus = "NEUTRAL"
            if chronos_p is not None and timesfm_p is not None:
                consensus = "BULLISH" if "UP" in chronos_dir and "UP" in timesfm_dir else \
                            "BEARISH" if "DOWN" in chronos_dir and "DOWN" in timesfm_dir else "CONFLICTED"

            col1.metric("Chronos Direction", chronos_dir)
            col2.metric("TimesFM Direction", timesfm_dir)
            col3.metric("Market Consensus", consensus)

            # Accuracy Metrics
            st.markdown("---")
            st.subheader("📊 Backtesting Accuracy (MAE)")
            m_col1, m_col2 = st.columns(2)

            if backtest_res:
                m_col1.metric("Chronos-Bolt MAE", f"{backtest_res.get('chronos_mae', 'N/A'):.2f}" if backtest_res.get('chronos_mae') else "N/A")
                m_col2.metric("TimesFM 2.5 MAE", f"{backtest_res.get('timesfm_mae', 'N/A'):.2f}" if backtest_res.get('timesfm_mae') else "N/A")
            else:
                st.warning("Insufficient data for backtesting.")

            st.info(f"💡 Detected {df['Is_Floor'].sum()} floor price days in the historical period. Models may show reduced variance in these zones.")

else:
    st.info("Enter a ticker and hit 'Launch Battle' to start local inference. Ensure you have the model weights cached or provided in the local paths.")
