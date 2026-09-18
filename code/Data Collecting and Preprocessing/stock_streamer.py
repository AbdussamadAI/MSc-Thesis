import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import logging
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataStreamer:
    def __init__(self, interval="5m", output_dir=None):
        self.interval = interval
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_data")
        else:
            self.output_dir = output_dir
            
        # Create market-specific directories
        self.market_dirs = {
            'US': os.path.join(self.output_dir, 'US'),
            'London': os.path.join(self.output_dir, 'London'), 
            'Pakistan': os.path.join(self.output_dir, 'Pakistan')
        }
        
        for market_dir in self.market_dirs.values():
            os.makedirs(market_dir, exist_ok=True)
            
        logger.info(f"Data will be saved to: {self.output_dir}")
        logger.info(f"Streaming interval set to: {interval}")
        
        # Define tickers for each market aligned with existing data
        self.market_tickers = {
            'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 
                   'ADBE', 'CRM', 'ORCL', 'AMD', 'INTC', 'CSCO', 'AVGO', 'TXN', 
                   'QCOM', 'INTU', 'MU', 'AMAT'],
            'London': ['AAL.L', 'AZN.L', 'BARC.L', 'BP.L', 'GLEN.L', 'GSK.L', 
                      'HSBA.L', 'LLOY.L', 'NWG.L', 'REL.L', 'RIO.L', 'SHEL.L', 
                      'SSE.L', 'ULVR.L', 'VOD.L'],
            'Pakistan': ['BAHL.KA', 'DGKC.KA', 'EFERT.KA', 'ENGRO.KA', 'FFC.KA', 
                        'HBL.KA', 'HUBC.KA', 'LUCK.KA', 'MARI.KA', 'MCB.KA', 
                        'MLCF.KA', 'NBP.KA', 'OGDC.KA', 'PPL.KA', 'UBL.KA']
        }
        
        # Market timezone configurations
        self.market_timezones = {
            'US': pytz.timezone('US/Eastern'),
            'London': pytz.timezone('Europe/London'),
            'Pakistan': pytz.timezone('Asia/Karachi')
        }
        
        # Market hours (local time for each market)
        self.market_hours = {
            'US': {'open': (9, 30), 'close': (16, 0)},      # 9:30 AM - 4:00 PM ET
            'London': {'open': (8, 0), 'close': (16, 30)},   # 8:00 AM - 4:30 PM GMT
            'Pakistan': {'open': (9, 15), 'close': (15, 30)} # 9:15 AM - 3:30 PM PKT
        }
        
        # Initialize datasets for each market
        self.datasets = {market: {} for market in self.market_tickers.keys()}
        
    def is_market_open(self, market):
        """Check if specific market is currently open"""
        tz = self.market_timezones[market]
        now = datetime.now(tz)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Weekend
            return False
            
        # Get market hours
        open_hour, open_min = self.market_hours[market]['open']
        close_hour, close_min = self.market_hours[market]['close']
        
        market_open = now.replace(hour=open_hour, minute=open_min, second=0, microsecond=0)
        market_close = now.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_active_markets(self):
        """Get list of currently active markets"""
        active = []
        for market in self.market_tickers.keys():
            if self.is_market_open(market):
                active.append(market)
        return active
    
    def process_data(self, hist, ticker, market):
        """Process and add features to the data"""
        try:
            df = hist.copy()
            if df.empty:
                logger.warning(f"Empty data received for {ticker}")
                return pd.DataFrame()
                
            # Add features matching existing project structure
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Price_MA'] = df['Close'].rolling(window=20).mean()
            df['Hour'] = df.index.hour
            df['Day'] = df.index.day
            df['Month'] = df.index.month
            df['Year'] = df.index.year
            df['DayOfWeek'] = df.index.dayofweek
            df['Ticker'] = ticker.replace('.L', '').replace('.KA', '')  # Clean ticker for consistency
            
            return df.fillna(method='ffill')
        except Exception as e:
            logger.error(f"Error processing data for {ticker}: {e}")
            return pd.DataFrame()
    
    def has_new_data(self, ticker, new_data, market):
        """Check if new data contains timestamps not in existing file"""
        try:
            clean_ticker = ticker.replace('.L', '').replace('.KA', '')
            filename = f"{clean_ticker}.csv" if market == 'US' else f"{clean_ticker}.{market[0]}.csv"
            filepath = os.path.join(self.market_dirs[market], filename)
            
            if not os.path.exists(filepath):
                return True  # File doesn't exist, so all data is new
            
            # Load existing data timestamps
            existing_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            existing_timestamps = set(existing_data.index)
            new_timestamps = set(new_data.index)
            
            # Check if there are any new timestamps
            new_count = len(new_timestamps - existing_timestamps)
            
            if new_count > 0:
                logger.info(f"{ticker}: {new_count} new data points available")
                return True
            else:
                logger.info(f"{ticker}: No new data (latest: {new_data.index.max()})")
                return False
                
        except Exception as e:
            logger.error(f"Error checking new data for {ticker}: {e}")
            return True  # Default to updating on error
    
    def fetch_market_data(self, market, period="1d"):
        """Fetch data for all tickers in a specific market"""
        logger.info(f"Fetching data for {market} market...")
        tickers = self.market_tickers[market]
        market_data = {}
        new_data_count = 0
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, interval=self.interval)
                
                if hist.empty:
                    logger.info(f"{ticker}: No data received from API")
                    continue
                
                # Process data
                processed_data = self.process_data(hist, ticker, market)
                if processed_data.empty:
                    continue
                
                # Check if this data contains new timestamps
                if self.has_new_data(ticker, processed_data, market):
                    market_data[ticker] = processed_data
                    new_data_count += 1
                    logger.info(f"{ticker}: {len(processed_data)} data points (will update)")
                    
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
        
        logger.info(f"{market} market: {new_data_count}/{len(tickers)} symbols have new data")
        return market_data
    
    def save_market_data(self, market, market_data):
        """Save market data to individual CSV files matching project structure"""
        if not market_data:
            logger.info(f"{market} market: No new data to save")
            return
            
        market_dir = self.market_dirs[market]
        updated_files = 0
        
        for ticker, data in market_data.items():
            try:
                # Clean ticker for filename
                clean_ticker = ticker.replace('.L', '').replace('.KA', '')
                filename = f"{clean_ticker}.csv" if market == 'US' else f"{clean_ticker}.{market[0]}.csv"
                filepath = os.path.join(market_dir, filename)
                
                # Load existing data if file exists
                if os.path.exists(filepath):
                    existing_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    
                    # Combine with new data and remove duplicates
                    combined_data = pd.concat([existing_data, data])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data = combined_data.sort_index()
                    
                    # Only save if we actually added new rows
                    if len(combined_data) > len(existing_data):
                        combined_data.to_csv(filepath)
                        new_rows = len(combined_data) - len(existing_data)
                        logger.info(f"{ticker}: Added {new_rows} new rows (total: {len(combined_data)})")
                        updated_files += 1
                    else:
                        logger.info(f"{ticker}: No new rows to add")
                else:
                    # New file
                    data.to_csv(filepath)
                    logger.info(f"{ticker}: Created new file with {len(data)} records")
                    updated_files += 1
                    
            except Exception as e:
                logger.error(f"Error saving {ticker} data: {e}")
        
        logger.info(f"{market} market: Updated {updated_files} files")
    
    def stream_data_continuous(self):
        """Continuously stream data for all markets every 5 minutes"""
        logger.info("Starting continuous multi-market data streaming...")
        logger.info(f"Markets: {list(self.market_tickers.keys())}")
        logger.info(f"Interval: {self.interval}")
        logger.info("Smart update: Only saves when new data is available")
        
        while True:
            try:
                current_time = datetime.now()
                logger.info(f"\n{'='*60}")
                logger.info(f"Streaming cycle started: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check which markets are currently active
                active_markets = self.get_active_markets()
                
                if not active_markets:
                    logger.info("All markets are currently closed")
                    # Check next market opening
                    next_open_time = self.get_next_market_open()
                    if next_open_time:
                        wait_seconds = (next_open_time - datetime.now()).total_seconds()
                        logger.info(f"Next market opens in {wait_seconds/3600:.1f} hours")
                        time.sleep(min(300, wait_seconds))  # Wait 5 min or until market opens
                    else:
                        time.sleep(300)  # Default 5-minute wait
                    continue
                
                logger.info(f"Active markets: {active_markets}")
                
                total_updates = 0
                # Fetch data for each active market
                for market in active_markets:
                    logger.info(f"\nProcessing {market} market...")
                    market_data = self.fetch_market_data(market)
                    
                    if market_data:
                        self.save_market_data(market, market_data)
                        total_updates += len(market_data)
                        logger.info(f"{market} market: {len(market_data)} symbols updated")
                    else:
                        logger.info(f"{market} market: No new data to update")
                
                if total_updates > 0:
                    logger.info(f"Streaming cycle completed: {total_updates} total updates")
                else:
                    logger.info(f"Streaming cycle completed: No updates needed")
                    
                logger.info("Waiting 5 minutes for next cycle...")
                time.sleep(300)  # Wait 5 minutes
                
            except KeyboardInterrupt:
                logger.info("Streamer stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                time.sleep(30)  # Wait 30 seconds before retrying
    
    def get_next_market_open(self):
        """Calculate when the next market opens"""
        earliest_open = None
        
        for market in self.market_tickers.keys():
            tz = self.market_timezones[market]
            now = datetime.now(tz)
            
            # Calculate next market open for this market
            open_hour, open_min = self.market_hours[market]['open']
            
            if now.weekday() >= 5:  # Weekend
                days_ahead = 7 - now.weekday()
                next_open = now + timedelta(days=days_ahead)
            else:
                if now.hour < open_hour or (now.hour == open_hour and now.minute < open_min):
                    next_open = now  # Today
                else:
                    next_open = now + timedelta(days=1)  # Tomorrow
            
            next_open = next_open.replace(hour=open_hour, minute=open_min, second=0, microsecond=0)
            
            # Convert to UTC for comparison
            next_open_utc = next_open.astimezone(pytz.UTC)
            
            if earliest_open is None or next_open_utc < earliest_open:
                earliest_open = next_open_utc
        
        return earliest_open.astimezone(pytz.timezone('UTC')).replace(tzinfo=None) if earliest_open else None
    
    def fetch_historical_data(self, days=30):
        """Fetch historical data for all markets to update existing datasets"""
        logger.info(f"Fetching {days} days of historical data for all markets...")
        
        for market in self.market_tickers.keys():
            logger.info(f"\nFetching historical data for {market} market...")
            market_data = self.fetch_market_data(market, period=f"{days}d")
            
            if market_data:
                self.save_market_data(market, market_data)
                logger.info(f"Historical data updated for {market} ({len(market_data)} symbols)")
        
        logger.info("Historical data fetch completed for all markets")

def main():
    """Main function to run the multi-market stock data streamer"""
    logger.info("Multi-Market Stock Data Streamer")
    logger.info("Markets: US, London, Pakistan")
    logger.info("Interval: 5 minutes")
    
    # Initialize streamer
    streamer = StockDataStreamer(interval="5m")
    
    try:
        # First, update with recent historical data
        streamer.fetch_historical_data(days=7)
        
        # Start continuous streaming
        streamer.stream_data_continuous()
        
    except KeyboardInterrupt:
        logger.info("Streamer stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
