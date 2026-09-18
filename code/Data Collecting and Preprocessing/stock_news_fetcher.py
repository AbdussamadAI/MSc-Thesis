import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import time
import logging
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockNewsFetcher:
    def __init__(self, output_dir=None):
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_news")
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
            
        logger.info(f"News will be saved to: {self.output_dir}")
        
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
        
    def process_news(self, news_list, ticker, market):
        """Process news data and add metadata"""
        try:
            if not news_list:
                logger.warning(f"No news received for {ticker}")
                return []
                
            processed_news = []
            for news in news_list:
                # Add processing timestamp and ticker info
                news['fetch_timestamp'] = datetime.now().isoformat()
                news['ticker'] = ticker.replace('.L', '').replace('.KA', '')
                news['market'] = market
                # Convert publish time to readable format if available
                if 'providerPublishTime' in news:
                    news['publish_datetime'] = datetime.fromtimestamp(news['providerPublishTime']).isoformat()
                processed_news.append(news)
                
            return processed_news
        except Exception as e:
            logger.error(f"Error processing news for {ticker}: {e}")
            return []
    
    def has_new_news(self, ticker, new_news, market):
        """Check if new news contains items not in existing file"""
        try:
            clean_ticker = ticker.replace('.L', '').replace('.KA', '')
            filename = f"{clean_ticker}_news.json"
            filepath = os.path.join(self.market_dirs[market], filename)
            
            if not os.path.exists(filepath):
                return True  # File doesn't exist, so all news is new
            
            # Load existing news
            with open(filepath, 'r') as f:
                existing_news = json.load(f)
            
            existing_titles = {item['title'] for item in existing_news}
            new_titles = {item['title'] for item in new_news}
            
            new_count = len(new_titles - existing_titles)
            
            if new_count > 0:
                logger.info(f"{ticker}: {new_count} new news articles available")
                return True
            else:
                logger.info(f"{ticker}: No new news")
                return False
                
        except Exception as e:
            logger.error(f"Error checking new news for {ticker}: {e}")
            return True  # Default to updating on error
    
    def fetch_market_news(self, market):
        """Fetch news for all tickers in a specific market"""
        logger.info(f"Fetching news for {market} market...")
        tickers = self.market_tickers[market]
        market_news = {}
        new_news_count = 0
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                news_list = stock.news
                
                if not news_list:
                    logger.info(f"{ticker}: No news received from API")
                    continue
                
                # Process news
                processed_news = self.process_news(news_list, ticker, market)
                if not processed_news:
                    continue
                
                # Check if this news contains new articles
                if self.has_new_news(ticker, processed_news, market):
                    market_news[ticker] = processed_news
                    new_news_count += 1
                    logger.info(f"{ticker}: {len(processed_news)} news articles (will update)")
                    
            except Exception as e:
                logger.error(f"Error fetching news for {ticker}: {e}")
        
        logger.info(f"{market} market: {new_news_count}/{len(tickers)} symbols have new news")
        return market_news
    
    def save_market_news(self, market, market_news):
        """Save market news to individual JSON files"""
        if not market_news:
            logger.info(f"{market} market: No new news to save")
            return
            
        market_dir = self.market_dirs[market]
        updated_files = 0
        
        for ticker, news_list in market_news.items():
            try:
                # Clean ticker for filename
                clean_ticker = ticker.replace('.L', '').replace('.KA', '')
                filename = f"{clean_ticker}_news.json"
                filepath = os.path.join(market_dir, filename)
                
                # Load existing news if file exists
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        existing_news = json.load(f)
                    
                    # Combine with new news, avoiding duplicates by title
                    existing_titles = {item['title'] for item in existing_news}
                    new_news = [item for item in news_list if item['title'] not in existing_titles]
                    
                    combined_news = existing_news + new_news
                    
                    # Sort by publish time if available
                    combined_news.sort(key=lambda x: x.get('providerPublishTime', 0), reverse=True)
                    
                    # Only save if we actually added new items
                    if new_news:
                        with open(filepath, 'w') as f:
                            json.dump(combined_news, f, indent=2)
                        logger.info(f"{ticker}: Added {len(new_news)} new articles (total: {len(combined_news)})")
                        updated_files += 1
                    else:
                        logger.info(f"{ticker}: No new articles to add")
                else:
                    # New file
                    with open(filepath, 'w') as f:
                        json.dump(news_list, f, indent=2)
                    logger.info(f"{ticker}: Created new file with {len(news_list)} articles")
                    updated_files += 1
                    
            except Exception as e:
                logger.error(f"Error saving news for {ticker}: {e}")
        
        logger.info(f"{market} market: Updated {updated_files} files")
    
    def fetch_all_news(self):
        """Fetch news for all markets"""
        logger.info("Fetching news for all markets...")
        
        for market in self.market_tickers.keys():
            logger.info(f"\nProcessing {market} market...")
            market_news = self.fetch_market_news(market)
            
            if market_news:
                self.save_market_news(market, market_news)
                logger.info(f"{market} market: {len(market_news)} symbols updated")
            else:
                logger.info(f"{market} market: No new news to update")
        
        logger.info("News fetch completed for all markets")

def main():
    """Main function to run the stock news fetcher"""
    logger.info("Stock News Fetcher")
    logger.info("Markets: US, London, Pakistan")
    
    # Initialize fetcher
    fetcher = StockNewsFetcher()
    
    try:
        # Fetch news for all markets
        fetcher.fetch_all_news()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()