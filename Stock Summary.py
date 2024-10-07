import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
from datetime import datetime, timedelta

class StockApp:
    def __init__(self, root):
        self.root = root

        # Start with a small window
        self.root.title("Stock Financial Summary")
        self.root.geometry("350x90")
        self.root.configure(bg='#2b2b2b')

        # UI Elements
        self.header_frame = tk.Frame(self.root, bg='#2b2b2b')
        self.header_frame.grid(row=0, column=0, columnspan=2, pady=int(20 * 0.9))

        self.ticker_label = tk.Label(self.header_frame, text="Ticker:", font=("Arial", int(14 * 0.9), "bold"), fg='white', bg='#2b2b2b')
        self.ticker_label.grid(row=0, column=0, padx=(0, 10))

        self.ticker_entry = tk.Entry(self.header_frame, font=("Arial", int(14 * 0.9)), bg='#3a3a3a', fg='white', insertbackground='white', width=10)
        self.ticker_entry.grid(row=0, column=1)

        self.get_data_button = tk.Button(self.header_frame, text="Populate Data", command=self.get_stock_data,
                                         font=("Arial", int(12 * 0.9)), bg='#3a3a3a', fg='white', activebackground='#3a3a3a', padx=int(10 * 0.9), pady=int(5 * 0.9))
        self.get_data_button.grid(row=0, column=2, padx=(10, 0))

        # Frames for data sections, initially hidden
        self.left_frame = tk.Frame(self.root, bg='#2b2b2b')
        self.middle_frame = tk.Frame(self.root, bg='#2b2b2b')
        self.right_frame = tk.Frame(self.root, bg='#2b2b2b')
        self.chart_frame = tk.Frame(self.root, bg='#2b2b2b')

    def classify_sentiment(self, score):
        if score <= -0.6:
            return "Strongly Negative"
        elif -0.6 < score <= -0.2:
            return "Negative"
        elif -0.2 < score <= 0.2:
            return "Neutral"
        elif 0.2 < score <= 0.6:
            return "Positive"
        else:
            return "Strongly Positive"

    # Diese Methode kommt in die StockApp Klasse
    def fetch_news_sentiment(self, ticker):
        import pandas as pd
        from bs4 import BeautifulSoup
        from urllib.request import urlopen, Request
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import random
        from datetime import datetime, timedelta
        import nltk

        nltk.download('vader_lexicon')

        # List of user agents for web scraping
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0',
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
        ]

        finwiz_url = f'https://finviz.com/quote.ashx?t={ticker}'
        user_agent = random.choice(user_agents)
        news_table = None

        try:
            req = Request(url=finwiz_url, headers={'user-agent': user_agent})
            resp = urlopen(req)
            html = BeautifulSoup(resp, features="lxml")
            news_table = html.find(id='news-table')
        except Exception as e:
            print(f"Error fetching news: {e}")
            return None, None

        if news_table is None:
            return None, None

        parsed_news = []
        for row in news_table.findAll('tr'):
            text = row.a.get_text()
            date_scrape = row.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]
                date = "Today"
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            parsed_news.append([ticker, date, time, text])

        news_df = pd.DataFrame(parsed_news, columns=['Ticker', 'Date', 'Time', 'Headline'])

        # Convert date format
        def parse_date(date_str):
            if date_str == "Today":
                return datetime.today().date()
            else:
                try:
                    return datetime.strptime(date_str, '%b-%d-%y').date()
                except ValueError:
                    return None

        news_df['Date'] = news_df['Date'].apply(parse_date)
        three_months_ago = datetime.today().date() - timedelta(days=90)
        news_df_filtered = news_df[news_df['Date'] >= three_months_ago]

        if news_df_filtered.empty:
            return None, None

        analyzer = SentimentIntensityAnalyzer()
        news_df_filtered['Sentiment'] = news_df_filtered['Headline'].apply(
            lambda x: analyzer.polarity_scores(x)['compound'])

        average_sentiment = round(news_df_filtered['Sentiment'].mean(), 2)

        return news_df_filtered, average_sentiment

    def get_stock_data(self):
        ticker = self.ticker_entry.get().strip().upper()
        if not ticker:
            messagebox.showerror("Input Error", "Please enter a stock ticker.")
            return

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            self.hist = hist  # Save hist for prediction use later

            if hist.empty:
                messagebox.showerror("Data Error", "No data found for ticker {}".format(ticker))
                return

            # Expand the window size for showing the results
            self.root.geometry("1150x950")

            # Extract stock information
            info = stock.info

            # Extract additional information
            industry = info.get('industryDisp', 'N/A')
            sector = info.get('sectorDisp', 'N/A')
            marketcap = info.get('marketCap', 'N/A')
            sharesoutstanding = info.get('sharesOutstanding', 'N/A')
            freecashflow = info.get('freeCashflow', 'N/A')
            heldpercentinsiders = info.get('heldPercentInsiders', 'N/A')
            heldpercentinstitutions = info.get('heldPercentInstitutions', 'N/A')

            # Valuation Ratios
            pe_ratio = round(info.get('trailingPE', 'N/A'), 2) if isinstance(info.get('trailingPE'), (int, float)) else 'N/A'
            ps_ratio = round(info.get('priceToSalesTrailing12Months', 'N/A'), 2) if isinstance(info.get('priceToSalesTrailing12Months'), (int, float)) else 'N/A'
            pb_ratio = round(info.get('priceToBook', 'N/A'), 2) if isinstance(info.get('priceToBook'), (int, float)) else 'N/A'

            # Dividend Information
            dividend_yield = round(info.get('dividendYield', 'N/A') * 100, 2) if isinstance(info.get('dividendYield'), (int, float)) else 'N/A'
            payout_ratio = round(info.get('payoutRatio', 'N/A') * 100, 2) if isinstance(info.get('payoutRatio'), (int, float)) else 'N/A'

            # Stock Performance Metrics
            beta = round(info.get('beta', 'N/A'), 2) if isinstance(info.get('beta'), (int, float)) else 'N/A'
            fifty_two_week_low = round(info.get('fiftyTwoWeekLow', 'N/A'), 2) if isinstance(info.get('fiftyTwoWeekLow'), (int, float)) else 'N/A'
            fifty_two_week_high = round(info.get('fiftyTwoWeekHigh', 'N/A'), 2) if isinstance(info.get('fiftyTwoWeekHigh'), (int, float)) else 'N/A'

            # Sentiment Analysis
            recommendation_mean = round(info.get('recommendationMean', 'N/A'), 2) if isinstance(info.get('recommendationMean'), (int, float)) else 'N/A'
            target_high_price = round(info.get('targetHighPrice', 'N/A'), 2) if isinstance(info.get('targetHighPrice'), (int, float)) else 'N/A'
            target_low_price = round(info.get('targetLowPrice', 'N/A'), 2) if isinstance(info.get('targetLowPrice'), (int, float)) else 'N/A'
            target_mean_price = round(info.get('targetMeanPrice', 'N/A'), 2) if isinstance(info.get('targetMeanPrice'), (int, float)) else 'N/A'

            for widget in self.left_frame.winfo_children():
                widget.destroy()
            for widget in self.middle_frame.winfo_children():
                widget.destroy()
            for widget in self.right_frame.winfo_children():
                widget.destroy()
            for widget in self.chart_frame.winfo_children():
                widget.destroy()

            # Now, we can grid the data frames
            self.left_frame.grid(row=1, column=0, padx=int(20 * 0.9), pady=int(10 * 0.9), sticky='nw')
            self.middle_frame.grid(row=1, column=1, padx=int(20 * 0.9), pady=int(10 * 0.9), sticky='n')
            self.right_frame.grid(row=1, column=2, padx=int(20 * 0.9), pady=int(10 * 0.9), sticky='ne')
            self.chart_frame.grid(row=2, column=0, columnspan=3, padx=int(20 * 0.9), pady=int(20 * 0.9))

            # Display additional information
            self.display_section(self.left_frame, "Company Overview", {
                "Industry": industry,
                "Sector": sector,
                "Market Cap": f"${marketcap:,.2f}" if isinstance(marketcap, (int, float)) else marketcap,
                "Total Shares Outstanding": f"{sharesoutstanding:,}" if isinstance(sharesoutstanding,
                                                                                   (int, float)) else sharesoutstanding,
                "Insider Holding": f"{heldpercentinsiders * 100:.2f}%" if isinstance(heldpercentinsiders, (
                int, float)) else heldpercentinsiders,
                "Institution Holding": f"{heldpercentinstitutions * 100:.2f}%" if isinstance(heldpercentinstitutions, (
                int, float)) else heldpercentinstitutions
            }, 0)

            # Berechnung der Total Assets unter Verwendung von Total Debt und Debt to Equity
            if info.get('totalDebt') and info.get('debtToEquity'):
                equity = info.get('totalDebt') / info.get('debtToEquity')  # Ableitung des Eigenkapitals (Equity)
                total_assets = equity + info.get('totalDebt')  # Berechnung der Total Assets
            else:
                total_assets = None  # Wenn Total Debt oder Debt to Equity fehlt

            # Berechnung der Current Liabilities, wenn Quick Ratio und Current Ratio vorhanden sind
            if info.get('quickRatio') and info.get('currentRatio') and info.get('totalCash'):
                current_liabilities = (info.get('totalCash') / info.get('quickRatio')) * info.get('currentRatio')
            else:
                current_liabilities = None

            # Berechnung der Cash Ratio, wenn Current Liabilities berechnet werden konnten
            if info.get('totalCash') and current_liabilities:
                cash_ratio = round(info.get('totalCash') / current_liabilities, 2)
            else:
                cash_ratio = None

            # Finanzkennzahlen
            financial_ratios = {
                "Return Ratios": {
                    "Return on Equity (ROE)": f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get(
                        'returnOnEquity') is not None else "N/A",
                    "Return on Assets (ROA)": f"{info.get('returnOnAssets', 0) * 100:.2f}%" if info.get(
                        'returnOnAssets') is not None else "N/A",
                    "Return on Invested Capital (ROIC)": f"{((info.get('netIncomeToCommon', 0) - info.get('dividendsPaid', 0)) / (info.get('totalDebt', 1) + equity)) * 100:.2f}%" if (
                                info.get('netIncomeToCommon') and info.get('totalDebt') and equity) else "N/A"
                },
                "Profitability Ratios": {
                    "Gross Margin": f"{info.get('grossMargins', 0) * 100:.2f}%" if info.get(
                        'grossMargins') is not None else "N/A",
                    "Operating Profit Margin": f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get(
                        'operatingMargins') is not None else "N/A",
                    "Net Profit Margin": f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get(
                        'profitMargins') is not None else "N/A",
                },
                "Liquidity Ratios": {
                    "Current Ratio": round(info.get("currentRatio", 'N/A'), 2) if isinstance(info.get("currentRatio"), (int, float)) else "N/A",
                    "Quick Ratio": round(info.get("quickRatio", 'N/A'), 2) if isinstance(info.get("quickRatio"), (int, float)) else "N/A",
                    "Cash Ratio": f"{cash_ratio:.2f}" if cash_ratio is not None else "N/A"
                },
                "Solvency Ratios": {
                    "Total Assets to Equity": f"{total_assets / equity:.2f}" if total_assets and equity else "N/A",
                    "Debt to Equity": round(info.get('debtToEquity', 'N/A'), 2) if isinstance(info.get('debtToEquity'), (int, float)) else "N/A",
                    "Debt to EBITDA": f"{info.get('totalDebt', 0) / info.get('ebitda', 1):.2f}" if info.get(
                        'totalDebt') and info.get('ebitda') else "N/A"
                },
                "Valuation Ratios": {
                    "Price to Earnings (P/E)": pe_ratio,
                    "Price to Sales (P/S)": ps_ratio,
                    "Price to Book (P/B)": pb_ratio
                },
                "Dividend Information": {
                    "Dividend Yield": f"{dividend_yield:.2f}%" if isinstance(dividend_yield, (int, float)) else "N/A",
                    "Payout Ratio": f"{payout_ratio:.2f}%" if isinstance(payout_ratio, (int, float)) else "N/A"
                },
                "Stock Performance Metrics": {
                    "Beta": beta,
                    "52-Week Low": fifty_two_week_low,
                    "52-Week High": fifty_two_week_high
                },
                "Sentiment Analysis": {
                    "Recommendation Mean": recommendation_mean,
                    "Target High Price": target_high_price,
                    "Target Low Price": target_low_price,
                    "Target Mean Price": target_mean_price
                }
            }

            news_df, avg_sentiment = self.fetch_news_sentiment(ticker)

            if news_df is not None:
                # Get sentiment classification
                sentiment_classification = self.classify_sentiment(avg_sentiment)

                # Display sentiment with classification
                sentiment_data = {
                    "Average news sentiment (3 months)": f"{avg_sentiment} ({sentiment_classification})",
                }
                self.display_section(self.left_frame, "News Sentiment", sentiment_data, 12)
            else:
                sentiment_data = {
                    "Sentiment scores for recent news": "No news found in the last 3 months"
                }
                self.display_section(self.left_frame, "News Sentiment", sentiment_data, 12)



            # Display additional information
            self.display_section(self.left_frame, "Company Overview", {
                "Industry": industry,
                "Sector": sector,
                "Market Cap": f"${marketcap:,.2f}" if isinstance(marketcap, (int, float)) else marketcap,
                "Total Shares Outstanding": f"{sharesoutstanding:,}" if isinstance(sharesoutstanding,
                                                                                   (int, float)) else sharesoutstanding,
                "Insider Holding": f"{heldpercentinsiders * 100:.2f}%" if isinstance(heldpercentinsiders, (
                int, float)) else heldpercentinsiders,
                "Institution Holding": f"{heldpercentinstitutions * 100:.2f}%" if isinstance(heldpercentinstitutions, (
                int, float)) else heldpercentinstitutions
            }, 0)

            # Display the financial ratios in a structured format
            self.display_section(self.middle_frame, "Return Ratios", financial_ratios["Return Ratios"], 0)
            self.display_section(self.middle_frame, "Profitability Ratios", financial_ratios["Profitability Ratios"], 6)
            self.display_section(self.middle_frame, "Liquidity Ratios", financial_ratios["Liquidity Ratios"], 12)
            self.display_section(self.middle_frame, "Solvency Ratios", financial_ratios["Solvency Ratios"], 18)
            self.display_section(self.right_frame, "Valuation Ratios", financial_ratios["Valuation Ratios"], 0)
            self.display_section(self.right_frame, "Dividend Information", financial_ratios["Dividend Information"], 6)
            self.display_section(self.right_frame, "Stock Performance Metrics", financial_ratios["Stock Performance Metrics"], 12)
            self.display_section(self.right_frame, "Analyst Commentary", financial_ratios["Sentiment Analysis"], 18)

            # Plotting stock price data
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            ax1.plot(self.hist.index, self.hist['Close'], label='Closing Price', color='#ba3b46', linewidth=1.2)
            ax1.set_title(f'{ticker} Stock Price (Last 1 Year)', fontsize=12, color='white')
            ax1.set_xlabel('Date', fontsize=9, color='white')
            ax1.set_ylabel('Price (USD)', fontsize=9, color='white')
            ax1.tick_params(colors='white')
            ax1.legend(facecolor='#2b2b2b', edgecolor='white', fontsize=7, labelcolor='white')
            fig1.patch.set_facecolor('#2b2b2b')  # Match the figure background to the prediction chart
            ax1.set_facecolor('#2b2b2b')  # Match the plot area background to the prediction chart
            fig1.tight_layout()
            ax1.grid(False)

            # Change the color of the spines (borders) to white, similar to prediction chart
            ax1.spines['top'].set_color('white')
            ax1.spines['bottom'].set_color('white')
            ax1.spines['left'].set_color('white')
            ax1.spines['right'].set_color('white')

            # Create a frame for the stock price chart and grid it to column 0
            price_chart_frame = tk.Frame(self.chart_frame, bg='white')
            price_chart_frame.grid(row=0, column=0, padx=int(20 * 0.9), pady=int(10 * 0.9))

            # Embedding the plot in Tkinter for the stock price
            canvas1 = FigureCanvasTkAgg(fig1, master=price_chart_frame)
            canvas1.draw()
            canvas1.get_tk_widget().grid()

            # Call the prediction function and plot future predictions
            self.predict_stock_price()


        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_section(self, frame, title, data, start_row):
            section_label = tk.Label(frame, text=title, font=("Arial", int(14 * 0.9), "bold"), fg='white', bg='#2b2b2b')
            section_label.grid(row=start_row, column=0, sticky='w', padx=int(10 * 0.9), pady=int(10 * 0.9))
            start_row += 1
            for key, value in data.items():
                label_key = tk.Label(frame, text=f"{key}:", font=("Arial", int(12 * 0.9)), fg='white', bg='#2b2b2b')
                label_key.grid(row=start_row, column=0, sticky='w', padx=int(20 * 0.9), pady=int(5 * 0.9))
                label_value = tk.Label(frame, text=value, font=("Arial", int(12 * 0.9)), fg='white', bg='#2b2b2b')
                label_value.grid(row=start_row, column=1, sticky='w', padx=int(10 * 0.9), pady=int(5 * 0.9))
                start_row += 1

    def predict_stock_price(self):
        if not hasattr(self, 'hist') or self.hist.empty:
            messagebox.showerror("Prediction Error", "Please load stock data first.")
            return

        try:
            # Prepare data for prediction for 6 months (180 days)
            self.hist['Prediction'] = self.hist['Close'].shift(-180)
            X = np.array(self.hist['Close']).reshape(-1, 1)  # Feature variable - closing price
            y = np.array(self.hist['Prediction'])  # Target variable - future closing price

            # Remove the last 180 rows which have NaN values in 'Prediction'
            X = X[:-180]
            y = y[:-180]

            # Split data into training and testing sets (80% training, 20% testing)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model using Linear Regression
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict future prices for the next 180 days (6 months)
            future_X = np.array(self.hist['Close'])[-180:].reshape(-1, 1)
            future_predictions = model.predict(future_X)

            # Plotting future predictions (inside predict_stock_price function)
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            # Setting the linestyle to '--' for dashed line
            ax2.plot(np.arange(1, 181), future_predictions, label='Future Predictions', color='#ba3b46',
                     linewidth=1.2, linestyle='--')
            ax2.set_title('Machine Learning Prediction for Next 6 Months', fontsize=12, color='white')
            ax2.set_xlabel('Days into the Future', fontsize=9, color='white')
            ax2.set_ylabel('Predicted Price (USD)', fontsize=9, color='white')
            ax2.tick_params(colors='white')
            ax2.legend(facecolor='#2b2b2b', edgecolor='white', fontsize=8, labelcolor='white')
            fig2.patch.set_facecolor('#2b2b2b')  # Keep the background dark
            ax2.set_facecolor('#2b2b2b')  # Keep the plot area dark
            fig2.tight_layout()
            ax2.grid(False)

            # Change the color of the spines (borders)
            ax2.spines['top'].set_color('white')
            ax2.spines['bottom'].set_color('white')
            ax2.spines['left'].set_color('white')
            ax2.spines['right'].set_color('white')

            # Create a frame for the stock price prediction chart and grid it to column 1
            prediction_chart_frame = tk.Frame(self.chart_frame, bg='white')  # Frame color
            prediction_chart_frame.grid(row=0, column=1, padx=int(20 * 0.9), pady=int(10 * 0.9))

            # Embedding the plot in Tkinter for the stock price prediction
            canvas2 = FigureCanvasTkAgg(fig2, master=prediction_chart_frame)
            canvas2.draw()
            canvas2.get_tk_widget().grid()

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()