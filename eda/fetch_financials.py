from yahooquery import Ticker
import pandas as pd
from tabulate import tabulate
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Suppress warnings
warnings.filterwarnings('ignore')

def get_price_on_date(ticker_obj, target_date_str):
    """
    Fetches the closing price for a ticker on or immediately after the target date.
    """
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    # Fetch a small window to ensure we catch a trading day
    end_date = (target_date + timedelta(days=10)).strftime('%Y-%m-%d')
    
    try:
        hist = ticker_obj.history(start=target_date_str, end=end_date)
        if not hist.empty:
            return hist.iloc[0]['Close']
    except Exception:
        pass
    return None

def get_financial_data(ticker_symbol):
    try:
        t_yq = Ticker(ticker_symbol)
        t_yf = yf.Ticker(ticker_symbol)
        
        # 1. Fetch PE Ratio (TTM)
        pe_val = None
        pe_str = "?"
        try:
            details = t_yq.summary_detail
            if isinstance(details, dict) and ticker_symbol in details:
                summary = details[ticker_symbol]
                if isinstance(summary, dict):
                    pe = summary.get('trailingPE') or summary.get('forwardPE')
                    if isinstance(pe, (int, float)):
                        pe_val = float(pe)
                        pe_str = f"{pe:.2f}"
        except:
            pass

        # 2. Fetch Quarterly Revenue (Latest Growth)
        growth_val = None
        growth_str = "?"
        try:
            df = t_yq.income_statement(frequency='q', trailing=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                if 'periodType' in df.columns:
                    df = df[df['periodType'] != 'TTM']
                
                rev_col = next((c for c in ['TotalRevenue', 'totalRevenue'] if c in df.columns), None)
                if rev_col and 'asOfDate' in df.columns:
                    rev_series = df.set_index('asOfDate')[rev_col].dropna()
                    rev_series.index = pd.to_datetime(rev_series.index)
                    rev_series = rev_series.sort_index(ascending=False)
                    
                    if len(rev_series) >= 2:
                        current_date = rev_series.index[0]
                        current_rev = rev_series.iloc[0]
                        for j in range(1, len(rev_series)):
                            prev_date = rev_series.index[j]
                            if 300 <= (current_date - prev_date).days <= 430:
                                prev_rev = rev_series.iloc[j]
                                if prev_rev and prev_rev != 0:
                                    growth = (current_rev / prev_rev - 1) * 100
                                    growth_val = float(growth)
                                    growth_str = f"{growth:.1f}%"
                                    break
        except:
            pass

        # 3. Fetch Historical Prices
        price_jan_2025 = get_price_on_date(t_yf, '2025-01-01')
        price_jan_2026 = get_price_on_date(t_yf, '2026-01-01')
        price_friday = get_price_on_date(t_yf, '2026-02-13')

        def fmt_p(p): return f"${p:.2f}" if p else "?"
        def fmt_pct(cur, prev):
            if cur and prev and prev != 0:
                diff = (cur / prev - 1) * 100
                return f"{diff:+.1f}%"
            return "?"

        row = [
            ticker_symbol,
            pe_str,
            growth_str,
            fmt_p(price_jan_2025),
            fmt_p(price_jan_2026),
            fmt_pct(price_jan_2026, price_jan_2025),
            fmt_p(price_friday),
            fmt_pct(price_friday, price_jan_2026)
        ]
        
        # Metadata for plotting
        meta = {
            'Ticker': ticker_symbol,
            'PE': pe_val,
            'Growth': growth_val
        }
        
        return row, meta
        
    except Exception:
        return [ticker_symbol] + ["?"] * 7, None

def create_opportunity_graph(plot_data):
    """
    Creates an Opportunity Graph: Revenue Growth vs PE Ratio.
    Ideal stocks are in the 'Opportunity Zone' (High Growth, Low PE).
    """
    df_plot = pd.DataFrame([d for d in plot_data if d['PE'] is not None and d['Growth'] is not None])
    
    if df_plot.empty:
        print("Not enough data to create the opportunity graph.")
        return

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create the scatter plot
    scatter = sns.scatterplot(
        data=df_plot, 
        x='Growth', 
        y='PE', 
        s=200, 
        color='blue', 
        alpha=0.7,
        edgecolor='black'
    )
    
    # Add labels to points
    for i in range(df_plot.shape[0]):
        plt.text(
            df_plot.Growth[i]+0.5, 
            df_plot.PE[i]+0.5, 
            df_plot.Ticker[i], 
            fontsize=12, 
            weight='bold'
        )

    # Median lines to define quadrants
    plt.axvline(df_plot['Growth'].median(), color='red', linestyle='--', alpha=0.5, label='Median Growth')
    plt.axhline(df_plot['PE'].median(), color='green', linestyle='--', alpha=0.5, label='Median PE')

    # Label Quadrants
    x_max = plt.xlim()[1]
    y_max = plt.ylim()[1]
    plt.text(x_max*0.75, y_max*0.1, "OPPORTUNITY\n(High Growth, Low PE)", color='green', fontweight='bold', ha='center')
    plt.text(x_max*0.1, y_max*0.9, "OVERVALUED?", color='red', fontweight='bold', ha='center')

    plt.title('Opportunity Graph: Revenue Growth vs PE Ratio', fontsize=16, pad=20)
    plt.xlabel('Revenue Growth (%)', fontsize=14)
    plt.ylabel('PE Ratio (TTM)', fontsize=14)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('opportunity_graph.png')
    print("\nOpportunity graph saved to 'opportunity_graph.png'")

def main():
    tickers = ['PLTR', 'SHOP', 'ADBE', 'CRM', 'DDOG', 'NOW', 'RBRK', 'TEAM', 'TTD', 'FIGS']
    print(f"Fetching data for: {', '.join(tickers)}")
    
    table_data = []
    plot_data = []
    for t in tickers:
        print(f"Processing {t}...", end=" ", flush=True)
        row, meta = get_financial_data(t)
        table_data.append(row)
        if meta:
            plot_data.append(meta)
        print("Done.")
    
    headers = [
        "Ticker", 
        "PE (TTM)", 
        "Rev Growth", 
        "Jan 1 2025", 
        "Jan 1 2026", 
        "25-26 %", 
        "Feb 13 2026", 
        "YTD %"
    ]
    
    # 1. Output to Console
    print("\n" + "="*110)
    print("FINANCIAL & STOCK PERFORMANCE SUMMARY".center(110))
    print("="*110)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*110)
    
    # 2. Export to Markdown
    md_table = tabulate(table_data, headers=headers, tablefmt="github")
    with open('report.md', 'w') as f:
        f.write("# Financial & Stock Performance Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(md_table)
        f.write("\n\n### Opportunity Analysis\n")
        f.write("![Opportunity Graph](opportunity_graph.png)\n\n")
        f.write("#### Notes:\n")
        f.write("- **Rev Growth**: Most recent Year-over-Year quarterly revenue growth.\n")
        f.write("- **Prices**: Closing price on the specified date or the next available trading day.\n")
        f.write("- **YTD %**: Percentage change from Jan 1, 2026 to Feb 13, 2026.\n")
        f.write("- **Opportunity Graph**: Stocks in the bottom-right quadrant have high growth and low PE relative to peers.\n")
    
    print(f"\nReport exported to 'report.md'")

    # 3. Create Graph
    create_opportunity_graph(plot_data)

if __name__ == "__main__":
    main()
