"""
Retail Sales Intelligence
IEEE LGU AI/ML Cohort One - Week 3: Data Analysis with NumPy & Pandas
AI/ML Leads: Abdullah Faisal & Alina Irshad
"""

import numpy as np
import pandas as pd

DATA_PATH = "data/sales.csv"


def load_data(path):
    """Load the raw sales CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df):
    """Clean the raw data and print what was changed at every step."""
    print("=" * 60)
    print("DATA CLEANING REPORT")
    print("=" * 60)
    print(f"Raw rows loaded            : {len(df)}")

    # 1. Remove duplicate orders (same order_id)
    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    print(f"Duplicate orders removed   : {before - len(df)}")

    # 2. Convert numeric columns; invalid text like 'abc' becomes NaN
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 3. Negative values are invalid -> treat as missing
    df.loc[df["units_sold"] <= 0, "units_sold"] = np.nan
    df.loc[df["unit_price"] <= 0, "unit_price"] = np.nan

    print(f"Missing units_sold         : {df['units_sold'].isna().sum()}")
    print(f"Missing/invalid unit_price : {df['unit_price'].isna().sum()}")
    print(f"Missing region             : {df['region'].isna().sum()}")

    # 4. Rows without valid units cannot be used -> drop them
    before = len(df)
    df = df.dropna(subset=["units_sold"])
    print(f"Rows dropped (bad units)   : {before - len(df)}")

    # 5. Fill missing price with the median price of the same product
    df["unit_price"] = df.groupby("product_name")["unit_price"].transform(
        lambda s: s.fillna(s.median())
    )
    # Any product with no price at all cannot be priced -> drop
    before = len(df)
    df = df.dropna(subset=["unit_price"])
    print(f"Rows dropped (no price)    : {before - len(df)}")

    # 6. Missing region -> label as 'Unknown'
    df["region"] = df["region"].fillna("Unknown")

    df["units_sold"] = df["units_sold"].astype(int)
    print(f"Clean rows remaining       : {len(df)}")
    return df.reset_index(drop=True)


def add_revenue(df):
    """Revenue = units_sold x unit_price (calculated with NumPy)."""
    df["revenue"] = np.multiply(df["units_sold"].to_numpy(), df["unit_price"].to_numpy())
    return df


def business_metrics(df):
    """Overall metrics using NumPy arrays."""
    revenue = df["revenue"].to_numpy()
    units = df["units_sold"].to_numpy()
    total_revenue = np.sum(revenue)
    total_orders = df["order_id"].nunique()
    return {
        "total_revenue": total_revenue,
        "aov": total_revenue / total_orders,
        "total_units": int(np.sum(units)),
        "total_orders": total_orders,
        "max_order": np.max(revenue),
    }


def print_report(df, m):
    category_rev = (
        df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    )
    product_rev = (
        df.groupby("product_name")["revenue"].sum().sort_values(ascending=False)
    )
    region_rev = df.groupby("region")["revenue"].sum().sort_values(ascending=False)

    print("\n" + "=" * 60)
    print("RETAIL SALES SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Orders        : {m['total_orders']}")
    print(f"Total Units Sold    : {m['total_units']}")
    print(f"Total Revenue       : ${m['total_revenue']:,.2f}")
    print(f"Average Order Value : ${m['aov']:,.2f}")
    print(f"Largest Single Order: ${m['max_order']:,.2f}")

    print("\n--- Category Ranking (by revenue) ---")
    for rank, (name, val) in enumerate(category_rev.items(), 1):
        print(f"{rank}. {name:<12} ${val:>10,.2f}")

    print("\n--- Product Ranking (by revenue) ---")
    for rank, (name, val) in enumerate(product_rev.items(), 1):
        print(f"{rank:>2}. {name:<22} ${val:>10,.2f}")

    print(f"\nHighest-performing product: {product_rev.index[0]} "
          f"(${product_rev.iloc[0]:,.2f})")

    print("\n--- Region Breakdown ---")
    for name, val in region_rev.items():
        share = val / m["total_revenue"] * 100
        print(f"{name:<8} ${val:>10,.2f}  ({share:.1f}%)")

    # Filtering examples
    north = df[df["region"] == "North"]
    print(f"\n--- Filter: North region orders ---")
    print(f"Orders: {len(north)}, Revenue: ${north['revenue'].sum():,.2f}")

    mask = (df["date"] >= "2026-09-01") & (df["date"] <= "2026-09-07")
    week1 = df[mask]
    print(f"\n--- Filter: 1 Sep to 7 Sep 2026 ---")
    print(f"Orders: {len(week1)}, Revenue: ${week1['revenue'].sum():,.2f}")

    high_value = df[df["revenue"] >= 300].sort_values("revenue", ascending=False)
    print(f"\n--- High-value orders (revenue >= $300) ---")
    print(high_value[["order_id", "product_name", "region", "revenue"]]
          .to_string(index=False))


def main():
    raw = load_data(DATA_PATH)
    clean = clean_data(raw)
    clean = add_revenue(clean)
    metrics = business_metrics(clean)
    print_report(clean, metrics)


if __name__ == "__main__":
    main()
