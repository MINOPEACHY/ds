import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os


def generate_and_save_data():
    """生成脱敏后的电商数据集"""
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start_date, end_date)

    # --- 客户表 (Customers) ---
    num_customers = 1500
    df_customers = pd.DataFrame({
        'customer_id': [f'CUST-{i:04d}' for i in range(num_customers)],
        'signup_date': [start_date + timedelta(days=np.random.randint(0, 300)) for _ in range(num_customers)],
        'region': np.random.choice(['US-East', 'US-West', 'Canada', 'US-Central'], num_customers),
        'email_subscriber': np.random.choice([True, False], num_customers, p=[0.7, 0.3]),
        'acquisition_channel': np.random.choice(['Organic', 'Paid Search', 'Social', 'Email'],
                                                num_customers, p=[0.3, 0.35, 0.25, 0.1])
    })

    # --- 订单表 (Orders) - 使用虚构产品名称 ---
    orders = []
    platforms = ['Amazon', 'Shopify']
    # 虚构产品名称
    skus = {
        'AudioPro X1': 179.95,
        'WaterSound S2': 149.95,
        'ComfortFit C3': 169.95,
        'ActiveMove A4': 79.95
    }

    for _, cust in df_customers.iterrows():
        num_purchases = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        for _ in range(num_purchases):
            p_date = cust['signup_date'] + timedelta(days=np.random.randint(0, 60))
            if p_date > end_date:
                continue

            # BFCM 期间订单量翻倍
            is_bfcm = (p_date.month == 11 and 20 <= p_date.day <= 30)
            loop = 2 if is_bfcm else 1
            discount = np.random.uniform(0.15, 0.25) if is_bfcm else np.random.uniform(0, 0.05)

            for _ in range(loop):
                platform = np.random.choice(platforms, p=[0.65, 0.35])
                sku = np.random.choice(list(skus.keys()), p=[0.4, 0.2, 0.2, 0.2])
                base_price = skus[sku]
                final_price = base_price * (1 - discount)

                # 模拟 5% 退货率
                is_returned = np.random.random() < 0.05
                actual_revenue = 0 if is_returned else final_price

                orders.append([
                    f'ORD-{len(orders):05d}',
                    cust['customer_id'],
                    p_date,
                    platform,
                    sku,
                    base_price,
                    discount,
                    actual_revenue,
                    is_returned
                ])

    df_orders = pd.DataFrame(orders, columns=[
        'order_id', 'customer_id', 'date', 'platform', 'sku',
        'base_price', 'discount_rate', 'revenue', 'returned'
    ])

    # --- 营销表 (Marketing) ---
    marketing = []
    for d in date_range:
        # BFCM 期间广告投入翻4倍
        base_spend = 800 if d.month != 11 else 3500
        actual_spend = round(np.random.normal(base_spend, base_spend * 0.1), 2)

        # CPC 在 0.7-1.2 美元之间波动
        cpc = round(np.random.uniform(0.7, 1.2), 2)
        clicks = int(actual_spend / cpc)

        # 转化率 2-4%
        conversions = int(clicks * np.random.uniform(0.02, 0.04))

        # 展示次数 (假设 CTR 为 2%)
        impressions = int(clicks / 0.02)

        marketing.append([
            d.strftime('%Y-%m-%d'),
            actual_spend,
            impressions,
            clicks,
            cpc,
            conversions
        ])

    df_marketing = pd.DataFrame(marketing, columns=[
        'date', 'ad_spend', 'impressions', 'clicks', 'cpc', 'conversions'
    ])

    # --- 导出数据 ---
    print("--- 正在导出数据 ---")

    # 导出为 Excel
    with pd.ExcelWriter('audio_tech_raw_data.xlsx') as writer:
        df_orders.to_excel(writer, sheet_name='Orders', index=False)
        df_customers.to_excel(writer, sheet_name='Customers', index=False)
        df_marketing.to_excel(writer, sheet_name='Marketing', index=False)
    print("✅ 已生成 Excel: audio_tech_raw_data.xlsx")

    # 导出为 CSV
    if not os.path.exists('csv_data'):
        os.makedirs('csv_data')
    df_orders.to_csv('csv_data/orders.csv', index=False)
    df_customers.to_csv('csv_data/customers.csv', index=False)
    df_marketing.to_csv('csv_data/marketing.csv', index=False)
    print("✅ 已生成 CSV 文件夹: /csv_data/")

    # 导出为 SQLite
    conn = sqlite3.connect('audio_tech_ecommerce.db')
    df_orders.to_sql('orders', conn, if_exists='replace', index=False)
    df_customers.to_sql('customers', conn, if_exists='replace', index=False)
    df_marketing.to_sql('marketing', conn, if_exists='replace', index=False)
    conn.close()
    print("✅ 已生成 SQL 数据库: audio_tech_ecommerce.db")

    # 数据摘要
    print("\n--- 数据概览 ---")
    print(f"总订单量: {len(df_orders)}")
    print(f"有效订单量: {len(df_orders[df_orders['returned'] == False])}")
    print(f"总销售额: ${df_orders['revenue'].sum():,.2f}")
    print(f"平均客单价 (AOV): ${df_orders[df_orders['revenue'] > 0]['revenue'].mean():.2f}")
    print(f"退货率: {(df_orders['returned'].sum() / len(df_orders) * 100):.2f}%")
    print(f"数据周期: {df_orders['date'].min()} 至 {df_orders['date'].max()}")


if __name__ == "__main__":
    generate_and_save_data()
