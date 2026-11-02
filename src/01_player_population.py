"""
Phase 1: Build player population from FBRef.
Uses system Chrome (channel='chrome') to bypass Cloudflare.
"""

import asyncio
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from playwright.async_api import async_playwright

SEASONS = list(range(2011, 2026))

def fbref_url(end_year):
    if end_year == 2025:
        return "https://fbref.com/en/comps/9/stats/Premier-League-Stats"
    return (
        f"https://fbref.com/en/comps/9/{end_year-1}-{end_year}"
        f"/stats/{end_year-1}-{end_year}-Premier-League-Stats"
    )

def season_label(end_year):
    return f"{end_year-1}-{end_year}"

async def scrape_all():
    os.makedirs("data/raw/seasons", exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="chrome",
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context()
        page = await context.new_page()
        all_dfs = []
        for yr in SEASONS:
            url = fbref_url(yr)
            season = season_label(yr)
            print(f"  Scraping {season}...", flush=True)
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_selector("#stats_standard", timeout=30000)
                html = await page.content()
                tables = pd.read_html(html, attrs={"id": "stats_standard"})
                df = tables[0]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
                df["season"] = season
                out = f"data/raw/seasons/{season}.csv"
                df.to_csv(out, index=False)
                print(f"    {len(df)} rows -> {out}", flush=True)
                all_dfs.append(df)
            except Exception as e:
                print(f"    Error: {e}", flush=True)
            await asyncio.sleep(5)
        await browser.close()

    if not all_dfs:
        print("No data collected.")
        exit(1)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("data/raw/fbref_pl_all_seasons_raw.csv", index=False)
    print(f"\nTotal rows: {len(combined):,}")
    print("Saved: data/raw/fbref_pl_all_seasons_raw.csv")

if __name__ == "__main__":
    asyncio.run(scrape_all())
