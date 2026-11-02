import asyncio
from playwright.async_api import async_playwright

async def debug():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        url = "https://fbref.com/en/comps/9/2023-2024/stats/2023-2024-Premier-League-Stats"
        print(f"Loading: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(5)
        html = await page.content()
        with open("debug_page.html", "w") as f:
            f.write(html)
        print(f"Saved {len(html)} chars to debug_page.html")
        # Quick checks
        print(f"  'stats_standard' in html: {'stats_standard' in html}")
        print(f"  'Cloudflare' in html: {'Cloudflare' in html}")
        print(f"  'cf-browser-verification' in html: {'cf-browser-verification' in html}")
        print(f"  '<table' in html: {'<table' in html}")
        # Print first 500 chars of <title>
        import re
        title = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        print(f"  Page title: {title.group(1) if title else 'no title'}")
        await browser.close()

asyncio.run(debug())
