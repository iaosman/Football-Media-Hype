import requests, pandas as pd, json, os, time
from pathlib import Path

GUARDIAN_KEY = os.environ.get("GUARDIAN_KEY", "")
if not GUARDIAN_KEY:
    raise EnvironmentError(
        "Set GUARDIAN_KEY before running.\n"
        "Free key from https://open-platform.theguardian.com/access/\n"
        "Then: export GUARDIAN_KEY=your-key-here"
    )
BASE_URL = "https://content.guardianapis.com/search"

path = "data/raw/player_population.csv"
df = pd.read_csv(path)

out_dir = Path("data/raw/coverage")
out_dir.mkdir(parents=True, exist_ok=True)

SEASONS = {
    "2016-17": ("2016-08-01", "2017-05-31"),
    "2017-18": ("2017-08-01", "2018-05-31"),
    "2018-19": ("2018-08-01", "2019-05-31"),
    "2019-20": ("2019-08-01", "2020-07-26"),
    "2020-21": ("2020-09-12", "2021-05-31"),
    "2021-22": ("2021-08-01", "2022-05-31"),
    "2022-23": ("2022-08-01", "2023-05-31"),
    "2023-24": ("2023-08-01", "2024-05-31"),
}
SEASON_ORDER = list(SEASONS.keys())

def search_name(name):
    parts = name.strip().split()
    return f"{parts[0]} {parts[-1]}" if len(parts) >= 2 else name

def fetch_season(name, from_date, to_date):
    query = search_name(name)
    headlines = []
    page = 1
    total = 0
    while page <= 4:  # cap at 200 articles per player-season
        r = requests.get(BASE_URL, params={
            "q": f'"{query}"',
            "from-date": from_date,
            "to-date": to_date,
            "section": "football",
            "show-fields": "headline",
            "page-size": 50,
            "page": page,
            "api-key": GUARDIAN_KEY,
        }, timeout=15)
        if r.status_code != 200:
            break
        resp = r.json()["response"]
        total = resp["total"]
        for art in resp["results"]:
            if "fields" in art:
                headlines.append(art["fields"].get("headline", ""))
        if page >= resp["pages"]:
            break
        page += 1
        time.sleep(0.12)
    return total, headlines

done = 0
skipped = 0
errors = 0

for i, row in df.iterrows():
    name = str(row["player"]).strip()
    debut = str(row["debut_season"])
    slug = name.lower().replace(" ", "_").replace("/", "_")
    out_file = out_dir / f"{slug}.json"

    # Resume: skip if already collected
    if out_file.exists():
        skipped += 1
        continue

    debut_idx = SEASON_ORDER.index(debut) if debut in SEASON_ORDER else 0
    seasons_to_query = SEASON_ORDER[debut_idx:]

    player_data = {}
    for season in seasons_to_query:
        from_d, to_d = SEASONS[season]
        try:
            total, headlines = fetch_season(name, from_d, to_d)
            player_data[season] = {"total": total, "headlines": headlines}
            time.sleep(0.1)
        except Exception as e:
            player_data[season] = {"total": 0, "headlines": [], "error": str(e)}
            errors += 1

    with open(out_file, "w") as f:
        json.dump({"player": name, "seasons": player_data}, f)

    done += 1
    if done % 50 == 0:
        print(f"  {done} players done ({skipped} skipped, {errors} errors)")

print(f"\nDone. Collected: {done}, Skipped (cached): {skipped}, Errors: {errors}")
