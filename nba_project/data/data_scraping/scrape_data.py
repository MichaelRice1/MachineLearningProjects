import os 
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import time
import sys
import asyncio
import nest_asyncio

nest_asyncio.apply()



SEASONS = list(range(2024,2025))

DATA_DIR = 'nba_project\data_scraping\data'
STANDINGS_DIR = os.path.join(DATA_DIR, 'standings')
SCORES_DIR = os.path.join(DATA_DIR, 'scores')
TEAMS_DIR = os.path.join(DATA_DIR, 'teams')
PLAYERS_DIR = os.path.join(DATA_DIR, 'players')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STANDINGS_DIR, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)
os.makedirs(TEAMS_DIR, exist_ok=True)
os.makedirs(PLAYERS_DIR, exist_ok=True)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def get_html(url, selector, sleep = 5, retries = 3):

    html = None

    for i in range(1,retries+1):
        time.sleep(sleep+1)

        try:
            async with async_playwright() as p:
                browser = await p.firefox.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout error on {url}, retrying {i}/{retries}")
            continue
        else:
            break

    return html


async def scrape_season(season):
    url = f'https://www.basketball-reference.com/leagues/NBA_{season}_games.html'
    selector = '#content .filter'
    html = await get_html(url, selector)
    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    hrefs = [l["href"] for l in links]  
    standings_pages = [f"https://www.basketball-reference.com{l}" for l in hrefs]

    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, "#all_schedule")
        with open(save_path, 'w+') as f:
            f.write(html)


async def scrape_games(standings_file):
    
    with open(standings_file, 'r') as f:
        html = f.read()

    soup = BeautifulSoup(html,'html.parser')
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links]

    box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]
    
    for url in box_scores:
        score_save_path = os.path.join(SCORES_DIR, url.split("/")[-1])

        if os.path.exists(score_save_path):
            continue

        html = await get_html(url, "#content")            

        if not html:
            continue
        
        with open(score_save_path, 'w+', encoding='utf-8') as f:
            f.write(html)



if __name__ == '__main__':



    #updating season data

    # for season in SEASONS:
    #     asyncio.run(scrape_season(season))


    #updating game data

    standings_files = os.listdir(STANDINGS_DIR)
    files = [s for s in standings_files]
    for f in files:
        file_path = os.path.join(STANDINGS_DIR, f)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        asyncio.run(scrape_games(file_path))
