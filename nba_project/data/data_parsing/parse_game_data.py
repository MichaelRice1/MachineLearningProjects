import os 
import pandas as pd
from bs4 import BeautifulSoup
import tqdm

SCORES_DIR = "nba_project/data_scraping/data/scores"

boxscores = os.listdir(SCORES_DIR)

box_scores = [os.path.join(SCORES_DIR, b) for b in boxscores if b.endswith(".html")]



def parse_html(box_score):
    with open(box_score, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr thead")]

    return soup

def read_score(soup):
    score = pd.read_html(str(soup), attrs={"id":"line_score"})[0]
    cols = list(score.columns)
    cols[0] = "Team"
    cols[-1] = "Total"
    score.columns = cols

    score = score[['Team','Total']]

    return score

def read_stats(soup,team, stat):

    df = pd.read_html(str(soup), attrs={"id":f"box-{team}-game-{stat}"},index_col=0)[0]
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [l["href"] for l in nav.find_all("a")]
    season_num = os.path.basename(hrefs[1]).split("_")[0]

    return season_num




if __name__ == '__main__':

    base_cols = None
    games = []

    for box_score in tqdm.tqdm(box_scores):
        soup = parse_html(box_score)
        score = read_score(soup)
        teams = list(score["Team"])

        summaries = []

        for team in teams:
            basic = read_stats(soup, team, "basic")
            advanced = read_stats(soup, team, "advanced")

            totals = pd.concat([basic.iloc[-1,:], advanced.iloc[-1,:]])
            totals.index = totals.index.str.lower()

            maxes = pd.concat([basic.iloc[:-1,:].max(), advanced.iloc[:-1,:].max()])
            maxes.index = maxes.index.str.lower() + "_max"

            summary = pd.concat([totals, maxes])

            if base_cols is None:
                base_cols = list(summary.index.drop_duplicates(keep='first'))
                base_cols = [b for b in base_cols if "bpm" not in b]

            summary = summary[base_cols]

            summaries.append(summary)
        
        game_summary = pd.concat(summaries, axis=1).T

        game = pd.concat([score, game_summary], axis=1)
        game["home"] = [0,1]

        game_opp = game.iloc[::-1].reset_index()
        game_opp.columns += "_opp"

        full_game = pd.concat([game, game_opp], axis=1)

        full_game["season"] = read_season_info(soup)

        full_game["date"] = os.path.basename(box_score)[:8]
        full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d")

        full_game["won"] = full_game["Total"] > full_game["Total_opp"]
        

        games.append(full_game)

    all_games_df = pd.concat(games, ignore_index=True)
    all_games_df.to_csv("nba_project/data_scraping/data/games.csv")






            
