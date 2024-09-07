import os 
import pandas as pd
from bs4 import BeautifulSoup
import tqdm


SCORES_DIR = "nba_project/data_scraping/data/scores"

PLAYERS_DIR = "nba_project/data_scraping/data/players"


boxscores = os.listdir(SCORES_DIR)
box_scores = [os.path.join(SCORES_DIR, b) for b in boxscores if b.endswith(".html")]


def parse_html(box_score):
    with open(box_score, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr thead")]

    return soup

def read_stats(soup,team, stat):

    df = pd.read_html(str(soup), attrs={"id":f"box-{team}-game-{stat}"},index_col=0)[0]
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def read_score(soup):
    score = pd.read_html(str(soup), attrs={"id":"line_score"})[0]
    cols = list(score.columns)
    cols[0] = "Team"
    cols[-1] = "Total"
    score.columns = cols

    score = score[['Team','Total']]

    return score

def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [l["href"] for l in nav.find_all("a")]
    season_num = os.path.basename(hrefs[1]).split("_")[0]

    return season_num


def get_player_data(box_scores=box_scores):

    performances = []

    for box_score in tqdm.tqdm(box_scores):
        soup = parse_html(box_score)
        score = read_score(soup)
        teams = list(score["Team"])

        for team in teams:
            basic = read_stats(soup, team, "basic")
            advanced = read_stats(soup, team, "advanced")

            player_names = basic.index.tolist()
            #get rid of Reserves and Team Totals from basic and advanced stats
            player_names = [p for p in player_names if p not in ["Reserves","Team Totals"]]

            for player in player_names:
                basic_stats = basic.loc[player]
                advanced_stats = advanced.loc[player]

                player_stats_one_game = pd.concat([basic_stats, advanced_stats])
                player_stats_one_game["Name"] = player

                player_stats_one_game["season"] = read_season_info(soup)

                player_stats_one_game["date"] = os.path.basename(box_score)[:8]
                player_stats_one_game["date"] = pd.to_datetime(player_stats_one_game["date"], format="%Y%m%d")

                if team == teams[0]:
                    player_stats_one_game["home"] = 0
                else:
                    player_stats_one_game["home"] = 1


                performances.append(player_stats_one_game)
    
    #all_performances = pd.concat(performances, axis=1).T
    all_performances.to_csv(os.path.join(PLAYERS_DIR, "player_data.csv"))


        

if __name__ == '__main__':
    get_player_data(box_scores)