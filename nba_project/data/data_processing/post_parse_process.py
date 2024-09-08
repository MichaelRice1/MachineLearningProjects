import pandas as pd



player_data = pd.read_csv("nba_project/data_scraping/data/players/player_data.csv")


def data_processing(player_data):

    player_data = player_data.sort_values(by = ["Name","date"])




def add_predictors(player_data):
    #last 2 games points,rebounds,assists per 38

    #last 5 games points,rebounds,assists per 38

    #last 10 games points,rebounds,assists per 38

    #last 25 games points,rebounds,assists per 38

    #last 50 games points,rebounds,assists per 38

    #last 100 games points,rebounds,assists per 38

    #last 250 games points,rebounds,assists per 38


    #players shot attempts per game last 5, 25, 50, 100 games

    #players true shooting percentage last 5, 25, 50, 100 games

    #players rebounding percentage last 5, 25, 50, 100 games

    #players assist percentage last 5, 25, 50, 100 games

    #players number of free throw attempts per game last 5, 25, 50, 100 games




    #opponents defensive rating

    #opponents pace

    #opponents points allowed per game

    #oppoonents total rebounds per game

    #opponents total assists allowed per game

    #opponents offensive rebounds allowed per game

    #opponents defensive rebounds allowed per game




    #non opponents pace

    #non opponents points scored per game

    #non opponents rebounds per game

    #non opponents defensive rebounds per game

    #non opponents offensive rebounds per game

    #non opponents assists per game


    #player usage rates

    #the superteam factor (overall team capabiliy)


    
    #days since last game

    #num of games in last 5 days

    #num of games in last 10 days

    #game importance


    pass


def combination_features(player_data):
    #combine features to create new features

    pass


def data_splitting(player_data):
    #split data into training, validation, and testing sets

    pass



