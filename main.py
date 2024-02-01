""" Contains the data preparation and model training, testing and prediction
methods.
"""

import polars as pl
from polars import DataFrame

from util.omdb_api import OMDB


def get_omdb_info(all_movies_df: DataFrame) -> DataFrame:
    """ Looks up OMDB to get additional data for all movies, and returns it in
    a new polars DataFrame.

    Args:
        all_movies_df (DataFrame): Contains all movies.

    Returns:
        all_movies_new_df (DataFrame): New DataFrame with the additional data
            from OMDB.
    """
    omdb_api = OMDB()
    imdb_ratings = []
    imdb_votes = []
    box_offices = []
    metascores = []
    rotten_tomatoes_ratings = []
    names = all_movies_df["Name"].to_list()
    years = all_movies_df["Year"].to_list()
    for idx, name in enumerate(names):
        omdb_info = omdb_api.get_movie_stats(name, years[idx])
        imdb_ratings.append(omdb_info["imdb_rating"])
        imdb_votes.append(omdb_info["imdb_votes"])
        box_offices.append(omdb_info["box_office"])
        metascores.append(omdb_info["metascore"])
        rotten_tomatoes_ratings.append(omdb_info["rotten_tomatoes_rating"])
    imdb_ratings_s = pl.Series("imdb_rating", imdb_ratings)
    imdb_votes_s = pl.Series("imdb_votes", imdb_votes)
    box_offices_s = pl.Series("box_office", box_offices)
    metascores_s = pl.Series("metascore", metascores)
    rotten_tomatoes_ratings_s = pl.Series("rotten_tomatoes_rating",
                                          rotten_tomatoes_ratings)
    all_movies_new_df = all_movies_df.with_columns(
        imdb_ratings_s, imdb_votes_s, box_offices_s, metascores_s,
        rotten_tomatoes_ratings_s
    )
    return all_movies_new_df


def process_movies(filename: str) -> (DataFrame, DataFrame):
    """ If the metadata file 'nominees_with_metadata_by_year.csv' doesn't
    exist, it loads the movie nominees dataset, looks up additional data from
    OMDB, and returns two DataFrames, one with previous years oscars nominees
    and another with this year's nominees.

    If the metadata file 'nominees_with_metadata_by_year.csv' already exists,
    the addiontal data from OMDB lookup step is skipped.

    Args:
        filename (str): The name of the csv file containing the movie info.

    Returns:
        previous_years_df (DataFrame): A polars DataFrame with the previous
            years nominees.
        new_nominees_df (DataFrame): A polars DataFrame with this year's
            nominees.
    """
    try:
        all_movies_df = pl.read_csv(filename)
    except FileNotFoundError:
        all_movies_df = pl.read_csv("data/nominees_by_year.csv")
        all_movies_df = get_omdb_info(all_movies_df)
        all_movies_df.write_csv(filename)
    new_nominees_df = all_movies_df.filter(
        pl.col("Won") == "Null"
    )
    previous_years_df = all_movies_df.filter(
        pl.col("Won") != "Null"
    )
    return (previous_years_df, new_nominees_df)


if __name__ == "__main__":
    new_nominees_df, previous_years_df = process_movies(
        "data/nominees_with_metadata_by_year.csv")
    print(new_nominees_df)
    print(previous_years_df)
