""" Module that contains the OMDB class to interact with all the OMDB API. """

import json

import polars as pl
import requests
from polars import DataFrame


class OMDB:
    """ Class that provides the methods to interact with the OMDB API. """
    def __init__(self):
        with open('config.json') as file:
            config = json.load(file)
        self._api_key = config["omdb"]["api_key"]

    @property
    def api_key(self) -> str:
        """ Property to store the OMDB api_key.

        Returns:
            str: The value of the OMDB api_key.
        """
        return self._api_key

    @api_key.setter
    def api_key(self, key: str) -> None:
        self._api_key = key

    def get_movie_stats(self, name: str, year: int) -> dict:
        """ Query the OMDB API and return the relevant movie info in a
        dictionary.

        Args:
            name (str): The name of the movie.
            year (int): The year the movie was released.

        Returns:
            data (dict): Contains the movie's imdb_rating, imdb_votes,
                and box_office.
        """
        print("Movie: ", name, "| Year: ", year)
        data = {}
        url = "http://www.omdbapi.com/?apikey=" + self.api_key + "\
&t=" + name + "&y=" + str(year)
        info = requests.get(url, timeout=10)
        omdb_info = json.loads(info.text)
        if omdb_info["Response"] == "False":
            raise ValueError("MovieNotFound")
        data["imdb_rating"] = float(
            omdb_info["imdbRating"]) if omdb_info["imdbRating"] != "N/A" else 0
        data["imdb_votes"] = int(
            omdb_info["imdbVotes"].replace(",", "")) if (
                omdb_info["imdbVotes"] != "N/A") else 0
        data["box_office"] = (
            int(omdb_info["BoxOffice"][1:].replace(",", "")) if (
                omdb_info["BoxOffice"] != "N/A") else 0)
        return data


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
    names = all_movies_df["Name"].to_list()
    years = all_movies_df["Year"].to_list()
    for idx, name in enumerate(names):
        omdb_info = omdb_api.get_movie_stats(name, years[idx])
        imdb_ratings.append(omdb_info["imdb_rating"])
        imdb_votes.append(omdb_info["imdb_votes"])
    imdb_ratings_s = pl.Series("imdb_rating", imdb_ratings)
    imdb_votes_s = pl.Series("imdb_votes", imdb_votes)
    all_movies_new_df = all_movies_df.with_columns(
        imdb_ratings_s, imdb_votes_s
    )
    return all_movies_new_df


def process_movies(filename: str) -> (DataFrame, DataFrame):
    """ If the metadata file 'nominees_with_metadata_by_year.csv' doesn't
    exist, it loads the movie nominees dataset, looks up additional data from
    OMDB, filters out movies with zero values, and returns two DataFrames:
    one with previous years oscars nominees, and another with this year's
    nominees.

    If the metadata file 'nominees_with_metadata_by_year.csv' already exists,
    the step to look up addiontal data from OMDB is skipped.

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
        all_movies_df = pl.read_csv("data/nominees_by_year.csv",
                                    null_values="Null")
        all_movies_df = get_omdb_info(all_movies_df)
        all_movies_df.write_csv(filename)

    # Filter out movies with zero values.
    all_movies_df = all_movies_df.filter(
        pl.col("imdb_rating") > 0
    )
    new_nominees_df = all_movies_df.filter(
        pl.col("Won").is_null()
    )
    previous_years_df = all_movies_df.filter(
        pl.col("Won").is_not_null()
    )
    return (previous_years_df, new_nominees_df)


if __name__ == "__main__":
    omdb = OMDB()
    omdb.get_movie_stats("Oppenheimer", 2023)
