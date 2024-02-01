""" Module that contains the OMDB class to interact with all the OMDB API. """

import json

import requests


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


if __name__ == "__main__":
    omdb = OMDB()
    omdb.get_movie_stats("Oppenheimer", 2023)
