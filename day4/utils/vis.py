from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


def plot_progress_over_years(countries: List[str], areas: List[float]):
    """
    Plot a curve showing progress over the years.

    Args:
        years (List[str]): List of years.
        progress (List[float]): List of progress values corresponding to each year.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(countries, areas, marker="o", color="b", linestyle="-")
    plt.title("Progress Over Years")
    plt.xlabel("Year")
    plt.ylabel("Progress")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return "Plotting..."


def plot_progress_over_years_for_one_countrie(years: List[str], areas: List[float]):
    """
    Plot a curve showing progress over the years.

    Args:
        years (List[str]): List of years.
        progress (List[float]): List of progress values corresponding to each year.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(years, areas, marker="o", color="b", linestyle="-")
    plt.title("Progress Over Years")
    plt.xlabel("Year")
    plt.ylabel("Progress")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return "Plotting..."


def plot_house_pricing_data(prices: List[float], areas: List[str]):
    """
    Plot a curve showing prices over the areas.

    Args:
        areas (List[str]): List of areas.
        prices (List[float]): List of prices values corresponding to each area.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(areas, prices, marker="o", color="b", linestyle="-")
    plt.title("Progress Over Prices")
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return "Plotting..."


def apply_python_script_on_df(df_: DataFrame):
    def fn(script: str):
        df = df_
        exec(script)
        return "Script applied"

    return fn


if __name__ == "__main__":
    years = ["2019", "2020", "2021", "2022"]
    progress = [0.2, 0.4, 0.6, 0.8]

    plot_progress_over_years(years, progress)
