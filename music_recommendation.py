# -*- coding: utf-8 -*-
"""Artist recommendations from Last.fm 360K play counts.

Original script by Bhavesh, 13 December 2018. The pipeline is unchanged:
load play counts and listener profiles, keep a leading row window, restrict
to one country, drop duplicate user-artist rows, and fit a brute-force
cosine nearest-neighbor model on the artist-user matrix.

The listening files are not stored in this repository. See README.md.
``--demo`` runs the same functions on a tiny synthetic table so the code
can be checked without those files. Demo output is not a listening model
and is not a quality benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

PLAY_COLUMNS = ["user", "musicbrainz-artist-id", "artist-name", "plays"]
PROFILE_COLUMNS = ["users", "gender", "age", "country", "signup"]

# Paths from the 2018 script. The profiles path was a file URI; the plays
# path was a plain filesystem path to the same directory. Both defaults
# below are plain paths to that directory.
ORIGINAL_PLAYS_PATH = (
    "C:/Users/Bhavesh/Pictures/Music_recommendation/"
    "lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv"
)
ORIGINAL_PROFILES_PATH = (
    "C:/Users/Bhavesh/Pictures/Music_recommendation/lastfm-dataset-360K/usersha1-profile.tsv"
)
LOCAL_PLAYS_PATH = Path("data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv")
LOCAL_PROFILES_PATH = Path("data/lastfm-dataset-360K/usersha1-profile.tsv")

# The 2018 script stored this number as popularity_threshold and then
# applied it with DataFrame.iloc, so it caps rows rather than play totals.
DEFAULT_ROW_CAP = 40_000
DEFAULT_NEIGHBORS = 6
DEFAULT_NROWS = 20_000_000
DEFAULT_COUNTRY = "United States"


@dataclass
class RecommendationRun:
    recommendations: list[dict]
    removed_rows: int
    initial_shape: tuple[int, int]
    new_shape: tuple[int, int]


def resolve_plays_path(explicit: str | None) -> str:
    """CLI value, then LASTFM_PLAYS, then data/, then the 2018 Windows path."""
    if explicit:
        return explicit
    env = os.environ.get("LASTFM_PLAYS")
    if env:
        return env
    if LOCAL_PLAYS_PATH.exists():
        return str(LOCAL_PLAYS_PATH)
    return ORIGINAL_PLAYS_PATH


def resolve_profiles_path(explicit: str | None) -> str:
    """CLI value, then LASTFM_PROFILES, then data/, then the 2018 Windows path."""
    if explicit:
        return explicit
    env = os.environ.get("LASTFM_PROFILES")
    if env:
        return env
    if LOCAL_PROFILES_PATH.exists():
        return str(LOCAL_PROFILES_PATH)
    return ORIGINAL_PROFILES_PATH


def local_filesystem_path(path: str) -> str:
    """Map a file URI to a local path. Plain paths are returned unchanged.

    ``file:///C:/Users/...`` becomes ``C:/Users/...``. ``file:///tmp/x`` becomes
    ``/tmp/x``. The 2018 profiles path used the Windows form.
    """
    if not path.startswith("file:"):
        return path
    local = unquote(urlparse(path).path)
    if len(local) >= 3 and local[0] == "/" and local[2] == ":":
        return local[1:]
    return local


def load_plays(path: str, nrows: int = DEFAULT_NROWS) -> pd.DataFrame:
    """Load usersha1-artmbid-artname-plays.tsv (no header, tab separated)."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        nrows=nrows,
        names=PLAY_COLUMNS,
    )


def load_profiles(path: str, nrows: int = DEFAULT_NROWS) -> pd.DataFrame:
    """Load usersha1-profile.tsv and keep the user id and country columns."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        nrows=nrows,
        names=PROFILE_COLUMNS,
        usecols=["users", "country"],
    )


def drop_missing_artist_names(user_data: pd.DataFrame) -> pd.DataFrame:
    if user_data["artist-name"].isnull().sum() > 0:
        return user_data.dropna(axis=0, subset=["artist-name"]).copy()
    return user_data.copy()


def attach_artist_totals(user_data: pd.DataFrame) -> pd.DataFrame:
    """Add total_artist_plays. Totals include rows that later filters may drop."""
    totals = (
        user_data.groupby(by=["artist-name"])["plays"]
        .sum()
        .reset_index()
        .rename(columns={"plays": "total_artist_plays"})
    )
    totals = totals[["artist-name", "total_artist_plays"]]
    return user_data.merge(totals, on="artist-name", how="left")


def take_leading_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Keep the first ``limit`` rows.

    This is the 2018 ``popularity_threshold`` behavior: ``iloc``, not a
    filter on ``total_artist_plays``.
    """
    if limit < 1:
        raise ValueError("row_cap must be at least 1")
    return frame.iloc[0:limit, :].copy()


def prepare_user_artist_frame(
    popular: pd.DataFrame,
    profiles: pd.DataFrame,
    country: str = DEFAULT_COUNTRY,
) -> pd.DataFrame:
    frame = popular.copy()
    frame["users"] = frame["user"]
    frame = frame.drop(columns=["user"])
    combined = frame.merge(profiles, on="users", how="left")
    return combined.loc[combined["country"] == country].copy()


def drop_duplicate_user_artists(usa_data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop extra (user, artist) rows. Plays on the extras are not summed."""
    duplicates = usa_data[usa_data.duplicated(["users", "artist-name"])]
    if duplicates.empty:
        return usa_data, 0
    initial_rows = int(usa_data.shape[0])
    cleaned = usa_data.drop_duplicates(["users", "artist-name"]).copy()
    return cleaned, initial_rows - int(cleaned.shape[0])


def artist_user_matrix(usa_data: pd.DataFrame) -> pd.DataFrame:
    return usa_data.pivot(index="artist-name", columns="users", values="plays").fillna(0)


def fit_cosine_neighbors(matrix: pd.DataFrame) -> NearestNeighbors:
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(csr_matrix(matrix.to_numpy()))
    return model


def choose_query_index(n_artists: int, query_index: int | None, seed: int | None) -> int:
    if query_index is not None:
        if query_index < 0 or query_index >= n_artists:
            raise ValueError(f"query_index must be in 0..{n_artists - 1}, got {query_index}")
        return query_index
    if seed is None:
        return int(np.random.choice(n_artists))
    return int(np.random.default_rng(seed).integers(0, n_artists))


def neighbor_recommendations(
    model: NearestNeighbors,
    matrix: pd.DataFrame,
    query_index: int,
    n_neighbors: int,
) -> list[dict]:
    row = matrix.iloc[query_index, :].to_numpy().reshape(1, -1)
    distances, indices = model.kneighbors(row, n_neighbors=n_neighbors)
    query_name = matrix.index[query_index]
    results = []
    for rank, (idx, dist) in enumerate(zip(indices.flatten(), distances.flatten(), strict=True)):
        results.append(
            {
                "rank": int(rank),
                "artist": matrix.index[idx],
                "distance": float(dist),
                "query": query_name,
            }
        )
    return results


def format_recommendations(results: list[dict]) -> str:
    """Match the 2018 printout: a header, a blank line, then numbered neighbors."""
    if not results:
        return ""
    lines = [f"Recommendations for {results[0]['query']}:\n"]
    for item in results:
        if item["rank"] == 0:
            continue
        lines.append(
            "{0}: {1}, with distance of {2}:".format(item["rank"], item["artist"], item["distance"])
        )
    return "\n".join(lines)


def recommend(
    plays: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    row_cap: int = DEFAULT_ROW_CAP,
    neighbors: int = DEFAULT_NEIGHBORS,
    country: str = DEFAULT_COUNTRY,
    query_index: int | None = None,
    seed: int | None = None,
) -> RecommendationRun:
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")
    prepared = drop_missing_artist_names(plays)
    capped = take_leading_rows(attach_artist_totals(prepared), row_cap)
    usa = prepare_user_artist_frame(capped, profiles, country)
    deduped, removed_rows = drop_duplicate_user_artists(usa)
    matrix = artist_user_matrix(deduped)
    if matrix.shape[0] < neighbors:
        raise ValueError(
            f"Need at least {neighbors} artists after filtering; got {matrix.shape[0]}."
        )
    model = fit_cosine_neighbors(matrix)
    index = choose_query_index(matrix.shape[0], query_index, seed)
    return RecommendationRun(
        recommendations=neighbor_recommendations(model, matrix, index, neighbors),
        removed_rows=removed_rows,
        initial_shape=(int(usa.shape[0]), int(usa.shape[1])),
        new_shape=(int(deduped.shape[0]), int(deduped.shape[1])),
    )


def demo_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic rows for ``--demo``. These are not Last.fm listens."""
    rows = [
        ("u1", "a", None, 1),
        ("u1", "1", "Alpha", 50),
        ("u1", "1", "Alpha", 50),
        ("u2", "1", "Alpha", 40),
        ("u3", "1", "Alpha", 10),
        ("u4", "1", "Alpha", 5),
        ("u1", "2", "Beta", 48),
        ("u2", "2", "Beta", 42),
        ("u3", "2", "Beta", 12),
        ("u4", "2", "Beta", 6),
        ("u1", "3", "Gamma", 5),
        ("u2", "3", "Gamma", 8),
        ("u3", "3", "Gamma", 40),
        ("u4", "3", "Gamma", 45),
        ("u1", "4", "Delta", 4),
        ("u2", "4", "Delta", 7),
        ("u3", "4", "Delta", 38),
        ("u4", "4", "Delta", 44),
        ("u1", "5", "Epsilon", 20),
        ("u2", "5", "Epsilon", 2),
        ("u3", "5", "Epsilon", 20),
        ("u4", "5", "Epsilon", 2),
        ("u1", "6", "Zeta", 19),
        ("u2", "6", "Zeta", 3),
        ("u3", "6", "Zeta", 21),
        ("u4", "6", "Zeta", 1),
        ("u9", "1", "Alpha", 999),
        ("u1", "7", "Late", 100_000),
    ]
    plays = pd.DataFrame(rows, columns=["user", "musicbrainz-artist-id", "artist-name", "plays"])
    profiles = pd.DataFrame(
        {
            "users": ["u1", "u2", "u3", "u4", "u9"],
            "country": [
                "United States",
                "United States",
                "United States",
                "United States",
                "Mexico",
            ],
        }
    )
    return plays, profiles


def _require_file(path: str, label: str) -> None:
    if os.path.exists(local_filesystem_path(path)):
        return
    print(
        f"Missing {label}: {path}\n"
        "The Last.fm 360K files are not in this repository. "
        "See README.md for the public source, license, and download steps.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Recommend artists with cosine nearest neighbors on Last.fm 360K play counts.")
    )
    parser.add_argument(
        "--plays",
        default=None,
        help=(
            "Path to usersha1-artmbid-artname-plays.tsv. "
            "Defaults to LASTFM_PLAYS, data/lastfm-dataset-360K/, "
            "or the original Windows path."
        ),
    )
    parser.add_argument(
        "--profiles",
        default=None,
        help=(
            "Path to usersha1-profile.tsv. "
            "Defaults to LASTFM_PROFILES, data/lastfm-dataset-360K/, "
            "or the original Windows path."
        ),
    )
    parser.add_argument("--nrows", type=int, default=DEFAULT_NROWS)
    parser.add_argument(
        "--row-cap",
        type=int,
        default=DEFAULT_ROW_CAP,
        help=(
            "Keep this many leading play rows before the country filter. "
            "The 2018 name for this value was popularity_threshold."
        ),
    )
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--country", default=DEFAULT_COUNTRY)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--query-index", type=int, default=None)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic rows. Does not read Last.fm files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo:
        plays, profiles = demo_frames()
    else:
        plays_path = resolve_plays_path(args.plays)
        profiles_path = resolve_profiles_path(args.profiles)
        _require_file(plays_path, "play-count file")
        _require_file(profiles_path, "profile file")
        plays = load_plays(plays_path, nrows=args.nrows)
        profiles = load_profiles(profiles_path, nrows=args.nrows)
    outcome = recommend(
        plays,
        profiles,
        row_cap=args.row_cap,
        neighbors=args.neighbors,
        country=args.country,
        query_index=args.query_index,
        seed=args.seed,
    )
    if outcome.removed_rows:
        print(f"Initial dataframe shape {outcome.initial_shape}")
        print(f"New dataframe shape {outcome.new_shape}")
        print(f"Removed {outcome.removed_rows} rows")
    print(format_recommendations(outcome.recommendations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
