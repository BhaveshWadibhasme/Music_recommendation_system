"""Tests for the Last.fm artist recommender. They use synthetic rows only."""

import pandas as pd
import pytest

import music_recommendation as mr


def test_default_paths_fall_back_to_original_windows_locations(monkeypatch):
    monkeypatch.delenv("LASTFM_PLAYS", raising=False)
    monkeypatch.delenv("LASTFM_PROFILES", raising=False)
    monkeypatch.setattr(mr, "LOCAL_PLAYS_PATH", mr.Path("/nonexistent/plays.tsv"))
    monkeypatch.setattr(mr, "LOCAL_PROFILES_PATH", mr.Path("/nonexistent/profiles.tsv"))
    assert mr.resolve_plays_path(None) == mr.ORIGINAL_PLAYS_PATH
    assert mr.resolve_profiles_path(None) == mr.ORIGINAL_PROFILES_PATH
    assert "usersha1-artmbid-artname-plays.tsv" in mr.ORIGINAL_PLAYS_PATH
    assert "usersha1-profile.tsv" in mr.ORIGINAL_PROFILES_PATH


def test_explicit_and_local_paths_override_the_windows_default(monkeypatch, tmp_path):
    plays = tmp_path / "plays.tsv"
    profiles = tmp_path / "profiles.tsv"
    plays.write_text("x", encoding="utf-8")
    profiles.write_text("y", encoding="utf-8")
    monkeypatch.delenv("LASTFM_PLAYS", raising=False)
    monkeypatch.setenv("LASTFM_PROFILES", "from-env.tsv")
    monkeypatch.setattr(mr, "LOCAL_PLAYS_PATH", plays)
    monkeypatch.setattr(mr, "LOCAL_PROFILES_PATH", profiles)
    assert mr.resolve_plays_path("given.tsv") == "given.tsv"
    assert mr.resolve_plays_path(None) == str(plays)
    assert mr.resolve_profiles_path(None) == "from-env.tsv"


def test_load_tsv_files(tmp_path):
    plays_path = tmp_path / "plays.tsv"
    plays_path.write_text(
        "u1\tmbid-1\tArtist A\t10\nu2\t\tArtist B\t4\nu3\tmbid-3\tArtist C\t1\n",
        encoding="utf-8",
    )
    profiles_path = tmp_path / "profiles.tsv"
    profiles_path.write_text(
        "u1\tm\t19\tUnited States\tApr 28, 2008\nu2\tf\t\tMexico\tMay 1, 2008\n",
        encoding="utf-8",
    )
    plays = mr.load_plays(str(plays_path), nrows=2)
    profiles = mr.load_profiles(str(profiles_path))
    assert list(plays.columns) == mr.PLAY_COLUMNS
    assert len(plays) == 2
    assert plays.loc[0, "artist-name"] == "Artist A"
    assert plays.loc[0, "plays"] == 10
    assert list(profiles.columns) == ["users", "country"]
    assert profiles.loc[0, "country"] == "United States"
    assert profiles.loc[1, "country"] == "Mexico"


def test_row_cap_keeps_leading_rows_not_the_highest_play_total():
    plays = pd.DataFrame(
        {
            "user": ["u1", "u1"],
            "musicbrainz-artist-id": ["1", "2"],
            "artist-name": ["Obscure", "Famous"],
            "plays": [1, 10_000],
        }
    )
    capped = mr.take_leading_rows(mr.attach_artist_totals(plays), 1)
    assert list(capped["artist-name"]) == ["Obscure"]
    assert int(capped["total_artist_plays"].iloc[0]) == 1


def test_totals_include_rows_the_country_filter_later_drops():
    plays, profiles = mr.demo_frames()
    cleaned = mr.drop_missing_artist_names(plays)
    assert int(cleaned["artist-name"].isnull().sum()) == 0
    assert len(cleaned) == len(plays) - 1
    totals = mr.attach_artist_totals(cleaned)
    alpha_total = int(totals.loc[totals["artist-name"] == "Alpha", "total_artist_plays"].iloc[0])
    assert alpha_total == 50 + 50 + 40 + 10 + 5 + 999

    usa = mr.prepare_user_artist_frame(totals, profiles, "United States")
    assert "u9" not in set(usa["users"])
    assert set(usa["country"]) == {"United States"}
    underscored = mr.prepare_user_artist_frame(totals, profiles, "United_States")
    assert underscored.empty


def test_duplicate_user_artist_rows_are_dropped_without_summing_plays():
    plays, profiles = mr.demo_frames()
    outcome = mr.recommend(
        plays,
        profiles,
        row_cap=100,
        neighbors=6,
        query_index=0,
    )
    assert outcome.removed_rows == 1
    assert outcome.new_shape[0] == outcome.initial_shape[0] - 1
    first = outcome.recommendations[0]
    assert first["rank"] == 0
    assert first["artist"] == first["query"]
    assert first["distance"] == pytest.approx(0.0, abs=1e-8)
    assert len(outcome.recommendations) == 6
    # The duplicate Alpha/u1 row stored 50 plays twice. Dedup keeps one row,
    # so the kept Alpha/u1 play count is 50 rather than 100.
    assert first["query"] == "Alpha"


def test_demo_cli_prints_the_original_recommendation_layout(capsys):
    assert mr.main(["--demo", "--seed", "0", "--query-index", "0"]) == 0
    out = capsys.readouterr().out
    assert "Removed 1 rows" in out
    assert "Recommendations for Alpha:" in out
    assert "with distance of " in out
    assert "1: " in out


def test_missing_dataset_exits_before_reading(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        mr.main(
            [
                "--plays",
                str(tmp_path / "no-plays.tsv"),
                "--profiles",
                str(tmp_path / "no-profiles.tsv"),
            ]
        )
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "README.md" in err
    assert "no-plays.tsv" in err


def test_file_uri_prefix_is_ignored_for_the_existence_check(tmp_path):
    plays = tmp_path / "plays.tsv"
    plays.write_text("u1\tmbid\tArtist\t3\n", encoding="utf-8")
    uri = "file://" + str(plays)
    mr._require_file(uri, "play-count file")
    loaded = mr.load_plays(str(plays))
    assert loaded.loc[0, "artist-name"] == "Artist"
    assert (
        mr.local_filesystem_path("file:///C:/Users/Bhavesh/plays.tsv")
        == "C:/Users/Bhavesh/plays.tsv"
    )
    assert mr.local_filesystem_path("/plain/path.tsv") == "/plain/path.tsv"
