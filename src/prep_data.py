from pathlib import Path
from typing import Dict

import polars as pl

INPUT_DATA_DIR = Path("data/bdb_2024/")


def get_players_df() -> pl.DataFrame:
    return (
        pl.read_csv(INPUT_DATA_DIR / "players.csv", null_values=["NA", "nan", "N/A", "NaN", ""])
        .with_columns(
            height_inches = pl.col("height")
            .str.split("-")
            .map_elements(lambda s: int(s[0]) * 12 + int(s[1]), return_dtype=int),
        ).with_columns(
            weight_Z = (pl.col("weight") - pl.col("weight").mean()) / pl.col("weight").std(),
            height_Z = (pl.col("height_inches") - pl.col("height_inches").mean()) / pl.col("height_inches").std(),
        )
    )


def get_plays_df() -> pl.DataFrame:
    return pl.read_csv(INPUT_DATA_DIR / "plays.csv", null_values=["NA", "nan", "N/A", "NaN", ""]).with_columns(
        distanceToGoal=(
            pl.when(pl.col("possessionTeam") == pl.col("yardlineSide"))
            .then(100 - pl.col("yardlineNumber"))
            .otherwise(pl.col("yardlineNumber"))
        )
    )


def get_tracking_df() -> pl.DataFrame:
    # don't include football rows for this project
    return pl.read_csv(INPUT_DATA_DIR / "tracking_week_*.csv", null_values=["NA", "nan", "N/A", "NaN", ""]).filter(
        pl.col("displayName") != "football"
    )


def add_features_to_tracking_df(
    tracking_df: pl.DataFrame,
    players_df: pl.DataFrame,
    plays_df: pl.DataFrame,
) -> pl.DataFrame:
    # add `is_ball_carrier`, `team_indicator`, and other features to tracking data
    og_len = len(tracking_df)
    tracking_df = (
        tracking_df.join(
            plays_df.select(
                [
                    "gameId",
                    "playId",
                    "ballCarrierId",
                    "possessionTeam",
                    "down",
                    "yardsToGo",
                    "distanceToGoal",
                    "playResult",
                ]
            ),
            on=["gameId", "playId"],
            how="inner",
        )
        .join(
            players_df.select(["nflId", "weight_Z", "height_Z"]).unique(),
            on="nflId",
            how="inner",
        )
        .with_columns(
            is_ball_carrier=(pl.col("nflId") == pl.col("ballCarrierId")).cast(int),
            side=pl.when(pl.col("club") == pl.col("possessionTeam"))
            .then(pl.lit(1))
            .otherwise(pl.lit(-1))
            .alias("side"),
        )
        .drop(["ballCarrierId", "possessionTeam"])
    )
    assert len(tracking_df) == og_len, "Lost rows when joining tracking data with play/player data"

    return tracking_df


def convert_tracking_to_cartesian(tracking_df: pl.DataFrame) -> pl.DataFrame:
    return (
        tracking_df.with_columns(
            dir=((pl.col("dir") - 90) * -1) % 360,
            o=((pl.col("o") - 90) * -1) % 360,
        )
        # convert polar vectors to cartesian ((s, dir) -> (vx, vy), (o) -> (ox, oy))
        .with_columns(
            vx=pl.col("s") * pl.col("dir").radians().cos(),
            vy=pl.col("s") * pl.col("dir").radians().sin(),
            ox=pl.col("o").radians().cos(),
            oy=pl.col("o").radians().sin(),
        )
    )


def standardize_tracking_directions(tracking_df: pl.DataFrame) -> pl.DataFrame:
    return tracking_df.with_columns(
        x=pl.when(pl.col("playDirection") == "right").then(pl.col("x")).otherwise(120 - pl.col("x")),
        y=pl.when(pl.col("playDirection") == "right").then(pl.col("y")).otherwise(53.3 - pl.col("y")),
        vx=pl.when(pl.col("playDirection") == "right").then(pl.col("vx")).otherwise(-1 * pl.col("vx")),
        vy=pl.when(pl.col("playDirection") == "right").then(pl.col("vy")).otherwise(-1 * pl.col("vy")),
        ox=pl.when(pl.col("playDirection") == "right").then(pl.col("ox")).otherwise(-1 * pl.col("ox")),
        oy=pl.when(pl.col("playDirection") == "right").then(pl.col("oy")).otherwise(-1 * pl.col("oy")),
    ).drop("playDirection")


def augment_mirror_tracking(tracking_df: pl.DataFrame) -> pl.DataFrame:
    # Augment data by mirroring the field (assuming all plays are moving to right)
    # There are arguments to not do this (e.g. most QBs are right-handed).
    # But I think more data is more important

    og_len = len(tracking_df)

    mirrored_tracking_df = tracking_df.clone().with_columns(
        # only flip y values
        y=53.3 - pl.col("y"),
        vy=-1 * pl.col("vy"),
        oy=-1 * pl.col("oy"),
        mirrored=pl.lit(True),
    )

    tracking_df = pl.concat(
        [
            tracking_df.with_columns(mirrored=pl.lit(False)),
            mirrored_tracking_df,
        ],
        how="vertical",
    )

    assert len(tracking_df) == og_len * 2, "Lost rows when mirroring tracking data"
    return tracking_df


def get_tackle_loc_target_df(tracking_df: pl.DataFrame) -> pl.DataFrame:
    # generate per-play target dataframe
    TACKLE_EVENT_ENUM = {v: k for k, v in enumerate(["tackle", "out_of_bounds", "touchdown", "qb_slide", "fumble"])}

    tackle_loc_df = (
        tracking_df.sort("frameId")
        .filter(pl.col("event").is_in(TACKLE_EVENT_ENUM.keys()) & (pl.col("is_ball_carrier") == 1))
        .group_by(["gameId", "playId", "mirrored"])
        .tail(1)
        .select(
            [
                "gameId",
                "playId",
                "mirrored",
                "nflId",
                "displayName",
                "frameId",
                "event",
                "x",
                "y",
                "x_rel",
                "y_rel",
                "play_origin_x",
                "play_origin_y",
                "playResult",
            ]
        )
        .rename(
            {
                "nflId": "ballCarrierNflId",
                "displayName": "ballCarrierName",
                "frameId": "tackle_frameId",
                "event": "tackle_event",
                "x": "tackle_x",
                "y": "tackle_y",
                "x_rel": "tackle_x_rel",
                "y_rel": "tackle_y_rel",
            }
        )
        .with_columns(
            pl.col("tackle_event").replace(TACKLE_EVENT_ENUM).cast(int).alias("tackle_event_enum"),
        )
    )

    # only keep plays in dataset that have a valid tackle location target
    og_play_count = len(tracking_df.select(["gameId", "playId"]).unique())
    tracking_df = tracking_df.join(
        tackle_loc_df.select(["gameId", "playId", "mirrored"]), on=["gameId", "playId", "mirrored"], how="inner"
    )
    new_play_count = len(tracking_df.select(["gameId", "playId"]).unique())
    print(f"Lost {(og_play_count - new_play_count)/og_play_count:.3%} plays when joining with tackle_loc_df")
    return tackle_loc_df, tracking_df


def split_train_test_val(tracking_df: pl.DataFrame, target_df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    tracking_df = tracking_df.sort(["gameId", "playId", "mirrored", "frameId"])
    target_df = target_df.sort(["gameId", "playId", "mirrored"])

    print(
        f"Total set: {tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{tracking_df.n_unique(['gameId', 'playId', 'mirrored', "frameId"])} frames",
    )

    test_val_ids = tracking_df.select(["gameId", "playId"]).unique(maintain_order=True).sample(fraction=0.3, seed=42)
    train_tracking_df = tracking_df.join(test_val_ids, on=["gameId", "playId"], how="anti")
    train_tgt_df = target_df.join(test_val_ids, on=["gameId", "playId"], how="anti")
    print(
        f"Train set: {train_tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{train_tracking_df.n_unique(['gameId', 'playId', 'mirrored', "frameId"])} frames",
    )

    test_ids = test_val_ids.sample(fraction=0.3, seed=42)  # roughly 70-20-10 split
    test_tracking_df = tracking_df.join(test_ids, on=["gameId", "playId"], how="inner")
    test_tgt_df = target_df.join(test_ids, on=["gameId", "playId"], how="inner")
    print(
        f"Test set: {test_tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{test_tracking_df.n_unique(['gameId', 'playId', 'mirrored', "frameId"])} frames",
    )

    val_ids = test_val_ids.join(test_ids, on=["gameId", "playId"], how="anti")
    val_tracking_df = tracking_df.join(val_ids, on=["gameId", "playId"], how="inner")
    val_tgt_df = target_df.join(val_ids, on=["gameId", "playId"], how="inner")
    print(
        f"Validation set: {val_tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{val_tracking_df.n_unique(['gameId', 'playId', 'mirrored', "frameId"])} frames",
    )

    return {
        "train_features": train_tracking_df,
        "train_targets": train_tgt_df,
        "test_features": test_tracking_df,
        "test_targets": test_tgt_df,
        "val_features": val_tracking_df,
        "val_targets": val_tgt_df,
    }


def add_relative_positions(tracking_df: pl.DataFrame) -> pl.DataFrame:
    return (
        tracking_df.sort("frameId")
        # x, y, having wide distribution of values is bad for training
        # use ball-carrier position at first frame as "origin" for relative positions
        # this should make each frame's feature look more standardized to a model too
        .with_columns(
            play_origin_x=pl.col("x")
            .filter(pl.col("is_ball_carrier") == 1)
            .first()
            .over(["gameId", "playId", "mirrored"]),
            play_origin_y=pl.col("y")
            .filter(pl.col("is_ball_carrier") == 1)
            .first()
            .over(["gameId", "playId", "mirrored"]),
        )
        .with_columns(
            x_rel=pl.col("x") - pl.col("play_origin_x"),
            y_rel=pl.col("y") - pl.col("play_origin_y"),
        )
    )


def main():
    players_df = get_players_df()
    plays_df = get_plays_df()
    tracking_df = get_tracking_df()

    tracking_df = add_features_to_tracking_df(tracking_df, players_df, plays_df)
    tracking_df = convert_tracking_to_cartesian(tracking_df)
    tracking_df = standardize_tracking_directions(tracking_df)
    tracking_df = augment_mirror_tracking(tracking_df)
    tracking_df = add_relative_positions(tracking_df)

    tkl_loc_tgt_df, tracking_df = get_tackle_loc_target_df(tracking_df)

    split_dfs = split_train_test_val(tracking_df, tkl_loc_tgt_df)

    out_dir = Path("data/split_prepped_data/")
    out_dir.mkdir(exist_ok=True, parents=True)

    for key, df in split_dfs.items():
        sort_keys = (
            ["gameId", "playId", "mirrored", "frameId"] if "features" in key else ["gameId", "playId", "mirrored"]
        )
        df.sort(sort_keys).write_parquet(out_dir / f"{key}.parquet")


if __name__ == "__main__":
    main()
