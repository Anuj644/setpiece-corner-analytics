import json
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers import events_df

from setpiece.ingest.events import EVENTS_COLUMNS, EVENTS_DTYPES
from setpiece.ingest.frames import CORNER_360_COLUMNS, CORNER_360_DTYPES
from setpiece.io import derived_dir, read_table, write_manifest, write_table

PLAYER = {"x": 100.0, "y": 40.0, "teammate": True, "actor": False, "keeper": False}


def ev(event_id, comp=43, season=106, match_id=1, **extra):
    """One partial ``events`` row of a slice."""
    return {"match_id": match_id, "event_id": event_id, "event_index": 1,
            "competition_id": comp, "season_id": season, **extra}  # fmt: skip


def corner_360_df(rows):
    """A ``corner_360`` frame from partial rows; missing columns are null."""
    full = [{col: row.get(col) for col in CORNER_360_COLUMNS} for row in rows]
    return pd.DataFrame(full, columns=CORNER_360_COLUMNS).astype(CORNER_360_DTYPES)


def frame_row(event_id, comp=43, season=106, **extra):
    return {"match_id": 1, "event_id": event_id, "competition_id": comp,
            "season_id": season, **extra}  # fmt: skip


def assert_dtypes_match(back, dtypes):
    """Every column that has a pandas dtype in ``dtypes`` came back with exactly that dtype."""
    for col, expected in dtypes.items():
        if expected == "object":
            assert back[col].dtype == object, col
        else:
            assert back[col].dtype == pd.api.types.pandas_dtype(expected), col


# --- round trip -------------------------------------------------------------------------------


def test_roundtrip_keeps_nulls_and_lists(cfg_tmp):
    df = events_df([{"match_id": 1, "event_id": "a", "event_index": 1,
                     "related_event_ids": ["b"], "competition_id": 43, "season_id": 106}])  # fmt: skip
    write_table(df, "events", cfg_tmp, (43, 106))
    back = read_table("events", cfg_tmp)
    assert back.related_event_ids.iloc[0].tolist() == ["b"]
    assert pd.isna(back.pass_type.iloc[0])


def test_roundtrip_keeps_column_order_dtypes_and_values(cfg_tmp):
    df = events_df([
        ev("a", related_event_ids=["b", "c"], pass_type="Corner", start_x=120.0, out=True,
           duration=1.5, event_details='{"k": 1}', player_id=7),
        ev("b", match_id=2, event_index=2),
    ])  # fmt: skip
    write_table(df, "events", cfg_tmp, (43, 106))
    back = read_table("events", cfg_tmp)

    # The partition columns sit in their original positions, not appended at the end.
    assert list(back.columns) == EVENTS_COLUMNS
    assert_dtypes_match(back, EVENTS_DTYPES)
    plain = [c for c in EVENTS_COLUMNS if EVENTS_DTYPES[c] != "object"]
    pd.testing.assert_frame_equal(back[plain], df[plain])
    assert back.related_event_ids.iloc[0].tolist() == ["b", "c"]
    assert back.related_event_ids.iloc[1] is None


def test_roundtrip_keeps_plain_numpy_dtypes(cfg_tmp):
    """An unpartitioned table of ordinary dtypes (what a stats table holds) comes back equal."""
    df = pd.DataFrame({"n": pd.array([1, 2], dtype="int64"), "ratio": [1.5, float("nan")],
                       "flag": [True, False], "name": ["a", None]})  # fmt: skip
    write_table(df, "frame_stats", cfg_tmp, None)
    pd.testing.assert_frame_equal(read_table("frame_stats", cfg_tmp), df)


def test_roundtrip_of_empty_frame_keeps_columns_and_dtypes(cfg_tmp):
    write_table(events_df([]), "events", cfg_tmp, (43, 106))
    back = read_table("events", cfg_tmp)
    assert len(back) == 0
    assert list(back.columns) == EVENTS_COLUMNS
    assert_dtypes_match(back, EVENTS_DTYPES)


def test_roundtrip_of_list_of_dicts_and_floats(cfg_tmp):
    df = corner_360_df([frame_row("a", players=[PLAYER, dict(PLAYER, x=90.5)],
                                  visible_area=[0.0, 1.5, 2.0, 3.5])])  # fmt: skip
    write_table(df, "corner_360", cfg_tmp, (43, 106))
    back = read_table("corner_360", cfg_tmp)
    assert list(back.columns) == CORNER_360_COLUMNS
    assert list(back.players.iloc[0]) == [PLAYER, dict(PLAYER, x=90.5)]
    assert list(back.visible_area.iloc[0]) == [0.0, 1.5, 2.0, 3.5]


def test_returned_paths(cfg_tmp):
    p = write_table(events_df([ev("a")]), "events", cfg_tmp, (43, 106))
    expected = Path("events", "competition_id=43", "season_id=106", "part.parquet")
    assert p == cfg_tmp.processed_dir / expected
    assert p.is_file()
    q = write_table(pd.DataFrame({"n": [1]}), "frame_stats", cfg_tmp, None)
    assert q == cfg_tmp.processed_dir / "frame_stats.parquet"
    assert q.is_file()


# --- partition columns ------------------------------------------------------------------------


def test_partition_columns_are_filled_from_the_partition(cfg_tmp):
    """A frame without (or with null) partition columns still reads back with them, as Int64."""
    write_table(pd.DataFrame({"v": [1, 2]}), "derived_a", cfg_tmp, (43, 106))
    back = read_table("derived_a", cfg_tmp)
    assert back.competition_id.tolist() == [43, 43] and back.season_id.tolist() == [106, 106]
    assert back.competition_id.dtype == "Int64" and back.season_id.dtype == "Int64"

    write_table(events_df([{"event_id": "a", "event_index": 1}]), "events", cfg_tmp, (43, 106))
    back = read_table("events", cfg_tmp)
    assert back.competition_id.tolist() == [43] and back.season_id.tolist() == [106]
    assert list(back.columns) == EVENTS_COLUMNS


def test_partition_columns_that_disagree_with_the_partition_raise(cfg_tmp):
    write_table(events_df([ev("old")]), "events", cfg_tmp, (43, 106))
    with pytest.raises(ValueError, match="season_id"):
        write_table(events_df([ev("new", season=107)]), "events", cfg_tmp, (43, 106))
    # The refused write changed nothing.
    assert read_table("events", cfg_tmp).event_id.tolist() == ["old"]


def test_hive_path_adds_no_second_partition_column(cfg_tmp):
    """The partition columns live in the file once; the directory names do not add another."""
    p = write_table(events_df([ev("a")]), "events", cfg_tmp, (43, 106))
    assert pq.read_schema(p).names.count("competition_id") == 1
    back = read_table("events", cfg_tmp)
    assert list(back.columns).count("competition_id") == 1
    assert list(back.columns).count("season_id") == 1


# --- rewriting partitions ---------------------------------------------------------------------


def test_rewrite_replaces_partition(cfg_tmp):
    write_table(events_df([{"match_id": 1, "event_id": "a", "event_index": 1,
                            "competition_id": 43, "season_id": 106}]), "events", cfg_tmp, (43, 106))  # fmt: skip
    write_table(events_df([{"match_id": 2, "event_id": "z", "event_index": 1,
                            "competition_id": 43, "season_id": 106}]), "events", cfg_tmp, (43, 106))  # fmt: skip
    back = read_table("events", cfg_tmp)
    assert back.event_id.tolist() == ["z"]


def test_rewrite_clears_the_partition_directory(cfg_tmp):
    p = write_table(events_df([ev("a")]), "events", cfg_tmp, (43, 106))
    stale = p.parent / "stale.parquet"
    stale.write_bytes(b"left over from an earlier run")
    (p.parent / "old_dir").mkdir()
    write_table(events_df([ev("z")]), "events", cfg_tmp, (43, 106))
    assert sorted(f.name for f in p.parent.iterdir()) == ["part.parquet"]


def test_rewrite_leaves_other_partitions_untouched(cfg_tmp):
    write_table(events_df([ev("a")]), "events", cfg_tmp, (43, 106))
    other = write_table(events_df([ev("b", comp=55, season=43)]), "events", cfg_tmp, (55, 43))
    before = (other.read_bytes(), other.stat().st_mtime_ns)

    write_table(events_df([ev("z")]), "events", cfg_tmp, (43, 106))

    assert (other.read_bytes(), other.stat().st_mtime_ns) == before
    back = read_table("events", cfg_tmp)
    assert sorted(back.event_id) == ["b", "z"]


def test_rewrite_with_unconvertible_frame_keeps_the_old_partition(cfg_tmp):
    write_table(events_df([ev("old")]), "events", cfg_tmp, (43, 106))
    bad = events_df([ev("new", related_event_ids=[object()])])  # no Arrow type for an object()
    with pytest.raises(pa.ArrowException):
        write_table(bad, "events", cfg_tmp, (43, 106))
    assert read_table("events", cfg_tmp).event_id.tolist() == ["old"]


def test_unpartitioned_table_is_overwritten(cfg_tmp):
    write_table(pd.DataFrame({"n": [1, 2]}), "frame_stats", cfg_tmp, None)
    assert read_table("frame_stats", cfg_tmp).n.tolist() == [1, 2]
    p = write_table(pd.DataFrame({"n": [3]}), "frame_stats", cfg_tmp, None)
    assert read_table("frame_stats", cfg_tmp).n.tolist() == [3]
    assert sorted(f.name for f in p.parent.iterdir()) == ["frame_stats.parquet"]


def test_failed_unpartitioned_write_keeps_the_old_file(cfg_tmp, monkeypatch):
    p = write_table(pd.DataFrame({"n": [1, 2]}), "frame_stats", cfg_tmp, None)

    def broken_write(table, where, **kwargs):
        Path(where).write_bytes(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr("setpiece.io.pq.write_table", broken_write)
    with pytest.raises(OSError, match="disk full"):
        write_table(pd.DataFrame({"n": [3]}), "frame_stats", cfg_tmp, None)
    monkeypatch.undo()

    assert read_table("frame_stats", cfg_tmp).n.tolist() == [1, 2]
    assert sorted(f.name for f in p.parent.iterdir()) == ["frame_stats.parquet"]


def test_a_table_cannot_be_both_partitioned_and_unpartitioned(cfg_tmp):
    write_table(events_df([ev("a")]), "events", cfg_tmp, (43, 106))
    with pytest.raises(ValueError, match="events"):
        write_table(events_df([ev("a")]), "events", cfg_tmp, None)
    write_table(pd.DataFrame({"n": [1]}), "frame_stats", cfg_tmp, None)
    with pytest.raises(ValueError, match="frame_stats"):
        write_table(pd.DataFrame({"n": [1]}), "frame_stats", cfg_tmp, (43, 106))


# --- empty and mixed-type partitions ----------------------------------------------------------


@pytest.mark.parametrize("typed", [True, False], ids=["typed-empty", "untyped-empty"])
@pytest.mark.parametrize("empty_partition", [(43, 105), (43, 107)])
def test_empty_partition_reads_back_only_the_other_rows(cfg_tmp, empty_partition, typed):
    """The empty one sorts before, then after, the non-empty one; it may even carry no dtypes."""
    empty = corner_360_df([]) if typed else pd.DataFrame(columns=CORNER_360_COLUMNS)
    write_table(empty, "corner_360", cfg_tmp, empty_partition)
    write_table(corner_360_df([frame_row("a", players=[PLAYER], visible_area=[1.0, 2.0])]),
                "corner_360", cfg_tmp, (43, 106))  # fmt: skip

    back = read_table("corner_360", cfg_tmp)

    assert back.event_id.tolist() == ["a"]
    assert list(back.players.iloc[0]) == [PLAYER]
    assert list(back.columns) == CORNER_360_COLUMNS
    assert_dtypes_match(back, CORNER_360_DTYPES)
    assert read_table("corner_360", cfg_tmp, slices=[empty_partition]).empty


def test_rewriting_a_partition_with_an_empty_frame_clears_it(cfg_tmp):
    write_table(events_df([ev("a")]), "events", cfg_tmp, (43, 106))
    write_table(events_df([ev("b", season=107)]), "events", cfg_tmp, (43, 107))
    write_table(events_df([]), "events", cfg_tmp, (43, 106))
    assert read_table("events", cfg_tmp).event_id.tolist() == ["b"]


@pytest.mark.parametrize("nothing", [None, []], ids=["all-None", "all-empty"])
@pytest.mark.parametrize("empty_first", [True, False], ids=["sparse-first", "sparse-last"])
def test_list_columns_with_unknown_element_type_in_one_partition(cfg_tmp, nothing, empty_first):
    """Arrow infers ``null`` or ``list<null>`` for a partition without a single element."""
    sparse = corner_360_df([frame_row("a", season=105, players=nothing, visible_area=nothing)])
    full = corner_360_df([frame_row("b", players=[PLAYER], visible_area=[1.0, 2.0])])
    first, second = ((sparse, 105), (full, 106)) if empty_first else ((full, 106), (sparse, 105))
    write_table(first[0], "corner_360", cfg_tmp, (43, first[1]))
    write_table(second[0], "corner_360", cfg_tmp, (43, second[1]))

    back = read_table("corner_360", cfg_tmp).set_index("event_id")

    assert list(back.index) == ["a", "b"]  # partition order, whichever was written first
    assert list(back.players["b"]) == [PLAYER]
    assert list(back.visible_area["b"]) == [1.0, 2.0]
    if nothing is None:
        assert back.players["a"] is None
    else:
        assert len(back.players["a"]) == 0
    assert_dtypes_match(back.reset_index(), CORNER_360_DTYPES)


def test_struct_fields_without_a_value_in_one_partition(cfg_tmp):
    """``actor`` and ``keeper`` are null in every player of one partition, a bool in the other."""
    nulls = dict(PLAYER, actor=None, keeper=None)
    write_table(corner_360_df([frame_row("a", season=105, players=[nulls], visible_area=[1.0])]),
                "corner_360", cfg_tmp, (43, 105))  # fmt: skip
    write_table(corner_360_df([frame_row("b", players=[PLAYER], visible_area=[1.0])]),
                "corner_360", cfg_tmp, (43, 106))  # fmt: skip

    back = read_table("corner_360", cfg_tmp)

    assert list(back.players.iloc[0]) == [nulls]
    assert list(back.players.iloc[1]) == [PLAYER]


@pytest.mark.parametrize("nothing", [None, []], ids=["all-None", "all-empty"])
def test_related_event_ids_with_unknown_element_type_in_one_partition(cfg_tmp, nothing):
    write_table(events_df([ev("a", season=105, related_event_ids=nothing)]),
                "events", cfg_tmp, (43, 105))  # fmt: skip
    write_table(events_df([ev("b", related_event_ids=["a"])]), "events", cfg_tmp, (43, 106))

    back = read_table("events", cfg_tmp)

    assert back.related_event_ids.iloc[1].tolist() == ["a"]
    assert list(back.columns) == EVENTS_COLUMNS
    assert_dtypes_match(back, EVENTS_DTYPES)


# --- reading ----------------------------------------------------------------------------------


def test_partitions_are_read_in_numeric_order(cfg_tmp):
    for comp, season in [(43, 106), (43, 9), (11, 2), (11, 10)]:
        frame = events_df([ev(f"{comp}-{season}", comp, season)])
        write_table(frame, "events", cfg_tmp, (comp, season))
    back = read_table("events", cfg_tmp)
    assert back.event_id.tolist() == ["11-2", "11-10", "43-9", "43-106"]
    assert back.index.tolist() == [0, 1, 2, 3]


def partitioned_slices(cfg):
    for comp, season in [(43, 43), (43, 106), (55, 43), (55, 106)]:
        frame = events_df([ev(f"{comp}-{season}", comp, season)])
        write_table(frame, "events", cfg, (comp, season))


def unpartitioned_slices(cfg):
    frame = pd.DataFrame({"competition_id": pd.array([43, 43, 55, 55, None], dtype="Int64"),
                          "season_id": pd.array([43, 106, 43, 106, 106], dtype="Int64"),
                          "event_id": ["43-43", "43-106", "55-43", "55-106", "null-106"]})  # fmt: skip
    write_table(frame, "events", cfg, None)


@pytest.mark.parametrize("writer", [partitioned_slices, unpartitioned_slices])
def test_slices_match_whole_pairs(cfg_tmp, writer):
    """(43,106) and (55,43) must not also select (43,43) or (55,106)."""
    writer(cfg_tmp)
    back = read_table("events", cfg_tmp, slices=[(43, 106), (55, 43)])
    assert back.event_id.tolist() == ["43-106", "55-43"]
    assert back.competition_id.dtype == "Int64" and back.season_id.dtype == "Int64"
    assert back.index.tolist() == [0, 1]


@pytest.mark.parametrize("writer", [partitioned_slices, unpartitioned_slices])
def test_slices_that_match_nothing_give_an_empty_frame_with_the_columns(cfg_tmp, writer):
    writer(cfg_tmp)
    everything = read_table("events", cfg_tmp)
    for slices in ([(1, 1)], []):
        back = read_table("events", cfg_tmp, slices=slices)
        assert len(back) == 0
        assert list(back.columns) == list(everything.columns)
        assert back.dtypes.tolist() == everything.dtypes.tolist()


def test_slices_on_an_unpartitioned_table_need_the_partition_columns(cfg_tmp):
    write_table(pd.DataFrame({"n": [1]}), "frame_stats", cfg_tmp, None)
    with pytest.raises(ValueError, match="competition_id"):
        read_table("frame_stats", cfg_tmp, slices=[(43, 106)])


def test_reading_a_table_that_does_not_exist_raises(cfg_tmp):
    with pytest.raises(FileNotFoundError, match="no_such_table"):
        read_table("no_such_table", cfg_tmp)
    write_table(events_df([ev("a")]), "events", cfg_tmp, (43, 106))
    with pytest.raises(FileNotFoundError, match="no_such_table"):
        read_table("no_such_table", cfg_tmp, slices=[(43, 106)])


def test_reads_never_hold_a_file_open(cfg_tmp):
    """On Windows an open file blocks the next rewrite; write, read, rewrite, read, both layouts."""
    for n in (1, 2):
        write_table(events_df([ev(f"e{n}")]), "events", cfg_tmp, (43, 106))
        assert read_table("events", cfg_tmp).event_id.tolist() == [f"e{n}"]
        write_table(pd.DataFrame({"n": [n]}), "frame_stats", cfg_tmp, None)
        assert read_table("frame_stats", cfg_tmp).n.tolist() == [n]


# --- manifests --------------------------------------------------------------------------------


def test_manifest_fields(cfg_tmp):
    p = write_manifest("ingest", cfg_tmp, {"events": 3})
    m = json.loads(p.read_text())
    assert m["sha"] == cfg_tmp.sha and m["config_hash"] == cfg_tmp.config_hash()
    assert m["counts"] == {"events": 3}


def test_manifest_layout_and_metadata(cfg_tmp):
    p = write_manifest("ingest", cfg_tmp, {"events": 3})
    assert p == cfg_tmp.processed_dir / "manifests" / "ingest.json"
    m = json.loads(p.read_text())
    assert sorted(m) == ["config_hash", "counts", "created_utc", "package_version", "sha", "stage"]
    assert m["stage"] == "ingest"
    assert m["package_version"] == version("setpiece")
    created = datetime.fromisoformat(m["created_utc"])
    assert created.utcoffset() == timedelta(0)
    assert abs(datetime.now(created.tzinfo) - created) < timedelta(minutes=1)
    assert sorted(f.name for f in p.parent.iterdir()) == ["ingest.json"]


def test_manifest_accepts_numpy_integer_counts(cfg_tmp):
    n_events = pd.Series([1, 2, 3]).sum()  # what pandas hands back: a numpy integer
    assert not isinstance(n_events, int)
    p = write_manifest("ingest", cfg_tmp, {"events": n_events, "corner_360": 4})
    assert '"events": 6' in p.read_text()
    assert json.loads(p.read_text())["counts"] == {"events": 6, "corner_360": 4}


def test_manifest_rejects_other_unserialisable_counts_and_keeps_the_old_manifest(cfg_tmp):
    p = write_manifest("ingest", cfg_tmp, {"events": 3})
    with pytest.raises(TypeError):
        write_manifest("ingest", cfg_tmp, {"events": object()})
    assert json.loads(p.read_text())["counts"] == {"events": 3}
    assert sorted(f.name for f in p.parent.iterdir()) == ["ingest.json"]


def test_manifest_is_overwritten_by_a_rerun(cfg_tmp):
    write_manifest("ingest", cfg_tmp, {"events": 3})
    p = write_manifest("ingest", cfg_tmp, {"events": 5})
    assert json.loads(p.read_text())["counts"] == {"events": 5}


def test_derived_dir(cfg_tmp):
    assert derived_dir(cfg_tmp) == cfg_tmp.processed_dir / "derived"
