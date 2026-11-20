
from statsbombpy import sb
import pandas as pd

# 2022 FIFA World Cup
competition_id = 43
season_id = 106

matches = sb.matches(
    competition_id=competition_id,
    season_id=season_id
)

print("Total matches:", len(matches))

results = []
snapshot_details = []

for i, match_id in enumerate(matches["match_id"], start=1):
    try:
        events = sb.events(match_id=match_id)
        frames = sb.frames(match_id=match_id)

        corners = events[
            (events["type"] == "Pass") &
            (events["pass_type"] == "Corner")
        ].copy()

        if frames.empty or "id" not in frames.columns:
            counts = pd.Series(dtype="int64")
        else:
            counts = frames.groupby("id").size()

        corners["visible_players"] = (
            corners["id"].map(counts).fillna(0).astype(int)
        )

        corners["has_360"] = corners["visible_players"] > 0

        # Exploratory visibility threshold, not a
        # guarantee of usable tactical information.
        corners["potentially_usable"] = (
            corners["visible_players"] >= 16
        )

        results.append({
            "match_id": match_id,
            "corners": len(corners),
            "with_360": int(corners["has_360"].sum()),
            "with_16_plus": int(
                corners["potentially_usable"].sum()
            )
        })

        snapshot_details.append(
            corners[[
                "id", "match_id", "team",
                "visible_players", "has_360",
                "potentially_usable"
            ]]
        )

        print(f"Processed {i}/{len(matches)}: {match_id}")

    except Exception as e:
        print(f"Failed for {match_id}: {e}")

summary = pd.DataFrame(results)

if not summary.empty:
    total = summary["corners"].sum()
    matched = summary["with_360"].sum()
    usable = summary["with_16_plus"].sum()

    print("\n--- WORLD CUP 2022 SUMMARY ---")
    print("Matches processed:", len(summary))
    print("Total corners:", total)
    print("Corners with 360:", matched)
    print("Corners with >=16 players:", usable)

    if total:
        print(f"360 coverage: {100 * matched / total:.2f}%")

    summary.to_csv("worldcup_360_summary.csv", index=False)

    pd.concat(snapshot_details).to_csv(
        "worldcup_corner_snapshots.csv",
        index=False
    )
