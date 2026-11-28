
from statsbombpy import sb

match_id = 3869685

# Load events and StatsBomb 360 data
events = sb.events(match_id=match_id)
frames = sb.frames(match_id=match_id)

print("Events columns:", events.columns.tolist())
print("360 columns:", frames.columns.tolist())

# Extract corner kicks
corners = events[
    (events["type"] == "Pass") &
    (events["pass_type"] == "Corner")
].copy()

# Check which corners have 360 data
frame_ids = set(frames["id"].dropna().unique())
corners["has_360"] = corners["id"].isin(frame_ids)

print("\n--- CORNER COVERAGE ---")
print("Total corners:", len(corners))
print("Corners with 360:", corners["has_360"].sum())

if len(corners) > 0:
    coverage = corners["has_360"].mean() * 100
    print(f"Coverage: {coverage:.2f}%")

# Each row in frames represents a visible player.
# Count visible players for each corner.
player_counts = (
    frames.groupby("id")
    .agg(
        visible_players=("location", "count"),
        visible_teammates=(
            "teammate",
            lambda x: x.eq(True).sum()
        ),
        visible_opponents=(
            "teammate",
            lambda x: x.eq(False).sum()
        )
    )
)

corner_analysis = corners[
    ["id", "team", "minute", "has_360"]
].merge(
    player_counts,
    left_on="id",
    right_index=True,
    how="left"
)

print("\n--- CORNER SNAPSHOT DETAILS ---")
print(corner_analysis.to_string(index=False))

print("\n--- VISIBLE PLAYER STATISTICS ---")
print(corner_analysis["visible_players"].describe())
