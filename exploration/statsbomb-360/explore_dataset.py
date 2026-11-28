
from statsbombpy import sb

# 1. Discover available competitions
competitions = sb.competitions()

print("Available competitions:")
print(
    competitions[
        ["competition_id", "season_id",
         "competition_name", "season_name"]
    ].head(10).to_string(index=False)
)

# 2. Select the 2022 FIFA World Cup
matches = sb.matches(
    competition_id=43,
    season_id=106
)

print("\nSample matches:")
print(
    matches[
        ["match_id", "home_team", "away_team"]
    ].head().to_string(index=False)
)

# 3. Select a match and retrieve its events
match_id = int(matches.iloc[0]["match_id"])
events = sb.events(match_id=match_id)

print(f"\nSelected match: {match_id}")
print(f"Total events: {len(events)}")

# 4. Inspect available attributes
print("\nAvailable columns:")
print(events.columns.tolist())

# 5. Find corner kicks
corners = events[
    (events["type"] == "Pass")
    & (events["pass_type"] == "Corner")
]

print(f"\nCorners found: {len(corners)}")

# 6. Look at a few corner events
columns = [
    "minute",
    "team",
    "player",
    "location",
    "pass_end_location",
    "pass_length",
    "pass_technique"
]

available = [c for c in columns if c in corners.columns]
print(corners[available].head().to_string(index=False))
