import pandas as pd

# === Step 1: Load data ===
salary_df = pd.read_csv("../Data/mlb_salaries_2024.csv")
batters_df = pd.read_csv("../Data/Batter_stats.csv")
pitchers_df = pd.read_csv("../Data/Pitcher_stats.csv")

# === Step 2: Normalize names ===
def to_first_last(name):
    if "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return name.strip()

salary_df["Normalized Name"] = salary_df["Player Name"].apply(to_first_last)
batters_df["Normalized Name"] = batters_df["last_name, first_name"].apply(to_first_last)
pitchers_df["Normalized Name"] = pitchers_df["last_name, first_name"].apply(to_first_last)

# === Step 3: Drop unneeded columns ===
drop_cols = ["last_name, first_name", "player_id", "year", "player_age"]
batters_df = batters_df.drop(columns=drop_cols, errors="ignore")
pitchers_df = pitchers_df.drop(columns=drop_cols, errors="ignore")

# === Step 4: Merge salary with batters ===
batters_merged = pd.merge(
    salary_df,
    batters_df,
    on="Normalized Name",
    how="inner"  # Only players in both salary & batters
)

# === Step 5: Merge salary with pitchers ===
pitchers_merged = pd.merge(
    salary_df,
    pitchers_df,
    on="Normalized Name",
    how="inner"  # Only players in both salary & pitchers
)

# === Step 6: Save both files ===
batters_merged.to_csv("mlb_batters_with_salary_2024.csv", index=False)
pitchers_merged.to_csv("mlb_pitchers_with_salary_2024.csv", index=False)

print("✅ Saved: mlb_batters_with_salary_2024.csv")
print("✅ Saved: mlb_pitchers_with_salary_2024.csv")


