import os
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import mariadb
from dotenv import load_dotenv


def get_db_connection() -> mariadb.Connection:
    load_dotenv("config.env")
    connection = mariadb.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    return connection


def discover_energy_outflow_column(cursor: mariadb.Cursor, db_name: str) -> Tuple[str, str]:
    """
    Try to find the L5S1 energy outflow column by scanning information_schema.
    Returns (table_name, column_name). Raises ValueError if not found.
    """
    like_clauses = [
        "(LOWER(COLUMN_NAME) LIKE '%l5s1%' OR LOWER(COLUMN_NAME) LIKE '%l5_s1%' OR LOWER(COLUMN_NAME) LIKE '%l5-%s1%')",
        "(LOWER(COLUMN_NAME) LIKE '%outflow%' OR LOWER(COLUMN_NAME) LIKE '%energy%')",
    ]
    query = f"""
        SELECT TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ?
          AND {like_clauses[0]}
          AND {like_clauses[1]}
        ORDER BY TABLE_NAME, COLUMN_NAME
    """
    cursor.execute(query, (db_name,))
    matches = cursor.fetchall()
    if matches:
        # Heuristic: prefer columns in poi first, then others
        matches_sorted = sorted(
            matches,
            key=lambda r: (0 if r[0].lower() == "poi" else 1, r[0], r[1]),
        )
        return matches_sorted[0][0], matches_sorted[0][1]

    # As a helpful fallback, surface any columns with l5s1 even if not clearly energy/outflow
    fallback_query = """
        SELECT TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ?
          AND (LOWER(COLUMN_NAME) LIKE '%l5s1%' OR LOWER(COLUMN_NAME) LIKE '%l5_s1%' OR LOWER(COLUMN_NAME) LIKE '%l5-%s1%')
        ORDER BY TABLE_NAME, COLUMN_NAME
    """
    cursor.execute(fallback_query, (db_name,))
    fallback_matches = cursor.fetchall()
    hint_text = ", ".join([f"{t}.{c}" for t, c in fallback_matches]) or "none"
    raise ValueError(
        f"Could not find an L5S1 energy outflow column. Candidate L5S1-related columns: {hint_text}"
    )


def discover_join_keys(cursor: mariadb.Cursor, db_name: str, table_name: str) -> List[str]:
    """
    Discover likely join keys present in a table.
    Returns list of candidate key column names ordered by preference.
    """
    cursor.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """,
        (db_name, table_name),
    )
    cols = [r[0] for r in cursor.fetchall()]

    preferred = []
    for candidate in [
        "session_trial",
        "session",
        "poi_id",
        "trial",
        "swing_id",
    ]:
        if candidate in cols:
            preferred.append(candidate)
    return preferred


def fetch_analysis_dataframe(connection: mariadb.Connection) -> Tuple[pd.DataFrame, str, str]:
    """
    Fetch a DataFrame joining energy outflow, bat speed, and other variables.
    Returns (df, energy_col_name, bat_speed_col_name).
    """
    db_name = os.getenv("DB_NAME")
    cursor = connection.cursor()

    energy_table, energy_column = discover_energy_outflow_column(cursor, db_name)
    join_keys = discover_join_keys(cursor, db_name, energy_table)

    # Base selection from poi and sessions
    # Using a limited set of columns to keep things manageable.
    base_select = """
        SELECT
            p.session_trial AS poi_session_trial,
            p.blast_bat_speed_mph AS poi_blast_bat_speed_mph,
            p.exit_velo_mph AS poi_exit_velo_mph,
            p.xfactor_angle_z_mhss AS poi_xfactor_angle_z_mhss,
            s.session AS session_id,
            s.height_meters AS session_height_meters,
            s.mass_kilograms AS session_mass_kilograms,
            s.date AS session_date
        FROM poi p
        JOIN sessions s
          ON s.session = CAST(SUBSTRING_INDEX(p.session_trial, '_', 1) AS UNSIGNED)
        WHERE s.date BETWEEN '2024-01-01' AND '2024-12-31'
    """

    # Materialize base into temp table for easier joining in MariaDB
    cursor.execute("DROP TEMPORARY TABLE IF EXISTS tmp_base")
    cursor.execute(
        "CREATE TEMPORARY TABLE tmp_base AS " + base_select
    )

    # Determine how to join energy table
    energy_select = None
    energy_alias = "e"
    energy_column_alias = "energy_outflow"

    if energy_table.lower() == "poi":
        # Energy column lives in poi, join tmp_base to poi
        energy_select = f"""
            SELECT b.*, p.{energy_column} AS {energy_column_alias}
            FROM tmp_base b
            JOIN poi p
              ON p.session_trial = b.poi_session_trial
            WHERE p.{energy_column} IS NOT NULL
        """
    else:
        join_done = False
        for key in join_keys:
            if key == "session_trial":
                energy_select = f"""
                    SELECT b.*, {energy_alias}.{energy_column} AS {energy_column_alias}
                    FROM tmp_base b
                    JOIN {energy_table} {energy_alias}
                      ON {energy_alias}.session_trial = b.poi_session_trial
                    WHERE {energy_alias}.{energy_column} IS NOT NULL
                """
                join_done = True
                break
            if key == "session":
                energy_select = f"""
                    SELECT b.*, {energy_alias}.{energy_column} AS {energy_column_alias}
                    FROM tmp_base b
                    JOIN {energy_table} {energy_alias}
                      ON {energy_alias}.session = b.session_id
                    WHERE {energy_alias}.{energy_column} IS NOT NULL
                """
                join_done = True
                break
        if not join_done:
            raise ValueError(
                f"Found energy column {energy_table}.{energy_column}, but no supported join key present."
            )

    cursor.execute(energy_select)
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=col_names)
    cursor.close()

    # Cleanups
    if df.empty:
        raise ValueError("Joined dataset is empty. Check date range or join keys.")

    # Filter bat speed > 0 and drop NaNs in key variables
    bat_speed_col = "poi_blast_bat_speed_mph"
    energy_col = energy_column_alias
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[bat_speed_col, energy_col])
    df = df[df[bat_speed_col] > 0]

    return df, energy_col, bat_speed_col


def compute_correlations(df: pd.DataFrame, energy_col: str) -> pd.Series:
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")
    corrs = corr_matrix[energy_col].drop(labels=[energy_col]).sort_values(ascending=False)
    return corrs


def compute_bivariate_stats(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[float, float]:
    try:
        from scipy.stats import pearsonr
    except Exception:
        # Fallback: just return coefficient from pandas, p-value as NaN
        r = df[x_col].corr(df[y_col])
        return float(r), float("nan")
    series = df[[x_col, y_col]].dropna()
    r, p = pearsonr(series[x_col], series[y_col])
    return float(r), float(p)


def compute_feature_importance(
    df: pd.DataFrame, energy_col: str, exclude_cols: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
    except Exception:
        return None

    exclude_cols = exclude_cols or []
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in numeric_df.columns if c != energy_col and c not in exclude_cols]
    X = numeric_df[feature_cols].fillna(numeric_df[feature_cols].median())
    y = numeric_df[energy_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances["model_r2"] = r2
    return importances


def save_plots(
    df: pd.DataFrame,
    energy_col: str,
    bat_speed_col: str,
    correlations: pd.Series,
    feature_importances: Optional[pd.DataFrame],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs("figures", exist_ok=True)

    # Scatter with regression line: energy vs bat speed
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=df,
        x=bat_speed_col,
        y=energy_col,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
    )
    r, p = compute_bivariate_stats(df, bat_speed_col, energy_col)
    plt.title(f"Energy Outflow vs Bat Speed (r={r:.3f}, p={p:.3g})")
    plt.xlabel("Bat Speed (mph)")
    plt.ylabel("L5S1 Energy Outflow")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "energy_vs_batspeed.png"), dpi=150)
    plt.close()

    # Correlation barplot (top 15)
    top_corrs = correlations.head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_corrs.values, y=top_corrs.index, orient="h")
    plt.title("Top correlations with Energy Outflow")
    plt.xlabel("Pearson r")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "top_correlations.png"), dpi=150)
    plt.close()

    # Feature importance
    if feature_importances is not None:
        top_feats = feature_importances.head(20)
        plt.figure(figsize=(9, 8))
        sns.barplot(
            x="importance",
            y="feature",
            data=top_feats,
            orient="h",
        )
        r2 = feature_importances["model_r2"].iloc[0]
        plt.title(f"Feature importance for Energy Outflow (RF, R^2={r2:.3f})")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "feature_importance.png"), dpi=150)
        plt.close()


def main() -> None:
    connection = get_db_connection()
    try:
        df, energy_col, bat_speed_col = fetch_analysis_dataframe(connection)

        # Correlations
        r, p = compute_bivariate_stats(df, bat_speed_col, energy_col)
        print(f"Correlation between {energy_col} and {bat_speed_col}: r={r:.4f}, p={p:.3g}")

        all_corrs = compute_correlations(df, energy_col)
        print("\nTop correlations with energy outflow:")
        print(all_corrs.head(20))

        # Feature importance
        feature_importances = compute_feature_importance(
            df, energy_col, exclude_cols=["session_id"]
        )
        if feature_importances is None:
            print("\nscikit-learn not available. Skipping feature importance.")
        else:
            print("\nTop feature importances (RandomForest):")
            print(feature_importances.head(20))

        # Plots
        save_plots(df, energy_col, bat_speed_col, all_corrs, feature_importances)
        print("\nSaved plots to ./figures")

    finally:
        connection.close()
        print("Connection closed")


if __name__ == "__main__":
    main()


