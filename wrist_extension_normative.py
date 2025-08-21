import os
from dotenv import load_dotenv
import mariadb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv("config.env")


def connect_to_database():
    try:
        return mariadb.connect(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT', 3306)),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def query_wrist_angles_at_event(connection, event_time_col, label, limit=100000):
    cursor = connection.cursor()
    query = f"""
    SELECT
        e.session_trial,
        e.{event_time_col} AS event_time,
        j.wrist_angle_x AS wrist_extension_at_{label},
        j.wrist_angle_y AS wrist_flexion_at_{label},
        j.wrist_angle_z AS wrist_radial_ulnar_at_{label},
        s.height_meters,
        s.mass_kilograms
    FROM events e
    JOIN joint_angles j ON j.session_trial = e.session_trial
        AND j.time = e.{event_time_col}
    JOIN sessions s ON s.session = CAST(SUBSTRING_INDEX(e.session_trial, '_', 1) AS UNSIGNED)
    WHERE e.{event_time_col} IS NOT NULL
        AND j.wrist_angle_x IS NOT NULL
    ORDER BY e.session_trial
    LIMIT {int(limit)}
    """
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(results, columns=cols)
        for col in [
            f'wrist_extension_at_{label}',
            f'wrist_flexion_at_{label}',
            f'wrist_radial_ulnar_at_{label}',
            'height_meters',
            'mass_kilograms',
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Error executing {label} query: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()


def summarize(series):
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return {}
    return {
        'count': int(series.size),
        'mean': float(series.mean()),
        'std': float(series.std()),
        'median': float(series.median()),
        'p5': float(series.quantile(0.05)),
        'p95': float(series.quantile(0.95)),
    }


def print_summary(label, stats):
    if not stats:
        print(f"No {label} results.")
        return
    print(f"{label} wrist extension normative range: {stats['p5']:.2f}° to {stats['p95']:.2f}° (n={stats['count']})")


def plot_combined(fp_series, br_series, fp_stats, br_stats, output_path='wrist_extension_fp_br_distribution.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Wrist Extension Distributions: Foot Plant vs Ball Release', fontsize=16)

    ax_fp = axes[0, 0]
    if not fp_series.empty:
        sns.histplot(fp_series, bins=50, kde=True, color='steelblue', edgecolor='black', ax=ax_fp)
        ax_fp.set_title('Foot Plant')
        ax_fp.set_xlabel('Degrees')
        ax_fp.set_ylabel('Count')
        ax_fp.axvline(fp_stats['mean'], color='red', linestyle='--', label=f"Mean: {fp_stats['mean']:.2f}")
        ax_fp.axvline(fp_stats['p5'], color='green', linestyle='--', label=f"P5: {fp_stats['p5']:.2f}")
        ax_fp.axvline(fp_stats['p95'], color='green', linestyle='--', label=f"P95: {fp_stats['p95']:.2f}")
        ax_fp.legend(); ax_fp.grid(True, alpha=0.25)
    else:
        ax_fp.text(0.5, 0.5, 'No FP data', ha='center', va='center', transform=ax_fp.transAxes)
        ax_fp.axis('off')

    ax_br = axes[0, 1]
    if not br_series.empty:
        sns.histplot(br_series, bins=50, kde=True, color='darkorange', edgecolor='black', ax=ax_br)
        ax_br.set_title('Ball Release')
        ax_br.set_xlabel('Degrees')
        ax_br.set_ylabel('Count')
        ax_br.axvline(br_stats['mean'], color='red', linestyle='--', label=f"Mean: {br_stats['mean']:.2f}")
        ax_br.axvline(br_stats['p5'], color='green', linestyle='--', label=f"P5: {br_stats['p5']:.2f}")
        ax_br.axvline(br_stats['p95'], color='green', linestyle='--', label=f"P95: {br_stats['p95']:.2f}")
        ax_br.legend(); ax_br.grid(True, alpha=0.25)
    else:
        ax_br.text(0.5, 0.5, 'No BR data', ha='center', va='center', transform=ax_br.transAxes)
        ax_br.axis('off')

    ax_fp_stats = axes[1, 0]; ax_fp_stats.axis('off')
    if fp_stats:
        fp_text = (
            f"Foot Plant (n={fp_stats['count']})\n"
            f"Mean: {fp_stats['mean']:.2f}°, SD: {fp_stats['std']:.2f}°\n"
            f"Median: {fp_stats['median']:.2f}°\n"
            f"P5-P95: {fp_stats['p5']:.2f}° to {fp_stats['p95']:.2f}°"
        )
        ax_fp_stats.text(0.02, 0.95, fp_text, va='top', fontsize=12,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray'))

    ax_br_stats = axes[1, 1]; ax_br_stats.axis('off')
    if br_stats:
        br_text = (
            f"Ball Release (n={br_stats['count']})\n"
            f"Mean: {br_stats['mean']:.2f}°, SD: {br_stats['std']:.2f}°\n"
            f"Median: {br_stats['median']:.2f}°\n"
            f"P5-P95: {br_stats['p5']:.2f}° to {br_stats['p95']:.2f}°"
        )
        ax_br_stats.text(0.02, 0.95, br_text, va='top', fontsize=12,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    print(f"Saved combined distribution figure: {output_path}")
    plt.show()


def main():
    print("=== Wrist Extension: Foot Plant and Ball Release ===")
    conn = connect_to_database()
    if not conn:
        return
    try:
        df_fp = query_wrist_angles_at_event(conn, 'FP_v5_time', 'foot_plant')
        df_br = query_wrist_angles_at_event(conn, 'BR_time', 'ball_release')

        fp_stats = summarize(df_fp.get('wrist_extension_at_foot_plant', pd.Series(dtype=float)))
        br_stats = summarize(df_br.get('wrist_extension_at_ball_release', pd.Series(dtype=float)))

        print()
        print_summary('Foot Plant', fp_stats)
        print_summary('Ball Release', br_stats)

        if not df_fp.empty:
            df_fp.to_csv('wrist_extension_at_foot_plant.csv', index=False)
            print('Saved: wrist_extension_at_foot_plant.csv')
        if not df_br.empty:
            df_br.to_csv('wrist_extension_at_ball_release.csv', index=False)
            print('Saved: wrist_extension_at_ball_release.csv')

        plot_combined(
            pd.to_numeric(df_fp.get('wrist_extension_at_foot_plant', pd.Series(dtype=float)), errors='coerce').dropna(),
            pd.to_numeric(df_br.get('wrist_extension_at_ball_release', pd.Series(dtype=float)), errors='coerce').dropna(),
            fp_stats,
            br_stats,
        )
    finally:
        conn.close()
        print("\nDatabase connection closed.")


if __name__ == '__main__':
    main()


