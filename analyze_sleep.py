#!/usr/bin/env python3
"""
WHOOP Sleep Analysis Report Generator.

Loads WHOOP export data, analyzes sleep patterns, and generates
a self-contained HTML report with embedded charts.
"""

import base64
import io
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = sorted(p for p in SCRIPT_DIR.glob("my_whoop_data_*") if p.is_dir())[-1]
OUTPUT_FILE = SCRIPT_DIR / "sleep_report.html"

sns.set_theme(style="white", palette="muted", font_scale=0.9)
COLORS = {
    "light": "#7fb3d8",
    "sws": "#2c5f8a",
    "rem": "#a86bd1",
    "perf": "#e07b39",
    "recovery": "#4caf50",
    "hrv": "#ff7043",
    "strain": "#ef5350",
    "efficiency": "#26a69a",
    "debt": "#ab47bc",
    "need": "#78909c",
}

def load_data():
    sleeps = pd.read_csv(DATA_DIR / "sleeps.csv")
    cycles = pd.read_csv(DATA_DIR / "physiological_cycles.csv")
    workouts = pd.read_csv(DATA_DIR / "workouts.csv")

    time_cols_sleep = ["Cycle start time", "Cycle end time", "Sleep onset", "Wake onset"]
    for col in time_cols_sleep:
        sleeps[col] = pd.to_datetime(sleeps[col], errors="coerce")

    time_cols_cycles = ["Cycle start time", "Cycle end time", "Sleep onset", "Wake onset"]
    for col in time_cols_cycles:
        cycles[col] = pd.to_datetime(cycles[col], errors="coerce")

    time_cols_workouts = ["Cycle start time", "Cycle end time", "Workout start time", "Workout end time"]
    for col in time_cols_workouts:
        workouts[col] = pd.to_datetime(workouts[col], errors="coerce")

    # Nap col
    sleeps["Nap"] = sleeps["Nap"].astype(str).str.lower() == "true"

    return sleeps, cycles, workouts


def prepare_data(sleeps, cycles, workouts):
    # Primary sleeps only (no naps)
    df = sleeps[~sleeps["Nap"]].copy().sort_values("Sleep onset").reset_index(drop=True)

    # Date key from sleep onset
    df["date"] = df["Sleep onset"].dt.date

    # Bedtime / wake as decimal hours
    df["bedtime_hour"] = df["Sleep onset"].dt.hour + df["Sleep onset"].dt.minute / 60
    # Shift so midnight-3am shows as 24-27 instead of 0-3 for plotting
    df["bedtime_hour_adj"] = df["bedtime_hour"].apply(lambda h: h + 24 if h < 12 else h)
    df["wake_hour"] = df["Wake onset"].dt.hour + df["Wake onset"].dt.minute / 60

    # Sleep stage percentages
    total = df["Asleep duration (min)"]
    df["light_pct"] = df["Light sleep duration (min)"] / total * 100
    df["deep_pct"] = df["Deep (SWS) duration (min)"] / total * 100
    df["rem_pct"] = df["REM duration (min)"] / total * 100

    # Asleep duration in hours
    df["asleep_hours"] = df["Asleep duration (min)"] / 60
    df["need_hours"] = df["Sleep need (min)"] / 60

    # Merge recovery/HRV from physiological cycles
    cyc = cycles[["Cycle start time", "Recovery score %", "Resting heart rate (bpm)",
                   "Heart rate variability (ms)", "Day Strain"]].copy()
    cyc = cyc.rename(columns={
        "Recovery score %": "recovery",
        "Resting heart rate (bpm)": "rhr",
        "Heart rate variability (ms)": "hrv",
        "Day Strain": "strain",
    })
    df = df.merge(cyc, on="Cycle start time", how="left")

    # Aggregate workouts per cycle
    wo = workouts.groupby("Cycle start time").agg(
        workout_count=("Duration (min)", "count"),
        total_workout_min=("Duration (min)", "sum"),
        max_workout_strain=("Activity Strain", "max"),
        total_workout_cal=("Energy burned (cal)", "sum"),
    ).reset_index()
    df = df.merge(wo, on="Cycle start time", how="left")
    df[["workout_count", "total_workout_min", "max_workout_strain", "total_workout_cal"]] = \
        df[["workout_count", "total_workout_min", "max_workout_strain", "total_workout_cal"]].fillna(0)

    df["has_workout"] = df["workout_count"] > 0

    # Week label for aggregates
    df["week"] = df["Sleep onset"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["Sleep onset"].dt.to_period("W").apply(lambda p: p.start_time)

    return df


# chart helpters

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_date_axis(ax, df):
    ax.set_xlim(df["date"].min(), df["date"].max())
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


# plotting

def chart_duration_performance(df):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df["date"], df["asleep_hours"], color=COLORS["sws"], linewidth=1.5, label="Asleep (hrs)")
    ax1.plot(df["date"], df["need_hours"], color=COLORS["need"], linewidth=1, linestyle="--", alpha=0.7, label="Sleep need")
    avg_sleep = df["asleep_hours"].mean()
    ax1.axhline(y=avg_sleep, color=COLORS["sws"], linestyle=":", alpha=0.4, linewidth=1)
    ax1.annotate(f"avg {avg_sleep:.1f}h", xy=(df["date"].iloc[0], avg_sleep), fontsize=7, color=COLORS["sws"], alpha=0.7, va="bottom")
    ax1.set_ylabel("Hours")
    ax1.set_ylim(0, max(df["asleep_hours"].max(), df["need_hours"].max()) + 1)
    make_date_axis(ax1, df)

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["Sleep performance %"], color=COLORS["perf"], linewidth=1.5, alpha=0.8, label="Performance %")
    avg_perf = df["Sleep performance %"].mean()
    ax2.axhline(y=avg_perf, color=COLORS["perf"], linestyle=":", alpha=0.4, linewidth=1)
    ax2.annotate(f"avg {avg_perf:.0f}%", xy=(df["date"].iloc[-1], avg_perf), fontsize=7, color=COLORS["perf"], alpha=0.7, va="bottom", ha="right")
    ax2.set_ylabel("Performance %")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
    return fig


def chart_sleep_stages(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(
        df["date"],
        df["Deep (SWS) duration (min)"],
        df["REM duration (min)"],
        df["Light sleep duration (min)"],
        labels=["SWS", "REM", "Light"],
        colors=[COLORS["sws"], COLORS["rem"], COLORS["light"]],
        alpha=0.85,
    )
    ax.set_ylabel("Minutes")
    ax.legend(loc="upper left", fontsize=8)
    make_date_axis(ax, df)
    return fig


def chart_stage_proportions(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.8
    x = np.arange(len(df))
    ax.bar(x, df["deep_pct"], width, label="SWS %", color=COLORS["sws"])
    ax.bar(x, df["rem_pct"], width, bottom=df["deep_pct"], label="REM %", color=COLORS["rem"])
    # bands
    ax.axhline(y=15, color=COLORS["sws"], linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axhline(y=25, color=COLORS["sws"], linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axhspan(15, 25, alpha=0.06, color=COLORS["sws"], label="SWS target (15-25%)")
    ax.set_ylabel("% of Sleep")
    # sparse x labels
    tick_idx = np.linspace(0, len(df) - 1, min(15, len(df)), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(df.iloc[i]["date"]) for i in tick_idx], rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    return fig


def chart_bedtime_consistency(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(df["date"], df["bedtime_hour_adj"], s=20, color=COLORS["sws"], label="Bedtime", zorder=3)
    ax.plot(df["date"], df["bedtime_hour_adj"], color=COLORS["sws"], linewidth=0.8, alpha=0.5)
    avg_bed = df["bedtime_hour_adj"].mean()
    ax.axhline(y=avg_bed, color=COLORS["sws"], linestyle=":", alpha=0.4, linewidth=1)
    ax.annotate(f"avg {int(avg_bed % 24):02d}:{int((avg_bed % 1) * 60):02d}", xy=(df["date"].iloc[0], avg_bed), fontsize=7, color=COLORS["sws"], alpha=0.7, va="bottom")

    ax.scatter(df["date"], df["wake_hour"], s=20, color=COLORS["perf"], label="Wake time", zorder=3)
    ax.plot(df["date"], df["wake_hour"], color=COLORS["perf"], linewidth=0.8, alpha=0.5)
    avg_wake = df["wake_hour"].mean()
    ax.axhline(y=avg_wake, color=COLORS["perf"], linestyle=":", alpha=0.4, linewidth=1)
    ax.annotate(f"avg {int(avg_wake):02d}:{int((avg_wake % 1) * 60):02d}", xy=(df["date"].iloc[0], avg_wake), fontsize=7, color=COLORS["perf"], alpha=0.7, va="bottom")

    ymin = int(df["wake_hour"].min())
    ymax = int(df["bedtime_hour_adj"].max()) + 2
    # Minor ticks every hour (gridlines), major ticks every 2 hours (labels)
    major_ticks = list(range(ymin - (ymin % 2), ymax + 1, 2))
    minor_ticks = list(range(ymin, ymax + 1))
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([f"{h % 24:02d}:00" for h in major_ticks])
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="minor", linestyle="-", alpha=0.15)
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.15)
    ax.set_ylabel("Time of Day")
    ax.legend(fontsize=8)
    make_date_axis(ax, df)
    return fig


def chart_sleep_debt(df):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(df["date"], df["Sleep debt (min)"], alpha=0.4, color=COLORS["debt"])
    ax.plot(df["date"], df["Sleep debt (min)"], color=COLORS["debt"], linewidth=1.5)
    avg_debt = df["Sleep debt (min)"].mean()
    ax.axhline(y=avg_debt, color=COLORS["debt"], linestyle=":", alpha=0.4, linewidth=1)
    ax.annotate(f"avg {avg_debt:.0f} min", xy=(df["date"].iloc[0], avg_debt), fontsize=7, color=COLORS["debt"], alpha=0.7, va="bottom")
    ax.set_ylabel("Sleep Debt (min)")
    make_date_axis(ax, df)
    return fig


def chart_efficiency(df):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df["date"], df["Sleep efficiency %"], color=COLORS["efficiency"], linewidth=1.5)
    ax.axhline(y=85, color="gray", linestyle="--", alpha=0.5, label="85% target")
    avg_eff = df["Sleep efficiency %"].mean()
    ax.axhline(y=avg_eff, color=COLORS["efficiency"], linestyle=":", alpha=0.4, linewidth=1)
    ax.annotate(f"avg {avg_eff:.0f}%", xy=(df["date"].iloc[0], avg_eff), fontsize=7, color=COLORS["efficiency"], alpha=0.7, va="bottom")
    ax.set_ylabel("Efficiency %")
    ax.set_ylim(50, 105)
    ax.legend(fontsize=8)
    make_date_axis(ax, df)
    return fig


def chart_recovery_hrv_corr(df):
    valid = df.dropna(subset=["recovery", "hrv"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Sleep performance vs recovery
    ax = axes[0]
    ax.scatter(valid["Sleep performance %"], valid["recovery"], s=25, alpha=0.6, color=COLORS["recovery"])
    z = np.polyfit(valid["Sleep performance %"].values, valid["recovery"].values, 1)
    xline = np.linspace(valid["Sleep performance %"].min(), valid["Sleep performance %"].max(), 50)
    ax.plot(xline, np.polyval(z, xline), color=COLORS["recovery"], linestyle="--", linewidth=1)
    r = valid["Sleep performance %"].corr(valid["recovery"])
    ax.set_xlabel("Sleep Performance %")
    ax.set_ylabel("Recovery %")
    ax.annotate(f"r={r:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=8, va="top")

    # Asleep duration vs HRV
    ax = axes[1]
    ax.scatter(valid["asleep_hours"], valid["hrv"], s=25, alpha=0.6, color=COLORS["hrv"])
    z = np.polyfit(valid["asleep_hours"].values, valid["hrv"].values, 1)
    xline = np.linspace(valid["asleep_hours"].min(), valid["asleep_hours"].max(), 50)
    ax.plot(xline, np.polyval(z, xline), color=COLORS["hrv"], linestyle="--", linewidth=1)
    r = valid["asleep_hours"].corr(valid["hrv"])
    ax.set_xlabel("Asleep Duration (hrs)")
    ax.set_ylabel("HRV (ms)")
    ax.annotate(f"r={r:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=8, va="top")

    # SWS vs recovery
    ax = axes[2]
    ax.scatter(valid["Deep (SWS) duration (min)"], valid["recovery"], s=25, alpha=0.6, color=COLORS["sws"])
    z = np.polyfit(valid["Deep (SWS) duration (min)"].values, valid["recovery"].values, 1)
    xline = np.linspace(valid["Deep (SWS) duration (min)"].min(), valid["Deep (SWS) duration (min)"].max(), 50)
    ax.plot(xline, np.polyval(z, xline), color=COLORS["sws"], linestyle="--", linewidth=1)
    r = valid["Deep (SWS) duration (min)"].corr(valid["recovery"])
    ax.set_xlabel("Slow-Wave Sleep (min)")
    ax.set_ylabel("Recovery %")
    ax.annotate(f"r={r:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=8, va="top")

    fig.tight_layout()
    return fig


def chart_training_impact(df):
    valid = df.dropna(subset=["strain"])
    has_strain = valid[valid["strain"] > 0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Day strain vs sleep performance that night
    ax = axes[0]
    if len(has_strain) > 2:
        ax.scatter(has_strain["strain"], has_strain["Sleep performance %"], s=25, alpha=0.6, color=COLORS["strain"])
        z = np.polyfit(has_strain["strain"].values, has_strain["Sleep performance %"].values, 1)
        xline = np.linspace(has_strain["strain"].min(), has_strain["strain"].max(), 50)
        ax.plot(xline, np.polyval(z, xline), color=COLORS["strain"], linestyle="--", linewidth=1)
        r = has_strain["strain"].corr(has_strain["Sleep performance %"])
        ax.annotate(f"r={r:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=8, va="top")
    ax.set_xlabel("Day Strain")
    ax.set_ylabel("Sleep Performance %")

    # Workout duration vs sleep performance
    ax = axes[1]
    wo_days = df[df["has_workout"]].copy()
    if len(wo_days) > 2:
        ax.scatter(wo_days["total_workout_min"], wo_days["Sleep performance %"], s=25, alpha=0.6, color=COLORS["perf"])
        z = np.polyfit(wo_days["total_workout_min"].values, wo_days["Sleep performance %"].values, 1)
        xline = np.linspace(wo_days["total_workout_min"].min(), wo_days["total_workout_min"].max(), 50)
        ax.plot(xline, np.polyval(z, xline), color=COLORS["perf"], linestyle="--", linewidth=1)
        r = wo_days["total_workout_min"].corr(wo_days["Sleep performance %"])
        ax.annotate(f"r={r:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=8, va="top")
    ax.set_xlabel("Workout Duration (min)")
    ax.set_ylabel("Sleep Performance %")

    # Workout days vs rest days
    ax = axes[2]
    metrics = ["Sleep performance %", "Sleep efficiency %", "deep_pct", "rem_pct"]
    labels = ["Perf %", "Efficiency %", "SWS %", "REM %"]
    wo_means = [df[df["has_workout"]][m].mean() for m in metrics]
    rest_means = [df[~df["has_workout"]][m].mean() for m in metrics]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, wo_means, 0.35, label="Workout days", color=COLORS["strain"], alpha=0.8)
    ax.bar(x + 0.18, rest_means, 0.35, label="Rest days", color=COLORS["need"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def chart_weekly_agg(df):
    weekly = df.groupby("week_start").agg(
        sleep_hrs=("asleep_hours", "mean"),
        perf=("Sleep performance %", "mean"),
        recovery=("recovery", "mean"),
        hrv=("hrv", "mean"),
    ).dropna().reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 4))
    dates = weekly["week_start"]
    ax1.plot(dates, weekly["sleep_hrs"], color=COLORS["sws"], linewidth=1.5, label="Avg Sleep (hrs)")
    ax1.set_ylabel("Hours")
    ax1.set_ylim(0, weekly["sleep_hrs"].max() + 2)

    ax2 = ax1.twinx()
    ax2.plot(dates, weekly["perf"], color=COLORS["perf"], linewidth=1.5, label="Avg Perf %")
    ax2.plot(dates, weekly["recovery"], color=COLORS["recovery"], linewidth=1.5, label="Avg Recovery %")
    ax2.plot(dates, weekly["hrv"], color=COLORS["hrv"], linewidth=1.5, label="Avg HRV (ms)")
    ax2.set_ylabel("% / ms")
    ax2.set_ylim(0, 110)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
    return fig


def build_summary(df):
    valid = df.dropna(subset=["recovery", "hrv"])
    metrics = {
        "Asleep Duration (hrs)": df["asleep_hours"],
        "Sleep Performance %": df["Sleep performance %"],
        "Sleep Efficiency %": df["Sleep efficiency %"],
        "Slow-Wave Sleep %": df["deep_pct"],
        "REM Sleep %": df["rem_pct"],
        "Sleep Debt (min)": df["Sleep debt (min)"],
        "Respiratory Rate (rpm)": df["Respiratory rate (rpm)"],
        "Recovery %": valid["recovery"],
        "HRV (ms)": valid["hrv"],
        "RHR (bpm)": valid["rhr"],
    }
    rows = []
    for name, series in metrics.items():
        rows.append({
            "Metric": name,
            "Mean": f"{series.mean():.1f}",
            "Median": f"{series.median():.1f}",
            "Min": f"{series.min():.1f}",
            "Max": f"{series.max():.1f}",
            "Std Dev": f"{series.std():.1f}",
        })
    return pd.DataFrame(rows)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WHOOP Sleep Analysis Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #333; background: #fafafa; }}
  h1 {{ border-bottom: 2px solid #2c5f8a; padding-bottom: 8px; }}
  h2 {{ color: #2c5f8a; margin-top: 40px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
  th {{ background: #2c5f8a; color: white; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  td:first-child {{ text-align: left; font-weight: 600; }}
  img {{ width: 100%; margin: 12px 0; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .meta {{ color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Sleep analysis report</h1>
<p class="meta">Generated {generated} &mdash; Data: {date_range} ({n_nights} nights)</p>

<h2>Summary statistics</h2>
{summary_table}

<h2>Sleep duration &amp; performance</h2>
<img src="data:image/png;base64,{chart_duration}" alt="Duration & Performance">

<h2>Sleep stages over time</h2>
<img src="data:image/png;base64,{chart_stages}" alt="Sleep Stages">

<h2>SWS + REM proportions</h2>
<img src="data:image/png;base64,{chart_proportions}" alt="Stage Proportions">

<h2>Bedtime &amp; wake time consistency</h2>
<img src="data:image/png;base64,{chart_consistency}" alt="Bedtime Consistency">

<h2>Sleep debt trend</h2>
<img src="data:image/png;base64,{chart_debt}" alt="Sleep Debt">

<h2>Sleep efficiency</h2>
<img src="data:image/png;base64,{chart_efficiency}" alt="Sleep Efficiency">

<h2>Recovery &amp; HRV correlations</h2>
<img src="data:image/png;base64,{chart_recovery}" alt="Recovery & HRV">

<h2>Training load impact</h2>
<img src="data:image/png;base64,{chart_training}" alt="Training Impact">

<h2>Weekly averages</h2>
<img src="data:image/png;base64,{chart_weekly}" alt="Weekly Averages">

</body>
</html>"""


def generate_report(df):
    summary = build_summary(df)

    charts = {
        "chart_duration": fig_to_base64(chart_duration_performance(df)),
        "chart_stages": fig_to_base64(chart_sleep_stages(df)),
        "chart_proportions": fig_to_base64(chart_stage_proportions(df)),
        "chart_consistency": fig_to_base64(chart_bedtime_consistency(df)),
        "chart_debt": fig_to_base64(chart_sleep_debt(df)),
        "chart_efficiency": fig_to_base64(chart_efficiency(df)),
        "chart_recovery": fig_to_base64(chart_recovery_hrv_corr(df)),
        "chart_training": fig_to_base64(chart_training_impact(df)),
        "chart_weekly": fig_to_base64(chart_weekly_agg(df)),
    }

    html = HTML_TEMPLATE.format(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        date_range=f"{df['date'].min()} to {df['date'].max()}",
        n_nights=len(df),
        summary_table=summary.to_html(index=False, classes="summary"),
        **charts,
    )

    OUTPUT_FILE.write_text(html)
    print(f"Report saved to {OUTPUT_FILE}")


def main():
    print(f"Loading data from {DATA_DIR} ...")
    sleeps, cycles, workouts = load_data()
    print(f"  {len(sleeps)} sleep records, {len(cycles)} cycles, {len(workouts)} workouts")

    df = prepare_data(sleeps, cycles, workouts)
    print(f"  {len(df)} primary sleep nights (naps excluded)")

    generate_report(df)


if __name__ == "__main__":
    main()
