import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STAT_REGISTRY = {}


def register_stat(name):
    def deco(fn):
        STAT_REGISTRY[name] = fn
        return fn
    return deco


class WorkerStatistics:
    def __init__(self, merged_detections, category_map, fps,
                 min_track_frames=15, min_coverage=0.5):
        """
        Parameters
        ----------
        merged_detections : dict[int, list[dict]]
            frame_id -> list of detections, each with "tracked_id", "action".
        category_map : dict[str, str]
            action -> category (e.g. "lifting" -> "working").
        fps : float
        min_track_frames : int
            Tracks shorter than this are treated as tracker noise (blips)
            and dropped before any stats are computed, rather than being
            counted as real workers.
        min_coverage : float in [0, 1]
            Threshold used to flag (not drop) tracks whose detections are
            sparse relative to their own frame span -- likely occlusion /
            re-ID issues worth reviewing manually.
        """
        if fps <= 0:
            raise ValueError("fps must be positive")

        self.fps = fps
        self.category_map = category_map
        self.min_track_frames = min_track_frames
        self.min_coverage = min_coverage

        raw_timeline = self._build_timeline(merged_detections)

        self.unmapped_actions = sorted(
            set(raw_timeline.loc[raw_timeline["category"] == "unknown", "action"])
        ) if not raw_timeline.empty else []

        quality = self._build_track_quality(raw_timeline)
        noisy_ids = quality.index[quality["n_frames"] < self.min_track_frames]

        self.dropped_tracks = quality.loc[noisy_ids]
        self.track_quality = quality.drop(index=noisy_ids)
        self.timeline = raw_timeline[
            ~raw_timeline["track_id"].isin(noisy_ids)
        ].reset_index(drop=True)

    # ---------- construction helpers ----------

    def _build_timeline(self, merged_detections):
        rows = [
            {
                "frame_id": frame_id,
                "time_sec": frame_id / self.fps,
                "track_id": d["tracked_id"],
                "action": d["action"],
                "category": self.category_map.get(d["action"], "unknown"),
            }
            for frame_id, dets in merged_detections.items()
            for d in dets
        ]
        df = pd.DataFrame(rows, columns=["frame_id", "time_sec", "track_id", "action", "category"])
        return df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)

    def _build_track_quality(self, timeline):
        cols = ["n_frames", "first_frame", "last_frame", "span_frames", "coverage"]
        if timeline.empty:
            return pd.DataFrame(columns=cols)
        g = timeline.groupby("track_id")["frame_id"]
        n_frames = g.size()
        first, last = g.min(), g.max()
        span = last - first + 1
        coverage = (n_frames / span).round(3)
        return pd.DataFrame({
            "n_frames": n_frames, "first_frame": first, "last_frame": last,
            "span_frames": span, "coverage": coverage,
        })[cols]

    def compute_all(self):
        return {name: fn(self) for name, fn in STAT_REGISTRY.items()}


# ---------------------------------------------------------------------
# registered stats
# ---------------------------------------------------------------------

@register_stat("time_per_category_sec")
def stat_time_per_category(stats):
    if stats.timeline.empty:
        return pd.DataFrame()
    frame_duration = 1.0 / stats.fps
    return (stats.timeline.groupby(["track_id", "category"]).size()
            .unstack(fill_value=0) * frame_duration).round(2)


@register_stat("time_per_action_sec")
def stat_time_per_action(stats):
    if stats.timeline.empty:
        return pd.DataFrame()
    frame_duration = 1.0 / stats.fps
    return (stats.timeline.groupby(["track_id", "action"]).size()
            .unstack(fill_value=0) * frame_duration).round(2)


@register_stat("video_duration_sec")
def stat_video_duration(stats):
    return round(stats.timeline["frame_id"].max() / stats.fps, 2) if len(stats.timeline) else 0.0


@register_stat("track_presence_sec")
def stat_track_presence(stats):
    if stats.track_quality.empty:
        return pd.Series(dtype=float, name="presence_sec")
    return (stats.track_quality["span_frames"] / stats.fps).round(2).rename("presence_sec")


@register_stat("track_quality_table")
def stat_track_quality(stats):
    q = stats.track_quality.copy()
    if q.empty:
        return q
    q["presence_sec"] = (q["span_frames"] / stats.fps).round(2)
    q["low_coverage_flag"] = q["coverage"] < stats.min_coverage
    return q.sort_values("coverage")


@register_stat("utilization_rate")
def stat_utilization_rate(stats):
    if stats.timeline.empty:
        return {"per_track": {}, "overall": 0.0}

    frame_duration = 1.0 / stats.fps
    presence = stat_track_presence(stats)

    working = (stats.timeline[stats.timeline["category"] == "working"]
               .groupby("track_id").size() * frame_duration)
    working = working.reindex(presence.index).fillna(0.0)

    per_track = (working / presence).replace([np.inf, -np.inf], 0).fillna(0).round(4)
    overall = round(working.sum() / presence.sum(), 4) if presence.sum() else 0.0

    return {"per_track": per_track.to_dict(), "overall": overall}


@register_stat("summary_table")
def stat_summary_table(stats):
    cat_time = stat_time_per_category(stats)
    if cat_time.empty:
        return cat_time

    presence = stat_track_presence(stats)
    working_col = cat_time["working"] if "working" in cat_time.columns else 0

    summary = cat_time.copy()
    summary["presence_sec"] = presence
    summary["total_tracked_sec"] = cat_time.sum(axis=1)
    summary["coverage"] = stats.track_quality["coverage"]
    summary["utilization_pct"] = (working_col / summary["presence_sec"] * 100).round(1)
    return summary.sort_index()


# ---------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------

def plot_worker_stats(cat_time, category_colors_hex, save_path=None):
    """Stacked bar: time per category, per worker."""
    if cat_time.empty:
        print("No data to plot.")
        return None
    colors = [category_colors_hex.get(c, "#999999") for c in cat_time.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_time.plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_xlabel("Track ID")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time per worker by category")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_utilization_rates(summary_table, save_path=None, threshold=None):
    """Horizontal bar of per-worker utilization %, sorted -- much easier to
    scan for outliers than a single aggregate percentage."""
    if summary_table.empty:
        print("No data to plot.")
        return None
    data = summary_table["utilization_pct"].sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(data))))
    colors = ["#d9534f" if v < (threshold or 0) else "#5cb85c" for v in data]
    ax.barh(data.index.astype(str), data.values, color=colors)
    ax.set_xlabel("Utilization (%)")
    ax.set_ylabel("Track ID")
    ax.set_title("Utilization rate by worker (own presence window)")
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
                    label=f"Target: {threshold}%")
        ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
