import pandas as pd
import matplotlib.pyplot as plt

STAT_REGISTRY = {}

def register_stat(name):
    def deco(fn):
        STAT_REGISTRY[name] = fn
        return fn
    return deco


class WorkerStatistics:
    def __init__(self, merged_detections, category_map, fps):
        self.fps = fps
        self.category_map = category_map
        self.timeline = self._build_timeline(merged_detections)

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
        return pd.DataFrame(rows).sort_values(["track_id", "frame_id"]).reset_index(drop=True)

    def compute_all(self):
        return {name: fn(self) for name, fn in STAT_REGISTRY.items()}


@register_stat("time_per_category_sec")
def stat_time_per_category(stats):
    frame_duration = 1.0 / stats.fps
    return (stats.timeline.groupby(["track_id", "category"]).size()
            .unstack(fill_value=0) * frame_duration).round(2)


@register_stat("time_per_action_sec")
def stat_time_per_action(stats):
    frame_duration = 1.0 / stats.fps
    return (stats.timeline.groupby(["track_id", "action"]).size()
            .unstack(fill_value=0) * frame_duration).round(2)


@register_stat("video_duration_sec")
def stat_video_duration(stats):
    return round(stats.timeline["frame_id"].max() / stats.fps, 2) if len(stats.timeline) else 0.0


@register_stat("utilization_rate")
def stat_utilization_rate(stats):
    duration = stat_video_duration(stats)
    n_workers = stats.timeline["track_id"].nunique()
    if duration == 0 or n_workers == 0:
        return 0.0
    working_seconds = (stats.timeline["category"] == "working").sum() / stats.fps
    return round(working_seconds / (duration * n_workers), 4)


@register_stat("summary_table")
def stat_summary_table(stats):
    cat_time = stat_time_per_category(stats)
    summary = cat_time.copy()
    summary["total_tracked_sec"] = cat_time.sum(axis=1)
    working_col = cat_time["working"] if "working" in cat_time.columns else 0
    summary["work_pct"] = (working_col / summary["total_tracked_sec"] * 100).round(1)
    return summary


def plot_worker_stats(cat_time, category_colors_hex):
    colors = [category_colors_hex.get(c, "#999999") for c in cat_time.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_time.plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_xlabel("Track ID")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time per worker by category")
    ax.legend(title="Category")
    plt.tight_layout()
    plt.show()