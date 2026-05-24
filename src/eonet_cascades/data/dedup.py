"""Cross-catalog spatio-temporal deduplication."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from datetime import timedelta

from eonet_cascades.data.schema import Event, Mark


class _Threshold:
    __slots__ = ("spatial_km", "temporal")

    def __init__(self, spatial_km: float, temporal: timedelta) -> None:
        self.spatial_km = spatial_km
        self.temporal = temporal


DEFAULT_THRESHOLDS: dict[Mark, _Threshold] = {
    Mark.EARTHQUAKE: _Threshold(5.0, timedelta(hours=1)),
    Mark.VOLCANIC_ERUPTION: _Threshold(5.0, timedelta(hours=24)),
    Mark.WILDFIRE: _Threshold(25.0, timedelta(hours=24)),
    Mark.TROPICAL_CYCLONE: _Threshold(200.0, timedelta(hours=12)),
    Mark.TORNADO: _Threshold(25.0, timedelta(hours=6)),
    Mark.SEVERE_STORM: _Threshold(25.0, timedelta(hours=6)),
    Mark.FLOOD: _Threshold(50.0, timedelta(hours=24)),
    Mark.LANDSLIDE: _Threshold(50.0, timedelta(hours=24)),
    Mark.DROUGHT: _Threshold(100.0, timedelta(days=7)),
    Mark.TEMPERATURE_EXTREME: _Threshold(100.0, timedelta(days=7)),
    Mark.DUST_HAZE: _Threshold(100.0, timedelta(days=7)),
    Mark.SEA_LAKE_ICE: _Threshold(100.0, timedelta(days=7)),
}


def assign_dedup_groups(
    events: Iterable[Event],
    thresholds: dict[Mark, _Threshold] | None = None,
) -> list[Event]:
    """Assign `dedup_group_id` to events that cluster within mark-specific spatial/temporal thresholds.

    Returns a new list of Event instances (Events are frozen) — does not mutate input.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    events = list(events)
    if not events:
        return []

    # Union-find over event indices, restricted to same-mark candidates.
    parent = list(range(len(events)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Bucket events by mark.
    by_mark: dict[Mark, list[int]] = {}
    for idx, ev in enumerate(events):
        by_mark.setdefault(ev.mark, []).append(idx)

    for mark, idxs in by_mark.items():
        t = th.get(mark)
        if t is None:
            continue
        # Sort by time and do a sliding-window comparison.
        # IMPORTANT: do not slice `idxs[i_pos+1:]` — it allocates a new list
        # every outer iteration and turns the algorithm O(N^2) in memory ops,
        # which becomes hours-long on 100k+ events. Use index-based iteration.
        idxs.sort(key=lambda i: events[i].time_start)
        n = len(idxs)
        for i_pos in range(n):
            i = idxs[i_pos]
            for j_pos in range(i_pos + 1, n):
                j = idxs[j_pos]
                if events[j].time_start - events[i].time_start > t.temporal:
                    break
                if _haversine_km(
                    events[i].longitude, events[i].latitude,
                    events[j].longitude, events[j].latitude,
                ) <= t.spatial_km:
                    union(i, j)

    # Materialize stable group ids: hash of sorted member event_ids.
    groups: dict[int, list[int]] = {}
    for i in range(len(events)):
        groups.setdefault(find(i), []).append(i)

    out: list[Event] = list(events)
    for _root, members in groups.items():
        member_ids = sorted(events[m].event_id for m in members)
        gid = "dg_" + hashlib.sha1("|".join(member_ids).encode()).hexdigest()[:12]
        for m in members:
            out[m] = events[m].model_copy(update={"dedup_group_id": gid})
    return out


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometers."""
    r_earth = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r_earth * math.asin(math.sqrt(a))
