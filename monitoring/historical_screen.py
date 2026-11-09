"""Train-only activation screening within one historical model trajectory."""
import math


def screen_trajectory(samples, *, steps, protected_classes, channels=128, bottom=16, cap=3):
    """Intersect low-RMS ranks across all declared scenes, steps and small classes."""
    if (len(steps) < 2 or sorted(set(steps)) != list(steps) or not protected_classes
            or not 1 <= cap <= bottom < channels):
        raise ValueError("Ordered checkpoints, explicit cohorts and bounded ranks required")
    expected = {(scene, step) for scene in protected_classes for step in steps}
    observed = [(row["scene"], row["step"]) for row in samples]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Every scene/checkpoint pair must occur exactly once")

    def ranks(values):
        if len(values) != channels or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Finite nonnegative RMS required for each channel")
        order = sorted(range(channels), key=lambda channel: (values[channel], channel))
        result = [0] * channels
        for rank, channel in enumerate(order, 1):
            result[channel] = rank
        return result

    global_ranks = []
    class_ranks = []
    for row in samples:
        expected_classes = {str(i) for i in protected_classes[row["scene"]]}
        if set(row["protected_rms"]) != expected_classes:
            raise ValueError("Every declared small-class bucket must have a measurement")
        global_ranks.append(ranks(row["rms"]))
        class_ranks.extend(ranks(row["protected_rms"][i]) for i in sorted(expected_classes))

    def eligible(rank_rows):
        passing = [c for c in range(channels) if all(row[c] <= bottom for row in rank_rows)]
        return sorted(passing, key=lambda c: (max(row[c] for row in rank_rows),
                                              sum(row[c] for row in rank_rows), c))

    scene_eligible = eligible(global_ranks)
    protected_eligible = eligible(global_ranks + class_ranks)
    return {"policy": "mon_historical_scene_and_small_class_rms_v1", "scene_eligible": scene_eligible,
            "eligible": protected_eligible, "shortlist": protected_eligible[:cap],
            "automatic_pruning": False, "online_gradients_available": False}
