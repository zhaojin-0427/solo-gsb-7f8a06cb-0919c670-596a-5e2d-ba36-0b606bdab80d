from __future__ import annotations

from tests.conftest import ack_event, get_events, post_records


def sample(sequence: int, at: str, turbidity: float, device: str = "sensor-a"):
    return {
        "device_id": device,
        "sequence": sequence,
        "sampled_at": at,
        "turbidity": turbidity,
    }


def exposure(client, device: str = "sensor-a", **overrides) -> dict:
    payload = {
        "from": "2026-09-23T08:00:00Z",
        "to": "2026-09-23T08:10:00Z",
        "threshold": 10.0,
    }
    payload.update(overrides)
    response = client.post(f"/devices/{device}/turbidity-exposure", json=payload)
    assert response.status_code == 200
    return response.json()


def test_linear_interpolation_crossing_threshold(client):
    # 0 -> 20 over ten minutes crosses the threshold of 10 at 08:05.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 0.0),
            sample(2, "2026-09-23T08:10:00Z", 20.0),
        ],
    )

    body = exposure(client)
    assert body["device_id"] == "sensor-a"
    assert body["from"] == "2026-09-23T08:00:00Z"
    assert body["to"] == "2026-09-23T08:10:00Z"
    assert body["threshold"] == 10.0
    assert body["covered_seconds"] == 600.0
    assert body["exceedance_seconds"] == 300.0
    # Triangle above the threshold: (0 + 10) / 2 * 300.
    assert body["excess_integral"] == 1500.0


def test_crossing_at_span_endpoint_is_not_split(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 0.0),
            sample(2, "2026-09-23T08:10:00Z", 20.0),
        ],
    )

    # The trace reaches 20 only at the final instant: strictly above never
    # happens, so nothing exceeds a threshold of 20.
    body = exposure(client, threshold=20.0)
    assert body["covered_seconds"] == 600.0
    assert body["exceedance_seconds"] == 0.0
    assert body["excess_integral"] == 0.0

    # A threshold of 0 is exceeded over the whole span except the first
    # instant, which has zero length.
    body = exposure(client, threshold=0.0)
    assert body["covered_seconds"] == 600.0
    assert body["exceedance_seconds"] == 600.0
    assert body["excess_integral"] == 6000.0


def test_gap_over_limit_is_not_interpolated(client):
    # Twenty minutes apart exceeds the 600 second limit: no interpolation
    # across the gap, so the window has no covered span at all.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 20.0),
            sample(2, "2026-09-23T08:20:00Z", 20.0),
        ],
    )

    body = exposure(client, to="2026-09-23T08:20:00Z")
    assert body["covered_seconds"] == 0.0
    assert body["exceedance_seconds"] == 0.0
    assert body["excess_integral"] == 0.0


def test_gap_equal_to_limit_still_interpolates(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 20.0),
            sample(2, "2026-09-23T08:10:00Z", 20.0),
        ],
    )

    body = exposure(client)
    assert body["covered_seconds"] == 600.0
    assert body["exceedance_seconds"] == 600.0
    assert body["excess_integral"] == 6000.0


def test_same_timestamp_keeps_highest_sequence(client):
    # Two samples share 08:00; the highest sequence (turbidity 20) wins, so
    # the whole span sits at 20 instead of rising from 0.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 0.0),
            sample(2, "2026-09-23T08:00:00Z", 20.0),
            sample(3, "2026-09-23T08:05:00Z", 20.0),
        ],
    )

    body = exposure(client, to="2026-09-23T08:05:00Z")
    assert body["covered_seconds"] == 300.0
    assert body["exceedance_seconds"] == 300.0
    assert body["excess_integral"] == 3000.0


def test_window_clips_segments_without_extrapolation(client):
    # Samples cover 08:05..08:15 only; the wider window must not extend the
    # trace beyond the first/last sample.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:05:00Z", 0.0),
            sample(2, "2026-09-23T08:15:00Z", 20.0),
        ],
    )

    body = exposure(
        client,
        to="2026-09-23T08:20:00Z",
        threshold=5.0,
    )
    # Clipped to [08:05, 08:15]: 0 -> 20 crosses 5 at 08:07:30.
    assert body["covered_seconds"] == 600.0
    assert body["exceedance_seconds"] == 450.0
    assert body["excess_integral"] == 3375.0


def test_flat_segment_at_threshold_never_exceeds(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 10.0),
            sample(2, "2026-09-23T08:10:00Z", 10.0),
        ],
    )

    body = exposure(client)
    assert body["covered_seconds"] == 600.0
    assert body["exceedance_seconds"] == 0.0
    assert body["excess_integral"] == 0.0


def test_flat_segment_above_threshold_fully_counts(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:10:00Z", 12.0),
        ],
    )

    body = exposure(client)
    assert body["covered_seconds"] == 600.0
    assert body["exceedance_seconds"] == 600.0
    assert body["excess_integral"] == 1200.0


def test_gaps_split_coverage_into_separate_segments(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 20.0),
            sample(2, "2026-09-23T08:05:00Z", 20.0),
            # 15 minute gap: no interpolation between 08:05 and 08:20.
            sample(3, "2026-09-23T08:20:00Z", 20.0),
            sample(4, "2026-09-23T08:25:00Z", 0.0),
        ],
    )

    body = exposure(client, to="2026-09-23T08:25:00Z")
    # [08:00, 08:05] flat at 20: 300 s above, excess 10 * 300.
    # [08:20, 08:25] 20 -> 0 crosses 10 at 08:22:30: 150 s above,
    # triangle (10 + 0) / 2 * 150.
    assert body["covered_seconds"] == 600.0
    assert body["exceedance_seconds"] == 450.0
    assert body["excess_integral"] == 3750.0


def test_fractional_seconds_are_not_truncated(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00.250000Z", 0.0),
            sample(2, "2026-09-23T08:00:00.750000Z", 1.0),
        ],
    )

    body = exposure(
        client,
        **{
            "from": "2026-09-23T08:00:00.250000Z",
            "to": "2026-09-23T08:00:00.750000Z",
            "threshold": 0.5,
        },
    )
    assert body["covered_seconds"] == 0.5
    assert body["exceedance_seconds"] == 0.25
    assert body["excess_integral"] == 0.0625


def test_unknown_device_returns_zeroes(client):
    body = exposure(client, device="sensor-unknown")
    assert body == {
        "device_id": "sensor-unknown",
        "from": "2026-09-23T08:00:00Z",
        "to": "2026-09-23T08:10:00Z",
        "threshold": 10.0,
        "covered_seconds": 0.0,
        "exceedance_seconds": 0.0,
        "excess_integral": 0.0,
    }


def test_single_sample_covers_nothing(client):
    post_records(client, [sample(1, "2026-09-23T08:05:00Z", 20.0)])

    body = exposure(client)
    assert body["covered_seconds"] == 0.0
    assert body["exceedance_seconds"] == 0.0
    assert body["excess_integral"] == 0.0


def test_late_write_is_reflected_in_the_next_query(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 20.0),
            sample(2, "2026-09-23T08:20:00Z", 20.0),
        ],
    )
    body = exposure(client, to="2026-09-23T08:20:00Z")
    assert body["covered_seconds"] == 0.0

    # The late sample fills the gap up to the limit on both sides.
    post_records(client, [sample(3, "2026-09-23T08:10:00Z", 20.0)])
    body = exposure(client, to="2026-09-23T08:20:00Z")
    assert body["covered_seconds"] == 1200.0
    assert body["exceedance_seconds"] == 1200.0
    assert body["excess_integral"] == 12000.0


def test_correction_is_reflected_in_the_next_query(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 0.0),
            sample(2, "2026-09-23T08:10:00Z", 0.0),
        ],
    )
    body = exposure(client)
    assert body["exceedance_seconds"] == 0.0
    assert body["excess_integral"] == 0.0

    response = client.post(
        "/records/correction",
        json={
            "device_id": "sensor-a",
            "sequence": 2,
            "old_sampled_at": "2026-09-23T08:10:00Z",
            "old_turbidity": 0.0,
            "new_sampled_at": "2026-09-23T08:10:00Z",
            "new_turbidity": 20.0,
        },
    )
    assert response.status_code == 200
    body = exposure(client)
    assert body["covered_seconds"] == 600.0
    assert body["exceedance_seconds"] == 300.0
    assert body["excess_integral"] == 1500.0


def test_query_does_not_modify_events_acks_or_change_stream(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:05:00Z", 12.0),
        ],
    )
    events_before = get_events(client, "sensor-a")
    assert ack_event(client, events_before["events"][0]).status_code == 200
    events_before = get_events(client, "sensor-a")
    changes_before = client.get(
        "/event-changes", params={"after": 0, "limit": 1000}
    ).json()

    body = exposure(client)
    assert body["exceedance_seconds"] == 300.0

    assert get_events(client, "sensor-a") == events_before
    assert (
        client.get("/event-changes", params={"after": 0, "limit": 1000}).json()
        == changes_before
    )


def test_window_echo_is_normalized_to_utc(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 20.0),
            sample(2, "2026-09-23T08:10:00Z", 20.0),
        ],
    )

    body = exposure(
        client,
        **{
            "from": "2026-09-23T16:00:00+08:00",
            "to": "2026-09-23T16:10:00+08:00",
        },
    )
    assert body["from"] == "2026-09-23T08:00:00Z"
    assert body["to"] == "2026-09-23T08:10:00Z"
    assert body["covered_seconds"] == 600.0


def test_invalid_window_returns_422(client):
    base = {
        "from": "2026-09-23T08:00:00Z",
        "to": "2026-09-23T08:10:00Z",
        "threshold": 10.0,
    }
    bad_windows = [
        ("2026-09-23T08:10:00Z", "2026-09-23T08:00:00Z"),  # from after to
        ("2026-09-23T08:00:00Z", "2026-09-23T08:00:00Z"),  # equal instants
        ("2026-09-23T10:00:00+02:00", "2026-09-23T08:00:00Z"),  # equal in UTC
        ("2026-09-23T08:00:00", "2026-09-23T08:10:00Z"),  # naive from
        ("2026-09-23T08:00:00Z", "2026-09-23T08:10:00"),  # naive to
    ]
    for window_from, window_to in bad_windows:
        response = client.post(
            "/devices/sensor-a/turbidity-exposure",
            json={**base, "from": window_from, "to": window_to},
        )
        assert response.status_code == 422, (window_from, window_to)


def test_invalid_threshold_returns_422(client):
    base = {
        "from": "2026-09-23T08:00:00Z",
        "to": "2026-09-23T08:10:00Z",
    }
    for threshold in (-1.0, -0.1, "high"):
        response = client.post(
            "/devices/sensor-a/turbidity-exposure",
            json={**base, "threshold": threshold},
        )
        assert response.status_code == 422, threshold

    # Infinity and NaN are non-standard JSON tokens, so send them raw; the
    # server parser accepts the tokens and the non-finite validator rejects
    # them with 422.
    for token in ("Infinity", "-Infinity", "NaN"):
        response = client.post(
            "/devices/sensor-a/turbidity-exposure",
            content=(
                '{"from":"2026-09-23T08:00:00Z",'
                '"to":"2026-09-23T08:10:00Z",'
                f'"threshold":{token}}}'
            ),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, token


def test_missing_and_extra_fields_return_422(client):
    base = {
        "from": "2026-09-23T08:00:00Z",
        "to": "2026-09-23T08:10:00Z",
        "threshold": 10.0,
    }
    for key in ("from", "to", "threshold"):
        payload = {name: value for name, value in base.items() if name != key}
        response = client.post(
            "/devices/sensor-a/turbidity-exposure", json=payload
        )
        assert response.status_code == 422, key
    response = client.post(
        "/devices/sensor-a/turbidity-exposure", json={**base, "extra": 1}
    )
    assert response.status_code == 422
