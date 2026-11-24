import dashboard


def test_latency_stats_summary():
    s = dashboard.LatencyStats()
    assert s.summary()["n"] == 0 and s.summary()["avg"] is None
    for v in (10, 20, 30, 40, 100):
        s.add(v)
    r = s.summary()
    assert (r["n"], r["last"], r["avg"], r["min"], r["max"]) == (5, 100.0, 40.0, 10.0, 100.0)
    assert r["p95"] == 40.0                      # index int(0.95*4)=3 of the sorted list


def test_state_snapshot_has_latency():
    st = dashboard.DashboardState()
    st.add_latency("rtt", 120.4)
    st.add_latency("oneway", 35.0)
    snap = st.snapshot()["latency"]
    assert snap["rtt"]["last"] == 120.4 and snap["oneway"]["n"] == 1
    assert snap["rtt_recent"] == [120.4]
