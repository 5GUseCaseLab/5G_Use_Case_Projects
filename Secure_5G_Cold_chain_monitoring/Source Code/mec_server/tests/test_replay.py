import time
import pytest
import security
from security import VerificationError


def pkt(seq, ts=None, temp=4.0, dev="uno_q_edge_01"):
    return {"device_id": dev, "seq": seq, "timestamp": int(ts or time.time()), "temperature": temp}


@pytest.fixture
def guard(tmp_path):
    return security.TelemetryGuard(str(tmp_path / "seq.json"))


def test_increasing_seq_ok(guard):
    guard.check_and_accept(pkt(1))
    guard.check_and_accept(pkt(2))


def test_duplicate_seq_rejected(guard):
    guard.check_and_accept(pkt(5))
    with pytest.raises(VerificationError, match="replayed"):
        guard.check_and_accept(pkt(5))


def test_older_seq_rejected(guard):
    guard.check_and_accept(pkt(10))
    with pytest.raises(VerificationError, match="replayed"):
        guard.check_and_accept(pkt(9))


def test_seq_survives_server_restart(tmp_path):
    path = str(tmp_path / "seq.json")
    security.TelemetryGuard(path).check_and_accept(pkt(7))
    with pytest.raises(VerificationError):
        security.TelemetryGuard(path).check_and_accept(pkt(7))     # "restarted" server


def test_stale_timestamp_rejected(guard):
    with pytest.raises(VerificationError, match="stale"):
        guard.check_and_accept(pkt(1, ts=time.time() - 3600))


def test_future_timestamp_rejected(guard):
    with pytest.raises(VerificationError, match="stale"):
        guard.check_and_accept(pkt(1, ts=time.time() + 3600))


def test_unknown_device_rejected(guard):
    with pytest.raises(VerificationError, match="unknown device"):
        guard.check_and_accept(pkt(1, dev="evil"))


@pytest.mark.parametrize("temp", [float("nan"), float("inf"), 999, -999, "4", True, None])
def test_bad_temperature_rejected(guard, temp):
    with pytest.raises(VerificationError):
        guard.check_and_accept(pkt(1, temp=temp))


@pytest.mark.parametrize("seq", [-1, "1", 1.5, True, None])
def test_bad_seq_rejected(guard, seq):
    with pytest.raises(VerificationError):
        guard.check_and_accept(pkt(seq))
