"""
Threat Tracker Module for 5G Assistive Wearable
================================================
Follows each obstacle across consecutive frames using bounding box IoU matching,
calculates closing velocity (m/s), estimates Time-To-Collision (TTC),
and manages per-object alert cooldowns.
"""

import time
import math

RANK = {"CLEAR": 0, "ADVISORY": 1, "WARNING": 2, "CRITICAL": 3}

def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

class ThreatTracker:
    """Follows each obstacle across frames and works out how fast it is getting closer."""
    def __init__(self, keep_s=1.5, drop_s=2.0):
        self.tracks = {}
        self.next_id = 0
        self.keep_s = keep_s
        self.drop_s = drop_s

    def update(self, threats, now=None, heading_deg=None):
        now = now or time.time()
        for t in threats:
            best, best_iou = None, 0.3
            for tid, tr in self.tracks.items():
                if tr['name'] == t['name']:
                    iou = _iou(tr['bbox'], t['bbox'])
                    if iou > best_iou:
                        best, best_iou = tid, iou
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {'name': t['name'], 'hist': [], 'last_alert': 0.0, 'last_alert_dist': 99.0}
            tr = self.tracks[best]
            tr['bbox'], tr['seen'] = t['bbox'], now
            tr['hits'] = tr.get('hits', 0) + 1
            t['hits'] = tr['hits']
            t['track_id'] = best

            x, z = t.get('x', 0.0), t['distance']
            if heading_deg is not None:                                     # D6: undo the user's turning
                tr.setdefault('ref_heading', heading_deg)
                dpsi = math.radians((heading_deg - tr['ref_heading'] + 540) % 360 - 180)
                x, z = x * math.cos(dpsi) - z * math.sin(dpsi), x * math.sin(dpsi) + z * math.cos(dpsi)
            t['x_ref'], t['z_ref'] = x, z

            # Fix 2: Only add fully visible positions to movement history to prevent edge distortion
            if not t.get('edge'):
                tr['hist'] = [s for s in tr['hist'] if now - s[0] <= self.keep_s] + [(now, x, z, t['distance'])]
            else:
                tr['hist'] = [s for s in tr['hist'] if now - s[0] <= self.keep_s]

            t['vx'], t['vz'] = self._velocity(tr['hist'])
            t['closing_mps'], t['ttc_s'] = self._ttc([(s[0], s[3]) for s in tr['hist']])   # old mode keeps working
        self.tracks = {k: v for k, v in self.tracks.items() if now - v.get('seen', 0) <= self.drop_s}
        return threats

    @staticmethod
    def _velocity(hist):
        # Fix 1: Require at least 6 samples over 0.8 s before estimating velocity
        if len(hist) < 6 or hist[-1][0] - hist[0][0] < 0.8:
            return 0.0, 0.0
        n = len(hist)
        mt = sum(s[0] for s in hist) / n
        den = sum((s[0] - mt) ** 2 for s in hist) or 1e-6
        mx = sum(s[1] for s in hist) / n
        mz = sum(s[2] for s in hist) / n
        return (sum((s[0] - mt) * (s[1] - mx) for s in hist) / den,
                sum((s[0] - mt) * (s[2] - mz) for s in hist) / den)

    def confirm_level(self, tid, level, now):
        """Rise needs 2 consecutive frames; fall needs 1 s. Returns (confirmed_level, rose_now)."""
        tr = self.tracks.get(tid)
        if not tr:
            return level, False
        cur = tr.get('level', 'CLEAR')
        if RANK[level] > RANK[cur]:
            tr['rise_count'] = tr.get('rise_count', 0) + 1 if tr.get('rise_to') == level else 1
            tr['rise_to'], tr['fall_since'] = level, None
            if tr['rise_count'] >= 2:
                tr['level'], tr['rise_count'] = level, 0
                return level, True
        elif RANK[level] < RANK[cur]:
            tr['rise_count'] = 0
            tr['fall_since'] = tr.get('fall_since') or now
            if now - tr['fall_since'] >= 1.0:
                tr['level'], tr['fall_since'] = level, None
                tr['spoken_rank'] = min(tr.get('spoken_rank', 0), RANK[level])
        else:
            tr['rise_count'], tr['fall_since'] = 0, None
        return tr.get('level', 'CLEAR'), False

    def needs_voice(self, tid, level, now, repeat_s=4.0):
        """True if this object's level hasn't been spoken yet, or it's still CRITICAL after repeat_s."""
        tr = self.tracks.get(tid)
        if not tr or RANK[level] < 2:
            return False
        if RANK[level] > tr.get('spoken_rank', 0):
            return True
        return RANK[level] == 3 and now - tr.get('spoken_at', 0) >= repeat_s

    def mark_spoken(self, tid, level, now):
        tr = self.tracks.get(tid)
        if tr:
            tr['spoken_rank'], tr['spoken_at'] = RANK[level], now

    @staticmethod
    def _ttc(hist):
        if len(hist) < 4 or hist[-1][0] - hist[0][0] < 0.5:
            return 0.0, float('inf')
        n = len(hist)
        mt = sum(t for t, _ in hist) / n
        md = sum(d for _, d in hist) / n
        den = sum((t - mt) ** 2 for t, _ in hist)
        speed = -sum((t - mt) * (d - md) for t, d in hist) / den if den > 0 else 0.0   # + means closing
        d_now = hist[-1][1]
        return speed, (d_now / speed if speed > 0.2 else float('inf'))

    def should_alert(self, t, now, cooldown_s=3.0):
        """Speak again for the same object only if it is clearly worse or the cooldown has passed."""
        tr = self.tracks.get(t.get('track_id'))
        if not tr:
            return True
        worse = t['distance'] < tr['last_alert_dist'] * 0.7
        if worse or now - tr['last_alert'] > cooldown_s:
            tr['last_alert'], tr['last_alert_dist'] = now, t['distance']
            return True
        return False
