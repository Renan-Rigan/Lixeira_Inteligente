"""Testes sem hardware e sem modelo: python -m unittest discover tests -v"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lixeira.actuators import MockServo, Sorter  # noqa: E402
from lixeira.classifier import FakeClassifier, decide  # noqa: E402
from lixeira.config import load_config  # noqa: E402
from lixeira.dataset import materialize, scan_source, split_items  # noqa: E402
from lixeira.event_log import EventLogger  # noqa: E402
from lixeira.metrics import summarize, wilson_interval  # noqa: E402
from lixeira.pipeline import SortingPipeline, State  # noqa: E402
from lixeira.presence import PresenceDetector  # noqa: E402

RNG = np.random.default_rng(0)
H, W = 240, 320


def empty():
    base = np.full((H, W, 3), 110, np.uint8)
    noise = RNG.integers(-4, 5, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def with_obj(color, x=120, y=80, w=80, h=70):
    f = empty()
    cv2.rectangle(f, (x, y), (x + w, y + h), color, -1)
    return f


BLUE, WHITE, RED = (220, 90, 40), (245, 245, 245), (40, 60, 220)  # BGR


def make(cfg_overrides=None):
    cfg = load_config(ROOT / "config.yaml")
    cfg["classifier"]["conf_threshold"] = 0.5
    for k, v in (cfg_overrides or {}).items():
        cfg[k].update(v)
    det = PresenceDetector(**cfg["presence"])
    det.learn_background([empty() for _ in range(10)])
    rot, hatch = MockServo("base", False), MockServo("escotilha", False)
    sorter = Sorter(rot, hatch, cfg, sleep=lambda s: None)
    pipe = SortingPipeline(det, FakeClassifier(cfg["classes"]), sorter, cfg)
    return cfg, pipe, rot, hatch


def run(pipe, frames):
    return [e for e in (pipe.step(f) for f in frames) if e]


class TestPresence(unittest.TestCase):
    def test_empty_not_present(self):
        _, pipe, *_ = make()
        for _ in range(20):
            r = pipe.det.update(empty())
        self.assertFalse(r.present)

    def test_object_present_and_still(self):
        _, pipe, *_ = make()
        pipe.det.update(with_obj(RED))
        r = pipe.det.update(with_obj(RED))
        self.assertTrue(r.present)
        self.assertTrue(r.still)
        x, y, w, h = r.bbox
        self.assertAlmostEqual(x, 120, delta=4)
        self.assertAlmostEqual(w, 81, delta=6)

    def test_moving_object_not_still(self):
        _, pipe, *_ = make()
        pipe.det.update(with_obj(RED, x=40))
        r = pipe.det.update(with_obj(RED, x=90))
        self.assertTrue(r.present)
        self.assertFalse(r.still)

    def test_slow_lighting_drift_adapts(self):
        _, pipe, *_ = make()
        for i in range(200):  # luz subindo devagar, plataforma vazia
            r = pipe.det.update(np.clip(empty().astype(int) + i // 10, 0, 255).astype(np.uint8))
        self.assertFalse(r.present)


class TestPipeline(unittest.TestCase):
    def test_full_cycle_each_class(self):
        for color, expected in [(BLUE, "metal"), (WHITE, "papel"), (RED, "plastico")]:
            cfg, pipe, rot, hatch = make()
            frames = [empty()] * 5 + [with_obj(color)] * 20 + [empty()] * 15
            events = run(pipe, frames)
            self.assertEqual(len(events), 1, expected)
            self.assertEqual(events[0].prediction.label, expected)
            self.assertIn(cfg["actuators"]["bin_angles"][expected], rot.history)
            self.assertEqual(pipe.state, State.VAZIO)

    def test_hand_moving_does_not_trigger_until_still(self):
        cfg, pipe, *_ = make()
        hand = [with_obj(RED, x=20 + 15 * i) for i in range(10)]  # passa se mexendo
        events = run(pipe, hand)
        self.assertEqual(events, [])
        events = run(pipe, [with_obj(RED, x=165)] * 20)
        self.assertEqual(len(events), 1)

    def test_object_removed_before_classification(self):
        _, pipe, *_ = make()
        events = run(pipe, [with_obj(RED)] * 3 + [empty()] * 10)
        self.assertEqual(events, [])
        self.assertEqual(pipe.state, State.VAZIO)

    def test_stuck_object_not_disposed_twice(self):
        _, pipe, *_ = make()
        events = run(pipe, [with_obj(RED)] * 80)  # escotilha falhou, objeto ficou
        self.assertEqual(len(events), 1)
        self.assertEqual(pipe.state, State.AGUARDANDO_LIBERAR)

    def test_low_confidence_goes_to_reject(self):
        cfg, pipe, rot, _ = make()
        pipe.conf_threshold = 0.99
        gray = (80, 170, 80)  # esverdeado: nenhuma classe fake se destaca
        events = run(pipe, [with_obj(gray)] * 20)
        self.assertEqual(events[0].prediction.label, "indefinido")
        self.assertIn(cfg["actuators"]["bin_angles"]["indefinido"], rot.history)

    def test_event_logging(self):
        _, pipe, *_ = make()
        ev = run(pipe, [with_obj(WHITE)] * 20)[0]
        with tempfile.TemporaryDirectory() as d:
            log = EventLogger(f"{d}/logs", True, f"{d}/inbox")
            row = log.log(ev, "papel")
            self.assertTrue(Path(row["image"]).exists())
            self.assertIn("papel", row["image"])
            self.assertEqual(len(log.csv_path.read_text().splitlines()), 2)


class TestSorter(unittest.TestCase):
    def test_hatch_closed_whenever_base_moves(self):
        cfg = load_config(ROOT / "config.yaml")
        timeline = []

        class Rec(MockServo):
            def set_angle(self, a):
                super().set_angle(a)
                timeline.append((self.name, a))

        rot, hatch = Rec("base", False), Rec("escotilha", False)
        s = Sorter(rot, hatch, cfg, sleep=lambda _: None)
        s.home()
        for lab in cfg["actuators"]["bin_angles"]:
            s.dispose(lab)
        closed, opened = cfg["actuators"]["hatch"]["closed_angle"], cfg["actuators"]["hatch"]["open_angle"]
        hatch_angle = None
        for name, a in timeline:
            if name == "escotilha":
                hatch_angle = a
            else:
                self.assertEqual(hatch_angle, closed, "base girou com escotilha aberta")
        self.assertIn(opened, hatch.history)
        self.assertEqual(hatch.angle, closed)
        self.assertEqual(rot.angle, cfg["actuators"]["home_angle"])

    def test_refuses_rotation_with_open_hatch(self):
        cfg = load_config(ROOT / "config.yaml")
        s = Sorter(MockServo("b", False), MockServo("h", False), cfg, sleep=lambda _: None)
        s.hatch_open = True
        with self.assertRaises(RuntimeError):
            s._rotate(30)


class TestClassifierVoting(unittest.TestCase):
    def test_votes_average(self):
        clf = FakeClassifier()
        p = decide(clf, [with_obj(RED), with_obj(RED), with_obj(BLUE)], 0.0, "indefinido")
        self.assertEqual(p.raw_label, "plastico")
        self.assertAlmostEqual(sum(p.probs.values()), 1.0, places=5)


class TestConfig(unittest.TestCase):
    def test_pi_config_extends_base(self):
        cfg = load_config(ROOT / "config.pi.yaml")
        self.assertEqual(cfg["camera"]["backend"], "picamera2")
        self.assertEqual(cfg["actuators"]["backend"], "gpiozero")
        self.assertEqual(cfg["presence"]["stable_frames"], 8)  # herdado

    def test_missing_bin_angle_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text("classes: [metal, vidro]\n")
            with self.assertRaises(ValueError):
                load_config(p)


class TestDataset(unittest.TestCase):
    def test_split_by_object_no_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            raw, pub = Path(d) / "raw", Path(d) / "pub"
            for cls in ("metal", "papel", "plastico"):
                (raw / cls).mkdir(parents=True)
                for obj in range(10):
                    for shot in range(5):
                        cv2.imwrite(str(raw / cls / f"obj{obj:04d}__{shot:03d}.jpg"), empty())
            for cls in ("plastic", "paper", "trash"):
                (pub / cls).mkdir(parents=True)
                for i in range(4):
                    cv2.imwrite(str(pub / cls / f"{cls}{i}.jpg"), empty())

            items = scan_source(raw, None, False, "s0")
            items += scan_source(pub, {"plastic": "plastico", "paper": "papel"}, True, "pub0")
            splits = split_items(items, 0.2, 0.2, seed=1)

            groups = {k: {it.group for it in v} for k, v in splits.items()}
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["val"] & groups["test"])
            self.assertTrue(all(not it.group.startswith("pub") for it in splits["test"]))
            self.assertFalse(any(it.label == "trash" for v in splits.values() for it in v))
            counts = materialize(splits, Path(d) / "ds")
            self.assertEqual(counts["test"]["metal"], 10)   # 2 objetos x 5 fotos
            self.assertEqual(counts["train"]["papel"], 30 + 4)


class TestMetrics(unittest.TestCase):
    def test_summary(self):
        s = summarize(["a", "a", "b", "b"], ["a", "x", "b", "a"], ["a", "b"], "x")
        self.assertAlmostEqual(s["accuracy"], 0.5)
        self.assertAlmostEqual(s["reject_rate"], 0.25)
        self.assertAlmostEqual(s["accuracy_accepted"], 2 / 3)
        self.assertAlmostEqual(s["per_class"]["a"]["precision"], 0.5)

    def test_wilson(self):
        lo, hi = wilson_interval(45, 50)
        self.assertTrue(lo < 0.9 < hi)


if __name__ == "__main__":
    unittest.main()
