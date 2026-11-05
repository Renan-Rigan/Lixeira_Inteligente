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
from lixeira.camera import NetworkCamera, open_camera  # noqa: E402
from lixeira.classifier import FakeClassifier, ProbSmoother, decide, from_probs  # noqa: E402
from lixeira.config import load_config  # noqa: E402
from lixeira.dataset import (cap_per_class, materialize, resolve_label, scan_source,  # noqa: E402
                             split_items)
from lixeira.quality import ERRO, OK, QualityParams, check_array, check_image, find_duplicates  # noqa: E402
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


BLUE, GREEN, RED = (220, 90, 40), (60, 200, 60), (40, 60, 220)  # BGR


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
        for color, expected in [(BLUE, "metal"), (RED, "plastico"), (GREEN, "resto")]:
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

    def test_low_confidence_goes_to_resto(self):
        cfg, pipe, rot, _ = make()
        pipe.conf_threshold = 0.99
        purple = (160, 90, 140)  # azul e vermelho parecidos: nenhuma classe se destaca
        events = run(pipe, [with_obj(purple)] * 20)
        p = events[0].prediction
        self.assertEqual(p.label, "resto")
        self.assertNotEqual(p.raw_label, "resto")
        self.assertIn(cfg["actuators"]["bin_angles"]["resto"], rot.history)

    def test_event_logging(self):
        _, pipe, *_ = make()
        ev = run(pipe, [with_obj(GREEN)] * 20)[0]
        with tempfile.TemporaryDirectory() as d:
            log = EventLogger(f"{d}/logs", True, f"{d}/inbox")
            row = log.log(ev, "resto")
            self.assertTrue(Path(row["image"]).exists())
            self.assertIn("resto", row["image"])
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

    def test_from_probs_threshold(self):
        names = ["metal", "plastico", "resto"]
        self.assertEqual(from_probs(np.array([0.8, 0.1, 0.1]), names, 0.7, "resto").label, "metal")
        p = from_probs(np.array([0.5, 0.4, 0.1]), names, 0.7, "resto")
        self.assertEqual((p.label, p.raw_label), ("resto", "metal"))

    def test_smoother_resists_single_bad_frame(self):
        sm = ProbSmoother(3, alpha=0.3)
        for _ in range(5):
            sm.update([0.9, 0.05, 0.05])
        out = sm.update([0.0, 1.0, 0.0])  # um frame com reflexo
        self.assertEqual(int(np.argmax(out)), 0)


class TestNetworkCamera(unittest.TestCase):
    """O celular é simulado por um arquivo de vídeo: o cv2.VideoCapture abre
    URL e arquivo do mesmo jeito."""

    def _video(self, d):
        path = str(Path(d) / "celular.avi")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 15, (W, H))
        for i in range(30):
            out.write(with_obj(RED, x=10 + 5 * i))
        out.release()
        return path

    def test_reads_resized_and_cropped_frames(self):
        with tempfile.TemporaryDirectory() as d:
            cam = NetworkCamera(self._video(d), 640, 480, warmup_frames=2, roi=[0, 0, 100, 50])
            try:
                self.assertEqual(cam.read().shape, (50, 100, 3))
                self.assertEqual(cam.read().shape, (50, 100, 3))  # frame novo, sem travar
            finally:
                cam.close()

    def test_open_camera_uses_url(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(ROOT / "config.celular.yaml")
            cfg["camera"].update(url=self._video(d), warmup_frames=0)
            cam = open_camera(cfg)
            try:
                self.assertIsInstance(cam, NetworkCamera)
                self.assertEqual(cam.read().shape, (480, 640, 3))
            finally:
                cam.close()


class TestConfig(unittest.TestCase):
    def test_pi_config_extends_base(self):
        cfg = load_config(ROOT / "config.pi.yaml")
        self.assertEqual(cfg["camera"]["backend"], "picamera2")
        self.assertEqual(cfg["actuators"]["backend"], "gpiozero")
        self.assertEqual(cfg["presence"]["stable_frames"], 8)  # herdado

    def test_classes_and_reject(self):
        cfg = load_config(ROOT / "config.yaml")
        self.assertEqual(cfg["classes"], ["metal", "plastico", "resto"])
        self.assertEqual(cfg["reject_label"], "resto")
        self.assertEqual(set(cfg["actuators"]["bin_angles"]), {"metal", "plastico", "resto"})

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
            for cls in ("metal", "plastico", "resto"):
                (raw / cls).mkdir(parents=True)
                for obj in range(10):
                    for shot in range(5):
                        cv2.imwrite(str(raw / cls / f"obj{obj:04d}__{shot:03d}.jpg"), empty())
            for cls in ("plastic", "food", "glass"):
                (pub / cls).mkdir(parents=True)
                for i in range(4):
                    cv2.imwrite(str(pub / cls / f"{cls}{i}.jpg"), empty())

            items = scan_source(raw, None, False, "s0")
            items += scan_source(pub, {"plastic": "plastico", "food": "resto"}, True, "pub0")
            splits = split_items(items, 0.2, 0.2, seed=1)

            groups = {k: {it.group for it in v} for k, v in splits.items()}
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["val"] & groups["test"])
            self.assertTrue(all(not it.group.startswith("pub") for it in splits["test"]))
            self.assertFalse(any(it.label == "glass" for v in splits.values() for it in v))
            counts = materialize(splits, Path(d) / "ds")
            self.assertEqual(counts["test"]["metal"], 10)   # 2 objetos x 5 fotos
            self.assertEqual(counts["train"]["resto"], 30 + 4)

    def test_resolve_label_bank_mapping(self):
        classes = ["metal", "plastico", "resto"]
        m = {"PET": "plastico", "glass": "", "*": "resto"}
        self.assertEqual(resolve_label("PET", m, classes), "plastico")
        self.assertEqual(resolve_label("pet", m, classes), "plastico")
        self.assertEqual(resolve_label("Metal", m, classes), "metal")      # já é classe
        self.assertEqual(resolve_label("Food Organics", m, classes), "resto")  # curinga
        self.assertIsNone(resolve_label("Glass", m, classes))              # descartada
        self.assertIsNone(resolve_label("cardboard", {"PET": "plastico"}, classes))
        self.assertEqual(resolve_label("qualquer", None, None), "qualquer")

    def test_cap_per_class_keeps_own_photos_and_whole_objects(self):
        with tempfile.TemporaryDirectory() as d:
            own, pub = Path(d) / "own" / "resto", Path(d) / "pub" / "resto"
            own.mkdir(parents=True)
            pub.mkdir(parents=True)
            for i in range(3):
                cv2.imwrite(str(own / f"obj{i:04d}__000.jpg"), empty())
            for i in range(50):
                cv2.imwrite(str(pub / f"img{i}.jpg"), empty())
            items = scan_source(own.parent, None, False, "s0") + scan_source(pub.parent, None, True, "pub0")
            capped = cap_per_class(items, 10)
            self.assertEqual(len(capped), 10)
            self.assertEqual(sum(not it.train_only for it in capped), 3)

    def test_materialize_max_side(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src" / "metal"
            src.mkdir(parents=True)
            cv2.imwrite(str(src / "big.png"), np.zeros((1500, 2000, 3), np.uint8))
            items = scan_source(src.parent, None, True, "pub0")
            materialize({"train": items}, Path(d) / "ds", max_side=640)
            out = list((Path(d) / "ds" / "train" / "metal").iterdir())
            self.assertEqual(cv2.imread(str(out[0])).shape[:2], (480, 640))


def scene(seed, h=960, w=1280):
    """Imagem com formas e textura (nítida), diferente para cada seed."""
    r = np.random.default_rng(seed)
    img = np.full((h, w, 3), 180, np.uint8)
    for _ in range(8):
        cx, cy = int(r.integers(0, w)), int(r.integers(0, h))
        cv2.circle(img, (cx, cy), int(r.integers(h // 20, h // 5)),
                   tuple(int(v) for v in r.integers(0, 255, 3)), -1)
    return img


class TestQuality(unittest.TestCase):
    P = QualityParams(min_side=224, ref_size=(640, 480))

    def test_good_image(self):
        r = check_array(scene(1), self.P)
        self.assertEqual(r.status, OK, r.problems)

    def test_too_small_is_error(self):
        r = check_array(cv2.resize(scene(1), (200, 150)), self.P)
        self.assertEqual(r.status, ERRO)
        self.assertTrue(any("pequena" in p for p in r.problems))

    def test_below_pi_resolution_is_warning(self):
        r = check_array(cv2.resize(scene(1), (400, 300)), self.P)
        self.assertEqual(r.status, "aviso")

    def test_blurry_and_dark(self):
        blurry = cv2.GaussianBlur(scene(1), (61, 61), 0)
        self.assertTrue(any("borrada" in p for p in check_array(blurry, self.P).problems))
        dark = (scene(1) * 0.1).astype(np.uint8)
        self.assertTrue(any("escura" in p for p in check_array(dark, self.P).problems))

    def test_sharpness_measured_at_pi_scale(self):
        # a mesma cena em 4000 px e em 640 px dá nitidez parecida
        big = cv2.resize(scene(2), (4000, 3000), interpolation=cv2.INTER_CUBIC)
        small = cv2.resize(big, (640, 480), interpolation=cv2.INTER_AREA)
        a, b = check_array(big, self.P).sharpness, check_array(small, self.P).sharpness
        self.assertAlmostEqual(a / b, 1.0, delta=0.15)

    def test_unreadable_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.jpg"
            f.write_bytes(b"nao e imagem")
            self.assertEqual(check_image(f, self.P).status, ERRO)

    def test_duplicates_across_classes(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = Path(d) / "a.jpg", Path(d) / "b.jpg", Path(d) / "c.jpg"
            cv2.imwrite(str(a), scene(3))
            cv2.imwrite(str(b), cv2.resize(scene(3), (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 60])
            cv2.imwrite(str(c), scene(4))
            groups = find_duplicates([(a, "metal"), (b, "resto"), (c, "metal")])
            self.assertEqual(len(groups), 1)
            self.assertEqual({p for p, _ in groups[0]}, {a, b})


class TestMetrics(unittest.TestCase):
    def test_summary(self):
        s = summarize(["a", "a", "b", "b"], ["a", "x", "b", "a"], ["a", "b"], "x")
        self.assertAlmostEqual(s["accuracy"], 0.5)
        self.assertAlmostEqual(s["reject_rate"], 0.25)
        self.assertAlmostEqual(s["accuracy_accepted"], 2 / 3)
        self.assertAlmostEqual(s["per_class"]["a"]["precision"], 0.5)

    def test_reject_label_is_also_a_class(self):
        # "resto" é classe e destino da rejeição: só conta como rejeitado o que
        # caiu abaixo do limiar, não o "resto" previsto com confiança
        y_true = ["resto", "metal", "resto", "plastico"]
        y_pred = ["resto", "resto", "resto", "plastico"]
        rejected = [False, True, True, False]
        s = summarize(y_true, y_pred, ["metal", "plastico", "resto"], "resto", rejected)
        self.assertAlmostEqual(s["accuracy"], 0.75)
        self.assertAlmostEqual(s["reject_rate"], 0.5)
        self.assertAlmostEqual(s["accuracy_accepted"], 1.0)
        self.assertEqual(len(s["labels"]), 3)

    def test_wilson(self):
        lo, hi = wilson_interval(45, 50)
        self.assertTrue(lo < 0.9 < hi)


if __name__ == "__main__":
    unittest.main()
