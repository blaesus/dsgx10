"""Synthetic CPU tests of the public worker; never load checkpoint weights or call CUDA.

Run: python -m unittest discover -s vision/tests -v
Images are generated in memory. Socket/HTTP requests and GPU qualification are not
part of this suite. Cache controls run the actual Encoder.encode method with an
explicit CPU fake for the vision tower, so they do not claim visual correctness.
"""
import base64
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import encoder
import reference_image_processor as processor
import torch
from PIL import Image


def png(color=(0, 255, 0), size=(64, 64)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def record(raw):
    return {"data": base64.b64encode(raw).decode("ascii")}


class EncoderCPU(unittest.TestCase):
    def setUp(self):
        self.no_network = mock.patch.object(encoder.urllib.request, "urlopen",
                                             side_effect=AssertionError("CPU test attempted network"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def test_raw_identity_preserved_for_supported_embedded_forms(self):
        raw = png()
        encoded = base64.b64encode(raw).decode("ascii")
        forms = [{"data": encoded}, {"source": {"data": encoded}},
                 {"url": "data:image/png;base64," + encoded}]
        for form in forms:
            self.assertEqual(encoder.image_bytes(form), raw)

    def test_local_paths_and_nonimage_data_urls_rejected(self):
        for form in [{"url":"file:///synthetic.png"}, {"path":"synthetic.png"},
                     {"url":"data:text/plain;base64,AA=="}, {}, []]:
            with self.subTest(form=form), self.assertRaises(ValueError):
                encoder.image_bytes(form)

    def test_bad_base64_and_corrupt_image_rejected(self):
        for form in [{"data":"!bad!"}, record(b"not a PNG"), {"data":""}]:
            with self.subTest(form=form), self.assertRaises((ValueError, OSError)):
                encoder.image_bytes(form)

    def test_encoded_byte_budget(self):
        raw = png()
        with mock.patch.object(encoder, "MAX_IMAGE_BYTES", len(raw)):
            self.assertEqual(encoder.image_bytes(record(raw)), raw)
        with mock.patch.object(encoder, "MAX_IMAGE_BYTES", len(raw)-1):
            with self.assertRaises(ValueError):
                encoder.image_bytes(record(raw))

    def test_pixel_budget_without_allocating_large_image(self):
        with mock.patch.object(encoder, "MAX_PIXELS", 63 * 64):
            with self.assertRaises(ValueError):
                encoder.image_bytes(record(png()))

    def test_animation_rejected(self):
        stream = io.BytesIO()
        Image.new("RGB", (8,8), "red").save(stream, format="GIF", save_all=True,
            append_images=[Image.new("RGB", (8,8), "blue")], duration=100, loop=0)
        with self.assertRaises(ValueError):
            encoder.image_bytes(record(stream.getvalue()))

    def test_actual_cpu_preprocessing_and_span_layout(self):
        cfg = json.loads((ROOT.parent / "ref/inference/config.json").read_text())
        args = SimpleNamespace(**cfg)
        patches, nh, nw, lh, lw = processor.load_image(record(png()), args)
        self.assertEqual(patches.device.type, "cpu")
        self.assertEqual(patches.dtype, torch.bfloat16)
        self.assertEqual(tuple(patches.shape), (nh*nw, 3, args.vision_patch_size, args.vision_patch_size))
        types = processor.image_token_types(lh, lw).tolist()
        self.assertEqual(types, [processor.IMAGE_START] +
            ([processor.IMAGE]*lw + [processor.IMAGE_NEW_LINE])*lh + [processor.IMAGE_END])
        self.assertLessEqual(len(types), 1024)
        self.assertTrue(torch.isfinite(patches).all().item())

    def test_large_history_geometry_exceeds_old_whole_request_cap(self):
        args = SimpleNamespace(**json.loads((ROOT.parent / "ref/inference/config.json").read_text()))
        geometry = [(1944,1215)]*49 + [(780,1688)]*5 + [(1200,750)]*4
        rows = []
        for width,height in geometry:
            lh,lw,_,_ = processor.plan_image_grid(width,height,args)
            rows.append(processor.num_image_tokens(lh,lw))
        self.assertEqual(len(rows),58)
        self.assertEqual(sum(rows),53416)
        self.assertEqual(sum(rows)*5120*2,546979840)
        self.assertGreater(sum(rows)*5120*2,512<<20)
        self.assertTrue(all(0<n<=1024 for n in rows))

    def test_encode_many_order_exact_limit_and_overflow(self):
        class Payload:
            def __init__(self, size): self.size=size
            def __len__(self): return self.size
        class Fake:
            def __init__(self): self.seen=[]
            def encode(self, value): self.seen.append(value); return ({"id":value}, Payload(value))
        f=Fake()
        self.assertEqual(len(encoder.encode_many(f,[8]*64)),64)
        self.assertEqual(f.seen,[8]*64)
        self.assertEqual(len(encoder.encode_many(f,[encoder.MAX_FEATURE_BYTES])),1)
        with self.assertRaises(encoder.ImageFeatureBudgetError):
            encoder.encode_many(f,[encoder.MAX_FEATURE_BYTES,1])
        for invalid in ([],None,{}):
            with self.assertRaises(ValueError): encoder.encode_many(f,invalid)

    def test_byte_lru_reuse_append_replacement_and_immutable_payload(self):
        # Bypass heavyweight __init__; keep the real encode/cache code and CPU BF16 tensors.
        e=encoder.Encoder.__new__(encoder.Encoder)
        e.identity="11"*32; e.cache=OrderedDict(); e.cache_limit=3*5120*2
        e.cache_bytes=0; e.cache_hits=e.cache_misses=e.cache_evictions=0
        e.args=None; e.device="cpu"
        calls=[]
        def model(*_):
            calls.append(1)
            return torch.zeros((1,5120),dtype=torch.bfloat16)
        e.model=model
        images=[png((i*50,255-i*50,0)) for i in range(3)]
        with mock.patch.object(encoder,"load_image",return_value=(torch.zeros(1),1,1,1,1)):
            first=[e.encode(record(raw)) for raw in images]
            self.assertEqual(len(calls),3)
            for raw,(_,expected) in zip(images,first):
                meta,actual=e.encode(record(raw))
                self.assertTrue(meta["cache_hit"])
                self.assertEqual(actual,expected)
                self.assertEqual(meta["embedding_sha256"],hashlib.sha256(actual).hexdigest())
            self.assertEqual(len(calls),3)
            e.encode(record(png((200,0,0))))
            self.assertEqual(len(calls),4)
            self.assertEqual(e.cache_evictions,1)
            self.assertEqual(e.cache_bytes,e.cache_limit)
            e.encode(record(images[0]))
            self.assertEqual(len(calls),5)
            self.assertEqual(len(e.cache),3)

    def test_nonfinite_features_never_enter_cache(self):
        e=encoder.Encoder.__new__(encoder.Encoder)
        e.identity="22"*32; e.cache=OrderedDict(); e.cache_limit=1<<20
        e.cache_bytes=0; e.cache_hits=e.cache_misses=e.cache_evictions=0
        e.args=None; e.device="cpu"
        e.model=lambda *_:torch.full((1,5120),float("nan"),dtype=torch.bfloat16)
        with mock.patch.object(encoder,"load_image",return_value=(torch.zeros(1),1,1,1,1)):
            with self.assertRaises(ValueError): e.encode(record(png()))
        self.assertEqual(e.cache_bytes,0)
        self.assertEqual(len(e.cache),0)


if __name__ == "__main__":
    unittest.main()
