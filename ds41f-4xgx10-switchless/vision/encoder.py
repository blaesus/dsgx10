"""Native DeepSeek V4.1 vision tower. No language model or caption substitution.

The Unix-only boundary accepts bounded image records and returns BF16 embeddings
including the learned start/newline/end rows. Pixel processing follows the retained
PIL reference used by the accepted SGLang deployment.
"""
import argparse
import base64
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import socketserver
import struct
import time
from types import SimpleNamespace
import urllib.request
import warnings

import PIL
from PIL import Image
import torch
from torch import nn
from safetensors import safe_open

from reference_vision import ViT, Aligner
from reference_image_processor import load_image, image_token_types

MAX_FEATURE_BYTES = 512 << 20
MAX_IMAGE_BYTES = 16 << 20
MAX_REQUEST = 256 << 20
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
warnings.simplefilter('error', Image.DecompressionBombWarning)


def read_exact(sock, size):
    result = bytearray()
    while len(result) < size:
        part = sock.recv(min(size - len(result), 1 << 20))
        if not part:
            raise ValueError('truncated vision request')
        result.extend(part)
    return bytes(result)


def image_bytes(record):
    if not isinstance(record, dict):
        raise ValueError('image must be an object')
    source = record.get('source') or {}
    data = record.get('data', source.get('data'))
    url = record.get('url', source.get('url'))
    if data is not None:
        if not isinstance(data, str) or len(data) > 4 * (MAX_IMAGE_BYTES // 3 + 1):
            raise ValueError('invalid or oversized base64 image')
        raw = base64.b64decode(data, validate=True)
    elif isinstance(url, str) and url.startswith('data:'):
        header, sep, data = url.partition(',')
        if not sep or not header.startswith('data:image/') or not header.endswith(';base64'):
            raise ValueError('expected base64 image data URL')
        return image_bytes({'data': data})
    elif isinstance(url, str) and url.startswith(('https://', 'http://')):
        # No local file paths or file:// access. HTTP image bodies are size/time bounded.
        request = urllib.request.Request(url, headers={'User-Agent': 'dsgb10-native-vision/1'})
        with urllib.request.urlopen(request, timeout=20) as response:
            if not response.geturl().startswith(('https://', 'http://')):
                raise ValueError('unsupported image redirect')
            raw = response.read(MAX_IMAGE_BYTES + 1)
    else:
        raise ValueError('image requires a data URL, base64 source, or HTTP(S) URL')
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError('empty or oversized image')
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError('image exceeds pixel limit')
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('animated images are not supported')
        image.verify()
    return raw


class NativeVision(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vision = ViT(args)
        self.aligner = Aligner(args)
        self.image_start = nn.Parameter(torch.empty(args.dim))
        self.image_newline = nn.Parameter(torch.empty(args.dim))
        self.image_end = nn.Parameter(torch.empty(args.dim))

    @torch.inference_mode()
    def forward(self, patches, h, w):
        features = self.aligner(self.vision(patches, h, w), h, w)
        r = self.aligner.downsample_ratio
        types = image_token_types((h + r - 1) // r, (w + r - 1) // r).to(features.device)
        span = torch.empty((len(types), self.image_start.numel()), dtype=torch.bfloat16, device=features.device)
        span[types == 0] = self.image_start
        span[types == 1] = features
        span[types == 2] = self.image_newline
        span[types == 3] = self.image_end
        return span


class Encoder:
    def __init__(self, checkpoint, revision, device='cuda'):
        self.checkpoint = Path(checkpoint)
        self.revision = revision
        self.device = device
        cfg = json.loads((self.checkpoint / 'config.json').read_text())
        v = cfg['vision_config']
        self.args = SimpleNamespace(
            dim=cfg['text_config']['hidden_size'],
            vision_n_layers=v['num_hidden_layers'], vision_dim=v['hidden_size'],
            vision_n_heads=v['num_attention_heads'], vision_inter_dim=v['intermediate_size'],
            vision_patch_size=v['patch_size'], vision_rope_theta=v['rope_theta'],
            vision_downsample_ratio=v['downsample_ratio'], vision_max_n_token=v['max_image_tokens'],
            vision_min_pixels=v['min_pixels'], vision_max_wh_ratio=v['max_wh_ratio'])
        assert self.args.dim == 5120 and cfg['image_token_id'] == 129264
        assert self.args.vision_max_n_token == 1024
        # Preserve each checkpoint dtype (including FP32 norms); no quantization.
        with torch.device('meta'):
            self.model = NativeVision(self.args)
        index = json.loads((self.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
        keys = sorted(k for k in index if k.startswith(('vision.', 'aligner.', 'image_')))
        expected = set(self.model.state_dict())
        assert set(keys) == expected, (sorted(set(keys) - expected), sorted(expected - set(keys)))
        state = {}
        digest = hashlib.sha256()
        digest.update(json.dumps(cfg, sort_keys=True).encode())
        digest.update(revision.encode())
        digest.update(PIL.__version__.encode())
        for name in ('encoder.py', 'reference_vision.py', 'reference_image_processor.py'):
            digest.update((Path(__file__).parent / name).read_bytes())
        for shard in sorted({index[k] for k in keys}):
            path = (self.checkpoint / shard).resolve()
            assert path.parent == self.checkpoint.resolve()
            with safe_open(path, framework='pt', device='cpu') as source:
                for name in (k for k in keys if index[k] == shard):
                    value = source.get_tensor(name)
                    assert tuple(value.shape) == tuple(self.model.state_dict()[name].shape), name
                    digest.update(name.encode())
                    digest.update(str(value.dtype).encode())
                    digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
                    state[name] = value
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model = self.model.to(device).eval()
        self.identity = digest.hexdigest()
        self.weight_bytes = sum(v.numel() * v.element_size() for v in state.values())
        self.cache = OrderedDict()
        # CPU BF16 bytes, not GPU tensors. Two complete max-budget histories fit.
        self.cache_limit = 2 * MAX_FEATURE_BYTES
        self.cache_bytes = 0
        self.cache_hits = self.cache_misses = self.cache_evictions = 0

    @torch.inference_mode()
    def encode(self, record):
        raw = image_bytes(record)
        identity = hashlib.sha256(bytes.fromhex(self.identity) + raw).hexdigest()
        if identity in self.cache:
            value = self.cache.pop(identity)
            self.cache[identity] = value
            self.cache_hits += 1
            return (dict(value[0], cache_hit=True, encode_ms=0.0), value[1])
        self.cache_misses += 1
        # Bound to the validated PIL path; no automatic alternate resizing backend.
        patches, h, w, lh, lw = load_image({'data': base64.b64encode(raw).decode()}, self.args)
        before = time.perf_counter()
        span = self.model(patches.to(self.device), h, w)
        if not torch.isfinite(span).all().item():
            raise ValueError('non-finite native vision embeddings')
        payload = span.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        meta = {'rows': int(span.shape[0]), 'dim': 5120, 'digest': identity,
                'embedding_sha256': hashlib.sha256(payload).hexdigest(),
                'vit_grid': [h, w], 'llm_grid': [lh, lw],
                'encode_ms': (time.perf_counter() - before) * 1000, 'cache_hit': False}
        assert 0 < meta['rows'] <= 1024 and len(payload) == meta['rows'] * 5120 * 2
        value = (meta, payload)
        self.cache[identity] = value
        self.cache_bytes += len(payload)
        while self.cache_bytes > self.cache_limit:
            _, evicted = self.cache.popitem(last=False)
            self.cache_bytes -= len(evicted[1])
            self.cache_evictions += 1
        return value


class ImageFeatureBudgetError(ValueError):
    pass


def encode_many(encoder, images):
    if not isinstance(images, list) or not images:
        raise ValueError('expected nonempty image list')
    outputs = []
    size = 0
    for record in images:
        value = encoder.encode(record)
        size += len(value[1])
        if size > MAX_FEATURE_BYTES:
            raise ImageFeatureBudgetError('image feature memory budget exceeded')
        outputs.append(value)
    return outputs


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(180)
        try:
            size, = struct.unpack('<I', read_exact(self.request, 4))
            if size > MAX_REQUEST:
                raise ValueError('vision request too large')
            request = json.loads(read_exact(self.request, size))
            encoder = self.server.encoder
            if request.get('op') == 'health':
                header = {'ok': True, 'identity': encoder.identity, 'revision': encoder.revision,
                          'weight_bytes': encoder.weight_bytes, 'device': encoder.device,
                          'cache': {'bytes': encoder.cache_bytes, 'limit_bytes': encoder.cache_limit,
                                    'entries': len(encoder.cache), 'hits': encoder.cache_hits,
                                    'misses': encoder.cache_misses, 'evictions': encoder.cache_evictions}}
                outputs = []
            else:
                outputs = encode_many(encoder, request.get('images'))
                header = {'ok': True, 'identity': encoder.identity, 'revision': encoder.revision,
                          'images': [meta for meta, _ in outputs]}
            encoded = json.dumps(header).encode()
            self.request.sendall(struct.pack('<I', len(encoded)) + encoded)
            for _, payload in outputs:
                self.request.sendall(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            # Do not echo URLs, image contents or exception payloads to clients/logs.
            encoded = json.dumps({'ok': False, 'code': 'image_feature_budget' if isinstance(error, ImageFeatureBudgetError) else 'image_encoding', 'error': type(error).__name__ + ': native image encoding failed'}).encode()
            try:
                self.request.sendall(struct.pack('<I', len(encoded)) + encoded)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    encoder = Encoder(args.checkpoint, args.revision, args.device)
    path = Path(args.socket)
    if path.exists():
        raise RuntimeError('vision socket already exists; refuse to replace a running encoder')
    previous = os.umask(0o177)
    try:
        with socketserver.UnixStreamServer(str(path), Handler) as server:
            server.encoder = encoder
            print(json.dumps({'ready': True, 'identity': encoder.identity, 'revision': encoder.revision,
                              'weight_bytes': encoder.weight_bytes}), flush=True)
            server.serve_forever(poll_interval=0.5)
    finally:
        os.umask(previous)
        path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
