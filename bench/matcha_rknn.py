#!/usr/bin/env python3
"""Matcha 的 NPU 推理路径，供 bench/run_tts.py 使用。

自 rkvoice-stream（Apache 2.0）`rkvoice_stream/backends/tts/matcha.py` 精简移植：
与 sherpa-onnx 逐 token 对齐的前端（83/83 校验）、vocos 定窗重叠合成、
ISTFT 边缘修复、utterance gain。

两个后端（--type 选择）：
- matcha-rknn  ：encoder+estimator 拆分 RKNN fp16（scripts/matcha-rknn/README.md 转换）。
                 ⚠️ 实测 estimator fp16 精度不足（逐步偏差 25%/37%/63%，ODE 累积后不可懂），
                 仅留作对照；上游生产也不用它（ORT CPU 声学 9/10 vs RKNN 拆分 7/10）。
- matcha-hybrid：声学模型 CPU ONNX fp32（动态长度，无需帧数估计）+ vocos NPU + ISTFT CPU。
                 这是上游事实上的生产形态（声学留 CPU），RTF 收益来自 vocos 卸载。

模型/前端文件：声学 models/matcha-zh-en/cpu/model-steps-3.onnx；
vocos models/matcha-zh-en/rknn/vocos-16khz-univ-fp16.rknn；
前端复用 models/matcha-zh-en/cpu/（lexicon.txt / tokens.txt / espeak-ng-data）。
"""
import re
import threading
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
NUM_MEL = 80
VOCOS_FRAMES = 600       # vocos 编译定长
MEL_SIGMA = 5.446792     # 拆分 rknn 路径的反归一化常数
MEL_MEAN = -2.9521978
N_TIME_BLOCKS = 6
TIME_EMB_DIM = 256
NOISE_SCALE = 0.667

# ---------------- 前端（与 sherpa-onnx 逐 token 对齐） ----------------

_IPA_MAP = {
    'aɪ': ('I',), 'eɪ': ('A',), 'oʊ': ('O',), 'aʊ': ('W',), 'ɔɪ': ('Y',),
    'tʃ': ('ʧ',), 'dʒ': ('ʤ',), 'ɚ': ('ə', 'ɹ'),
}
_IPA_MAX_KEY = max(len(k) for k in _IPA_MAP)
_IPA_STRIP = frozenset('ː')

_PUNCT_NORMALIZE = {'：': ',', ':': ',', '，': ',', '、': ',',
                    '。': '.', '！': '!', '？': '?', '；': ';'}
_PUNCT_DROP = frozenset("-'（）《》")

_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'\-]*")
_ITEM_RE = re.compile(r"[一-鿿]+|[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'\-]*|\s+|.")


def parse_tokens_file(path):
    """icefall tokens.txt → {token: id}。token 可能是空格本身，id 从右侧切。"""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            token, sep, id_str = line.rpartition(' ')
            if not sep:
                continue
            try:
                out[token] = int(id_str)
            except ValueError:
                continue
    return out


def ipa_to_token_strings(ipa):
    out, i, n = [], 0, len(ipa)
    while i < n:
        for width in range(min(_IPA_MAX_KEY, n - i), 0, -1):
            key = ipa[i:i + width]
            if key in _IPA_MAP:
                out.extend(_IPA_MAP[key])
                i += width
                break
        else:
            ch = ipa[i]
            i += 1
            if ch in _IPA_STRIP:
                continue
            if ch == ' ':
                out.append(' ')
            elif ch.isspace():
                continue
            else:
                out.append(ch)
    return out


def _load_espeak_lib():
    """libespeak-ng：优先系统库，否则用 piper-phonemize wheel 自带的那个。"""
    import ctypes
    try:
        return ctypes.CDLL('libespeak-ng.so.1')
    except OSError:
        pass
    try:
        import piper_phonemize
        libs = Path(piper_phonemize.__file__).parent.parent / 'piper_phonemize.libs'
        for so in sorted(libs.glob('libespeak-ng*.so*')):
            return ctypes.CDLL(str(so))
    except Exception:
        pass
    raise RuntimeError('libespeak-ng 不可用（系统无、piper-phonemize 也未装）')


class EspeakPhonemizer:
    """espeak-ng C API 逐词 IPA（CLI 会给孤立虚词加重读，与 sherpa 不一致，故不用 CLI）。"""

    _ESPEAK_CHARS_UTF8 = 1
    _ESPEAK_PHONEMES_IPA = 2
    _AUDIO_OUTPUT_RETRIEVAL = 1

    def __init__(self, data_dir=None):
        import ctypes
        self._ct = ctypes
        self._lock = threading.Lock()
        self._lib = _load_espeak_lib()
        self._lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self._lib.espeak_Initialize.restype = ctypes.c_int
        self._lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self._lib.espeak_SetVoiceByName.restype = ctypes.c_int
        self._lib.espeak_TextToPhonemes.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
        self._lib.espeak_TextToPhonemes.restype = ctypes.c_char_p

        # espeak_Initialize 的 path 是「包含 espeak-ng-data 的目录」
        path = None
        if data_dir:
            d = Path(data_dir)
            parent = d.parent if d.name == 'espeak-ng-data' else d
            if parent.is_dir():
                path = str(parent).encode()
        if self._lib.espeak_Initialize(self._AUDIO_OUTPUT_RETRIEVAL, 0, path, 0) < 0:
            raise RuntimeError('espeak_Initialize 失败')
        if self._lib.espeak_SetVoiceByName(b'en-us') != 0:
            raise RuntimeError("espeak_SetVoiceByName('en-us') 失败")

    def phonemize(self, word):
        ctypes = self._ct
        with self._lock:
            buf = ctypes.c_char_p(word.encode('utf-8'))
            ptr = ctypes.cast(ctypes.byref(buf), ctypes.POINTER(ctypes.c_void_p))
            out = []
            while True:
                res = self._lib.espeak_TextToPhonemes(
                    ptr, self._ESPEAK_CHARS_UTF8, self._ESPEAK_PHONEMES_IPA)
                if res:
                    out.append(res.decode('utf-8'))
                if not buf.value:
                    break
            return ''.join(out)


_MEL_FRAMES_PER_TOKEN = {'phoneme': 4.7, 'pinyin': 11.6, 'punct': 16.4, 'boundary': 0.0}
_MEL_FRAMES_CONST = 40.0
_PINYIN_RE = re.compile(r'^[a-z]+[1-5]$')


def classify_token(token):
    if token == ' ':
        return 'boundary'
    if len(token) == 1 and not token.isalnum():
        return 'punct'
    if _PINYIN_RE.match(token):
        return 'pinyin'
    return 'phoneme'


# ---------------- 基类：前端 + 分句 + vocos + ISTFT + 增益 ----------------

class MatchaBase:
    """子类只需实现 load() 与 _run_matcha(tokens, speed) → (mel, mel_frames)。"""

    TOKEN_BUDGET = 10**9  # 子类覆盖（RKNN 声学有烘焙上限）

    def __init__(self, frontend_dir):
        self.frontend_dir = Path(frontend_dir)

    # ---- 加载 ----

    def _load_frontend(self):
        self._lexicon = {}
        with open(self.frontend_dir / 'lexicon.txt', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self._lexicon[parts[0]] = parts[1:]
        self._token_to_id = parse_tokens_file(self.frontend_dir / 'tokens.txt')
        self._id_to_class = {tid: classify_token(tok) for tok, tid in self._token_to_id.items()}
        self._espeak = EspeakPhonemizer(self.frontend_dir / 'espeak-ng-data')

    @staticmethod
    def _load_rknn(path):
        from rknnlite.api import RKNNLite
        m = RKNNLite(verbose=False)
        if m.load_rknn(str(path)) != 0:
            raise RuntimeError(f'load_rknn 失败: {path}')
        if m.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError(f'init_runtime 失败: {path}')
        return m

    # ---- 前端 ----

    def _chinese_token_strings(self, text):
        out, i = [], 0
        while i < len(text):
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i + length]
                if word in self._lexicon:
                    out.extend(self._lexicon[word])
                    i += length
                    break
            else:
                i += 1
        return out

    def _phonemize_english(self, word):
        return ipa_to_token_strings(self._espeak.phonemize(word))

    def text_to_tokens(self, text):
        items = []
        for raw in _ITEM_RE.findall(text):
            if not raw or raw.isspace():
                continue
            if _CJK_RE.match(raw):
                items.append((False, self._chinese_token_strings(raw)))
                continue
            if _WORD_RE.fullmatch(raw):
                items.append((True, self._phonemize_english(raw)))
                continue
            ch = _PUNCT_NORMALIZE.get(raw, raw)
            if ch in _PUNCT_DROP:
                continue
            if not ch.isalnum() and ch in self._token_to_id:
                items.append((False, [ch]))
                continue
            items.append((False, self._phonemize_english(raw)))  # 数字等交给 espeak 读出来

        boundary = self._token_to_id.get(' ')
        tokens = []
        for idx, (is_word, strings) in enumerate(items):
            for s in strings:
                tid = self._token_to_id.get(s)
                if tid is not None:
                    tokens.append(tid)
            if (is_word and strings and boundary is not None
                    and idx + 1 < len(items) and items[idx + 1][1]):
                tokens.append(boundary)
        return tokens

    # ---- 分句（句末标点 → 软标点 → token 预算硬切） ----

    def _split_text(self, text):
        segments = re.split(r'([。！？；;]|[!?.](?=\s|$))', text)
        merged = []
        for i in range(0, len(segments), 2):
            seg = segments[i] + (segments[i + 1] if i + 1 < len(segments) else '')
            seg = seg.strip()
            if seg:
                merged.append(seg)
        if not merged:
            return [text]

        final = []
        for seg in merged:
            if len(self.text_to_tokens(seg)) <= self.TOKEN_BUDGET:
                final.append(seg)
                continue
            subs = re.split(r'([，,、：:])', seg)
            for j in range(0, len(subs), 2):
                s = subs[j] + (subs[j + 1] if j + 1 < len(subs) else '')
                s = s.strip()
                if not s:
                    continue
                if len(self.text_to_tokens(s)) <= self.TOKEN_BUDGET:
                    final.append(s)
                else:  # 硬切
                    cur = ''
                    for atom in re.findall(r'[一-鿿]|\S+\s*', s):
                        if cur and len(self.text_to_tokens(cur + atom)) > self.TOKEN_BUDGET:
                            final.append(cur.strip())
                            cur = atom
                        else:
                            cur += atom
                    if cur.strip():
                        final.append(cur.strip())
        return final or [text]

    # ---- vocos / ISTFT / 增益 ----

    def _run_vocos(self, mel, total_frames):
        def emit(window, lo, hi):
            buf = np.zeros((1, NUM_MEL, VOCOS_FRAMES), dtype=np.float32)
            buf[:, :, :window.shape[2]] = window
            out = self._vocos.inference(inputs=[buf])
            return out[0][0][:, lo:hi], out[1][0][:, lo:hi], out[2][0][:, lo:hi]

        if total_frames <= VOCOS_FRAMES:
            return emit(mel[:, :, :total_frames], 0, total_frames)
        ctx = min(32, VOCOS_FRAMES // 8)
        stride = VOCOS_FRAMES - 2 * ctx
        mags, xs, ys, pos = [], [], [], 0
        while pos < total_frames:
            w_end = min(total_frames, max(pos - ctx, 0) + VOCOS_FRAMES)
            w_start = max(0, w_end - VOCOS_FRAMES)
            keep_end = min(total_frames, pos + stride)
            m, cx, sy = emit(mel[:, :, w_start:w_end], pos - w_start, keep_end - w_start)
            mags.append(m); xs.append(cx); ys.append(sy)
            pos = keep_end
        return (np.concatenate(mags, axis=1), np.concatenate(xs, axis=1),
                np.concatenate(ys, axis=1))

    @staticmethod
    def _istft(mag, x, y):
        spec = mag * (x + 1j * y)
        n_frames = spec.shape[1]
        out_len = (n_frames - 1) * HOP_LENGTH + N_FFT
        audio = np.zeros(out_len, dtype=np.float32)
        window = np.hanning(N_FFT)
        wsum = np.zeros(out_len, dtype=np.float32)
        for i in range(n_frames):
            frame = np.fft.irfft(spec[:, i], n=N_FFT) * window
            s = i * HOP_LENGTH
            audio[s:s + N_FFT] += frame
            wsum[s:s + N_FFT] += window ** 2
        # 边缘覆盖不足处清零而非除小量（防末端尖峰劫持整体归一化）
        steady = float(np.median(wsum[wsum > 0])) if np.any(wsum > 0) else 0.0
        floor = max(steady * 1e-2, 1e-8)
        valid = wsum > floor
        audio[valid] /= wsum[valid]
        audio[~valid] = 0.0
        return audio

    @staticmethod
    def _gain(audio):
        if len(audio) == 0:
            return audio
        peak = float(np.abs(audio).max())
        if peak <= 0:
            return audio
        p999 = float(np.percentile(np.abs(audio), 99.9))
        if p999 > 0 and p999 / peak < 0.25:  # 尖峰离群：按 p99.9 归一化并削峰
            return np.clip(audio * (0.95 / p999), -1.0, 1.0)
        return audio * (0.95 / peak)

    # ---- 合成主流程 ----

    def synthesize(self, text, speed=1.0):
        parts = []
        t0 = time.perf_counter()
        for seg in self._split_text(text):
            tokens = self.text_to_tokens(seg)
            if not tokens:
                continue
            if len(tokens) > self.TOKEN_BUDGET:  # 分句漏网，截断留痕
                print(f'[matcha] WARN 超预算截断 {len(tokens)}>{self.TOKEN_BUDGET}: {seg[:30]}')
                tokens = tokens[:self.TOKEN_BUDGET]
            mel, frames = self._run_matcha(tokens, speed)
            mag, x, y = self._run_vocos(mel, frames)
            audio = self._istft(mag, x, y)
            parts.append(audio[:frames * HOP_LENGTH])
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return self._gain(audio).astype(np.float32), time.perf_counter() - t0

    # run_tts.py 的 OfflineTts.generate 接口适配
    def generate(self, text, sid=0, speed=1.0):
        audio, _ = self.synthesize(text, speed)
        return type('Audio', (), {'samples': audio, 'sample_rate': SAMPLE_RATE})()


# ---------------- 全 NPU 拆分版（对照用；estimator fp16 精度不足，勿用于报告数字） ----------------

class MatchaRKNN(MatchaBase):
    """encoder(NPU) + estimator(NPU) + ODE 环(CPU fp32) + vocos(NPU)。"""

    TOKEN_BUDGET = 64  # s64 bucket 烘焙的 x_len

    def __init__(self, rknn_dir, frontend_dir, ode_steps=1):
        super().__init__(frontend_dir)
        self.rknn_dir = Path(rknn_dir)
        self.ode_steps = ode_steps

    def load(self):
        self._load_frontend()
        self._enc = self._load_rknn(self.rknn_dir / 'matcha-encoder-fp16.rknn')
        self._est = self._load_rknn(self.rknn_dir / 'matcha-estimator-fp16.rknn')
        self._vocos = self._load_rknn(self.rknn_dir / 'vocos-16khz-univ-fp16.rknn')
        self._time_emb = [np.load(self.rknn_dir / f'time_emb_step{i}.npy') for i in range(3)]
        # 从 encoder 输入张量探测烘焙的 seq_len（x: [1, seq_len]，s64=80）
        attr = self._enc.rknn_runtime.get_tensor_attr(0, False)
        self.seq_len = int(list(attr.dims[:attr.n_dims])[-1])
        print(f'[matcha-rknn] encoder seq_len={self.seq_len}, token budget={self.TOKEN_BUDGET}')

    def _estimate_mel_frames(self, tokens, length_scale=1.0):
        total = _MEL_FRAMES_CONST
        for tid in tokens:
            total += _MEL_FRAMES_PER_TOKEN[self._id_to_class.get(tid, 'phoneme')]
        return min(int(total * length_scale * 1.2 + 10 + 0.5), 600)

    def _run_matcha(self, tokens, speed=1.0):
        n = len(tokens)
        x = np.zeros((1, self.seq_len), dtype=np.int64)
        x[0, :n] = tokens
        xl = np.array([n], dtype=np.int64)
        ns = np.array([NOISE_SCALE], dtype=np.float32)
        ls = np.array([1.0 / speed], dtype=np.float32)

        mu, mask, z = self._enc.inference(inputs=[x, xl, ns, ls])[:3]
        z = z.astype(np.float32)  # ODE 环 CPU fp32
        dt = np.float32(1.0 / self.ode_steps)
        for step in range(self.ode_steps):
            te = self._time_emb[min(step, len(self._time_emb) - 1)]
            feeds = [z, mu, mask] + [te[i].reshape(1, TIME_EMB_DIM, 1).astype(np.float32)
                                     for i in range(N_TIME_BLOCKS)]
            v = self._est.inference(inputs=feeds)[0]
            z = z + dt * v

        mel = z * np.float32(MEL_SIGMA) + np.float32(MEL_MEAN)
        mel = np.clip(mel, -25.0, 8.0)
        return mel, min(self._estimate_mel_frames(tokens, 1.0 / speed), mel.shape[2])

    def release(self):
        for m in (self._enc, self._est, self._vocos):
            try:
                m.release()
            except Exception:
                pass


# ---------------- 混合版：声学 CPU ONNX + vocos NPU ----------------

class MatchaHybrid(MatchaBase):
    """声学模型走 CPU ONNX fp32（动态长度，免帧数估计），vocos 走 NPU。"""

    TOKEN_BUDGET = 256  # ORT 动态 shape；仅作分句兜底

    def __init__(self, acoustic_onnx, vocos_rknn, frontend_dir):
        super().__init__(frontend_dir)
        self.acoustic_onnx = str(acoustic_onnx)
        self._vocos_path = str(vocos_rknn)

    def load(self):
        import onnxruntime as ort
        self._load_frontend()
        self._acoustic = ort.InferenceSession(self.acoustic_onnx, providers=['CPUExecutionProvider'])
        self._vocos = self._load_rknn(self._vocos_path)
        print('[matcha-hybrid] 声学=CPU ONNX fp32, vocos=NPU RKNN')

    def _run_matcha(self, tokens, speed=1.0):
        out = self._acoustic.run(None, {
            'x': np.array([tokens], dtype=np.int64),
            'x_length': np.array([len(tokens)], dtype=np.int64),
            'noise_scale': np.array([NOISE_SCALE], dtype=np.float32),
            'length_scale': np.array([1.0 / speed], dtype=np.float32),
        })
        mel = out[0]
        return mel, mel.shape[2]  # 动态输出，真实帧数

    def release(self):
        try:
            self._vocos.release()
        except Exception:
            pass
