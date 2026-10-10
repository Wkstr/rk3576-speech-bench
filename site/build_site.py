#!/usr/bin/env python3
"""从 results/ 的 JSON 生成静态对比单页 site/index.html。

- 数据内联为 JS 常量，页面无构建依赖，Chart.js 走 CDN。
- 中英文切换：右上角按钮，localStorage 记忆，默认中文。
- TTS 音频试听：`--copy-audio N` 把每模型前 N 条样本 wav 拷到 site/audio/ 并挂播放器；
  site/audio/ 不存在则自动省略试听区（wav 默认 gitignore，发布与否由用户定）。
用法：.venv/bin/python site/build_site.py [--copy-audio 3]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASR_DIR = ROOT / "results/asr"
TTS_DIR = ROOT / "results/tts"
SITE = ROOT / "site"
OUT = SITE / "index.html"

ASR_NAMES = {
    "zipformer-small__rknn-2.3.2": ("Zipformer small", "NPU fp16"),
    "zipformer-small__cpu96": ("Zipformer small", "CPU fp32"),
    "zipformer-large__rknn-2.3.2": ("Zipformer large", "NPU fp16"),
    "zipformer-large__cpu": ("Zipformer large", "CPU fp32"),
}
TTS_NAMES = {
    "piper-zh-huayan__zh": ("Piper zh (huayan)", "VITS", "zh"),
    "piper-en-ryan__en": ("Piper en (ryan)", "VITS", "en"),
    "matcha-zh-en__zh-en": ("Matcha + vocos", "Matcha", "zh+en"),
    "kokoro__zh-en": ("Kokoro 82M v1.1", "Kokoro", "zh+en"),
    "mms-eng__en": ("MMS en", "VITS", "en"),
}
REPO = "https://github.com/Wkstr/rk3576-speech-bench"


def load(d: Path) -> dict[str, dict]:
    return {json.loads(f.read_text())["config"]["tag"]: json.loads(f.read_text())
            for f in sorted(d.glob("*.json")) if not f.name.endswith(".sysmon.jsonl")}


def npu_stats(tag: str, d: Path):
    f = d / f"{tag}.sysmon.jsonl"
    if not f.exists():
        return None, None
    vals = []
    for line in f.read_text().splitlines():
        s = json.loads(line).get("npu_load")
        if s:
            # 格式 "NPU load:  Core0:  0%, Core1:  0%,"——取各核百分比均值
            cores = [int(x) for x in re.findall(r"Core\d+:\s*(\d+)%", str(s))]
            if cores:
                vals.append(sum(cores) / len(cores))
    return (round(sum(vals) / len(vals), 1), round(max(vals), 1)) if vals else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-audio", type=int, default=0, help="每模型拷贝前 N 条 TTS 样本到 site/audio/")
    args = ap.parse_args()

    asr_j, tts_j = load(ASR_DIR), load(TTS_DIR)
    meta_src = next(iter(asr_j.values()))["config"]
    dates = sorted(r["config"]["date_utc"][:10] for r in list(asr_j.values()) + list(tts_j.values()))

    asr = []
    order = ["zipformer-small__rknn-2.3.2", "zipformer-small__cpu96",
             "zipformer-large__rknn-2.3.2", "zipformer-large__cpu"]
    for base in order:
        for lang in ("zh", "en"):
            r = asr_j.get(f"{base}__{lang}")
            if not r:
                continue
            a, s = r["aggregate"], r["sysmon"]
            e = a.get("cer") or a.get("wer")
            nm, np_ = npu_stats(r["config"]["tag"], ASR_DIR)
            asr.append(dict(model=ASR_NAMES[base][0], backend=ASR_NAMES[base][1], lang=lang,
                            errKind="CER" if "cer" in a else "WER",
                            err=e["center"], errLo=e["wilson95"][0], errHi=e["wilson95"][1],
                            rtf=a["rtf_mean"], rtfx=a["rtfx"],
                            ft50=a["first_token_s"]["p50"], ft95=a["first_token_s"]["p95"],
                            cpu=s["cpu_pct_mean"], rss=s["rss_mb_peak"], npuMean=nm, npuPeak=np_))
    tts = []
    for tag, (name, arch, langs) in TTS_NAMES.items():
        r = tts_j.get(tag)
        if not r:
            continue
        a, s = r["aggregate"], r["sysmon"]
        rt = a["roundtrip_error_rate"]
        audio = []
        adir = SITE / "audio" / tag
        if args.copy_audio:
            adir.mkdir(parents=True, exist_ok=True)
            src = TTS_DIR / "samples" / tag
            for w in sorted(src.glob("*.wav"))[: args.copy_audio]:
                shutil.copy2(w, adir / w.name)
        if adir.exists():
            audio = sorted(f"audio/{tag}/{p.name}" for p in adir.glob("*.wav"))
        tts.append(dict(tag=tag, name=name, arch=arch, langs=langs, n=len(r["items"]),
                        rtf=a["rtf_mean"], rtfx=a["rtfx"],
                        rt=rt["center"], rtLo=rt["wilson95"][0], rtHi=rt["wilson95"][1],
                        rss=s["rss_mb_peak"], audio=audio))

    html = TEMPLATE
    html = html.replace("__META__", json.dumps(dict(
        board=meta_src["board"], kernel=meta_src["kernel"], d0=dates[0], d1=dates[-1],
        repo=REPO), ensure_ascii=False))
    html = html.replace("__ASR__", json.dumps(asr, ensure_ascii=False))
    html = html.replace("__TTS__", json.dumps(tts, ensure_ascii=False))
    OUT.write_text(html)
    print(f"[done] {OUT}（audio: {'on' if any(t['audio'] for t in tts) else 'off'}）")


TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RK3576 ASR / TTS Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root { --ink:#1a2332; --mut:#5b6b7f; --line:#dde4ec; --npu:#0d9488; --cpu:#b45309; --bg:#f7f9fb; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.7 -apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC",sans-serif; color:var(--ink); background:var(--bg); }
main { max-width:1040px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:26px; margin:0 0 4px; }
h2 { font-size:19px; margin:36px 0 10px; border-left:4px solid var(--npu); padding-left:10px; }
.sub { color:var(--mut); font-size:13px; margin-bottom:20px; }
header { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap; }
#lang-btn { border:1px solid var(--line); background:#fff; border-radius:99px; padding:5px 14px;
  font-size:13px; cursor:pointer; color:var(--ink); }
#lang-btn:hover { border-color:var(--npu); color:var(--npu); }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:18px 0; }
.card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.card b { display:block; font-size:22px; margin-top:2px; }
.card span { color:var(--mut); font-size:12.5px; }
table { width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; font-size:13.5px; margin:10px 0; }
th,td { padding:8px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
th { background:#eef2f7; cursor:pointer; user-select:none; font-weight:600; }
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
tr:last-child td { border-bottom:0; }
tr:hover td { background:#f2f7f6; }
.tag { font-size:11px; padding:1px 7px; border-radius:99px; background:#e6fffa; color:var(--npu); border:1px solid #99f6e4; }
.tag.cpu { background:#fff7ed; color:var(--cpu); border-color:#fed7aa; }
.chart { background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px; margin:12px 0; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
@media (max-width:760px){ .grid2{ grid-template-columns:1fr; } }
audio { width:100%; height:32px; margin:2px 0; }
.audio-item { margin:8px 0; }
.audio-item code { font-size:12px; color:var(--mut); }
.note { background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:10px 14px; font-size:13.5px; }
.plain { background:#fff; border-color:var(--line); }
.plain a { display:inline-block; margin:2px 10px 2px 0; }
ul { padding-left:20px; }
footer { color:var(--mut); font-size:12.5px; margin-top:40px; border-top:1px solid var(--line); padding-top:14px; }
a { color:var(--npu); }
</style>
</head>
<body><main>
<header>
  <div>
    <h1>RK3576 ASR / TTS Benchmark</h1>
    <div class="sub" id="meta"></div>
  </div>
  <button id="lang-btn" onclick="switchLang()">English</button>
</header>

<div class="cards">
  <div class="card"><span data-i18="c1"></span><b id="c-asr-fast"></b></div>
  <div class="card"><span data-i18="c2"></span><b id="c-asr-acc"></b></div>
  <div class="card"><span data-i18="c3"></span><b id="c-tts-fast"></b></div>
  <div class="card"><span data-i18="c4"></span><b id="c-tts-rt"></b></div>
</div>

<div class="note" data-i18-html="note"></div>

<h2 data-i18="h_asr"></h2>
<div class="grid2">
  <div class="chart"><canvas id="ch-asr-err"></canvas></div>
  <div class="chart"><canvas id="ch-asr-rtf"></canvas></div>
</div>
<table id="t-asr"><thead><tr>
<th data-i18="th_model"></th><th data-i18="th_backend"></th><th data-i18="th_lang"></th><th data-k="err" data-i18="th_err"></th><th>95% CI</th>
<th data-k="rtf">RTF</th><th data-k="rtfx" data-i18="th_rtfx"></th><th data-k="ft50" data-i18="th_ft50"></th><th data-k="ft95">p95</th>
<th data-k="cpu" data-i18="th_cpu"></th><th data-k="rss">RSS (MB)</th><th data-k="npuMean" data-i18="th_npu"></th>
</tr></thead><tbody></tbody></table>

<h2 data-i18="h_tts"></h2>
<div class="grid2">
  <div class="chart"><canvas id="ch-tts-rtf"></canvas></div>
  <div class="chart"><canvas id="ch-tts-rt"></canvas></div>
</div>
<table id="t-tts"><thead><tr>
<th data-i18="th_model"></th><th data-i18="th_arch"></th><th data-i18="th_lang"></th><th data-k="n" data-i18="th_n"></th>
<th data-k="rtf">RTF</th><th data-k="rtfx" data-i18="th_rtfx"></th><th data-k="rt" data-i18="th_rt"></th><th>95% CI</th><th data-k="rss">RSS (MB)</th>
</tr></thead><tbody></tbody></table>

<div id="audio-sec"></div>

<h2 data-i18="h_find"></h2>
<ul>
<li data-i18-html="f1"></li>
<li data-i18-html="f2"></li>
<li data-i18-html="f3"></li>
<li data-i18-html="f4"></li>
</ul>

<h2 data-i18="h_about"></h2>
<div class="note plain">
  <a href="https://github.com/Wkstr/rk3576-speech-bench">GitHub repo</a>
  <a href="https://github.com/Wkstr/rk3576-speech-bench/blob/main/report/REPORT.md" data-i18="lk_report"></a>
  <a href="https://github.com/k2-fsa/sherpa-onnx">sherpa-onnx</a>
  <a href="https://github.com/k2-fsa/sherpa-onnx/issues/2861" data-i18="lk_issue"></a>
  <a href="https://github.com/airockchip/rknn-toolkit2">rknn-toolkit2 2.3.2</a>
  <a href="https://www.openslr.org/12">LibriSpeech</a>
  <a href="https://huggingface.co/datasets/google/fleurs">FLEURS</a>
</div>

<footer data-i18-html="foot"></footer>
</main>

<script>
const META = __META__;
const ASR = __ASR__;
const TTS = __TTS__;
const pct = x => (x*100).toFixed(1) + "%";

const I18N = {
zh: {
  btn: "English",
  c1: "ASR 最快", c2: "ASR 最准（zh）", c3: "TTS 最快", c4: "TTS 最可懂（下界）",
  note: `口径速读：zh 报 CER、en 报 WER，CI 列为 Wilson 95% 置信区间；RTF 越低越好（&lt;1 即快于实时）；
    NPU 版为自转 fp16（官方 RKNN 坏掉，<a href="https://github.com/k2-fsa/sherpa-onnx/issues/2861">sherpa-onnx#2861</a>）；
    TTS「回译」是<b>可懂度下界</b>（数字口读与同音词计为错误，横向比较公平）。
    <a href="https://github.com/Wkstr/rk3576-speech-bench/blob/main/report/REPORT.md">完整报告与原始 JSON →</a>`,
  h_asr: "ASR 语音识别", h_tts: "TTS 语音合成（CPU）", h_find: "要点", h_about: "关于",
  th_model: "模型", th_backend: "后端", th_lang: "语言", th_err: "错误率", th_rtfx: "倍速",
  th_ft50: "首字 p50 (s)", th_cpu: "CPU 均值", th_npu: "NPU 占用", th_arch: "架构", th_n: "条数",
  th_rt: "回译错误率（下界）",
  ch_asr_err: "ASR 错误率 %（zh=CER / en=WER）", ch_asr_rtf: "ASR RTF（越低越好，<1 快于实时）",
  ch_tts_rtf: "TTS RTF（CPU，越低越好）", ch_tts_rt: "TTS 回译错误率 %（可懂度下界）",
  h_audio: "TTS 试听",
  f1: "NPU 自转 fp16 与 CPU fp32 输出近乎逐条一致（zh 30/30、29/30，en 27/30、30/30，差异仅 1 词级）——fp16 零退化。",
  f2: "NPU 对 large 无加速（RTF 0.41 = CPU）；small NPU RTF 0.14–0.15（约 7 倍实时）。",
  f3: "TTS：Matcha 最快（RTF 0.25）且中英双语，但英文可懂度差；Piper 均衡；Kokoro RTF 2.87 仅批处理可用；MMS 仅作地板。",
  f4: "Kokoro zh 遇数字串会读成英文；Matcha en 被回译裁判听成中文音节——均为模型真实缺陷。",
  lk_report: "完整报告 REPORT.md", lk_issue: "上游 bug #2861",
  foot: `测量层：sherpa-onnx + librknnrt 2.2.0，双 pass（吞吐+实时节奏），10Hz 资源采样 ·
    呈现层只读 <code>results/*.json</code>，重跑 harness 后 <code>build_site.py</code> 重新生成 ·
    数据集：LibriSpeech test-clean / FLEURS cmn_hans_cn（各抽 30 条）/ 自建 TTS 文本`,
  meta: m => `${m.board} · RK3576 8GB · Debian 12 · kernel ${m.kernel} · 测量 ${m.d0} ~ ${m.d1}`,
},
en: {
  btn: "中文",
  c1: "Fastest ASR", c2: "Most accurate ASR (zh)", c3: "Fastest TTS", c4: "Most intelligible TTS (floor)",
  note: `How to read: zh = CER, en = WER, CI columns are Wilson 95% intervals; lower RTF is better (&lt;1 = faster than realtime).
    NPU builds are self-converted fp16 (the vendor's official RKNN zipformer is broken upstream, <a href="https://github.com/k2-fsa/sherpa-onnx/issues/2861">sherpa-onnx#2861</a>).
    TTS "roundtrip" is an <b>intelligibility lower bound</b> (verbalized digits and homophones count as errors; comparisons across models stay fair).
    <a href="https://github.com/Wkstr/rk3576-speech-bench/blob/main/report/REPORT.md">Full report &amp; raw JSON →</a>`,
  h_asr: "ASR (speech recognition)", h_tts: "TTS (speech synthesis, CPU)", h_find: "Highlights", h_about: "About",
  th_model: "Model", th_backend: "Backend", th_lang: "Lang", th_err: "Error rate", th_rtfx: "Speed",
  th_ft50: "1st-token p50 (s)", th_cpu: "CPU mean", th_npu: "NPU load", th_arch: "Arch", th_n: "Clips",
  th_rt: "Roundtrip err (floor)",
  ch_asr_err: "ASR error rate % (zh=CER / en=WER)", ch_asr_rtf: "ASR RTF (lower is better, <1 = realtime)",
  ch_tts_rtf: "TTS RTF on CPU (lower is better)", ch_tts_rt: "TTS roundtrip error % (intelligibility floor)",
  h_audio: "Listen to TTS samples",
  f1: "Self-converted fp16 NPU output matches CPU fp32 near-verbatim (zh 30/30, 29/30; en 27/30, 30/30 — single-word diffs) — zero fp16 degradation.",
  f2: "NPU gives large no speedup (RTF 0.41 = CPU); small on NPU reaches RTF 0.14–0.15 (~7× realtime).",
  f3: "TTS: Matcha is fastest (RTF 0.25) and bilingual, but its English is unintelligible; Piper is balanced; Kokoro RTF 2.87 is batch-only; MMS is a floor reference.",
  f4: "Kokoro reads digit strings in English inside zh sentences; Matcha's English is heard as Chinese syllables by the judge — both are genuine model defects.",
  lk_report: "Full report REPORT.md", lk_issue: "upstream bug #2861",
  foot: `Measurement: sherpa-onnx + librknnrt 2.2.0, dual pass (throughput + realtime pacing), 10 Hz resource sampling ·
    The presentation layer reads only <code>results/*.json</code>; rerun the harness, then <code>build_site.py</code> ·
    Datasets: LibriSpeech test-clean / FLEURS cmn_hans_cn (30 clips each) / in-house TTS texts`,
  meta: m => `${m.board} · RK3576 8GB · Debian 12 · kernel ${m.kernel} · measured ${m.d0} ~ ${m.d1}`,
},
};
let LANG = localStorage.getItem("lang") || "zh";
const t = k => I18N[LANG][k];

// 卡片与表格（模型名为语言中性，数值无关语言）
const byRtf = (a,b)=>a.rtf-b.rtf;
const asrFast=[...ASR].sort(byRtf)[0], ttsFast=[...TTS].sort(byRtf)[0];
const asrZhAcc=[...ASR].filter(r=>r.lang==='zh').sort((a,b)=>a.err-b.err)[0];
const ttsRt=[...TTS].sort((a,b)=>a.rt-b.rt)[0];
function drawCards(){
  document.getElementById('c-asr-fast').textContent=`${asrFast.model} ${asrFast.backend} RTF ${asrFast.rtf}`;
  document.getElementById('c-asr-acc').textContent=`${asrZhAcc.model} ${asrZhAcc.backend} ${pct(asrZhAcc.err)}`;
  document.getElementById('c-tts-fast').textContent=`${ttsFast.name} RTF ${ttsFast.rtf}`;
  document.getElementById('c-tts-rt').textContent=`${ttsRt.name} ${pct(ttsRt.rt)}`;
}

function fillTable(id, rows, render){
  const tb=document.querySelector(`#${id} tbody`);
  const draw=rs=>{ tb.innerHTML=rs.map(render).join(''); };
  draw(rows);
  document.querySelectorAll(`#${id} th[data-k]`).forEach(th=>{
    let asc=1;
    th.onclick=()=>{ const k=th.dataset.k;
      const rs=[...rows].sort((a,b)=>((a[k]??-1)-(b[k]??-1))*asc); asc*=-1; draw(rs); };
  });
}
fillTable('t-asr', ASR, r=>`<tr>
<td>${r.model}</td><td><span class="tag ${r.backend.includes('NPU')?'':'cpu'}">${r.backend}</span></td>
<td>${r.lang}</td><td>${pct(r.err)} <small>${r.errKind}</small></td><td><small>${pct(r.errLo)}–${pct(r.errHi)}</small></td>
<td>${r.rtf}</td><td>${r.rtfx}×</td><td>${r.ft50}</td><td>${r.ft95}</td>
<td>${r.cpu}%</td><td>${r.rss}</td><td>${r.npuMean==null?'–':r.npuMean+'% / '+r.npuPeak+'%'}</td></tr>`);
fillTable('t-tts', TTS, r=>`<tr>
<td>${r.name}</td><td>${r.arch}</td><td>${r.langs}</td><td>${r.n}</td>
<td>${r.rtf}</td><td>${r.rtfx}×</td><td>${pct(r.rt)}</td><td><small>${pct(r.rtLo)}–${pct(r.rtHi)}</small></td><td>${r.rss}</td></tr>`);

// 图表（标题随语言切换，故留引用）
const C={npu:'#0d9488', cpu:'#b45309'};
const label=r=>`${r.model.replace('Zipformer ','')} ${r.backend.includes('NPU')?'NPU':'CPU'} ${r.lang}`;
const charts=[];
function mkChart(id, key, rows, val, color, tip){
  const c=new Chart(document.getElementById(id),{type:'bar',
   data:{labels:rows.map(id==='ch-asr-err'||id==='ch-asr-rtf'?label:r=>r.name),
    datasets:[{data:rows.map(val),backgroundColor:rows.map(color)}]},
   options:{plugins:{legend:{display:false},title:{display:true,text:t(key)},
    ...(tip?{tooltip:{callbacks:{label:tip}}}:{})},
    scales:{y:{beginAtZero:true}},indexAxis:'y'}});
  charts.push({c,key});
}
function drawCharts(){
  mkChart('ch-asr-err','ch_asr_err',ASR,r=>r.err*100,r=>r.backend.includes('NPU')?C.npu:C.cpu,
    c=>` ${c.raw.toFixed(1)}% [${(ASR[c.dataIndex].errLo*100).toFixed(1)}–${(ASR[c.dataIndex].errHi*100).toFixed(1)}]`);
  mkChart('ch-asr-rtf','ch_asr_rtf',ASR,r=>r.rtf,r=>r.backend.includes('NPU')?C.npu:C.cpu);
  mkChart('ch-tts-rtf','ch_tts_rtf',TTS,r=>r.rtf,()=>'#155e75');
  mkChart('ch-tts-rt','ch_tts_rt',TTS,r=>r.rt*100,()=>'#7c3aed',
    c=>` ${c.raw.toFixed(1)}% [${(TTS[c.dataIndex].rtLo*100).toFixed(1)}–${(TTS[c.dataIndex].rtHi*100).toFixed(1)}]`);
}
drawCharts();

function drawAudio(){
  const withAudio=TTS.filter(r=>r.audio&&r.audio.length);
  document.getElementById('audio-sec').innerHTML = withAudio.length
    ? `<h2>${t('h_audio')}</h2>`+withAudio.map(r=>
      `<div class="audio-item"><b>${r.name}</b> `+r.audio.map(u=>
        `<div><code>${u.split('/').pop()}</code><audio controls preload="none" src="${u}"></audio></div>`).join('')
      +'</div>').join('')
    : '';
}

function applyLang(){
  document.documentElement.lang = LANG==='zh'?'zh':'en';
  document.getElementById('lang-btn').textContent = t('btn');
  document.querySelectorAll('[data-i18]').forEach(el=>{ el.textContent = t(el.dataset.i18); });
  document.querySelectorAll('[data-i18-html]').forEach(el=>{ el.innerHTML = t(el.dataset.i18Html); });
  document.getElementById('meta').textContent = I18N[LANG].meta(META);
  charts.forEach(({c,key})=>{ c.options.plugins.title.text=t(key); c.update(); });
  drawCards(); drawAudio();
}
function switchLang(){ LANG = LANG==='zh'?'en':'zh'; localStorage.setItem('lang',LANG); applyLang(); }
applyLang();
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
