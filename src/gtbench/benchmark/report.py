from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .regression import VARIANT_LABELS


def write_json(results: dict, path):
    Path(path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )


def _f(v, nd=4):
    return "N/A" if v is None else f"{v:.{nd}f}"


def write_markdown(results: dict, path):
    m = results["meta"]
    v = results["variants"]
    tr = results["tracking"]
    nb = results.get("native_benchmark", {})
    best = min((x["absrel"] for x in v.values() if x.get("absrel") is not None), default=None)
    fv = results["meta"]["final_variant"]
    nv = results["meta"]["naive_variant"]
    baseline = v.get("stereo_only", {}).get("absrel")
    fused = v.get(fv, {}).get("absrel")
    naive = v.get(nv, {}).get("absrel")
    cov_base = v.get("stereo_only", {}).get("coverage")
    cov_fused = v.get(fv, {}).get("coverage")

    L = []
    A = L.append
    A("# PerceptionGT-Bench 评测报告")
    A("")
    A(f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ seed={m['seed']} ｜ "
      f"分辨率 {m['resolution'][0]}×{m['resolution'][1]} ｜ {m['frames']} 帧深度评测 + {m['track_frames']} 帧跟踪评测")
    A("")
    A("## 1. 评测设置")
    A("")
    A(f"- 场景切片：{', '.join(m['slices'])}")
    A(f"- 评测方案：{', '.join(VARIANT_LABELS[k] for k in m['variants'])}")
    A("- 数据：仓库内置确定性合成双目渲染器，深度真值精确已知；不依赖任何需注册下载的数据集")
    A("- 指标口径：仅在有效像素上计算，深度 = 相机 Z 向距离（米）")
    A("")
    A("## 2. 伪真值源 / 融合方案对比")
    A("")
    A("| 方案 | AbsRel ↓ | RMSE ↓ | SIlog ↓ | δ<1.25 ↑ | 有效覆盖率 |")
    A("|---|---|---|---|---|---|")
    for k, x in v.items():
        mark = " **← 最优**" if x.get("absrel") == best else ""
        A(f"| {x['label']}{mark} | {_f(x.get('absrel'),5)} | {_f(x.get('rmse'))} | "
          f"{_f(x.get('silog'))} | {_f(x.get('delta_1_25'))} | {_f(x.get('coverage'))} |")
    A("")
    A("**关键发现：不做源级可靠性校准，融合反而会变差。**")
    A("")
    if naive is not None and baseline:
        A(f"- 三个伪真值源的 confidence 来自完全不同的口径（峰值比 / 重投影误差 / 手工先验），"
          f"直接等权融合会让低精度源把深度拉偏：等权融合 AbsRel = **{_f(naive,5)}**，"
          f"比单用双目（{_f(baseline,5)}）恶化 {abs((baseline-naive)/baseline*100):.1f}%。")
    if fused is not None:
        A(f"- 用独立留出标定集估计各源经验误差并逆误差平方定权后，最终方案 AbsRel = **{_f(fused,5)}**，"
          f"有效覆盖率由 {_f(cov_base)} 提升到 **{_f(cov_fused)}**。")
        cd = results.get("common_domain", {})
        A("- 融合的价值不是「把单源指标刷得更好」，而是**在双目失效区域（弱纹理、遮挡、远距离）"
          "用 SfM / 先验补上伪标签**。代价是共同有效域上 AbsRel 由 "
          f"{_f(cd.get('stereo_only'),5)} 升到 {_f(cd.get(fv),5)}，换得覆盖率 "
          f"{_f(cov_base)} → {_f(cov_fused)}；该权衡可用置信度阈值按业务调节（见 2.2）。")
    A("")
    A("### 2.1 评测口径：为什么不能直接比全图 AbsRel")
    A("")
    A(f"`stereo_only` 只在 **{_f(cov_base)}** 的像素上给出预测，而最终方案覆盖 **{_f(cov_fused)}**。"
      "有效像素集不同，AbsRel 就不可直接比较——稀疏方案把弱纹理、遮挡等困难区域排除在外，天然显得更准。"
      "真值系统评测必须同时报覆盖率，并给出共同有效域上的公平对比。")
    A("")
    A("| 方案 | 全图 AbsRel | 共同有效域 AbsRel |")
    A("|---|---|---|")
    cd = results.get("common_domain", {})
    for k, x in v.items():
        A(f"| {x['label']} | {_f(x.get('absrel'),5)} | {_f(cd.get(k),5)} |")
    A("")
    A("")
    A("### 2.2 覆盖率-精度权衡（risk-coverage 曲线）")
    A("")
    A("对最终方案按置信度阈值逐步放宽，得到可调节的精度 / 覆盖曲线：")
    A("")
    A("| 置信度阈值 | 覆盖率 | AbsRel |")
    A("|---|---|---|")
    for r in results.get("risk_coverage", []):
        A(f"| ≥ {r['confidence_threshold']:.1f} | {_f(r['coverage'])} | {_f(r['absrel'],5)} |")
    A("")
    A("**工程含义**：真值系统不需要固定一个阈值。生成训练伪标签时取高置信阈值保精度；"
      "做难例挖掘与补采清单时取低置信阈值保覆盖，把高误差区域交给 LiDAR / 人工复核。"
      "这条曲线就是「评测结果如何反哺数据闭环」的具体接口。")
    A("")
    A("## 3. 源级可靠性校准")
    A("")
    sc = results.get("source_calibration", {})
    A(f"留出标定集：{sc.get('calibration_frames')} 帧，seed={sc.get('calibration_seed')}（与评测帧完全隔离）。"
      f"权重 = 1/(0.05+标定 AbsRel)²，归一化到最优源为 1.0。")
    A("")
    A("| 伪真值源 | 标定集 AbsRel | 可靠性权重 |")
    A("|---|---|---|")
    for name, err in (sc.get("source_absrel") or {}).items():
        A(f"| {name} | {_f(err,5)} | {_f((sc.get('reliability_weights') or {}).get(name),5)} |")
    A("")
    A("## 4. 场景切片难度排序（最终方案）")
    A("")
    A("| 排名 | 场景切片 | AbsRel ↓ | 帧数 | 考察点 |")
    A("|---|---|---|---|---|")
    for i, s in enumerate(results["slice_ranking"], 1):
        A(f"| {i} | {s['slice']} | {_f(s['absrel'],5)} | {s['frames']} | {s['description']} |")
    A("")
    A("## 5. 3D 多目标跟踪评测")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---|")
    for k in ("MOTA", "MOTP", "IDF1", "IDSW", "TP", "FP", "FN", "DetA", "AssA", "HOTA_single_threshold"):
        A(f"| {k} | {tr.get(k)} |")
    A("")
    A("> HOTA 为单一 IoU 阈值下的实现（未做多阈值积分），与官方 HOTA 数值口径不完全等同。")
    A("")
    A("## 6. 置信度校准（最终方案，示例帧）")
    A("")
    if results.get("calibration"):
        A("| 置信度区间 | 像素数 | 实测平均 AbsRel |")
        A("|---|---|---|")
        for c in results["calibration"]:
            A(f"| {c['conf_bin']} | {c['pixels']} | {_f(c['mean_absrel'])} |")
        A("")
        A("置信度越高误差应越低；若出现反转，说明该源的置信度定义需要重新标定。")
    else:
        A("本帧有效像素不足，未产生校准分箱。")
    A("")
    A("## 7. Hard-case 挖掘（Top 10）")
    A("")
    A("| 切片 | 帧 | 图块 (x,y) | AbsRel | 覆盖率 |")
    A("|---|---|---|---|---|")
    for b in results["badcases"]:
        A(f"| {b['slice']} | {b['frame_idx']} | ({b['x']},{b['y']}) | {_f(b['absrel'])} | {_f(b['coverage'],3)} |")
    A("")
    A("## 8. C++17 内核")
    A("")
    A(f"- 原生库加载：{'成功' if results['native'].get('native_loaded') else '未加载（使用 NumPy 回退）'}")
    if nb.get("median3"):
        z = nb["median3"]
        A(f"- 3×1 中值后处理（单算子）：C++ {_f(z.get('native_ms'),3)} ms / NumPy {_f(z.get('numpy_ms'),3)} ms "
          f"→ {_f(z.get('speedup'),2)}×")
    if nb.get("combo"):
        z = nb["combo"]
        A(f"- **LRC + 置信度 + AbsRel 组合链路**：C++ {_f(z.get('native_ms'),3)} ms / "
          f"NumPy {_f(z.get('numpy_ms'),3)} ms → **{_f(z.get('speedup'),2)}×**")
    A("")
    if nb.get("median3") and nb["median3"].get("speedup", 1) < 1:
        A("> 性能边界分析：单算子未体现优势，是因为小数组上 ctypes 调用与输出数组分配成为主导；"
          "组合链路把多步运算收进同一侧、减少 Python↔C 边界穿越与临时数组，才获得加速。"
          "**native 化的收益来自批处理与减少边界开销，不是单个循环比 NumPy 快。**")
        A("")
    A("## 9. 已知限制（如实说明）")
    A("")
    A("1. 评测数据为程序化合成场景，**不是 KITTI 真实数据**；仓库提供 `KITTIAdapter` 接口，"
      "切换到真实数据只需实现字段映射，下游模块无需改动。")
    A("2. 基础模型支路默认使用**手写的离线单目先验**，不是 Depth Anything V2 / SAM；"
      "两个真实大模型的 backend 为显式集成占位，未下载权重、未参与本次数值。")
    A("3. SfM 支路由双目角点三角化实现（含本质矩阵校验与重投影误差），"
      "不是完整 COLMAP 增量式重建 pipeline。")
    A("4. C++ 内核覆盖立体后处理与指标计算，未覆盖立体匹配代价体本身。")
    A("")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def write_html(results: dict, path):
    md_path = Path(path).with_suffix(".md")
    rows = "".join(
        f"<tr><td>{x['label']}</td><td>{_f(x.get('absrel'),5)}</td>"
        f"<td>{_f(x.get('rmse'))}</td><td>{_f(x.get('delta_1_25'))}</td>"
        f"<td>{_f(x.get('coverage'))}</td></tr>"
        for x in results["variants"].values()
    )
    slices = "".join(
        f"<tr><td>{s['slice']}</td><td>{_f(s['absrel'],5)}</td><td>{s['description']}</td></tr>"
        for s in results["slice_ranking"]
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>PerceptionGT-Bench 评测报告</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;
color:#20242a;line-height:1.6}}h1{{font-size:24px}}h2{{font-size:17px;margin-top:32px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #dfe3e8;
padding:8px 10px;text-align:left}}th{{background:#f4f6f8}}img{{max-width:100%;border:1px solid #dfe3e8}}
.note{{background:#f4f6f8;padding:12px 16px;border-radius:8px;font-size:14px}}</style></head><body>
<h1>PerceptionGT-Bench 评测报告</h1>
<p class="note">完整文字版见 <a href="{md_path.name}">{md_path.name}</a>；
本页为可视化摘要，图表由 <code>viz.py</code> 生成。</p>
<h2>方案对比</h2><table><tr><th>方案</th><th>AbsRel ↓</th><th>RMSE ↓</th><th>δ&lt;1.25 ↑</th><th>覆盖率</th></tr>{rows}</table>
<h2>场景切片难度</h2><table><tr><th>切片</th><th>AbsRel ↓</th><th>考察点</th></tr>{slices}</table>
<h2>可视化</h2><p><img src="depth_panel.png" alt="深度对比"><br><img src="slice_bars.png" alt="切片对比"><br>
<img src="variant_bars.png" alt="方案对比"><br><img src="risk_coverage.png" alt="覆盖率-精度权衡"></p>
<h2>3D 跟踪</h2><p>MOTA {results['tracking']['MOTA']} ｜ IDF1 {results['tracking']['IDF1']} ｜
IDSW {results['tracking']['IDSW']} ｜ HOTA(单阈值) {results['tracking']['HOTA_single_threshold']}</p>
</body></html>"""
    Path(path).write_text(html, encoding="utf-8")
