import os
import re
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from survival_model import (
    MOBILITY_PROFILES,
    build_cost_grid,
    evaluate_path,
    stationary_cost,
)
from pathfinder import (
    astar_route,
    naive_shortest_route,
    suggest_safe_alternatives,
    extract_turn_by_turn_steps,
)

st.set_page_config(
    page_title="Mars EVA Hayatta Kalma Kılavuzu",
    page_icon="🚀",
    layout="wide",
)

st.markdown("""
<style>
.hud-banner {
    background: linear-gradient(90deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.85) 60%, rgba(255,107,53,0.18) 100%);
    border: 1px solid rgba(0, 245, 212, 0.3);
    border-left: 5px solid #ff6b35;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.hud-card {
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.hud-title {
    color: #38bdf8;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.step-box {
    background: rgba(2, 6, 23, 0.65);
    border-left: 3px solid #00f5d4;
    padding: 7px 10px;
    margin-bottom: 6px;
    border-radius: 4px;
    font-size: 0.86rem;
}
.step-box-ret {
    border-left-color: #fee440;
}
</style>
""", unsafe_allow_html=True)


def fmt_html_step(txt: str) -> str:
    txt = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", txt)
    txt = re.sub(r"`(.*?)`", r"<code>\1</code>", txt)
    return txt


@st.cache_data
def load_sites_meta():
    meta_path = os.path.join("processed", "sites.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_region_layers(region_key: str):
    npz_path = os.path.join("processed", f"{region_key}.npz")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


@st.cache_data(show_spinner=False)
def compute_mission_routes(
    region_key: str,
    cell_m: float,
    lander_xy: tuple,
    target_xy: tuple,
    max_slope: float,
    max_crater: float,
    min_temp_limit: float,
    out_hour: int,
    ret_hour: int,
    solar_storm: bool,
    thermal_w: float,
    rad_w: float,
    profile_name: str,
):
    layers = load_region_layers(region_key)
    out_model = build_cost_grid(
        layers, cell_m, max_slope, max_crater, out_hour, solar_storm,
        thermal_w, rad_w, profile_name, region_key=region_key, min_temp_limit_c=min_temp_limit,
    )
    ret_model = build_cost_grid(
        layers, cell_m, max_slope, max_crater, ret_hour, solar_storm,
        thermal_w, rad_w, profile_name, region_key=region_key, min_temp_limit_c=min_temp_limit,
    )
    path_out, err_out = astar_route(out_model["cost_grid"], lander_xy, target_xy)
    path_ret, err_ret = (
        astar_route(ret_model["cost_grid"], target_xy, lander_xy)
        if not err_out
        else (None, err_out)
    )
    naive_out, _ = naive_shortest_route(layers, lander_xy, target_xy)
    return out_model, ret_model, path_out, err_out, path_ret, err_ret, naive_out


def find_target_image(region_key: str, target: dict) -> str:
    explicit = target.get("image_path", "")
    if explicit and os.path.exists(explicit):
        return explicit
    tid = str(target.get("id", "")).lower()
    for ext in ["jpg", "jpeg", "png", "webp"]:
        cand = os.path.join("images", f"{region_key}_{tid}.{ext}")
        if os.path.exists(cand):
            return cand
    return ""


def build_survival_guide_markdown(
    region_name: str,
    profile_name: str,
    lander_xy: tuple,
    target_xy: tuple,
    target: dict,
    out_hour: int,
    ret_hour: int,
    solar_storm: bool,
    target_stats: dict,
    mission_summary: dict | None,
    out_steps: list,
    ret_steps: list,
) -> str:
    lines = [
        "# MARS YÜZEY KEŞİF HAYATTA KALMA KILAVUZU",
        f"- **Bölge:** {region_name}",
        f"- **Mobilite Profili:** {profile_name}",
        f"- **Roket Koordinatı:** {lander_xy} | **Hedef:** {target['name']} {target_xy}",
        f"- **Sol Saati:** Gidiş `{out_hour:02d}:00` → Dönüş `{ret_hour:02d}:00` | **Güneş Fırtınası:** {'AKTİF (4x)' if solar_storm else 'Normal'}",
        "",
        "## 1. Hedef Noktası Koşulları",
        f"- Yükseklik: {target_stats['elev']:.1f} m | Eğim: {target_stats['slope']:.1f}° | Krater Derinliği: {target_stats['depth']:.1f} m",
        f"- Sıcaklık: {target_stats['temp']:.1f} °C | Radyasyon: {target_stats['rad']:.2f} mSv/gün",
        "",
    ]
    if mission_summary:
        lines.extend([
            "## 2. Hayatta Kalma Bütçesi",
            f"- **Durum:** {'GÜVENLİ DÖNÜŞ MÜMKÜN' if mission_summary['survived'] else 'KRİTİK RİSK - BATARYA/O2 YETMİYOR'}",
            f"- Toplam Mesafe: {mission_summary['dist']:.0f} m | Toplam Süre: {mission_summary['time']:.0f} dk",
            f"- Batarya Tüketimi: {mission_summary['wh']:.0f} Wh (Kalan: %{mission_summary['rem_batt']:.0f})",
            "",
        ])
    if out_steps:
        lines.append("## 3. Adım Adım Gidiş Rotası (Roket → Hedef)")
        lines.extend([f"- {s}" for s in out_steps])
        lines.append("")
    if ret_steps:
        lines.append("## 4. Adım Adım Dönüş Rotası (Hedef → Roket)")
        lines.extend([f"- {s}" for s in ret_steps])
        lines.append("")
    lines.extend([
        "## 5. Saha Güvenlik Kuralları",
        f"- **Hedef Açıklaması:** {target['description']}",
        "- Kalan O2 veya batarya %35 sınırına inmeden dönüş yürüyüşünü başlatın.",
    ])
    return "\n".join(lines)


sites_meta = load_sites_meta()
if not sites_meta:
    st.error("`processed/sites.json` bulunamadı! Önce `.venv\\Scripts\\python.exe data_prep.py` çalıştırın.")
    st.stop()

st.markdown("""
<div class="hud-banner">
  <h2 style="margin:0; color:#f8fafc;">🚀 MARS EVA GÖREV KONTROLÜ & HAYATTA KALMA KILAVUZU</h2>
  <p style="margin:4px 0 0 0; color:#cbd5e1; font-size:0.92rem;">
    NASA Mars Trek 3D DEM Topografyası + REMS/MEDA Saatlik Termal & RAD Radyasyon Modeli | Hiper-Yerel (3 km × 3 km) Gidiş-Dönüş Planlayıcı
  </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🛰️ Temel Görev Ayarları")
    region_keys = list(sites_meta.keys())
    region_key = st.selectbox(
        "İniş Bölgesi",
        region_keys,
        format_func=lambda k: sites_meta[k]["display_name"],
    )
    meta = sites_meta[region_key]
    layers = load_region_layers(region_key)
    grid_n = meta["grid_size"]
    cell_m = float(meta["cell_size_m"])
    coords_label = "5.4°G, 137.8°D" if region_key == "gale" else "18.4°K, 77.5°D"
    st.caption(f"📍 Mars Konumu: {coords_label} | Ölçek: 3 km × 3 km (1 br = {cell_m:.0f} m)")

    use_custom_lander = st.checkbox("Roket koordinatını elle gir", value=False)
    if use_custom_lander:
        c_lx, c_ly = st.columns(2)
        lx = c_lx.number_input("Roket X", 0, grid_n - 1, int(meta["lander"][0]))
        ly = c_ly.number_input("Roket Y", 0, grid_n - 1, int(meta["lander"][1]))
        lander_xy = (int(lx), int(ly))
    else:
        lander_xy = tuple(meta["lander"])

    target_options = {t["id"]: t for t in meta["targets"]}
    target_mode = st.radio("Hedef Seçimi", ["Bilimsel Hedefler", "Özel Koordinat"], horizontal=True)
    if target_mode == "Bilimsel Hedefler":
        sel_tid = st.selectbox(
            "Hedef Noktası",
            list(target_options.keys()),
            format_func=lambda tid: f"{tid}: {target_options[tid]['name']}",
        )
        sel_target = target_options[sel_tid]
        target_xy = (int(sel_target["x"]), int(sel_target["y"]))
    else:
        c_tx, c_ty = st.columns(2)
        tx = c_tx.number_input("Hedef X", 0, grid_n - 1, int(meta["targets"][0]["x"]))
        ty = c_ty.number_input("Hedef Y", 0, grid_n - 1, int(meta["targets"][0]["y"]))
        target_xy = (int(tx), int(ty))
        sel_target = {
            "id": "ÖZEL",
            "name": f"Özel Hedef ({tx}, {ty})",
            "description": "Kullanıcı tarafından belirlenen özel yüzey keşif koordinatı.",
            "image_path": "",
        }

    profile_name = st.selectbox("Astronot / Ekipman Profili", list(MOBILITY_PROFILES.keys()))
    default_temp_lim = int(MOBILITY_PROFILES[profile_name]["min_temp_c"])

    out_hour = st.slider("Mars Gidiş Saati (Sol Saati)", 0, 23, 13, format="%02d:00")
    return_colder = st.checkbox("Dönüşte Geç Saat Soğuması (+5 saat)", value=True)
    ret_hour = (out_hour + 5) % 24 if return_colder else out_hour
    st.caption(f"🕒 Gidiş: `{out_hour:02d}:00` | Dönüş: `{ret_hour:02d}:00`")

    solar_storm = st.checkbox("☀️ Güneş Fırtınası (4x Radyasyon)", value=False)

    with st.expander("⚙️ Gelişmiş Kıyafet & Arazi Limitleri", expanded=False):
        max_slope = st.slider("Maks. Tırmanılabilir Eğim (°)", 5, 45, 18)
        max_crater = st.slider("Maks. Krater Derinliği (m)", 3, 40, 9)
        min_temp_limit = st.slider("Giysi Min. Sıcaklık Sınırı (°C)", -90, -20, default_temp_lim)
        suit_battery_wh = st.slider("Batarya Kapasitesi (Wh)", 300, 3000, 1200, step=50)
        suit_o2_min = st.slider("Oksijen Süresi (dk)", 60, 480, 240, step=15)
        sample_time_min = st.slider("Veri Toplama Süresi (dk)", 5, 60, 15, step=5)
        thermal_w = st.slider("Termal Koruma Ağırlığı", 0.0, 5.0, 2.0, step=0.5)
        rad_w = st.slider("Radyasyon Koruma Ağırlığı", 0.0, 5.0, 1.5, step=0.5)

out_model, ret_model, path_out, err_out, path_ret, err_ret, naive_out = compute_mission_routes(
    region_key, cell_m, lander_xy, target_xy, max_slope, max_crater,
    min_temp_limit, out_hour, ret_hour, solar_storm, thermal_w, rad_w, profile_name
)

alt_suggestions = {}
mission_summary = None
out_steps, ret_steps = [], []

if err_out or err_ret:
    st.error(f"🚫 ROTA HESAPLANAMADI: {err_out or err_ret}")
    alt_suggestions = suggest_safe_alternatives(
        layers, out_model["cost_grid"], lander_xy, target_xy,
        max_slope, max_crater, temp_map=out_model["temp_map"], temp_limit_c=min_temp_limit
    )
    msg_parts = []
    if alt_suggestions.get("thermal_block"):
        msg_parts.append("• **Aşırı Soğuk Engeli:** Seçilen Sol saatinde başlangıç veya hedef giysi sınırının altında. Daha sıcak bir saate (**10:00–16:00**) geçin veya *Rover Destekli* profili seçin.")
    if "nearest_safe_start" in alt_suggestions:
        msg_parts.append(f"• Roket için en yakın güvenli iniş noktası: **{alt_suggestions['nearest_safe_start']}**")
    if "nearest_reachable_pt" in alt_suggestions:
        pt = alt_suggestions["nearest_reachable_pt"]
        gap_m = alt_suggestions["nearest_reachable_gap_cells"] * cell_m
        msg_parts.append(f"• Mevcut kısıtlarla hedefe en yakın güvenli ulaşım noktası: **{pt}** (hedefe `{gap_m:.0f} m` mesafede — mor çizgi).")
    if "min_slope_needed" in alt_suggestions:
        msg_parts.append(f"• Tam ulaşım için minimum ayar: **Eğim ≥ {alt_suggestions['min_slope_needed']}°** ve **Krater Derinliği ≥ {alt_suggestions['min_crater_needed']} m**.")
    if msg_parts:
        st.info("💡 **Doğru Yol Önerisi:**\n" + "\n".join(msg_parts))
else:
    eval_out = evaluate_path(path_out, layers, cell_m, out_model["temp_map"], out_model["rad_map"], solar_storm=solar_storm, profile_name=profile_name)
    eval_ret = evaluate_path(path_ret, layers, cell_m, ret_model["temp_map"], ret_model["rad_map"], solar_storm=solar_storm, profile_name=profile_name)
    stat_wh, stat_msv = stationary_cost(
        float(out_model["temp_map"][target_xy[1], target_xy[0]]),
        float(layers["radiation"][target_xy[1], target_xy[0]]),
        sample_time_min, solar_storm=solar_storm, profile_name=profile_name,
    )
    total_dist = eval_out["distance_m"] + eval_ret["distance_m"]
    total_time = eval_out["duration_min"] + sample_time_min + eval_ret["duration_min"]
    total_wh = eval_out["energy_wh"] + stat_wh + eval_ret["energy_wh"]
    total_msv = eval_out["radiation_msv"] + stat_msv + eval_ret["radiation_msv"]
    rem_batt_pct = max(0.0, (suit_battery_wh - total_wh) / suit_battery_wh * 100.0)
    rem_o2_pct = max(0.0, (suit_o2_min - total_time) / suit_o2_min * 100.0)
    survived = (total_wh <= suit_battery_wh) and (total_time <= suit_o2_min)
    mission_summary = {
        "survived": survived, "dist": total_dist, "time": total_time, "wh": total_wh,
        "rem_batt": rem_batt_pct, "rem_o2": rem_o2_pct,
        "min_temp": min(eval_out["min_temp_c"], eval_ret["min_temp_c"]), "rad": total_msv,
    }
    out_steps = extract_turn_by_turn_steps(path_out, cell_m, out_model["temp_map"], layers["slope_deg"])
    ret_steps = extract_turn_by_turn_steps(path_ret, cell_m, ret_model["temp_map"], layers["slope_deg"])

col_map, col_hud = st.columns([2.15, 1.0], gap="medium")

with col_map:
    c_sel1, c_sel2, c_sel3 = st.columns([1.8, 1.4, 1.1])
    with c_sel1:
        layer_choice = st.selectbox(
            "Katman",
            [
                "Yüzey Sıcaklığı (°C)",
                "Krater Derinliği (m)",
                "Topografya / Yükseklik (m)",
                "Radyasyon Dozu (mSv/gün)",
                "Geçilebilirlik / Engel Maskesi",
                "Jeolojik Bölgeler",
            ],
            label_visibility="collapsed",
        )
    with c_sel2:
        view_mode = st.radio("Görünüm", ["2D Taktik Harita", "3D Arazi"], horizontal=True, label_visibility="collapsed")
    with c_sel3:
        show_naive = st.checkbox("Kör Rota Kıyasla", value=True)

    if layer_choice == "Yüzey Sıcaklığı (°C)":
        z_data, colorscale, unit = out_model["temp_map"], "Turbo", "°C"
    elif layer_choice == "Krater Derinliği (m)":
        z_data, colorscale, unit = layers["crater_depth_m"], "Inferno", "m"
    elif layer_choice == "Topografya / Yükseklik (m)":
        z_data, colorscale, unit = layers["elevation"], "Turbo", "m"
    elif layer_choice == "Radyasyon Dozu (mSv/gün)":
        z_data, colorscale, unit = out_model["rad_map"], "Plasma", "mSv/gün"
    elif layer_choice == "Geçilebilirlik / Engel Maskesi":
        z_data = np.where(out_model["blocked"], 1.0, 0.0)
        colorscale, unit = [[0, "#065f46"], [1, "#dc2626"]], "0=Açık, 1=Engel"
    else:
        z_data, colorscale, unit = layers["geology_mask"], "Viridis", "Sınıf"

    fig = go.Figure()

    if view_mode == "2D Taktik Harita":
        fig.add_trace(go.Heatmap(
            z=z_data, colorscale=colorscale, colorbar=dict(title=unit, thickness=14),
            hovertemplate="X: %{x}<br>Y: %{y}<br>Değer: %{z:.2f} " + unit + "<extra></extra>",
        ))
        if show_naive and naive_out:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in naive_out], y=[p[1] for p in naive_out],
                mode="lines", name="Kör En Kısa Rota", line=dict(color="#fb923c", width=2.5, dash="dot"),
            ))
        if path_out:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in path_out], y=[p[1] for p in path_out],
                mode="lines", name="Güvenli Gidiş", line=dict(color="#00f5d4", width=4.5),
            ))
        if path_ret:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in path_ret], y=[p[1] for p in path_ret],
                mode="lines", name="Güvenli Dönüş", line=dict(color="#fee440", width=3, dash="dash"),
            ))
        if alt_suggestions.get("nearest_reachable_path"):
            alt_p = alt_suggestions["nearest_reachable_path"]
            fig.add_trace(go.Scatter(
                x=[p[0] for p in alt_p], y=[p[1] for p in alt_p],
                mode="lines", name="Alternatif Yaklaşım", line=dict(color="#c77dff", width=3, dash="dashdot"),
            ))
        fig.add_trace(go.Scatter(
            x=[lander_xy[0]], y=[lander_xy[1]], mode="markers+text", text=["🚀 Roket"], textposition="top center",
            textfont=dict(color="#ffffff", size=12),
            marker=dict(size=14, color="#22c55e", symbol="triangle-up", line=dict(color="black", width=1.5)),
            name="Roket",
        ))
        for tid, tinfo in target_options.items():
            fig.add_trace(go.Scatter(
                x=[tinfo["x"]], y=[tinfo["y"]], mode="markers+text", text=[tid], textposition="bottom center",
                textfont=dict(color="#ffffff", size=12),
                marker=dict(size=11, color="#f43f5e", symbol="diamond", line=dict(color="white", width=1.5)),
                name=f"{tid}",
            ))
        if target_mode != "Bilimsel Hedefler":
            fig.add_trace(go.Scatter(
                x=[target_xy[0]], y=[target_xy[1]], mode="markers+text", text=["🎯 Hedef"], textposition="top center",
                textfont=dict(color="#ffffff", size=12),
                marker=dict(size=14, color="#f43f5e", symbol="star"), name="Özel Hedef",
            ))
        fig.update_layout(
            height=680,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title=f"X (1 br = {cell_m:.0f} m)", constrain="domain"),
            yaxis=dict(title=f"Y (1 br = {cell_m:.0f} m)", scaleanchor="x", scaleratio=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
            margin=dict(l=10, r=10, t=35, b=10),
        )
    else:
        elev = layers["elevation"]
        coords_sub = np.arange(0, grid_n, 2)
        fig.add_trace(go.Surface(
            x=coords_sub, y=coords_sub, z=elev[::2, ::2],
            surfacecolor=z_data[::2, ::2].astype(float), colorscale=colorscale,
            colorbar=dict(title=unit, thickness=14), opacity=0.96,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Yükseklik: %{z:.1f} m<extra></extra>",
        ))
        if show_naive and naive_out:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in naive_out], y=[p[1] for p in naive_out],
                z=[float(elev[p[1], p[0]]) + 2.5 for p in naive_out],
                mode="lines", name="Kör En Kısa", line=dict(color="#fb923c", width=4, dash="dot"),
            ))
        if path_out:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in path_out], y=[p[1] for p in path_out],
                z=[float(elev[p[1], p[0]]) + 4.5 for p in path_out],
                mode="lines", name="Güvenli Gidiş", line=dict(color="#00f5d4", width=7),
            ))
        if path_ret:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in path_ret], y=[p[1] for p in path_ret],
                z=[float(elev[p[1], p[0]]) + 6.0 for p in path_ret],
                mode="lines", name="Güvenli Dönüş", line=dict(color="#fee440", width=5, dash="dash"),
            ))
        if alt_suggestions.get("nearest_reachable_path"):
            alt_p = alt_suggestions["nearest_reachable_path"]
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in alt_p], y=[p[1] for p in alt_p],
                z=[float(elev[p[1], p[0]]) + 4.5 for p in alt_p],
                mode="lines", name="Alternatif", line=dict(color="#c77dff", width=6, dash="dashdot"),
            ))
        fig.add_trace(go.Scatter3d(
            x=[lander_xy[0]], y=[lander_xy[1]], z=[float(elev[lander_xy[1], lander_xy[0]]) + 9.0],
            mode="markers+text", text=["🚀 Roket"], textposition="top center",
            textfont=dict(color="#00f5d4", size=13), marker=dict(size=7, color="#22c55e", symbol="diamond"),
            name="Roket",
        ))
        for tid, tinfo in target_options.items():
            tx_i, ty_i = int(tinfo["x"]), int(tinfo["y"])
            fig.add_trace(go.Scatter3d(
                x=[tx_i], y=[ty_i], z=[float(elev[ty_i, tx_i]) + 9.0],
                mode="markers+text", text=[tid], textposition="top center",
                textfont=dict(color="#ff2a85", size=13), marker=dict(size=6, color="#ff2a85", symbol="diamond"),
                name=tid,
            ))
        fig.update_layout(
            height=680,
            dragmode="turntable",
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis_title=f"X (1 br={cell_m:.0f}m)",
                yaxis_title=f"Y (1 br={cell_m:.0f}m)",
                zaxis_title="Yükseklik (m)",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.18),
                camera=dict(eye=dict(x=1.05, y=-1.05, z=0.75), up=dict(x=0, y=0, z=1)),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
            margin=dict(l=5, r=5, t=35, b=5),
        )

    st.plotly_chart(fig, use_container_width=True)

with col_hud:
    if mission_summary:
        if mission_summary["survived"]:
            st.success("✅ GÜVENLİ DÖNÜŞ MÜMKÜN")
        else:
            st.error("⚠️ DÖNÜŞTE BATARYA/O2 TÜKENİYOR!")

        m1, m2 = st.columns(2)
        m1.metric("Toplam Mesafe", f"{mission_summary['dist']:.0f} m")
        m2.metric("Süre / Kalan O2", f"{mission_summary['time']:.0f} dk", f"%{mission_summary['rem_o2']:.0f} O2")
        m3, m4 = st.columns(2)
        m3.metric("Batarya", f"{mission_summary['wh']:.0f} Wh", f"%{mission_summary['rem_batt']:.0f} kaldı")
        m4.metric("Min Sıcaklık", f"{mission_summary['min_temp']:.1f} °C", f"{mission_summary['rad']:.3f} mSv")

        st.markdown('<div class="hud-card"><div class="hud-title">🧭 GİDİŞ YOL TARİFİ (ROKET → HEDEF)</div>', unsafe_allow_html=True)
        for s in out_steps:
            st.markdown(f'<div class="step-box">{fmt_html_step(s)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="hud-card"><div class="hud-title">🔙 DÖNÜŞ YOL TARİFİ (HEDEF → ROKET)</div>', unsafe_allow_html=True)
        for s in ret_steps:
            st.markdown(f'<div class="step-box step-box-ret">{fmt_html_step(s)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="hud-card"><div class="hud-title">⚠️ ROTA BEKLEMEDE</div>Güvenli geçiş için soldan Sol saatini/koordinatı değiştirin veya Gelişmiş Ayarlardan limitleri gevşetin.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📊 Rota Telemetrisi & Karşılaştırma",
    "🧭 Detaylı Hayatta Kalma Kılavuzu & Fotoğraf",
    "ℹ️ NASA Veri Kaynakları & Şeffaflık",
])

with tab1:
    if not (err_out or err_ret):
        df_out = pd.DataFrame(eval_out["telemetry"])
        df_ret = pd.DataFrame(eval_ret["telemetry"])
        df_ret["dist_m"] = df_ret["dist_m"] + eval_out["distance_m"]
        df_ret["cum_energy_wh"] = df_ret["cum_energy_wh"] + eval_out["energy_wh"] + stat_wh
        df_all = pd.concat([df_out, df_ret], ignore_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Rota Boyunca Sıcaklık & Eğim")
            st.line_chart(df_all.set_index("dist_m")[["temp_c", "slope_deg"]])
        with col_b:
            st.subheader("Kümülatif Batarya Tüketimi (Wh)")
            st.line_chart(df_all.set_index("dist_m")[["cum_energy_wh"]])

        if show_naive and naive_out:
            eval_naive = evaluate_path(naive_out, layers, cell_m, out_model["temp_map"], out_model["rad_map"], solar_storm=solar_storm, profile_name=profile_name)
            st.markdown("#### Güvenli Rota vs. Kör En Kısa Rota (Gidiş Karşılaştırması)")
            comp_df = pd.DataFrame([
                {
                    "Rota": "Güvenli Hayatta Kalma Rotası",
                    "Mesafe (m)": round(eval_out["distance_m"], 1),
                    "Enerji (Wh)": round(eval_out["energy_wh"], 1),
                    "Maks. Eğim (°)": round(eval_out["max_slope_deg"], 1),
                    "Maks. Krater Derinliği (m)": round(eval_out["max_crater_depth_m"], 1),
                    "En Soğuk (°C)": round(eval_out["min_temp_c"], 1),
                },
                {
                    "Rota": "Kör En Kısa Rota",
                    "Mesafe (m)": round(eval_naive["distance_m"], 1),
                    "Enerji (Wh)": round(eval_naive["energy_wh"], 1),
                    "Maks. Eğim (°)": round(eval_naive["max_slope_deg"], 1),
                    "Maks. Krater Derinliği (m)": round(eval_naive["max_crater_depth_m"], 1),
                    "En Soğuk (°C)": round(eval_naive["min_temp_c"], 1),
                },
            ])
            st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Rota tamamlanamadığı için telemetri grafiği oluşturulamadı.")

with tab2:
    tx, ty = target_xy
    target_stats = {
        "elev": float(layers["elevation"][ty, tx]),
        "slope": float(layers["slope_deg"][ty, tx]),
        "depth": float(layers["crater_depth_m"][ty, tx]),
        "temp": float(out_model["temp_map"][ty, tx]),
        "rad": float(out_model["rad_map"][ty, tx]),
    }

    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        st.subheader(f"Hedef Brifingi: {sel_target['name']}")
        st.write(sel_target["description"])
        st.write(
            f"**Hedef Koşulları:** Yükseklik `{target_stats['elev']:.1f} m` | "
            f"Eğim `{target_stats['slope']:.1f}°` | "
            f"Krater Derinliği `{target_stats['depth']:.1f} m` | "
            f"Sıcaklık `{target_stats['temp']:.1f} °C` | "
            f"Radyasyon `{target_stats['rad']:.2f} mSv/gün`"
        )

        if mission_summary and mission_summary["min_temp"] < -55.0:
            st.warning("**Termal Donma Uyarısı:** Dönüş veya gölge sıcaklığı -55°C altında. Krater diplerinde duraklama süresini kısaltın.")
        else:
            st.success("**Termal Durum:** Seçilen Sol penceresi uygun; krater gölgelerinde ani soğumaya karşı batarya rezervi bırakın.")

        if solar_storm:
            st.error("**Güneş Fırtınası Protokolü:** Radyasyon 4x! Açık sırtlarda kalmayın, krater duvarı dibi doğal korumasını kullanın.")

        guide_md = build_survival_guide_markdown(
            meta["display_name"], profile_name, lander_xy, target_xy, sel_target,
            out_hour, ret_hour, solar_storm, target_stats, mission_summary, out_steps, ret_steps
        )
        st.download_button(
            "📥 Tam Görev Kılavuzunu İndir (.md)",
            data=guide_md,
            file_name=f"mars_eva_kilavuz_{region_key}_{sel_target['id']}.md",
            mime="text/markdown",
        )

    with col_g2:
        img_file = find_target_image(region_key, sel_target)
        if img_file:
            st.image(img_file, caption=f"{sel_target['name']} — NASA Yüzey Görünümü", use_container_width=True)
        else:
            st.info(
                f"📸 **Yüzey Fotoğrafı Slotu:**\n\n"
                f"`images/{region_key}_{sel_target['id'].lower()}.jpg` eklendiğinde gerçek NASA rover görüntüsü burada açılır."
            )

with tab3:
    st.markdown(f"""
    ### Veri Kaynakları ve Mühendislik Modeli ({meta['display_name']})
    1. **Topografya / Krater / Eğim (Ham NASA Verisi):** Doğrudan NASA JPL Mars Trek 3D DEM yüzey ağından (`.stl`, `{grid_n}×{grid_n}` grid, `1 br = {cell_m:.0f} m`) gerçek XY/Z oranı korunarak çıkarılmıştır.
    2. **Saatlik Termal Katman (NASA REMS / MEDA Hibrit Modeli):** Curiosity REMS (Gale) ve Perseverance MEDA (Jezero) 24 saatlik yüzey sıcaklık eğrisi temel alınarak, seçilen Sol saatindeki güneş azimut/yükseklik açısına göre dinamik gölgelenme (`hillshade`) ve krater içi soğuk hava birikimiyle hesaplanmıştır.
    3. **Radyasyon Katmanı (NASA RAD Modeli):** Curiosity RAD (~0.64 mSv/gün) gerçek yüzey radyasyonu temel alınarak topografik gökyüzü açıklığına göre modellenmiştir.
    """)