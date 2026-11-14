import os
import re
import json
import math
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
    page_title="Mars EVA Mission Control & Survival Guide",
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


def translate_nav_step(step_txt: str, to_en: bool) -> str:
    if not to_en:
        return step_txt
    repl = [
        ("Kuzeydoğu", "Northeast"), ("Kuzeybatı", "Northwest"),
        ("Güneydoğu", "Southeast"), ("Güneybatı", "Southwest"),
        ("Kuzey", "North"), ("Güney", "South"), ("Doğu", "East"), ("Batı", "West"),
        ("yönünde", "heading"), ("ilerle", "advance"),
        ("Dik eğim", "Steep slope"), ("Soğuk", "Cold"),
    ]
    res = step_txt
    for tr, en in repl:
        res = res.replace(tr, en)
    return res


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


# 3D ROKET
def build_3d_rocket(lx: float, ly: float, lz: float, h_m: float = 38.0, r_m: float = 1.8):
    lines_x, lines_y, lines_z = [], [], []
    angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    tip_z = lz + h_m
    shoulder_z = lz + h_m * 0.72
    base_z = lz + h_m * 0.18

    for a in angles:
        bx = lx + r_m * math.cos(a)
        by = ly + r_m * math.sin(a)
        leg_x = lx + (r_m * 2.6) * math.cos(a)
        leg_y = ly + (r_m * 2.6) * math.sin(a)
        lines_x.extend([bx, leg_x, None])
        lines_y.extend([by, leg_y, None])
        lines_z.extend([base_z, lz, None])
        lines_x.extend([bx, bx, lx, None])
        lines_y.extend([by, by, ly, None])
        lines_z.extend([base_z, shoulder_z, tip_z, None])

    for z_ring in [base_z, shoulder_z]:
        ring_pts = 16
        for i in range(ring_pts + 1):
            theta = 2 * math.pi * (i / ring_pts)
            lines_x.append(lx + r_m * math.cos(theta))
            lines_y.append(ly + r_m * math.sin(theta))
            lines_z.append(z_ring)
        lines_x.append(None); lines_y.append(None); lines_z.append(None)

    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode="lines",
        line=dict(color="#00f5d4", width=5),
        name="3D Lander",
        hoverinfo="name",
    )


# 3D ASTRONOT
def build_3d_astronaut(ax: float, ay: float, az: float, h_m: float = 24.0):
    lines_x, lines_y, lines_z = [], [], []
    head_z = az + h_m * 0.88
    neck_z = az + h_m * 0.75
    pelvis_z = az + h_m * 0.42
    w_m = 1.1

    lines_x.extend([ax - w_m, ax, ax + w_m, None])
    lines_y.extend([ay, ay, ay, None])
    lines_z.extend([az, pelvis_z, az, None])

    lines_x.extend([ax, ax, None])
    lines_y.extend([ay, ay, None])
    lines_z.extend([pelvis_z, neck_z, None])

    lines_x.extend([ax - w_m * 1.3, ax, ax + w_m * 1.3, None])
    lines_y.extend([ay, ay, ay, None])
    lines_z.extend([az + h_m * 0.48, neck_z, az + h_m * 0.60, None])

    lines_x.extend([ax, ax, None])
    lines_y.extend([ay - w_m * 0.8, ay - w_m * 0.8, None])
    lines_z.extend([pelvis_z + h_m * 0.08, neck_z - h_m * 0.05, None])

    trace_body = go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode="lines",
        line=dict(color="#ffffff", width=6),
        name="3D Astronaut",
        hoverinfo="name",
    )
    trace_head = go.Scatter3d(
        x=[ax], y=[ay], z=[head_z],
        mode="markers",
        marker=dict(size=6, color="#fee440", symbol="circle"),
        name="Helmet",
        hoverinfo="name",
    )
    return trace_body, trace_head


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
    is_en: bool = False,
) -> str:
    if is_en:
        lines = [
            "# MARS SURFACE EVA MISSION SURVIVAL GUIDE",
            f"- **Region:** {region_name}",
            f"- **Mobility Profile:** {profile_name}",
            f"- **Lander Coordinates:** {lander_xy} | **Target:** {target['name']} {target_xy}",
            f"- **Sol Hour:** Outbound `{out_hour:02d}:00` → Return `{ret_hour:02d}:00` | **Solar Storm:** {'ACTIVE (4x)' if solar_storm else 'Normal'}",
            "",
            "## 1. Target Site Conditions",
            f"- Elevation: {target_stats['elev']:.1f} m | Slope: {target_stats['slope']:.1f}° | Crater Depth: {target_stats['depth']:.1f} m",
            f"- Temperature: {target_stats['temp']:.1f} °C | Radiation: {target_stats['rad']:.2f} mSv/day",
            "",
        ]
        if mission_summary:
            lines.extend([
                "## 2. Survival Telemetry Budget",
                f"- **Status:** {'SAFE RETURN CONFIRMED' if mission_summary['survived'] else 'CRITICAL RISK - INSUFFICIENT BATTERY/O2'}",
                f"- Total Distance: {mission_summary['dist']:.0f} m | Total Time: {mission_summary['time']:.0f} min",
                f"- Battery Consumption: {mission_summary['wh']:.0f} Wh (Remaining: %{mission_summary['rem_batt']:.0f})",
                "",
            ])
        if out_steps:
            lines.append("## 3. Outbound Navigation Steps (Lander → Target)")
            lines.extend([f"- {s}" for s in out_steps])
            lines.append("")
        if ret_steps:
            lines.append("## 4. Return Navigation Steps (Target → Lander)")
            lines.extend([f"- {s}" for s in ret_steps])
            lines.append("")
        lines.extend([
            "## 5. Surface Safety Directives",
            f"- **Target Description:** {target['description']}",
            "- Initiate return journey before remaining O2 or Battery falls below 35% reserve.",
        ])
    else:
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


# ==========================================================
# BAŞLATMA
# ==========================================================
sites_meta = load_sites_meta()
if not sites_meta:
    st.error("`processed/sites.json` not found! Run `python data_prep.py` first.")
    st.stop()

# SIDEBAR BAŞLANGICI VE DİL SEÇİMİ
with st.sidebar:
    lang = st.radio("🌐 Dil / Language", ["TR", "EN"], horizontal=True)
    is_en = (lang == "EN")

    st.header("🛰️ Mission Settings" if is_en else "🛰️ Temel Görev Ayarları")
    region_keys = list(sites_meta.keys())
    region_key = st.selectbox(
        "Landing Site" if is_en else "İniş Bölgesi",
        region_keys,
        format_func=lambda k: sites_meta[k]["display_name"],
    )
    meta = sites_meta[region_key]
    layers = load_region_layers(region_key)
    grid_n = meta["grid_size"]
    cell_m = float(meta["cell_size_m"])
    coords_label = "5.4°S, 137.8°E" if (region_key == "gale" and is_en) else ("5.4°G, 137.8°D" if region_key == "gale" else ("18.4°N, 77.5°E" if is_en else "18.4°K, 77.5°D"))
    st.caption(f"📍 Mars: {coords_label} | Scale: 3 km × 3 km (1 px = {cell_m:.0f} m)" if is_en else f"📍 Mars Konumu: {coords_label} | Ölçek: 3 km × 3 km (1 br = {cell_m:.0f} m)")

    use_custom_lander = st.checkbox("Custom Lander Coordinates" if is_en else "Roket koordinatını elle gir", value=False)
    if use_custom_lander:
        c_lx, c_ly = st.columns(2)
        lx = c_lx.number_input("Lander X" if is_en else "Roket X", 0, grid_n - 1, int(meta["lander"][0]))
        ly = c_ly.number_input("Lander Y" if is_en else "Roket Y", 0, grid_n - 1, int(meta["lander"][1]))
        lander_xy = (int(lx), int(ly))
    else:
        lander_xy = tuple(meta["lander"])

    st.header("🎯 Target Selection" if is_en else "🎯 Hedef Belirleme")
    target_options = {t["id"]: t for t in meta["targets"]}
    target_modes = ["🔬 Scientific Sites", "🧭 Compass (Bearing & Dist)", "🎯 Manual (X, Y)"] if is_en else ["🔬 Bilimsel Hedefler", "🧭 Pusula (Yön & Mesafe)", "🎯 Hassas Koordinat (X, Y)"]
    target_mode = st.radio("Targeting Method" if is_en else "Hedef Belirleme Yöntemi", target_modes)

    if target_mode in ["🔬 Scientific Sites", "🔬 Bilimsel Hedefler"]:
        sel_tid = st.selectbox(
            "Select Site" if is_en else "Hazır Bilimsel Hedef",
            list(target_options.keys()),
            format_func=lambda tid: f"{tid}: {target_options[tid]['name']}",
        )
        sel_target = target_options[sel_tid]
        target_xy = (int(sel_target["x"]), int(sel_target["y"]))

    elif target_mode in ["🧭 Compass (Bearing & Dist)", "🧭 Pusula (Yön & Mesafe)"]:
        compass_dirs = [
            "North (N)", "Northeast (NE)", "East (E)", "Southeast (SE)",
            "South (S)", "Southwest (SW)", "West (W)", "Northwest (NW)",
        ] if is_en else [
            "Kuzey (N)", "Kuzeydoğu (NE)", "Doğu (E)", "Güneydoğu (SE)",
            "Güney (S)", "Güneybatı (SW)", "Batı (W)", "Kuzeybatı (NW)",
        ]
        compass_dir = st.selectbox("Exploration Heading" if is_en else "Keşif Yönü", compass_dirs, index=1)
        dist_m_choice = st.slider("Distance from Lander (m)" if is_en else "Roketten Mesafe (m)", 300, 1800, 900, step=100)

        dir_angles = {
            compass_dirs[0]: 90, compass_dirs[1]: 45, compass_dirs[2]: 0, compass_dirs[3]: 315,
            compass_dirs[4]: 270, compass_dirs[5]: 225, compass_dirs[6]: 180, compass_dirs[7]: 135,
        }
        rad = math.radians(dir_angles[compass_dir])
        dx_cells = (dist_m_choice / cell_m) * math.cos(rad)
        dy_cells = (dist_m_choice / cell_m) * math.sin(rad)

        calc_tx = int(np.clip(round(lander_xy[0] + dx_cells), 5, grid_n - 6))
        calc_ty = int(np.clip(round(lander_xy[1] + dy_cells), 5, grid_n - 6))
        target_xy = (calc_tx, calc_ty)

        sel_target = {
            "id": "COMPASS" if is_en else "PUSULA",
            "name": f"{compass_dir} ({dist_m_choice} m)",
            "description": f"Tactical reconnaissance point ~{dist_m_choice}m heading {compass_dir}." if is_en else f"Roketten {compass_dir} yönünde yaklaşık {dist_m_choice} metre uzaklıktaki taktik keşif hedefi.",
            "image_path": "",
        }
        st.info(f"{'Calculated Target' if is_en else 'Hesaplanan Hedef'}: **({calc_tx}, {calc_ty})**")

    else:
        c_tx, c_ty = st.columns(2)
        tx = c_tx.number_input("Target X" if is_en else "Hedef X", 0, grid_n - 1, int(meta["targets"][0]["x"]))
        ty = c_ty.number_input("Target Y" if is_en else "Hedef Y", 0, grid_n - 1, int(meta["targets"][0]["y"]))
        target_xy = (int(tx), int(ty))
        sel_target = {
            "id": "CUSTOM" if is_en else "ÖZEL",
            "name": f"Custom Target ({tx}, {ty})" if is_en else f"Özel Hedef ({tx}, {ty})",
            "description": "User-defined custom exploration waypoint." if is_en else "Kullanıcı tarafından belirlenen özel yüzey keşif koordinatı.",
            "image_path": "",
        }

    st.header("⚙️ Environment & Suit" if is_en else "⚙️ Çevre & Astronot")
    profile_name = st.selectbox("Astronaut / Suit Profile" if is_en else "Astronot / Ekipman Profili", list(MOBILITY_PROFILES.keys()))
    default_temp_lim = int(MOBILITY_PROFILES[profile_name]["min_temp_c"])

    out_hour = st.slider("Mars Departure Hour (Sol)" if is_en else "Mars Gidiş Saati (Sol Saati)", 0, 23, 13, format="%02d:00")
    return_colder = st.checkbox("Late Return Cooling (+5 hrs)" if is_en else "Dönüşte Geç Saat Soğuması (+5 saat)", value=True)
    ret_hour = (out_hour + 5) % 24 if return_colder else out_hour
    st.caption(f"🕒 Outbound: `{out_hour:02d}:00` | Return: `{ret_hour:02d}:00`" if is_en else f"🕒 Gidiş: `{out_hour:02d}:00` | Dönüş: `{ret_hour:02d}:00`")

    solar_storm = st.checkbox("☀️ Solar Storm (4x Radiation)" if is_en else "☀️ Güneş Fırtınası (4x Radyasyon)", value=False)

    with st.expander("⚙️ Advanced Suit & Terrain Limits" if is_en else "⚙️ Gelişmiş Kıyafet & Arazi Limitleri", expanded=False):
        max_slope = st.slider("Max Climbable Slope (°)" if is_en else "Maks. Tırmanılabilir Eğim (°)", 5, 45, 18)
        max_crater = st.slider("Max Crater Depth (m)" if is_en else "Maks. Krater Derinliği (m)", 3, 40, 9)
        min_temp_limit = st.slider("Suit Min. Temp Limit (°C)" if is_en else "Giysi Min. Sıcaklık Sınırı (°C)", -90, -20, default_temp_lim)
        suit_battery_wh = st.slider("Battery Capacity (Wh)" if is_en else "Batarya Kapasitesi (Wh)", 300, 3000, 1200, step=50)
        suit_o2_min = st.slider("Oxygen Duration (min)" if is_en else "Oksijen Süresi (dk)", 60, 480, 240, step=15)
        sample_time_min = st.slider("Sample Collection Time (min)" if is_en else "Veri Toplama Süresi (dk)", 5, 60, 15, step=5)
        thermal_w = st.slider("Thermal Weight" if is_en else "Termal Koruma Ağırlığı", 0.0, 5.0, 2.0, step=0.5)
        rad_w = st.slider("Radiation Weight" if is_en else "Radyasyon Koruma Ağırlığı", 0.0, 5.0, 1.5, step=0.5)

# BANNER
banner_title = "🚀 MARS EVA MISSION CONTROL & SURVIVAL GUIDE" if is_en else "🚀 MARS EVA GÖREV KONTROLÜ & HAYATTA KALMA KILAVUZU"
banner_sub = (
    "NASA Mars Trek 3D DEM Topography + REMS/MEDA Hourly Thermal & RAD Radiation Model | Hyper-Local (3 km × 3 km) Round-Trip Planner"
    if is_en
    else "NASA Mars Trek 3D DEM Topografyası + REMS/MEDA Saatlik Termal & RAD Radyasyon Modeli | Hiper-Yerel (3 km × 3 km) Gidiş-Dönüş Planlayıcı"
)
st.markdown(f"""
<div class="hud-banner">
  <h2 style="margin:0; color:#f8fafc;">{banner_title}</h2>
  <p style="margin:4px 0 0 0; color:#cbd5e1; font-size:0.92rem;">{banner_sub}</p>
</div>
""", unsafe_allow_html=True)

# ROTA HESAPLAMASI
out_model, ret_model, path_out, err_out, path_ret, err_ret, naive_out = compute_mission_routes(
    region_key, cell_m, lander_xy, target_xy, max_slope, max_crater,
    min_temp_limit, out_hour, ret_hour, solar_storm, thermal_w, rad_w, profile_name
)

alt_suggestions = {}
mission_summary = None
out_steps, ret_steps = [], []

if err_out or err_ret:
    st.error(f"🚫 {'ROUTE FAILED' if is_en else 'ROTA HESAPLANAMADI'}: {err_out or err_ret}")
    alt_suggestions = suggest_safe_alternatives(
        layers, out_model["cost_grid"], lander_xy, target_xy,
        max_slope, max_crater, temp_map=out_model["temp_map"], temp_limit_c=min_temp_limit
    )
    msg_parts = []
    if alt_suggestions.get("thermal_block"):
        msg_parts.append("• " + ("Extreme Cold: Selected hour is below suit limit. Switch to a warmer window (10:00–16:00) or choose Rover Assisted." if is_en else "Aşırı Soğuk Engeli: Seçilen saatte sıcaklık giysi sınırının altında. Daha sıcak bir saate (10:00–16:00) geçin veya Rover Destekli profili seçin."))
    if "nearest_safe_start" in alt_suggestions:
        msg_parts.append("• " + (f"Nearest safe landing spot: **{alt_suggestions['nearest_safe_start']}**" if is_en else f"Roket için en yakın güvenli iniş noktası: **{alt_suggestions['nearest_safe_start']}**"))
    if "nearest_reachable_pt" in alt_suggestions:
        pt = alt_suggestions["nearest_reachable_pt"]
        gap_m = alt_suggestions["nearest_reachable_gap_cells"] * cell_m
        msg_parts.append("• " + (f"Closest reachable waypoint: **{pt}** ({gap_m:.0f} m gap - purple line)." if is_en else f"Mevcut kısıtlarla hedefe en yakın güvenli ulaşım noktası: **{pt}** (hedefe `{gap_m:.0f} m` mesafede — mor çizgi)."))
    if "min_slope_needed" in alt_suggestions:
        msg_parts.append("• " + (f"Minimum clearance required: Slope ≥ {alt_suggestions['min_slope_needed']}°, Crater Depth ≥ {alt_suggestions['min_crater_needed']} m." if is_en else f"Tam ulaşım için minimum ayar: Eğim ≥ {alt_suggestions['min_slope_needed']}° ve Krater Derinliği ≥ {alt_suggestions['min_crater_needed']} m."))
    if msg_parts:
        st.info(("💡 **Tactical Recommendation:**\n" if is_en else "💡 **Doğru Yol Önerisi:**\n") + "\n".join(msg_parts))
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
    raw_out = extract_turn_by_turn_steps(path_out, cell_m, out_model["temp_map"], layers["slope_deg"])
    raw_ret = extract_turn_by_turn_steps(path_ret, cell_m, ret_model["temp_map"], layers["slope_deg"])
    out_steps = [translate_nav_step(s, is_en) for s in raw_out]
    ret_steps = [translate_nav_step(s, is_en) for s in raw_ret]

# ==========================================================
# ANA HARİTA VE HUD SÜTUNLARI
# ==========================================================
col_map, col_hud = st.columns([2.15, 1.0], gap="medium")

with col_map:
    c_sel1, c_sel2, c_sel3 = st.columns([1.8, 1.4, 1.1])
    layer_options_en = [
        "Surface Temperature (°C)",
        "Crater Depth (m)",
        "Topography / Elevation (m)",
        "Radiation Dose (mSv/day)",
        "Traversability / Hazard Mask",
        "Geological Units",
    ]
    layer_options_tr = [
        "Yüzey Sıcaklığı (°C)",
        "Krater Derinliği (m)",
        "Topografya / Yükseklik (m)",
        "Radyasyon Dozu (mSv/gün)",
        "Geçilebilirlik / Engel Maskesi",
        "Jeolojik Bölgeler",
    ]
    with c_sel1:
        layer_choice = st.selectbox(
            "Layer" if is_en else "Katman",
            layer_options_en if is_en else layer_options_tr,
            label_visibility="collapsed",
        )
    with c_sel2:
        view_modes = ["2D Tactical Map", "3D Terrain"] if is_en else ["2D Taktik Harita", "3D Arazi"]
        view_mode = st.radio("View", view_modes, horizontal=True, label_visibility="collapsed")
    with c_sel3:
        show_naive = st.checkbox("Compare Blind Path" if is_en else "Kör Rota Kıyasla", value=True)

    layer_idx = (layer_options_en if is_en else layer_options_tr).index(layer_choice)
    if layer_idx == 0:
        z_data, colorscale, unit = out_model["temp_map"], "Turbo", "°C"
    elif layer_idx == 1:
        z_data, colorscale, unit = layers["crater_depth_m"], "Inferno", "m"
    elif layer_idx == 2:
        z_data, colorscale, unit = layers["elevation"], "Turbo", "m"
    elif layer_idx == 3:
        z_data, colorscale, unit = out_model["rad_map"], "Plasma", ("mSv/day" if is_en else "mSv/gün")
    elif layer_idx == 4:
        z_data = np.where(out_model["blocked"], 1.0, 0.0)
        colorscale, unit = [[0, "#065f46"], [1, "#dc2626"]], ("0=Pass, 1=Hazard" if is_en else "0=Açık, 1=Engel")
    else:
        z_data, colorscale, unit = layers["geology_mask"], "Viridis", ("Class" if is_en else "Sınıf")

    fig = go.Figure()

    if view_mode in ["2D Tactical Map", "2D Taktik Harita"]:
        fig.add_trace(go.Heatmap(
            z=z_data, colorscale=colorscale, colorbar=dict(title=unit, thickness=14),
            hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z:.2f} " + unit + "<extra></extra>",
        ))
        if show_naive and naive_out:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in naive_out], y=[p[1] for p in naive_out],
                mode="lines", name=("Blind Shortest Path" if is_en else "Kör En Kısa Rota"), line=dict(color="#fb923c", width=2.5, dash="dot"),
            ))
        if path_out:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in path_out], y=[p[1] for p in path_out],
                mode="lines", name=("Safe Outbound" if is_en else "Güvenli Gidiş"), line=dict(color="#00f5d4", width=4.5),
            ))
        if path_ret:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in path_ret], y=[p[1] for p in path_ret],
                mode="lines", name=("Safe Return" if is_en else "Güvenli Dönüş"), line=dict(color="#fee440", width=3, dash="dash"),
            ))
        if alt_suggestions.get("nearest_reachable_path"):
            alt_p = alt_suggestions["nearest_reachable_path"]
            fig.add_trace(go.Scatter(
                x=[p[0] for p in alt_p], y=[p[1] for p in alt_p],
                mode="lines", name=("Alternative Path" if is_en else "Alternatif Yaklaşım"), line=dict(color="#c77dff", width=3, dash="dashdot"),
            ))

        fig.add_trace(go.Scatter(
            x=[lander_xy[0]], y=[lander_xy[1]], mode="text", text=["🚀"],
            textfont=dict(size=30), name=("Lander" if is_en else "Roket"),
        ))

       

        fig.add_trace(go.Scatter(
            x=[target_xy[0]], y=[target_xy[1]], mode="text", text=["👨‍🚀"],
            textfont=dict(size=30), name=("Target Waypoint" if is_en else "Astronot Hedefi"),
        ))

        fig.update_layout(
            height=680,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title=f"X (1 px = {cell_m:.0f} m)", constrain="domain"),
            yaxis=dict(title=f"Y (1 px = {cell_m:.0f} m)", scaleanchor="x", scaleratio=1),
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
            hovertemplate="X: %{x}<br>Y: %{y}<br>Elev: %{z:.1f} m<extra></extra>",
        ))
        if show_naive and naive_out:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in naive_out], y=[p[1] for p in naive_out],
                z=[float(elev[p[1], p[0]]) + 2.5 for p in naive_out],
                mode="lines", name=("Blind Shortest" if is_en else "Kör En Kısa"), line=dict(color="#fb923c", width=4, dash="dot"),
            ))
        if path_out:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in path_out], y=[p[1] for p in path_out],
                z=[float(elev[p[1], p[0]]) + 4.5 for p in path_out],
                mode="lines", name=("Safe Outbound" if is_en else "Güvenli Gidiş"), line=dict(color="#00f5d4", width=7),
            ))
        if path_ret:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in path_ret], y=[p[1] for p in path_ret],
                z=[float(elev[p[1], p[0]]) + 6.0 for p in path_ret],
                mode="lines", name=("Safe Return" if is_en else "Güvenli Dönüş"), line=dict(color="#fee440", width=5, dash="dash"),
            ))
        if alt_suggestions.get("nearest_reachable_path"):
            alt_p = alt_suggestions["nearest_reachable_path"]
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in alt_p], y=[p[1] for p in alt_p],
                z=[float(elev[p[1], p[0]]) + 4.5 for p in alt_p],
                mode="lines", name=("Alternative" if is_en else "Alternatif"), line=dict(color="#c77dff", width=6, dash="dashdot"),
            ))

        lz = float(elev[lander_xy[1], lander_xy[0]])
        fig.add_trace(build_3d_rocket(lander_xy[0], lander_xy[1], lz, h_m=38.0, r_m=2.0))
        fig.add_trace(go.Scatter3d(
            x=[lander_xy[0]], y=[lander_xy[1]], z=[lz + 44.0],
            mode="text", text=["🚀 LANDER" if is_en else "🚀 ROKET"], textfont=dict(color="#00f5d4", size=14),
            name="Lander", hoverinfo="skip",
        ))

        tz = float(elev[target_xy[1], target_xy[0]])
        ast_body, ast_head = build_3d_astronaut(target_xy[0], target_xy[1], tz, h_m=26.0)
        fig.add_trace(ast_body)
        fig.add_trace(ast_head)
        fig.add_trace(go.Scatter3d(
            x=[target_xy[0]], y=[target_xy[1]], z=[tz + 32.0],
            mode="text", text=["👨‍🚀 ASTRONAUT" if is_en else "👨‍🚀 ASTRONOT"], textfont=dict(color="#fee440", size=14),
            name="Astronaut", hoverinfo="skip",
        ))

        
        fig.update_layout(
            height=680,
            dragmode="turntable",
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis_title=f"X (1 px={cell_m:.0f}m)",
                yaxis_title=f"Y (1 px={cell_m:.0f}m)",
                zaxis_title="Elevation (m)",
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
            st.success("✅ SAFE RETURN CONFIRMED" if is_en else "✅ GÜVENLİ DÖNÜŞ MÜMKÜN")
        else:
            st.error("⚠️ CRITICAL: BATTERY / O2 DEPLETED!" if is_en else "⚠️ DÖNÜŞTE BATARYA/O2 TÜKENİYOR!")

        m1, m2 = st.columns(2)
        m1.metric("Total Distance" if is_en else "Toplam Mesafe", f"{mission_summary['dist']:.0f} m")
        m2.metric("Time / Rem. O2" if is_en else "Süre / Kalan O2", f"{mission_summary['time']:.0f} min" if is_en else f"{mission_summary['time']:.0f} dk", f"%{mission_summary['rem_o2']:.0f} O2")
        m3, m4 = st.columns(2)
        m3.metric("Battery" if is_en else "Batarya", f"{mission_summary['wh']:.0f} Wh", f"%{mission_summary['rem_batt']:.0f} left" if is_en else f"%{mission_summary['rem_batt']:.0f} kaldı")
        m4.metric("Min Temp" if is_en else "Min Sıcaklık", f"{mission_summary['min_temp']:.1f} °C", f"{mission_summary['rad']:.3f} mSv")

        hud_out_title = "🧭 OUTBOUND WAYPOINTS (LANDER → TARGET)" if is_en else "🧭 GİDİŞ YOL TARİFİ (ROKET → HEDEF)"
        st.markdown(f'<div class="hud-card"><div class="hud-title">{hud_out_title}</div>', unsafe_allow_html=True)
        for s in out_steps:
            st.markdown(f'<div class="step-box">{fmt_html_step(s)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        hud_ret_title = "🔙 RETURN WAYPOINTS (TARGET → LANDER)" if is_en else "🔙 DÖNÜŞ YOL TARİFİ (HEDEF → ROKET)"
        st.markdown(f'<div class="hud-card"><div class="hud-title">{hud_ret_title}</div>', unsafe_allow_html=True)
        for s in ret_steps:
            st.markdown(f'<div class="step-box step-box-ret">{fmt_html_step(s)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        standby_msg = "Adjust Sol hour, coordinates or relax constraints in Advanced Settings." if is_en else "Güvenli geçiş için soldan Sol saatini/koordinatı değiştirin veya Gelişmiş Ayarlardan limitleri gevşetin."
        st.markdown(f'<div class="hud-card"><div class="hud-title">⚠️ STANDBY</div>{standby_msg}</div>', unsafe_allow_html=True)

# ==========================================================
# ALT SEKMELER
# ==========================================================
tabs_labels = [
    "📊 Telemetry & Benchmark" if is_en else "📊 Rota Telemetrisi & Karşılaştırma",
    "🧭 Survival Guide & Site Context" if is_en else "🧭 Detaylı Hayatta Kalma Kılavuzu & Fotoğraf",
    "ℹ️ NASA Data & Model Transparency" if is_en else "ℹ️ NASA Veri Kaynakları & Şeffaflık",
]
tab1, tab2, tab3 = st.tabs(tabs_labels)

with tab1:
    if not (err_out or err_ret):
        df_out = pd.DataFrame(eval_out["telemetry"])
        df_ret = pd.DataFrame(eval_ret["telemetry"])
        df_ret["dist_m"] = df_ret["dist_m"] + eval_out["distance_m"]
        df_ret["cum_energy_wh"] = df_ret["cum_energy_wh"] + eval_out["energy_wh"] + stat_wh
        df_all = pd.concat([df_out, df_ret], ignore_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Route Temperature & Slope" if is_en else "Rota Boyunca Sıcaklık & Eğim")
            st.line_chart(df_all.set_index("dist_m")[["temp_c", "slope_deg"]])
        with col_b:
            st.subheader("Cumulative Battery Consumption (Wh)" if is_en else "Kümülatif Batarya Tüketimi (Wh)")
            st.line_chart(df_all.set_index("dist_m")[["cum_energy_wh"]])

        if show_naive and naive_out:
            eval_naive = evaluate_path(naive_out, layers, cell_m, out_model["temp_map"], out_model["rad_map"], solar_storm=solar_storm, profile_name=profile_name)
            st.markdown("#### " + ("Safe Survival Route vs. Blind Shortest Path (Outbound Benchmark)" if is_en else "Güvenli Rota vs. Kör En Kısa Rota (Gidiş Karşılaştırması)"))
            comp_df = pd.DataFrame([
                {
                    "Route" if is_en else "Rota": "Safe Survival Route" if is_en else "Güvenli Hayatta Kalma Rotası",
                    "Distance (m)" if is_en else "Mesafe (m)": round(eval_out["distance_m"], 1),
                    "Energy (Wh)" if is_en else "Enerji (Wh)": round(eval_out["energy_wh"], 1),
                    "Max Slope (°)" if is_en else "Maks. Eğim (°)": round(eval_out["max_slope_deg"], 1),
                    "Max Crater (m)" if is_en else "Maks. Krater Derinliği (m)": round(eval_out["max_crater_depth_m"], 1),
                    "Coldest (°C)" if is_en else "En Soğuk (°C)": round(eval_out["min_temp_c"], 1),
                },
                {
                    "Route" if is_en else "Rota": "Blind Shortest Path" if is_en else "Kör En Kısa Rota",
                    "Distance (m)" if is_en else "Mesafe (m)": round(eval_naive["distance_m"], 1),
                    "Energy (Wh)" if is_en else "Enerji (Wh)": round(eval_naive["energy_wh"], 1),
                    "Max Slope (°)" if is_en else "Maks. Eğim (°)": round(eval_naive["max_slope_deg"], 1),
                    "Max Crater (m)" if is_en else "Maks. Krater Derinliği (m)": round(eval_naive["max_crater_depth_m"], 1),
                    "Coldest (°C)" if is_en else "En Soğuk (°C)": round(eval_naive["min_temp_c"], 1),
                },
            ])
            st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Route calculation pending; telemetry chart unavailable." if is_en else "Rota tamamlanamadığı için telemetri grafiği oluşturulamadı.")

with tab2:
    tx, ty = target_xy
    target_stats = {
        "elev": float(layers["elevation"][ty, tx]),
        "slope": float(layers["slope_deg"][ty, tx]),
        "depth": float(layers["crater_depth_m"][ty, tx]),
        "temp": float(out_model["temp_map"][ty, tx]),
        "rad": float(layers["radiation"][ty, tx]),
    }

    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        st.subheader(f"{'Target Briefing' if is_en else 'Hedef Brifingi'}: {sel_target['name']}")
        st.write(sel_target["description"])
        st.write(
            f"**{'Target Conditions' if is_en else 'Hedef Koşulları'}:** "
            f"Elevation `{target_stats['elev']:.1f} m` | Slope `{target_stats['slope']:.1f}°` | Crater Depth `{target_stats['depth']:.1f} m` | Temp `{target_stats['temp']:.1f} °C` | Rad `{target_stats['rad']:.2f} mSv/day`"
            if is_en else
            f"**Hedef Koşulları:** Yükseklik `{target_stats['elev']:.1f} m` | Eğim `{target_stats['slope']:.1f}°` | Krater Derinliği `{target_stats['depth']:.1f} m` | Sıcaklık `{target_stats['temp']:.1f} °C` | Radyasyon `{target_stats['rad']:.2f} mSv/gün`"
        )

        if mission_summary and mission_summary["min_temp"] < -55.0:
            st.warning("**Thermal Freeze Alert:** Return or crater shadow temp drops below -55°C. Minimize stationary sample time in depressions." if is_en else "**Termal Donma Uyarısı:** Dönüş veya gölge sıcaklığı -55°C altında. Krater diplerinde duraklama süresini kısaltın.")
        else:
            st.success("**Thermal Status:** Selected Sol window is favorable; preserve reserve battery for cold shadows." if is_en else "**Termal Durum:** Seçilen Sol penceresi uygun; krater gölgelerinde ani soğumaya karşı batarya rezervi bırakın.")

        if solar_storm:
            st.error("**Solar Storm Protocol:** Radiation 4x! Avoid open ridges, utilize crater wall natural shielding." if is_en else "**Güneş Fırtınası Protokolü:** Radyasyon 4x! Açık sırtlarda kalmayın, krater duvarı dibi doğal korumasını kullanın.")

        guide_md = build_survival_guide_markdown(
            meta["display_name"], profile_name, lander_xy, target_xy, sel_target,
            out_hour, ret_hour, solar_storm, target_stats, mission_summary, out_steps, ret_steps, is_en=is_en
        )
        st.download_button(
            "📥 Download Mission Guide (.md)" if is_en else "📥 Tam Görev Kılavuzunu İndir (.md)",
            data=guide_md,
            file_name=f"mars_eva_guide_{region_key}_{sel_target['id']}.md" if is_en else f"mars_eva_kilavuz_{region_key}_{sel_target['id']}.md",
            mime="text/markdown",
        )

    with col_g2:
        img_file = find_target_image(region_key, sel_target)
        if img_file:
            st.image(img_file, caption=f"{sel_target['name']} — NASA Context View", use_container_width=True)
        else:
            st.info(
                "📸 **Surface Imagery Slot:**\n\nAdd `images/{region_key}_{target_id}.jpg` to view authentic rover imagery here."
                if is_en else
                f"📸 **Yüzey Fotoğrafı Slotu:**\n\n`images/{region_key}_{sel_target['id'].lower()}.jpg` eklendiğinde gerçek NASA rover görüntüsü burada açılır."
            )

with tab3:
    if is_en:
        st.markdown(f"""
        ### Data Sources & Engineering Methodology ({meta['display_name']})
        1. **Topography / Craters / Slope (Raw NASA Data):** Extracted directly from NASA JPL Mars Trek 3D DEM surface mesh (`.stl`, `{grid_n}×{grid_n}` grid, `1 px = {cell_m:.0f} m`) preserving real XY/Z aspect ratio.
        2. **Hourly Thermal Modeling (NASA REMS/MEDA Hybrid):** Built upon Curiosity REMS (Gale) and Perseverance MEDA (Jezero) 24-hour surface temperature curves, combining real-time solar elevation/azimuth dynamic shadow (`hillshade`) and cold-air drainage in craters.
        3. **Surface Radiation (NASA RAD Model):** Calibrated to Curiosity RAD ground baseline (~0.64 mSv/day) with topographic sky-view factor and 4x solar storm surge factor.
        """)
    else:
        st.markdown(f"""
        ### Veri Kaynakları ve Mühendislik Modeli ({meta['display_name']})
        1. **Topografya / Krater / Eğim (Ham NASA Verisi):** Doğrudan NASA JPL Mars Trek 3D DEM yüzey ağından (`.stl`, `{grid_n}×{grid_n}` grid, `1 br = {cell_m:.0f} m`) gerçek XY/Z oranı korunarak çıkarılmıştır.
        2. **Saatlik Termal Katman (NASA REMS/MEDA Hibrit Modeli):** Curiosity REMS (Gale) ve Perseverance MEDA (Jezero) 24 saatlik yüzey sıcaklık eğrisi temel alınarak, seçilen Sol saatindeki güneş azimut/yükseklik açısına göre dinamik gölgelenme (`hillshade`) ve krater içi soğuk hava birikimiyle hesaplanmıştır.
        3. **Radyasyon Katmanı (NASA RAD Modeli):** Curiosity RAD (~0.64 mSv/gün) gerçek yüzey radyasyonu temel alınarak topografik gökyüzü açıklığına göre modellenmiştir.
        """)