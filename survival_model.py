import numpy as np

MARS_G = 3.72  # m/s^2

MOBILITY_PROFILES = {
    "Standart EVA (Tek Astronot)": {
        "speed_mult": 1.0,
        "power_mult": 1.0,
        "slope_sensitivity": 1.0,
        "mass_kg": 165.0,  # Astronot + uzay giysisi + yaşam destek
        "min_temp_c": -68.0,
        "desc": "Standart uzay giysisi ve temel el aletleri ile yaya keşif.",
    },
    "Ağır Bilimsel Ekipmanlı": {
        "speed_mult": 0.78,
        "power_mult": 1.28,
        "slope_sensitivity": 1.35,
        "mass_kg": 195.0,
        "min_temp_c": -65.0,
        "desc": "Delici/spektrometre taşıma yükü nedeniyle eğimde daha hızlı yorulma ve yüksek batarya tüketimi.",
    },
    "Rover Destekli Keşif": {
        "speed_mult": 1.30,
        "power_mult": 0.72,
        "slope_sensitivity": 0.85,
        "mass_kg": 150.0,
        "min_temp_c": -78.0,
        "desc": "Ekipman ve yedek güç taşıyan otonom rover refakati ile artırılmış termal dayanım ve menzil.",
    },
}

# NASA Curiosity REMS (Gale) & Perseverance MEDA (Jezero) tipik Sol saatlik sıcaklık eğrileri (°C)
# Saatler: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
_HOURS = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], dtype=np.float32)
_REMS_GALE_C = np.array([-74, -78, -81, -82, -62, -32, -12, -4, -9, -30, -54, -67, -74], dtype=np.float32)
_MEDA_JEZERO_C = np.array([-76, -80, -83, -84, -65, -36, -18, -10, -15, -34, -57, -70, -76], dtype=np.float32)


def _resolve_sol_hour(time_spec) -> float:
    """String ('Gündüz'/'Gece') veya sayısal Sol saatini (0-24) float saate çevirir."""
    if isinstance(time_spec, (int, float)):
        return float(time_spec) % 24.0
    mapping = {"Gündüz": 14.0, "Alacakaranlık": 18.0, "Gece": 2.0}
    return mapping.get(str(time_spec), 14.0)


def get_temperature_map(layers: dict, time_spec, region_key: str = "gale") -> np.ndarray:
    """
    NASA REMS/MEDA saatlik baz sıcaklığı + güneş açısına bağlı dinamik gölgelenme (hillshade)
    ve krater içi soğuk hava birikimiyle °C cinsinden yüzey sıcaklık haritasını üretir.
    """
    hour = _resolve_sol_hour(time_spec)
    curve = _MEDA_JEZERO_C if region_key == "jezero" else _REMS_GALE_C
    base_temp_c = float(np.interp(hour, _HOURS, curve))

    norm_elev = layers["norm_elev"]
    depth = layers["crater_depth_m"]
    elev_norm = (norm_elev - float(norm_elev.mean())) / max(10.0, float(norm_elev.max() - norm_elev.min()) / 2.0)
    depth_norm = depth / max(1.0, float(depth.max()))

    # Mars güneşi yaklaşık 06:00'da doğar (90° Doğu), 12:00'de tepede, 18:00'de batar (270° Batı)
    if 6.0 <= hour <= 18.0 and "slope_rad" in layers and "aspect_rad" in layers:
        day_frac = (hour - 6.0) / 12.0  # 0..1
        sun_alt = np.radians(72.0 * np.sin(np.pi * day_frac))
        sun_az = np.radians(90.0 + 180.0 * day_frac)

        slope_rad = layers["slope_rad"]
        aspect_rad = layers["aspect_rad"]
        hillshade = (
            np.sin(sun_alt) * np.cos(slope_rad)
            + np.cos(sun_alt) * np.sin(slope_rad) * np.cos(sun_az - aspect_rad)
        )
        hillshade = np.clip(hillshade, 0.0, 1.0)
        solar_factor = float(np.sin(np.pi * day_frac))  # Öğlen 1.0, şafak/akşam 0.0
        solar_heating = (hillshade - 0.5) * (18.0 * solar_factor)
    else:
        solar_heating = 0.0

    # Krater tabanları gündüz gölge nedeniyle, gece soğuk hava çökelmesi nedeniyle daha soğuktur
    crater_cooling = depth_norm * (13.0 if 7.0 <= hour <= 17.0 else 9.0)
    elev_effect = elev_norm * (-2.0 if 8.0 <= hour <= 17.0 else 2.5)

    temp_map = base_temp_c + solar_heating - crater_cooling + elev_effect
    return temp_map.astype(np.float32)


def build_cost_grid(
    layers: dict,
    cell_size_m: float,
    max_slope_deg: float,
    max_crater_depth_m: float,
    time_spec,
    solar_storm: bool,
    thermal_weight: float,
    radiation_weight: float,
    profile_name: str = "Standart EVA (Tek Astronot)",
    region_key: str = "gale",
    min_temp_limit_c: float | None = None,
) -> dict:
    """Her grid hücresi için geçilebilirlik (blocked) ve rota maliyet katsayısını hesaplar."""
    prof = MOBILITY_PROFILES.get(profile_name, MOBILITY_PROFILES["Standart EVA (Tek Astronot)"])
    slope = layers["slope_deg"]
    depth = layers["crater_depth_m"]
    rad = layers["radiation"] * (4.0 if solar_storm else 1.0)
    temp = get_temperature_map(layers, time_spec, region_key=region_key)

    temp_limit = float(min_temp_limit_c) if min_temp_limit_c is not None else prof["min_temp_c"]

    # Geçilemez alanlar: Dik eğim VEYA derin krater çukuru VEYA giysi termal limiti altı
    blocked = (slope > max_slope_deg) | (depth > max_crater_depth_m) | (temp < temp_limit)

    slope_penalty = ((slope / max(1.0, max_slope_deg)) ** 2) * 1.8 * prof["slope_sensitivity"]
    # Krater kenarına/çukura yaklaştıkça karesel artan ceza (etrafından dolanmayı teşvik eder)
    depth_penalty = ((depth / max(1.0, max_crater_depth_m)) ** 2) * 2.2

    cold_excess = np.maximum(0.0, -20.0 - temp) / 25.0
    hot_excess = np.maximum(0.0, temp - 20.0) / 20.0
    thermal_penalty = (cold_excess + hot_excess) * thermal_weight

    rad_norm = rad / max(0.01, float(np.mean(layers["radiation"])))
    rad_penalty = np.maximum(0.0, (rad_norm - 0.8) * radiation_weight)

    cost_grid = (1.0 + slope_penalty + depth_penalty + thermal_penalty + rad_penalty).astype(np.float32)
    cost_grid[blocked] = np.inf

    return {
        "cost_grid": cost_grid,
        "blocked": blocked,
        "temp_map": temp.astype(np.float32),
        "rad_map": rad.astype(np.float32),
        "temp_limit_c": temp_limit,
    }


def evaluate_path(
    path: list,
    layers: dict,
    cell_size_m: float,
    temp_map: np.ndarray,
    rad_map: np.ndarray,
    walk_speed_mps: float = 1.0,
    solar_storm: bool = False,
    profile_name: str = "Standart EVA (Tek Astronot)",
) -> dict:
    """Verilen [(x, y), ...] rotası boyunca fiziksel mesafe, süre, enerji ve radyasyon hesabı yapar."""
    if not path or len(path) < 2:
        return {
            "distance_m": 0.0,
            "duration_min": 0.0,
            "energy_wh": 0.0,
            "radiation_msv": 0.0,
            "min_temp_c": float("nan"),
            "max_slope_deg": 0.0,
            "max_crater_depth_m": 0.0,
            "telemetry": [],
        }

    prof = MOBILITY_PROFILES.get(profile_name, MOBILITY_PROFILES["Standart EVA (Tek Astronot)"])
    elev = layers["elevation"]
    slope = layers["slope_deg"]
    depth = layers["crater_depth_m"]
    rad_mult = 4.0 if solar_storm else 1.0

    total_dist_m = 0.0
    total_sec = 0.0
    total_energy_wh = 0.0
    total_rad_msv = 0.0
    min_temp = float(temp_map[path[0][1], path[0][0]])
    max_slope_seen = float(slope[path[0][1], path[0][0]])
    max_depth_seen = float(depth[path[0][1], path[0][0]])

    telemetry = []

    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]

        step_cells = np.hypot(x1 - x0, y1 - y0)
        dist_m = step_cells * cell_size_m
        dh = float(elev[y1, x1] - elev[y0, x0])
        local_slope = float(slope[y1, x1])
        local_temp = float(temp_map[y1, x1])
        local_rad_day = float(layers["radiation"][y1, x1]) * rad_mult
        local_depth = float(depth[y1, x1])

        base_speed = walk_speed_mps * prof["speed_mult"]
        speed = max(0.22, base_speed * (1.0 - 0.02 * local_slope * prof["slope_sensitivity"]))
        dt_sec = dist_m / speed

        # Sabit yaşam destek + yatay yürüyüş gücü (Watt)
        base_life_support_w = 180.0
        locomotion_w = (110.0 * (1.0 + local_slope / 15.0)) * prof["power_mult"]
        heating_w = max(0.0, -20.0 - local_temp) * 4.2
        cooling_w = max(0.0, local_temp - 20.0) * 3.0
        total_w = base_life_support_w + locomotion_w + heating_w + cooling_w

        # Yokuş yukarı tırmanma için gerçek potansiyel enerji (Joule -> Wh, %25 kas/motor verimi)
        climb_wh = (prof["mass_kg"] * MARS_G * max(0.0, dh)) / (0.25 * 3600.0)

        step_wh = (total_w * dt_sec) / 3600.0 + climb_wh
        step_msv = (local_rad_day / 86400.0) * dt_sec

        total_dist_m += dist_m
        total_sec += dt_sec
        total_energy_wh += step_wh
        total_rad_msv += step_msv

        min_temp = min(min_temp, local_temp)
        max_slope_seen = max(max_slope_seen, local_slope)
        max_depth_seen = max(max_depth_seen, local_depth)

        telemetry.append({
            "dist_m": total_dist_m,
            "elev_m": float(elev[y1, x1]),
            "temp_c": local_temp,
            "slope_deg": local_slope,
            "crater_depth_m": local_depth,
            "cum_energy_wh": total_energy_wh,
            "cum_rad_msv": total_rad_msv,
        })

    return {
        "distance_m": total_dist_m,
        "duration_min": total_sec / 60.0,
        "energy_wh": total_energy_wh,
        "radiation_msv": total_rad_msv,
        "min_temp_c": min_temp,
        "max_slope_deg": max_slope_seen,
        "max_crater_depth_m": max_depth_seen,
        "telemetry": telemetry,
    }


def stationary_cost(
    temp_c: float,
    rad_msv_day: float,
    duration_min: float,
    solar_storm: bool = False,
    profile_name: str = "Standart EVA (Tek Astronot)",
) -> tuple:
    """Bilimsel numune toplarken sabit duruşta harcanan batarya (Wh) ve alınan radyasyon (mSv)."""
    dt_sec = duration_min * 60.0
    heating_w = max(0.0, -20.0 - temp_c) * 4.2
    total_w = (180.0 + heating_w) * (0.85 if "Rover" in profile_name else 1.0)
    energy_wh = (total_w * dt_sec) / 3600.0
    rad_mult = 4.0 if solar_storm else 1.0
    rad_msv = ((rad_msv_day * rad_mult) / 86400.0) * dt_sec
    return energy_wh, rad_msv