import os
import json
import glob
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, distance_transform_edt, label
from stl import mesh

# ==========================================================
# KALİBRASYON PARAMETRELERİ (Hiper-Yerel EVA Ölçeği)
# ==========================================================
GRID_SIZE = 200
AREA_SIZE_M = 3000.0                 # 3 km x 3 km yürüyüş alanı
CELL_SIZE_M = AREA_SIZE_M / GRID_SIZE  # 15 metre / hücre

REGION_CONFIG = {
    "gale": {
        "display_name": "Gale Krateri - Sharp Dağı (Curiosity REMS)",
        "base_datum_m": -4500.0,
        "temp_day_mean_c": -6.0,
        "temp_night_mean_c": -81.0,
        "rad_base_msv_day": 0.64,
        "target_names": [
            ("Boxwork Sırt Formasyonu", "Curiosity'nin 2025'te incelediği, antik yeraltı sularının oluşturduğu mineral ağ yapıları."),
            ("Kil-Sülfat Geçiş Zonu", "Eski göl tabanından kurak döneme geçişi gösteren katmanlı tortul kayaçlar."),
            ("Krater Tabanı Çakıl Yatağı", "Akarsu aşınmasıyla taşınmış yuvarlak çakıllar ve su izleri."),
        ],
    },
    "jezero": {
        "display_name": "Jezero Krateri - Antik Delta (Perseverance MEDA)",
        "base_datum_m": -2500.0,
        "temp_day_mean_c": -12.0,
        "temp_night_mean_c": -83.0,
        "rad_base_msv_day": 0.66,
        "target_names": [
            ("Antik Nehir Deltası Yelpazesi", "Mikrobiyal yaşam izleri barındırma ihtimali en yüksek ince taneli tortullar."),
            ("Karbonat Kıyı Şeridi", "Eski göl kıyısında oluşmuş karbonat mineral birikintileri."),
            ("Krater Kenarı Bazaltik Çıkıntı", "Volkanik geçmişi ve darbe kronolojisini aydınlatan kayaç yüzeyi."),
        ],
    },
}


def load_stl_to_grid(stl_path: str, target_size: int = GRID_SIZE) -> tuple:
    """Mars Trek .stl dosyasından üst arazi yüzeyini çıkarır ve gerçek XY/Z oranını korur."""
    m = mesh.Mesh.from_file(stl_path)
    up_mask = m.normals[:, 2] > 0.05
    pts = m.vectors[up_mask].reshape(-1, 3)

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    z_raw_min, z_raw_max = pts[:, 2].min(), pts[:, 2].max()

    xy_span = max(float(x_max - x_min), float(y_max - y_min), 1e-6)
    z_span = max(float(z_raw_max - z_raw_min), 1e-6)

    # STL'in kendi geometrik Z/XY oranından gerçekçi rölyef (metre) hesapla
    stl_relief_m = (z_span / xy_span) * AREA_SIZE_M
    relief_m = float(np.clip(stl_relief_m, 80.0, 650.0))

    nx = max(10, int(round(x_max - x_min)) + 1)
    ny = max(10, int(round(y_max - y_min)) + 1)

    ix = np.clip(np.rint(pts[:, 0] - x_min).astype(int), 0, nx - 1)
    iy = np.clip(np.rint(pts[:, 1] - y_min).astype(int), 0, ny - 1)

    raw_grid = np.full((ny, nx), -np.inf, dtype=np.float64)
    np.maximum.at(raw_grid, (iy, ix), pts[:, 2])

    missing = ~np.isfinite(raw_grid)
    if np.any(missing):
        _, indices = distance_transform_edt(missing, return_indices=True)
        raw_grid = raw_grid[tuple(indices)]

    zoom_y = target_size / raw_grid.shape[0]
    zoom_x = target_size / raw_grid.shape[1]
    resized = zoom(raw_grid, (zoom_y, zoom_x), order=1)

    # STL üçgen basamak gürültüsünü temizlemek için hafif pürüzsüzleştirme
    resized = gaussian_filter(resized, sigma=0.7)

    z_min, z_max = resized.min(), resized.max()
    if z_max > z_min:
        norm_elev = (resized - z_min) / (z_max - z_min) * relief_m
    else:
        norm_elev = np.zeros_like(resized)

    return norm_elev.astype(np.float32), relief_m


def compute_layers(norm_elev: np.ndarray, cfg: dict, relief_m: float) -> dict:
    """Topografyadan eğim, bakı, krater derinliği, termal, radyasyon ve jeoloji katmanlarını türetir."""
    elevation = norm_elev + cfg["base_datum_m"]

    dz_dy, dz_dx = np.gradient(norm_elev, CELL_SIZE_M)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy)).astype(np.float32)
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    aspect_rad = np.arctan2(-dz_dy, dz_dx).astype(np.float32)

    local_surface = gaussian_filter(norm_elev, sigma=10.0)
    crater_depth_m = np.maximum(0.0, local_surface - norm_elev).astype(np.float32)

    # Öğleden sonra varsayılan güneş açısı (statik hillshade)
    sun_az = np.radians(315.0)
    sun_alt = np.radians(35.0)
    hillshade = (
        np.sin(sun_alt) * np.cos(slope_rad)
        + np.cos(sun_alt) * np.sin(slope_rad) * np.cos(sun_az - aspect_rad)
    )
    hillshade = np.clip(hillshade, 0.0, 1.0).astype(np.float32)

    elev_norm = (norm_elev - norm_elev.mean()) / max(10.0, relief_m / 2.0)
    depth_norm = crater_depth_m / max(1.0, float(crater_depth_m.max()))

    temperature_day = (
        cfg["temp_day_mean_c"]
        + (hillshade - 0.5) * 18.0
        - depth_norm * 14.0
        - elev_norm * 2.5
    ).astype(np.float32)

    temperature_night = (
        cfg["temp_night_mean_c"]
        - depth_norm * 9.0
        + elev_norm * 3.0
    ).astype(np.float32)

    shielding_factor = np.clip(0.22 * depth_norm + 0.08 * (slope_deg / 45.0), 0.0, 0.28)
    radiation = (
        cfg["rad_base_msv_day"] * (1.05 + 0.08 * elev_norm - shielding_factor)
    ).astype(np.float32)

    geology_mask = np.zeros_like(norm_elev, dtype=np.int8)
    geology_mask[crater_depth_m > np.percentile(crater_depth_m, 65)] = 1
    geology_mask[(slope_deg >= 6.0) & (slope_deg <= 17.0)] = 2
    geology_mask[norm_elev > np.percentile(norm_elev, 75)] = 3

    return {
        "elevation": elevation.astype(np.float32),
        "norm_elev": norm_elev.astype(np.float32),
        "slope_rad": slope_rad,
        "aspect_rad": aspect_rad,
        "slope_deg": slope_deg,
        "crater_depth_m": crater_depth_m,
        "hillshade": hillshade,
        "temperature_day": temperature_day,
        "temperature_night": temperature_night,
        "radiation": radiation,
        "geology_mask": geology_mask,
    }


def pick_default_points(layers: dict, cfg: dict) -> dict:
    """En büyük güvenli yürünebilir ağ içinde birbirine KESİN ulaşılabilir İniş ve Hedefler seçer."""
    slope = layers["slope_deg"]
    depth = layers["crater_depth_m"]
    geol = layers["geology_mask"]
    h, w = slope.shape
    margin = 12

    walkable = np.zeros_like(slope, dtype=bool)
    walkable[margin:-margin, margin:-margin] = (slope[margin:-margin, margin:-margin] <= 20.0) & (
        depth[margin:-margin, margin:-margin] <= 26.0
    )

    labeled, num_feat = label(walkable, structure=np.ones((3, 3), dtype=int))
    if num_feat > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        main_label = int(np.argmax(counts))
        reachable = labeled == main_label
    else:
        reachable = np.zeros_like(slope, dtype=bool)
        reachable[margin:-margin, margin:-margin] = True

    flat_land = reachable & (slope < 8.0) & (depth < 10.0)
    if np.any(flat_land):
        ys, xs = np.where(flat_land)
        dist_pref = (xs - w * 0.35) ** 2 + (ys - h * 0.35) ** 2
        idx = int(np.argmin(dist_pref))
        lander_xy = [int(xs[idx]), int(ys[idx])]
    else:
        ys, xs = np.where(reachable)
        best = int(np.argmin(slope[ys, xs]))
        lander_xy = [int(xs[best]), int(ys[best])]

    targets = []
    desired_classes = [2, 3, 1]
    used_pts = [tuple(lander_xy)]

    for k, cls_id in enumerate(desired_classes):
        cand = reachable & (geol == cls_id) & (slope <= 18.5)
        ys, xs = np.where(cand)
        if len(xs) == 0:
            ys, xs = np.where(reachable & (slope <= 18.5))
        if len(xs) == 0:
            ys, xs = np.where(reachable)

        dists_from_lander = np.hypot(xs - lander_xy[0], ys - lander_xy[1])
        valid = (dists_from_lander >= 25) & (dists_from_lander <= 110)
        if np.any(valid):
            xs, ys, dists_from_lander = xs[valid], ys[valid], dists_from_lander[valid]

        order = np.argsort(dists_from_lander)
        pick_idx = order[min(len(order) - 1, int(len(order) * (0.35 + 0.2 * k)))]
        for idx_cand in order[len(order) // 3 :]:
            cx, cy = int(xs[idx_cand]), int(ys[idx_cand])
            if all(np.hypot(cx - ux, cy - uy) >= 20 for ux, uy in used_pts):
                pick_idx = idx_cand
                break

        tx, ty = int(xs[pick_idx]), int(ys[pick_idx])
        used_pts.append((tx, ty))

        name, desc = cfg["target_names"][k]
        targets.append({
            "id": f"T{k+1}",
            "name": name,
            "description": desc,
            "x": tx,
            "y": ty,
            "image_path": "",
        })

    return {"lander": lander_xy, "targets": targets}


def main():
    os.makedirs("processed", exist_ok=True)
    stl_files = sorted(glob.glob("raw/*.stl"))

    if not stl_files:
        print("HATA: raw/ klasöründe .stl dosyası bulunamadı!")
        return

    sites_meta = {}

    for stl_path in stl_files:
        fname = os.path.basename(stl_path).lower()
        region_key = "jezero" if "jezero" in fname else "gale"
        cfg = REGION_CONFIG[region_key]

        print(f"\nİşleniyor: {stl_path} -> Bölge: {region_key}")
        norm_elev, relief_m = load_stl_to_grid(stl_path, GRID_SIZE)
        layers = compute_layers(norm_elev, cfg, relief_m)
        points = pick_default_points(layers, cfg)

        out_npz = os.path.join("processed", f"{region_key}.npz")
        np.savez_compressed(out_npz, **layers)

        sites_meta[region_key] = {
            "display_name": cfg["display_name"],
            "grid_size": GRID_SIZE,
            "area_size_m": AREA_SIZE_M,
            "cell_size_m": CELL_SIZE_M,
            "relief_m": round(relief_m, 1),
            "lander": points["lander"],
            "targets": points["targets"],
        }

        print(f"Kaydedildi: {out_npz} (Rölyef: {relief_m:.1f}m)")
        print(f"  Eğim yüzdelikleri (%25/%50/%75): {np.round(np.percentile(layers['slope_deg'], [25, 50, 75]), 1)}")
        print(f"  Krater derinliği maks: {layers['crater_depth_m'].max():.1f}m")
        print(f"  Varsayılan İniş : {points['lander']}")
        for t in points["targets"]:
            print(f"  Hedef {t['id']} ({t['name']}): ({t['x']}, {t['y']})")

    meta_path = os.path.join("processed", "sites.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sites_meta, f, ensure_ascii=False, indent=2)
    print(f"\nMetadata kaydedildi: {meta_path}")


if __name__ == "__main__":
    main()