import heapq
import math
from collections import deque
import numpy as np


def astar_route(cost_grid: np.ndarray, start_xy: tuple, goal_xy: tuple) -> tuple:
    """
    2D cost_grid üzerinde (x, y) koordinatları için A* en düşük maliyetli rotayı bulur.
    Dönüş: (path_list, error_message)
    """
    h, w = cost_grid.shape
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])

    if not (0 <= sx < w and 0 <= sy < h):
        return None, f"Başlangıç noktası ({sx}, {sy}) harita sınırları (0-{w-1}, 0-{h-1}) dışında!"
    if not (0 <= gx < w and 0 <= gy < h):
        return None, f"Hedef noktası ({gx}, {gy}) harita sınırları (0-{w-1}, 0-{h-1}) dışında!"

    if not np.isfinite(cost_grid[sy, sx]):
        return None, f"Başlangıç (Roket) noktası ({sx}, {sy}) geçilemez krater, dik eğim veya aşırı soğuk bölgede!"
    if not np.isfinite(cost_grid[gy, gx]):
        return None, f"Hedef noktası ({gx}, {gy}) geçilemez krater, dik eğim veya aşırı soğuk bölgede!"

    if (sx, sy) == (gx, gy):
        return [(sx, sy)], None

    finite_vals = cost_grid[np.isfinite(cost_grid)]
    min_cost = float(np.min(finite_vals)) if finite_vals.size > 0 else 1.0

    start_idx = sy * w + sx
    goal_idx = gy * w + gx

    g_score = np.full(h * w, np.inf, dtype=np.float64)
    came_from = np.full(h * w, -1, dtype=np.int32)
    closed = np.zeros(h * w, dtype=bool)

    g_score[start_idx] = 0.0
    h0 = math.hypot(gx - sx, gy - sy) * min_cost
    heap = [(h0, start_idx)]

    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.41421356), (1, -1, 1.41421356),
        (-1, 1, 1.41421356), (1, 1, 1.41421356),
    ]

    while heap:
        _, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        if cur == goal_idx:
            path = []
            node = goal_idx
            while node != -1:
                ny, nx = divmod(int(node), w)
                path.append((nx, ny))
                node = came_from[node]
            path.reverse()
            return path, None

        closed[cur] = True
        cy, cx = divmod(cur, w)
        cur_g = g_score[cur]

        for dx, dy, step_len in neighbors:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                n_idx = ny * w + nx
                if closed[n_idx]:
                    continue
                cell_cost = float(cost_grid[ny, nx])
                if not math.isfinite(cell_cost):
                    continue

                tentative_g = cur_g + step_len * cell_cost
                if tentative_g < g_score[n_idx]:
                    g_score[n_idx] = tentative_g
                    came_from[n_idx] = cur
                    f_val = tentative_g + math.hypot(gx - nx, gy - ny) * min_cost
                    heapq.heappush(heap, (f_val, n_idx))

    return None, "Bu eğim, krater ve sıcaklık kısıtları altında güvenli bir geçit bulunamadı."


def naive_shortest_route(layers: dict, start_xy: tuple, goal_xy: tuple) -> tuple:
    """Sıcaklık, radyasyon ve krater derinliğini yoksayarak en kısa mesafeyi bulur (karşılaştırma için)."""
    naive_cost = np.ones_like(layers["slope_deg"], dtype=np.float32)
    naive_cost[layers["slope_deg"] > 45.0] = np.inf
    return astar_route(naive_cost, start_xy, goal_xy)


def suggest_safe_alternatives(
    layers: dict,
    cost_grid: np.ndarray,
    start_xy: tuple,
    goal_xy: tuple,
    current_max_slope: float,
    current_max_crater: float,
    temp_map: np.ndarray | None = None,
    temp_limit_c: float = -70.0,
) -> dict:
    """
    Rota tıkandığında en yakın güvenli yaklaşım noktasını ve gereken minimum
    eğim / krater / sıcaklık gevşetmesini hesaplar.
    """
    h, w = cost_grid.shape
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    suggestions = {}

    if temp_map is not None:
        if float(temp_map[sy, sx]) < temp_limit_c or float(temp_map[gy, gx]) < temp_limit_c:
            suggestions["thermal_block"] = True

    if not np.isfinite(cost_grid[sy, sx]):
        safe_ys, safe_xs = np.where(np.isfinite(cost_grid))
        if len(safe_xs) > 0:
            idx = int(np.argmin((safe_xs - sx) ** 2 + (safe_ys - sy) ** 2))
            suggestions["nearest_safe_start"] = (int(safe_xs[idx]), int(safe_ys[idx]))
        return suggestions

    visited = np.zeros((h, w), dtype=bool)
    q = deque([(sx, sy)])
    visited[sy, sx] = True
    best_pt = (sx, sy)
    best_dist = math.hypot(gx - sx, gy - sy)

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
    while q:
        cx, cy = q.popleft()
        d = math.hypot(gx - cx, gy - cy)
        if d < best_dist:
            best_dist = d
            best_pt = (cx, cy)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and np.isfinite(cost_grid[ny, nx]):
                visited[ny, nx] = True
                q.append((nx, ny))

    if best_pt != (sx, sy):
        near_path, _ = astar_route(cost_grid, start_xy, best_pt)
        suggestions["nearest_reachable_pt"] = best_pt
        suggestions["nearest_reachable_gap_cells"] = round(best_dist, 1)
        suggestions["nearest_reachable_path"] = near_path

    slope = layers["slope_deg"]
    depth = layers["crater_depth_m"]
    for add_s in [3, 6, 10, 15, 22, 30]:
        test_slope = current_max_slope + add_s
        test_crater = max(current_max_crater, float(depth[gy, gx]) + 2.0, float(depth[sy, sx]) + 2.0)
        test_cost = np.ones_like(slope, dtype=np.float32)
        test_cost[(slope > test_slope) | (depth > test_crater)] = np.inf
        p, _ = astar_route(test_cost, start_xy, goal_xy)
        if p is not None:
            suggestions["min_slope_needed"] = int(test_slope)
            suggestions["min_crater_needed"] = int(math.ceil(test_crater))
            break

    return suggestions


def _compass_direction(dx: float, dy: float) -> str:
    ang = math.degrees(math.atan2(dy, dx))
    if -22.5 <= ang < 22.5:
        return "Doğu"
    elif 22.5 <= ang < 67.5:
        return "Kuzeydoğu"
    elif 67.5 <= ang < 112.5:
        return "Kuzey"
    elif 112.5 <= ang < 157.5:
        return "Kuzeybatı"
    elif ang >= 157.5 or ang < -157.5:
        return "Batı"
    elif -157.5 <= ang < -112.5:
        return "Güneybatı"
    elif -112.5 <= ang < -67.5:
        return "Güney"
    return "Güneydoğu"


def extract_turn_by_turn_steps(
    path: list,
    cell_m: float,
    temp_map: np.ndarray,
    slope_deg: np.ndarray,
    max_steps: int = 7,
) -> list:
    """
    Rotayı sabit parçalara bölmek yerine gerçek yön değişimlerini (krater/engel dönüşlerini)
    algılayarak Google Maps tarzı adım adım Türkçe yol tarifi üretir.
    """
    if not path or len(path) < 2:
        return []

    # 1) Küçük piksel zikzaklarını temizlemek için pencere bazlı yön değişimi tespiti
    win = max(3, min(6, len(path) // 6))
    key_indices = [0]
    last_dir = None

    for i in range(0, len(path) - 1, max(1, win // 2)):
        j = min(len(path) - 1, i + win)
        dx = path[j][0] - path[i][0]
        dy = path[j][1] - path[i][1]
        if dx == 0 and dy == 0:
            continue
        cur_dir = _compass_direction(dx, dy)
        if last_dir is None:
            last_dir = cur_dir
        elif cur_dir != last_dir and (i - key_indices[-1]) >= win:
            key_indices.append(i)
            last_dir = cur_dir

    if key_indices[-1] != len(path) - 1:
        key_indices.append(len(path) - 1)

    # Çok fazla kırılım varsa en uzun/önemli adımları koruyarak sadeleştir
    while len(key_indices) - 1 > max_steps:
        seg_lens = [key_indices[k + 1] - key_indices[k] for k in range(len(key_indices) - 1)]
        shortest = int(np.argmin(seg_lens))
        del_idx = shortest + 1 if shortest == 0 else shortest
        if del_idx < len(key_indices) - 1:
            key_indices.pop(del_idx)
        else:
            break

    steps = []
    for k in range(len(key_indices) - 1):
        i0, i1 = key_indices[k], key_indices[k + 1]
        seg = path[i0 : i1 + 1]
        p0, p1 = seg[0], seg[-1]
        direction = _compass_direction(p1[0] - p0[0], p1[1] - p0[1])

        seg_dist = sum(
            math.hypot(seg[m][0] - seg[m - 1][0], seg[m][1] - seg[m - 1][1]) * cell_m
            for m in range(1, len(seg))
        )
        seg_min_temp = min(float(temp_map[py, px]) for px, py in seg)
        seg_max_slope = max(float(slope_deg[py, px]) for px, py in seg)

        warn = ""
        if seg_max_slope >= 14.0:
            warn += f" ⚠️ Dik eğim ({seg_max_slope:.0f}°)"
        if seg_min_temp <= -60.0:
            warn += f" ❄️ Soğuk ({seg_min_temp:.0f}°C)"

        steps.append(
            f"**{k+1}. {direction}** yönünde **~{seg_dist:.0f} m** ilerle `({p0[0]},{p0[1]}) → ({p1[0]},{p1[1]})`{warn}"
        )

    return steps