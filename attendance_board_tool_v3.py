
import json
import cv2
import numpy as np

DEFAULT_CONFIG = {
    "board_corners": [[72, 18], [292, 25], [302, 744], [72, 718]],
    "warp_size": [320, 920],
    "row_labels": [
        "Taryn", "Sean", "Noah", "OM Andrew", "Aman", "Andrew", "Breanna", "Darren",
        "Eric", "Hailey", "Kaitlyn", "Kam", "Kari", "Keena", "Pam", "Rachel", "Sara",
        "Sarah I", "Teresa", "Vanessa", "Redacted 1", "Redacted 2", "Kim", "Harlos", "Dermot"
    ],
    "row_count": 25,
    "header_search_max_y_frac": 0.16,
    "cell_margin_x_frac": 0.12,
    "cell_margin_y_frac": 0.18,
    "threshold_block_size": 31,
    "threshold_C": 15,
    "vertical_kernel_h": 41,
    "horizontal_kernel_w": 39,
    "separator_refine_search_px": 18,
    "separator_refine_band_half_height": 22,
    "piecewise_output_size": [320, 920]
}


def contiguous_clusters(indices):
    if len(indices) == 0:
        return []
    indices = np.array(indices, dtype=int)
    groups = []
    start = prev = int(indices[0])
    for v in indices[1:]:
        v = int(v)
        if v <= prev + 1:
            prev = v
        else:
            groups.append((start, prev))
            start = prev = v
    groups.append((start, prev))
    return groups


def rectify_board(image, corners, warp_size):
    w, h = warp_size
    src = np.float32(corners)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped, M


def adaptive_binary(gray, block_size=31, C=15):
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block_size, C)


def detect_vertical_separators(binary, config):
    h, w = binary.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(config['vertical_kernel_h'])))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)), iterations=1)
    xproj = vert.sum(axis=0) / 255.0
    candidate_idx = np.where(xproj > np.percentile(xproj, 90))[0]
    clusters = contiguous_clusters(candidate_idx)
    feats = []
    for a, b in clusters:
        center = (a + b) / 2
        strength = float(xproj[a:b+1].max())
        coverage = float(xproj[a:b+1].sum())
        feats.append({"a": a, "b": b, "center": center, "strength": strength, "coverage": coverage})
    feats_sorted = sorted(feats, key=lambda d: d['strength'], reverse=True)
    top = sorted(feats_sorted[:8], key=lambda d: d['center'])
    centers = [d['center'] for d in top]
    merged = []
    for c in centers:
        if not merged or abs(c - merged[-1]) > 10:
            merged.append(c)
        else:
            merged[-1] = (merged[-1] + c) / 2
    if not merged or merged[0] > 15:
        merged = [2.0] + merged
    if not merged or merged[-1] < w - 15:
        merged = merged + [w - 2.0]
    if len(merged) >= 5:
        left = merged[0]
        right = merged[-1]
        inner = merged[1:-1]
        targets = [w*0.47, w*0.61, w*0.76]
        chosen = []
        for t in targets:
            if inner:
                c = min(inner, key=lambda x: abs(x - t))
                chosen.append(c)
                inner = [x for x in inner if abs(x-c) > 10]
        sep = [left] + sorted(chosen) + [right]
    else:
        sep = [2.0, w*0.47, w*0.61, w*0.76, w-2.0]
    return [int(round(x)) for x in sep], vert, xproj, feats


def detect_row_lines(binary, config):
    h, w = binary.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(config['horizontal_kernel_w']), 1))
    horz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    horz = cv2.dilate(horz, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)), iterations=1)
    yproj = horz.sum(axis=1) / 255.0
    cand = np.where(yproj > np.percentile(yproj, 88))[0]
    clusters = contiguous_clusters(cand)
    peaks = []
    for a, b in clusters:
        center = (a + b) / 2
        strength = float(yproj[a:b+1].max())
        peaks.append({"a": a, "b": b, "center": center, "strength": strength})
    peaks = sorted(peaks, key=lambda d: d['center'])
    max_y = int(h * float(config['header_search_max_y_frac']))
    top_peaks = [p for p in peaks if p['center'] <= max_y and p['strength'] > np.percentile(yproj, 94)]
    header_candidates = [p for p in top_peaks if p['center'] > 20]
    header_bottom = int(round(header_candidates[0]['center'])) if header_candidates else int(round(h*0.065))
    row_like = [p for p in peaks if p['center'] > header_bottom + 10]
    row_centers = np.array([p['center'] for p in row_like], dtype=float)
    diffs = np.diff(row_centers) if len(row_centers) >= 2 else np.array([])
    plausible = diffs[(diffs >= 24) & (diffs <= 45)]
    pitch = float(np.median(plausible)) if len(plausible) else (h - header_bottom) / config['row_count']
    n = int(config['row_count'])
    lines = [header_bottom + i*pitch for i in range(n+1)]
    detected = np.array([p['center'] for p in peaks], dtype=float)
    snapped = []
    for pred in lines:
        if len(detected) == 0:
            snapped.append(pred)
            continue
        idx = np.argmin(np.abs(detected - pred))
        if abs(detected[idx] - pred) <= max(6, pitch * 0.18):
            snapped.append(detected[idx])
        else:
            snapped.append(pred)
    snapped = np.array(snapped, dtype=int)
    snapped[0] = max(0, min(h-2, snapped[0]))
    for i in range(1, len(snapped)):
        snapped[i] = max(snapped[i-1] + 1, min(h-1, snapped[i]))
    return snapped.tolist(), horz, yproj, peaks, header_bottom, pitch


def refine_separator_mesh(vertical_mask, base_separators, row_lines, config):
    h, w = vertical_mask.shape[:2]
    search = int(config.get('separator_refine_search_px', 18))
    band_h = int(config.get('separator_refine_band_half_height', 22))
    mesh = np.zeros((len(row_lines), len(base_separators)), dtype=np.float32)
    for i, y in enumerate(row_lines):
        y1 = max(0, int(y - band_h))
        y2 = min(h, int(y + band_h + 1))
        band = vertical_mask[y1:y2, :]
        for j, x0 in enumerate(base_separators):
            if j == 0 or j == len(base_separators)-1:
                x = max(0, min(w-1, x0))
            else:
                xl = max(0, int(x0 - search))
                xr = min(w, int(x0 + search + 1))
                region = band[:, xl:xr]
                colsum = region.sum(axis=0) / 255.0
                if colsum.size == 0 or np.all(colsum == 0):
                    x = x0
                else:
                    kernel = np.array([1,2,3,2,1], dtype=np.float32)
                    sm = np.convolve(colsum, kernel/kernel.sum(), mode='same')
                    x = xl + int(np.argmax(sm))
            mesh[i, j] = float(x)
    for i in range(mesh.shape[0]):
        for j in range(1, mesh.shape[1]):
            mesh[i, j] = max(mesh[i, j], mesh[i, j-1] + 8)
    return mesh


def build_target_mesh(separator_mesh, row_lines, config):
    out_w, out_h = config.get('piecewise_output_size', config['warp_size'])
    sep_target = np.median(separator_mesh, axis=0)
    left = sep_target[0]
    right = sep_target[-1]
    span = max(1.0, right - left)
    margin = 2.0
    sep_target = (sep_target - left) / span * (out_w - 2*margin) + margin
    rows = np.array(row_lines, dtype=np.float32)
    top = rows[0]
    bottom = rows[-1]
    y_span = max(1.0, bottom - top)
    y_target = (rows - top) / y_span * (out_h - 2*margin) + margin
    return sep_target.astype(np.float32), y_target.astype(np.float32)


def piecewise_grid_warp(image, separator_mesh, row_lines, target_x, target_y):
    out_w = int(round(target_x[-1] + 2))
    out_h = int(round(target_y[-1] + 2))
    out = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    mask_acc = np.zeros((out_h, out_w), dtype=np.uint8)
    n_rows = len(row_lines) - 1
    n_cols = separator_mesh.shape[1] - 1
    for r in range(n_rows):
        y_top = float(row_lines[r])
        y_bot = float(row_lines[r+1])
        for c in range(n_cols):
            src = np.float32([
                [separator_mesh[r, c], y_top],
                [separator_mesh[r, c+1], y_top],
                [separator_mesh[r+1, c+1], y_bot],
                [separator_mesh[r+1, c], y_bot],
            ])
            dst = np.float32([
                [target_x[c], target_y[r]],
                [target_x[c+1], target_y[r]],
                [target_x[c+1], target_y[r+1]],
                [target_x[c], target_y[r+1]],
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            warped_patch = cv2.warpPerspective(image, M, (out_w, out_h))
            patch_mask = np.zeros((out_h, out_w), dtype=np.uint8)
            cv2.fillConvexPoly(patch_mask, np.round(dst).astype(np.int32), 255)
            inv = cv2.bitwise_not(patch_mask)
            out_bg = cv2.bitwise_and(out, out, mask=inv)
            patch_fg = cv2.bitwise_and(warped_patch, warped_patch, mask=patch_mask)
            out = cv2.add(out_bg, patch_fg)
            mask_acc = cv2.bitwise_or(mask_acc, patch_mask)
    return out, mask_acc


def build_cells_from_target_grid(target_x, target_y, labels, config):
    cells = []
    margin_x_frac = float(config['cell_margin_x_frac'])
    margin_y_frac = float(config['cell_margin_y_frac'])
    for i in range(len(target_y)-1):
        y1, y2 = float(target_y[i]), float(target_y[i+1])
        row_h = max(1.0, y2 - y1)
        my = row_h * margin_y_frac
        in_w = max(1.0, target_x[2] - target_x[1])
        out_w = max(1.0, target_x[3] - target_x[2])
        mx_in = in_w * margin_x_frac
        mx_out = out_w * margin_x_frac
        in_cell = [target_x[1] + mx_in, y1 + my, target_x[2] - mx_in, y2 - my]
        out_cell = [target_x[2] + mx_out, y1 + my, target_x[3] - mx_out, y2 - my]
        cells.append({
            'label': labels[i] if i < len(labels) else f'Row {i+1}',
            'row_index': i,
            'row_bounds': [int(round(y1)), int(round(y2))],
            'in_cell': [int(round(v)) for v in in_cell],
            'out_cell': [int(round(v)) for v in out_cell],
        })
    return cells


def cell_darkness(gray, rect):
    x1, y1, x2, y2 = map(int, rect)
    x1 = max(0, min(gray.shape[1]-1, x1))
    x2 = max(0, min(gray.shape[1], x2))
    y1 = max(0, min(gray.shape[0]-1, y1))
    y2 = max(0, min(gray.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = gray[y1:y2, x1:x2]
    return float(255.0 - roi.mean())


def analyze(image_path, config_path=None, output_prefix='attendance_board_v3'):
    config = DEFAULT_CONFIG if config_path is None else json.load(open(config_path, 'r', encoding='utf-8'))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    warped, _ = rectify_board(image, config['board_corners'], config['warp_size'])
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    binary = adaptive_binary(gray, config['threshold_block_size'], config['threshold_C'])
    base_separators, vertical_mask, _, _ = detect_vertical_separators(binary, config)
    row_lines, horizontal_mask, _, _, header_bottom, pitch = detect_row_lines(binary, config)
    separator_mesh = refine_separator_mesh(vertical_mask, base_separators, row_lines, config)
    target_x, target_y = build_target_mesh(separator_mesh, row_lines, config)
    piecewise_warp, piecewise_mask = piecewise_grid_warp(warped, separator_mesh, row_lines, target_x, target_y)
    labels = config['row_labels']
    cells = build_cells_from_target_grid(target_x, target_y, labels, config)
    pw_gray = cv2.cvtColor(piecewise_warp, cv2.COLOR_BGR2GRAY)

    source_overlay = warped.copy()
    for i in range(len(row_lines)):
        for j in range(len(base_separators)):
            x = int(round(separator_mesh[i, j])); y = int(round(row_lines[i]))
            cv2.circle(source_overlay, (x, y), 2, (0, 0, 255), -1)
    for i in range(len(row_lines)):
        pts = np.array([[int(round(separator_mesh[i, j])), int(round(row_lines[i]))] for j in range(len(base_separators))], dtype=np.int32)
        cv2.polylines(source_overlay, [pts], False, (0,255,255), 1)
    for j in range(len(base_separators)):
        pts = np.array([[int(round(separator_mesh[i, j])), int(round(row_lines[i]))] for i in range(len(row_lines))], dtype=np.int32)
        cv2.polylines(source_overlay, [pts], False, (255,255,0), 1)

    piecewise_overlay = piecewise_warp.copy()
    for x in target_x.astype(int):
        cv2.line(piecewise_overlay, (int(x),0), (int(x), piecewise_overlay.shape[0]-1), (0,255,255), 1)
    for y in target_y.astype(int):
        cv2.line(piecewise_overlay, (0,int(y)), (piecewise_overlay.shape[1]-1, int(y)), (255,255,0), 1)

    rows = []
    for cell in cells:
        in_d = cell_darkness(pw_gray, cell['in_cell'])
        out_d = cell_darkness(pw_gray, cell['out_cell'])
        delta = None if in_d is None or out_d is None else round(in_d - out_d, 3)
        status = 'IN darker' if delta is not None and delta > 0 else 'OUT darker'
        rows.append({
            'label': cell['label'],
            'row_index': cell['row_index'],
            'row_bounds': cell['row_bounds'],
            'in_cell': cell['in_cell'],
            'out_cell': cell['out_cell'],
            'in_darkness': None if in_d is None else round(in_d, 3),
            'out_darkness': None if out_d is None else round(out_d, 3),
            'delta_in_minus_out': delta,
            'result': status,
        })
        ix1, iy1, ix2, iy2 = cell['in_cell']
        ox1, oy1, ox2, oy2 = cell['out_cell']
        cv2.rectangle(piecewise_overlay, (ix1, iy1), (ix2, iy2), (255,0,0), 1)
        cv2.rectangle(piecewise_overlay, (ox1, oy1), (ox2, oy2), (0,0,255), 1)
        cy = int((cell['row_bounds'][0] + cell['row_bounds'][1]) / 2)
        cv2.putText(piecewise_overlay, cell['label'], (4, min(piecewise_overlay.shape[0]-5, cy+4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 180, 0), 1, cv2.LINE_AA)

    result = {
        'image': image_path,
        'warp_size': config['warp_size'],
        'piecewise_output_size': config.get('piecewise_output_size', config['warp_size']),
        'board_corners': config['board_corners'],
        'base_separators': base_separators,
        'row_lines': row_lines,
        'target_x': [float(round(x,3)) for x in target_x],
        'target_y': [float(round(y,3)) for y in target_y],
        'header_bottom': int(header_bottom),
        'row_pitch': round(float(pitch), 3),
        'separator_mesh': [[float(round(v,3)) for v in row] for row in separator_mesh],
        'rows': rows,
    }

    cv2.imwrite(f'{output_prefix}_warp.png', warped)
    cv2.imwrite(f'{output_prefix}_binary.png', binary)
    cv2.imwrite(f'{output_prefix}_verticals.png', vertical_mask)
    cv2.imwrite(f'{output_prefix}_horizontals.png', horizontal_mask)
    cv2.imwrite(f'{output_prefix}_source_mesh_overlay.png', source_overlay)
    cv2.imwrite(f'{output_prefix}_piecewise_warp.png', piecewise_warp)
    cv2.imwrite(f'{output_prefix}_piecewise_overlay.png', piecewise_overlay)
    cv2.imwrite(f'{output_prefix}_piecewise_mask.png', piecewise_mask)
    with open(f'{output_prefix}_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Attendance board mapper with piecewise grid warping')
    parser.add_argument('image', nargs='?', default='image.png')
    parser.add_argument('--config', default=None)
    parser.add_argument('--out', default='attendance_board_v3')
    args = parser.parse_args()
    analyze(args.image, args.config, args.out)
