# RANSAC Electrode Grouping Design

## Problem

The current single-linkage hierarchical clustering approach for grouping
LoG-detected contacts into electrodes fails catastrophically:

- **Over-merges** 16 electrodes into ~3 clusters (single-linkage chaining)
- **Misplaces contacts** by reprojecting onto fitted axis instead of keeping
  original LoG positions (origin bug in `orient_deepest_first`)
- **Flips electrode orientation** using unreliable RAS origin heuristic

Crossings are very common in SEEG data (6-25 electrodes, 6-18 contacts each).

## Solution: RANSAC Line Fitting

Replace the grouping pipeline with iterative RANSAC. Each iteration finds the
best-supported line through the contact point cloud, claims its inliers, removes
them, and repeats.

### Why RANSAC

- **Handles crossings naturally** — contacts from electrode B are perpendicular
  outliers to electrode A's line, even when they're close in 3D
- **Preserves LoG positions** — contacts are assigned to groups, never
  reprojected
- **No pairwise distance matrix** — O(N) per iteration, not O(N^2)
- **Simple to implement and debug** — core loop is ~50 lines

### Algorithm

```
pool = all LoG contact positions (N x 3, RAS)
electrodes = []

while len(pool) >= min_contacts:
    best_inliers = None

    for trial in range(max_iterations):
        # Sample 2 random points, define candidate line
        p1, p2 = random_sample(pool, 2)
        direction = normalize(p2 - p1)

        # Find inliers: perpendicular distance < distance_tolerance
        distances = point_to_line_distance(pool, p1, direction)
        inlier_mask = distances < distance_tolerance
        inliers = pool[inlier_mask]

        # Validate: enough contacts + plausible spacing
        if len(inliers) < min_contacts:
            continue
        projections = project_onto_line(inliers, p1, direction)
        median_spacing = median_neighbor_spacing(sorted(projections))
        if not (expected_spacing * 0.5 < median_spacing < expected_spacing * 2.0):
            continue

        # Score: most inliers wins
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inlier_mask

    if best_inliers is None:
        break

    # Refit via PCA on inliers, then re-check for additional inliers
    # Remove claimed contacts from pool
    electrodes.append(build_electrode(pool[best_inliers]))
    pool = pool[~best_inliers]
```

### Post-Grouping (per electrode)

1. **PCA refit** on inlier coordinates (existing `fit_electrode_axis()`)
2. **Project + sort** contacts along axis
3. **Orient deepest-first** using brain mask centroid (not RAS origin)
4. **Analyze spacing** for gap detection (existing `analyze_spacing()`)
5. **Build Electrode** with original LoG positions (no reprojection)

### Gap Handling

Gapped electrodes work naturally with RANSAC. The gap is along the line, not
perpendicular to it — all contacts still lie within `distance_tolerance` of the
same line. The existing `analyze_spacing()` detects bimodal spacing after
grouping.

### Deepest-First Orientation

Replace RAS origin heuristic with brain mask centroid:

```python
def orient_deepest_first(contact_coords, sorted_indices, direction, brain_centroid):
    first_ras = contact_coords[sorted_indices[0]]
    last_ras = contact_coords[sorted_indices[-1]]
    if norm(first_ras - brain_centroid) > norm(last_ras - brain_centroid):
        return sorted_indices[::-1], -direction
    return sorted_indices, direction
```

Fallback to (0,0,0) if no brain mask available.

## Parameters (all exposed in GUI)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_tolerance` | 2.0 mm | Max perpendicular distance to claim contact |
| `max_iterations` | 1000 | RANSAC trials per electrode |
| `min_contacts` | 3 | Minimum contacts per electrode |
| `expected_spacing` | 3.5 mm | Expected contact spacing |
| `spacing_tolerance` | 0.5-2.0x | Accepted range around expected_spacing |

## Code Changes

### Deleted

- `cluster_into_electrodes()` — single-linkage clustering
- `merge_collinear_clusters()` — collinear gap merging
- Old `orient_deepest_first()` — RAS origin heuristic
- Reprojection logic in `detect_electrodes()`

### Kept

- `fit_electrode_axis()` — PCA (used for refit after RANSAC)
- `analyze_spacing()` — gap detection (well-tested)
- `Electrode`, `Contact`, `ElectrodeParams` data structures
- `_filter_contact_mask()` — unchanged
- All LoG detection code

### New

- `ransac_group_contacts()` — core RANSAC loop
- Updated `orient_deepest_first()` — brain centroid
- Updated `detect_electrodes()` — calls RANSAC, preserves LoG positions
- GUI widgets for RANSAC parameters

## Deferred

- **Bug #5** (`_filter_contact_mask` drops merged blobs): Low impact since LoG
  finds centers independently of component size.
- **Bug #6** (LoG sigma anisotropy): Post-implant CTs are near-isotropic;
  sigma 1.2-1.5 works well in practice.
