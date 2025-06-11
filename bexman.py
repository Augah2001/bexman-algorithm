import logging
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# Configure module‐level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def batched_cdist(A: np.ndarray, B: np.ndarray, batch_size: int = None) -> np.ndarray:
    """
    Computes pairwise Euclidean distances between A and B, optionally in batches.
    """
    # --- Input validation ---
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("batched_cdist: A and B must be numpy arrays.")
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(
            f"batched_cdist: A and B must be 2D with same n_features; "
            f"got A.shape={A.shape}, B.shape={B.shape}."
        )
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
        raise ValueError("batched_cdist: batch_size must be a positive integer or None.")

    # --- Compute distances ---
    try:
        if batch_size is None:
            return np.linalg.norm(A[:, None] - B[None, :], axis=2)
        m, n = A.shape[0], B.shape[0]
        distances = np.empty((m, n), dtype=np.float64)
        for i in range(0, m, batch_size):
            end = min(i + batch_size, m)
            distances[i:end] = np.linalg.norm(A[i:end, None] - B[None, :], axis=2)
        return distances
    except Exception as e:
        logger.error("batched_cdist failed: %s", e, exc_info=True)
        raise

def lightweight_kmeans(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int = 10,
    random_state: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple KMeans++ + fixed Lloyd iterations.
    """
    # --- Input validation ---
    if not isinstance(X, np.ndarray):
        raise ValueError("lightweight_kmeans: X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("lightweight_kmeans: X must be 2D.")
    n_samples, _ = X.shape
    if n_samples == 0:
        raise ValueError("lightweight_kmeans: X must contain at least one sample.")
    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("lightweight_kmeans: n_clusters must be a positive integer.")
    if n_clusters > n_samples:
        raise ValueError(
            f"lightweight_kmeans: n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})."
        )
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("lightweight_kmeans: max_iter must be a positive integer.")

    # --- Initialization (k-means++) ---
    try:
        if random_state is not None:
            np.random.seed(random_state)
        centers = [X[np.random.randint(n_samples)]]
        for _ in range(1, n_clusters):
            dist_sq = np.min(cdist(X, np.vstack(centers)) ** 2, axis=1)
            total = dist_sq.sum()
            if total <= 0:
                probs = np.full(n_samples, 1.0 / n_samples)
            else:
                probs = dist_sq / total
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            idx = np.searchsorted(cumprobs, r)
            centers.append(X[idx])
        centers = np.vstack(centers)
    except Exception as e:
        logger.error("lightweight_kmeans init failed: %s", e, exc_info=True)
        raise

    # --- Lloyd iterations ---
    try:
        labels = np.zeros(n_samples, dtype=int)
        for it in range(max_iter):
            dists = cdist(X, centers)
            new_labels = np.argmin(dists, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                pts = X[new_labels == k]
                if pts.size:
                    new_centers[k] = pts.mean(axis=0)
                else:
                    # no points assigned → keep old center
                    new_centers[k] = centers[k]
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
            labels = new_labels
        return centers, labels
    except Exception as e:
        logger.error("lightweight_kmeans Lloyd failed: %s", e, exc_info=True)
        raise

class Bexman:
    def __init__(
        self,
        n_kmeans_clusters: int = 80,
        k: int = 5,
        alpha: float = 1.0,
        sparse_alpha: float = 1.0,
        min_cluster_size: int = 10,
        gamma: float = 2.3,
        ratio_threshold: float = 1.0,
        beta: float = 0.0,
        random_state: int = 42,
        batch_size: int = None
    ):
        # --- Parameter validation ---
        if not isinstance(n_kmeans_clusters, int) or n_kmeans_clusters <= 0:
            raise ValueError("n_kmeans_clusters must be a positive integer.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if alpha < 0 or sparse_alpha < 0 or gamma < 0 or ratio_threshold < 0:
            raise ValueError("alpha, sparse_alpha, gamma, and ratio_threshold must be non‑negative.")
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least 1.")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError("batch_size must be a positive integer or None.")

        # --- Store ---
        self.n_kmeans_clusters = n_kmeans_clusters
        self.k = k
        self.alpha = alpha
        self.sparse_alpha = sparse_alpha
        self.min_cluster_size = min_cluster_size
        self.gamma = gamma
        self.ratio_threshold = ratio_threshold
        self.beta = beta
        self.random_state = random_state
        self.batch_size = batch_size

        # Placeholders
        self.points = None
        self.kmeans_centers = None
        self.kmeans_labels = None
        self.center_triangle = None
        self.best_costs = None
        self.old_to_new_center_map = None
        self.center_to_final_cluster = None
        self.final_labels = None
        self.final_labels_noise = None
        self.cluster_thresholds = None

    def fit_predict(self, points: np.ndarray) -> np.ndarray:
        """
        Runs the full Bexman pipeline on `points` and returns final labels (with noise = -1).
        """
        try:
            # --- Input checks ---
            if not isinstance(points, np.ndarray):
                raise ValueError("fit_predict: `points` must be a numpy array.")
            if points.ndim != 2:
                raise ValueError("fit_predict: `points` must be 2D (n_samples, n_features).")
            n_samples, _ = points.shape
            if n_samples == 0:
                raise ValueError("fit_predict: cannot fit on empty dataset.")
            if n_samples < self.n_kmeans_clusters:
                raise ValueError(
                    f"fit_predict: n_kmeans_clusters ({self.n_kmeans_clusters}) "
                    f"> number of samples ({n_samples})."
                )
            self.points = points

            # --- Step 1: initial KMeans ---
            centers, labels = lightweight_kmeans(
                points,
                n_clusters=self.n_kmeans_clusters,
                max_iter=10,
                random_state=self.random_state
            )
            self.kmeans_centers = centers
            self.kmeans_labels = labels

            # --- Step 1.5: filter sparse centers & reassign ---
            num_centers = centers.shape[0]
            if num_centers < self.k + 1:
                raise ValueError(
                    f"Not enough centers ({num_centers}) to find {self.k} neighbors."
                )
            nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(centers)
            dists_c, idxs_c = nbrs.kneighbors(centers)
            avg_d = dists_c[:, 1:].mean(axis=1)
            thresh = np.median(avg_d) + self.sparse_alpha * np.std(avg_d)
            dense_mask = avg_d <= thresh
            if not dense_mask.any():
                raise ValueError("All centers filtered out as sparse; adjust `sparse_alpha`.")
            dense_idxs = np.flatnonzero(dense_mask)
            self.old_to_new_center_map = {
                old: new for new, old in enumerate(dense_idxs)
            }
            self.kmeans_centers = centers[dense_mask]

            # reassign labels
            new_labels = np.array(
                [self.old_to_new_center_map.get(l, -1) for l in labels],
                dtype=int
            )
            to_reassign = np.where(new_labels == -1)[0]
            if to_reassign.size:
                nbrs_dense = NearestNeighbors(n_neighbors=1).fit(self.kmeans_centers)
                _, new_idx = nbrs_dense.kneighbors(points[to_reassign])
                new_labels[to_reassign] = new_idx.flatten()
            self.kmeans_labels = new_labels

            # --- Step 2: best triangles & costs ---
            num_centers = self.kmeans_centers.shape[0]
            if num_centers < 3:
                raise ValueError("Need at least 3 centers to form any triangle.")
            k_n = min(self.k, num_centers - 1)
            nbrs2 = NearestNeighbors(n_neighbors=k_n + 1).fit(self.kmeans_centers)
            d2, i2 = nbrs2.kneighbors(self.kmeans_centers)
            center_tri = {}
            best_costs = np.empty(num_centers, dtype=float)
            for i in range(num_centers):
                neigh = i2[i, 1:]
                neigh_d = d2[i, 1:]
                pairwise = cdist(self.kmeans_centers[neigh], self.kmeans_centers[neigh])
                comb = np.triu_indices(len(neigh), k=1)
                costs = neigh_d[comb[0]] + neigh_d[comb[1]] + self.beta * pairwise[comb]
                if costs.size == 0:
                    raise ValueError(f"Cannot form triangle around center {i}.")
                mi = np.argmin(costs)
                best_costs[i] = costs[mi]
                a, b = comb[0][mi], comb[1][mi]
                center_tri[i] = tuple(sorted((i, neigh[a], neigh[b])))
            self.center_triangle = center_tri
            self.best_costs = best_costs.tolist()

            # --- Step 2b: local thresholds ---
            full_d = batched_cdist(
                self.kmeans_centers, self.kmeans_centers, batch_size=self.batch_size
            )
            np.fill_diagonal(full_d, np.inf)
            nearest = np.argsort(full_d, axis=1)[:, :self.k]
            local_thresh = (
                np.median(full_d[np.arange(num_centers)[:, None], nearest], axis=1)
                + self.alpha
                * np.std(full_d[np.arange(num_centers)[:, None], nearest], axis=1)
            )

            # --- Step 3: union-find merge ---
            parent = np.arange(num_centers)
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for i, tri in self.center_triangle.items():
                if best_costs[i] <= local_thresh[i]:
                    for v in tri:
                        union(i, v)
            roots = np.array([find(i) for i in range(num_centers)])
            merged = {}
            for r in np.unique(roots):
                merged[r] = np.where(roots == r)[0].tolist()
            self.center_to_final_cluster = {
                idx: root for root, members in merged.items() for idx in members
            }

            # --- Step 4: assign & filter small clusters ---
            final_lbl = np.array(
                [self.center_to_final_cluster.get(l, l) for l in self.kmeans_labels],
                dtype=int
            )
            self.final_labels = final_lbl.copy()

            # densities & candidate values
            clusters = np.unique(final_lbl)
            densities = {}
            for cl in clusters:
                idxs = np.where(final_lbl == cl)[0]
                if idxs.size == 0:
                    densities[cl] = np.inf
                else:
                    pts = points[idxs]
                    cent_idxs = self.kmeans_labels[idxs]
                    ctrs = self.kmeans_centers[cent_idxs]
                    densities[cl] = np.linalg.norm(pts - ctrs, axis=1).mean()
            global_med = np.median(list(densities.values()))
            cand = {}
            for root, members in merged.items():
                size = np.sum(final_lbl == root)
                dens = densities[root]
                val = (size / (len(members))) * (global_med / (dens))
                cand[root] = 0.0 if val > 1e9 else val
            # keep only finite
            cand = {c: v for c, v in cand.items() if np.isfinite(v)}

            valid = [c for c, v in cand.items() if v >= self.min_cluster_size]
            small = [c for c, v in cand.items() if v < self.min_cluster_size]
            if valid:
                valid_idxs = [i for i, cl in self.center_to_final_cluster.items() if cl in valid]
                nbrs_v = NearestNeighbors(n_neighbors=1).fit(self.kmeans_centers[valid_idxs])
                for i in range(final_lbl.shape[0]):
                    if final_lbl[i] in small:
                        _, new_i = nbrs_v.kneighbors(points[i].reshape(1, -1))
                        nearest_center = valid_idxs[new_i[0][0]]
                        final_lbl[i] = self.center_to_final_cluster[nearest_center]
            self.final_labels = final_lbl.copy()

            # --- Step 5: distance-based noise detection ---
            noise_lbl = self.final_labels.copy()
            non_noise = np.where(self.final_labels != -1)[0]
            if non_noise.size:
                pts = points[non_noise]
                cent_idxs = self.kmeans_labels[non_noise]
                ctrs = self.kmeans_centers[cent_idxs]
                d = np.linalg.norm(pts - ctrs, axis=1)
                uniq, inv = np.unique(self.final_labels[non_noise], return_inverse=True)
                means = np.array([d[inv == i].mean() for i in range(len(uniq))])
                thr = means[inv] * self.gamma
                noise_lbl[non_noise[d > thr]] = -1
            self.final_labels_noise = noise_lbl

            # --- Step 6: competing-center noise ---
            valid_mask = self.final_labels_noise != -1
            valid_idxs = np.where(valid_mask)[0]
            if valid_idxs.size:
                pts = points[valid_idxs]
                all_d = batched_cdist(pts, self.kmeans_centers, batch_size=self.batch_size)
                part = np.argpartition(all_d, 1, axis=1)[:, :2]
                order = np.argsort(np.take_along_axis(all_d, part, axis=1), axis=1)
                two = np.take_along_axis(part, order, axis=1)
                d1 = all_d[np.arange(two.shape[0]), two[:, 0]]
                d2 = all_d[np.arange(two.shape[0]), two[:, 1]]
                ratios = np.maximum(d1, d2) / (np.minimum(d1, d2) + 1e-9)
                cond = ratios > self.ratio_threshold
                cl1 = np.array([self.center_to_final_cluster[i] for i in two[:, 0]])
                cl2 = np.array([self.center_to_final_cluster[i] for i in two[:, 1]])
                noise = cond & (cl1 != cl2)
                self.final_labels_noise[valid_idxs[noise]] = -1

            # --- Step 7: compute per-cluster thresholds ---
            non_nz = np.where(self.final_labels != -1)[0]
            if non_nz.size:
                pts = points[non_nz]
                cent_idxs = self.kmeans_labels[non_nz]
                ctrs = self.kmeans_centers[cent_idxs]
                d = np.linalg.norm(pts - ctrs, axis=1)
                uniq, inv = np.unique(self.final_labels[non_nz], return_inverse=True)
                means2 = np.array([d[inv == i].mean() for i in range(len(uniq))])
                self.cluster_thresholds = {
                    cl: means2[i] * self.gamma for i, cl in enumerate(uniq)
                }
            else:
                self.cluster_thresholds = {}

            # --- Reassign final cluster labels sequentially starting at 1 ---
            # Build a mapping for non-noise labels, leaving the noise label (-1) unchanged.
            final = self.final_labels_noise.copy()
            unique_clusters = sorted(set(final) - {-1})
            label_mapping = {old: new for new, old in enumerate(unique_clusters, start=1)}
            for i in range(final.shape[0]):
                if final[i] != -1:
                    final[i] = label_mapping[final[i]]
            self.final_labels_noise = final

            return self.final_labels_noise

        except Exception as e:
            logger.error("Bexman.fit_predict failed: %s", e, exc_info=True)
            raise

    def predict(self, points: np.ndarray, apply_noise_detection: bool = True) -> np.ndarray:
        """
        Assign new points to clusters (optionally applying the same noise thresholds).
        """
        if self.kmeans_centers is None or self.center_to_final_cluster is None:
            raise ValueError("Model has not been fitted yet.")
        if not isinstance(points, np.ndarray) or points.ndim != 2:
            raise ValueError("predict: points must be a 2D numpy array.")

        n_samples, _ = points.shape
        labels = np.empty(n_samples, dtype=int)
        for i, p in enumerate(points):
            d = np.linalg.norm(self.kmeans_centers - p, axis=1)
            idx = np.argmin(d)
            lab = self.center_to_final_cluster.get(int(idx), int(idx))
            if apply_noise_detection and lab in self.cluster_thresholds:
                if d[idx] > self.cluster_thresholds[lab]:
                    lab = -1
            labels[i] = lab

        # Optionally reassign the labels sequentially (excluding noise) if desired.
        unique_clusters = sorted(set(labels) - {-1})
        label_mapping = {old: new for new, old in enumerate(unique_clusters, start=1)}
        for i in range(labels.shape[0]):
            if labels[i] != -1:
                labels[i] = label_mapping[labels[i]]
        return labels

    def plot_clusters(self, use_noise=True, figsize=(10, 8)):
        if self.points is None:
            raise ValueError("Model has not been fitted.")
        labels = self.final_labels_noise if use_noise else self.final_labels
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        color_map = cm.get_cmap('viridis', num_labels)
        colors = {lbl: (0, 0, 0, 1) if lbl == -1 else color_map(i / (num_labels - 1))
                  for i, lbl in enumerate(sorted(unique_labels))}
        plt.figure(figsize=figsize)
        for lbl in unique_labels:
            pts = self.points[labels == lbl]
            label_str = "Noise" if lbl == -1 else f"Cluster {lbl}"
            plt.scatter(pts[:, 0], pts[:, 1], color=colors[lbl], s=20, label=label_str, zorder=3)
        plt.title("Robust Final Clusters" + (" with Noise" if use_noise else ""))
        plt.legend()
        plt.axis('equal')
        plt.show()

    def plot_best_triangles(self, figsize=(16, 12)):
        if self.kmeans_centers is None or self.center_triangle is None:
            raise ValueError("Model must be fitted before plotting best triangles.")
        plt.figure(figsize=figsize)
        plt.scatter(self.kmeans_centers[:, 0], self.kmeans_centers[:, 1],
                    color='blue', s=50, label='KMeans Centers')
        for i, triangle in self.center_triangle.items():
            pts = self.kmeans_centers[list(triangle)]
            center_pt, neighbor1_pt, neighbor2_pt = pts[0], pts[1], pts[2]
            plt.plot([center_pt[0], neighbor1_pt[0]], [center_pt[1], neighbor1_pt[1]],
                     color='black', linewidth=1, label='Center-Neighbor Edge' if i == 0 else "")
            plt.plot([center_pt[0], neighbor2_pt[0]], [center_pt[1], neighbor2_pt[1]],
                     color='black', linewidth=1)
            plt.plot([neighbor1_pt[0], neighbor2_pt[0]], [neighbor1_pt[1], neighbor2_pt[1]],
                     color='red', linewidth=1, label='Neighbor-Neighbor Edge' if i == 0 else "")
            plt.text(self.kmeans_centers[i, 0], self.kmeans_centers[i, 1], str(i),
                     fontsize=8, color='green')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
        plt.title("Best Triangles from KMeans Centers")
        plt.axis('equal')
        plt.show()
        