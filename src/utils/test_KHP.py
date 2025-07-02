import torch

class HierarchicalKDTreePartitioner:
    def __init__(self, points, max_depth=5, leaf_size=10, device='cpu'):
        """
        points: (N, 3) torch.Tensor des positions des points
        max_depth: profondeur maximale de la hiérarchie
        leaf_size: nombre minimum de points dans une feuille
        device: 'cpu' ou 'cuda'
        """
        assert points.ndim == 2 and points.size(1) == 3, "Les points doivent être de forme (N, 3)"
        self.points = points.to(device)
        self.max_depth = max_depth
        self.leaf_size = leaf_size
        self.device = device

        self.node_to_points = {}  # (depth, node_id) -> Tensor d'indices
        self.point_to_node = {}   # point_idx -> (depth, node_id)
        self._build_tree()

    def _build_tree(self):
        def recursive_split(indices, depth, node_id):
            self.node_to_points[(depth, node_id)] = indices

            for idx in indices.tolist():
                self.point_to_node[idx] = (depth, node_id)

            if depth >= self.max_depth or indices.numel() <= self.leaf_size:
                return

            pts = self.points[indices]
            axis = depth % pts.shape[1]
            sorted_vals, sorted_idx = torch.sort(pts[:, axis])
            sorted_indices = indices[sorted_idx]

            median_idx = len(sorted_indices) // 2
            left_indices = sorted_indices[:median_idx]
            right_indices = sorted_indices[median_idx:]

            recursive_split(left_indices, depth + 1, node_id * 2)
            recursive_split(right_indices, depth + 1, node_id * 2 + 1)

        all_indices = torch.arange(self.points.shape[0], device=self.device)
        recursive_split(all_indices, depth=0, node_id=1)

    def query_point_cluster(self, point_idx, depth):
        """
        Retourne l'ID du cluster contenant le point donné à une profondeur spécifiée.
        """
        if point_idx not in self.point_to_node:
            raise ValueError(f"Point index {point_idx} non trouvé.")

        # Si on demande une profondeur plus fine que la construction, remonter
        current_depth, current_node_id = self.point_to_node[point_idx]
        while current_depth > depth:
            current_depth -= 1
            current_node_id //= 2

        return current_node_id

    def query_points_clusters(self, point_indices, depth):
        """
        Retourne un dictionnaire {point_idx: cluster_id} pour plusieurs points.
        """
        return {int(idx): self.query_point_cluster(int(idx), depth) for idx in point_indices}

    def get_cluster_points(self, node_id, depth):
        """
        Retourne les indices des points d'un cluster spécifique (torch.Tensor).
        """
        return self.node_to_points.get((depth, node_id), None)


points = torch.rand(10000, 3)  # Exemple : 10k points uniformes
partitioner = HierarchicalKDTreePartitioner(points, max_depth=6, leaf_size=20, device='cpu')

# Requête
point_idx = 42
depth = 3
cluster_id = partitioner.query_point_cluster(point_idx, depth)
print(f"Le point {point_idx} est dans le cluster {cluster_id} à la profondeur {depth}.")

# Tous les points d'un cluster
cluster_points = partitioner.get_cluster_points(cluster_id, depth)
print(f"Cluster {cluster_id} contient {cluster_points.shape[0]} points.")
