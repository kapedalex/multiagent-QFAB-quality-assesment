from typing import Set, Tuple, Dict, FrozenSet, List
from qbaf import QBAFramework # noinspection PyUnresolvedReferences

class ClusteredQBAF:
    """
    Represents a Quantitative Bipolar Argumentation Framework (QBAF) where
    arguments are frozensets of strings (clusters).
    """

    def __init__(self, arguments: Set[FrozenSet[str]], attacks: Set[Tuple[FrozenSet[str], FrozenSet[str]]],
                 supports: Set[Tuple[FrozenSet[str], FrozenSet[str]]], strengths: Dict[FrozenSet[str], float]):
        self.arguments = arguments
        self.attacks = attacks
        self.supports = supports
        self.strengths = strengths

    def get_children(self, parent_arg_cluster: FrozenSet[str]) -> List[Tuple[FrozenSet[str], str]]:
        """
        Returns a list of (child_cluster, relation_type) where child_cluster
        is an argument that attacks or supports the parent_arg_cluster.
        """
        children_relations = []
        for attacker, attacked in self.attacks:
            if attacked == parent_arg_cluster:
                children_relations.append((attacker, 'attack'))
        for supporter, supported in self.supports:
            if supported == parent_arg_cluster:
                children_relations.append((supporter, 'support'))
        return children_relations

    def get_relations_to_parents(self, child_arg_cluster: FrozenSet[str]) -> List[
        Tuple[FrozenSet[str], FrozenSet[str], str]]:
        """
        Returns a list of (child_cluster, parent_cluster, relation_type) where
        child_cluster attacks or supports parent_cluster.
        """
        parent_relations = []
        for attacker, attacked in self.attacks:
            if attacker == child_arg_cluster:
                parent_relations.append((child_arg_cluster, attacked, 'attack'))
        for supporter, supported in self.supports:
            if supporter == child_arg_cluster:
                parent_relations.append((child_arg_cluster, supported, 'support'))
        return parent_relations

    def get_max_depth(self, root_arg_cluster: FrozenSet[str]) -> int:
        """
        Calculates the maximum depth of the argumentation tree rooted at root_arg_cluster
        by traversing downwards to children. The root itself is at depth 0.
        """
        if root_arg_cluster not in self.arguments:
            return -1  # Root not found in this QBAF

        # If the root has no children, its depth is 0
        if not self.get_children(root_arg_cluster):
            return 0

        max_d = 0
        # Queue for BFS: (argument_cluster, current_depth)
        q = [(root_arg_cluster, 0)]
        visited = {root_arg_cluster} # Keep track of visited clusters to prevent cycles and redundant processing

        head = 0
        while head < len(q):
            current_arg_cluster, current_depth = q[head]
            head += 1

            max_d = max(max_d, current_depth) # Update max_d with the current_depth of the deepest argument processed so far

            for child_arg_cluster, _ in self.get_children(current_arg_cluster):
                if child_arg_cluster not in visited:
                    visited.add(child_arg_cluster)
                    q.append((child_arg_cluster, current_depth + 1))
        return max_d

    def to_qba_framework(self, claim_argument: str, default_strength: float = 0.5,
                         semantics: str = "QuadraticEnergy_model") -> QBAFramework:
        """
        Converts the ClusteredQBAF back into a QBAFramework, assuming the original
        QBAF was built around a single, identifiable claim_argument (root).

        This inversion is only exact if all clusters in ClusteredQBAF are singletons
        and the claim_argument corresponds to exactly one cluster.
        """

        # 1. Map Clusters back to Atoms (Strict Singleton Check)
        cluster_to_atom_map: Dict[FrozenSet[str], str] = {}
        atom_to_cluster_map: Dict[str, FrozenSet[str]] = {}

        for cluster in self.arguments:
            if len(cluster) == 1:
                atom = next(iter(cluster))
                cluster_to_atom_map[cluster] = atom
                atom_to_cluster_map[atom] = cluster
            else:
                # If we find a multi-element cluster, we cannot uniquely invert the structure
                # created by the combination algorithm (Algorithm 1, Def 3, Property 2).
                raise ValueError(
                    f"Cannot reliably convert back: Cluster {cluster} is not a singleton. "
                    "The original QBAFramework structure cannot be uniquely recovered without knowing the exact merge rules."
                )

        # Check if the specified claim argument exists and corresponds to a cluster
        if claim_argument not in atom_to_cluster_map:
            # Handle the case where the claim argument itself might have been merged
            # with other arguments due to similarity (if similarity threshold was low).

            # For simple inversion, we assume the claim_argument maps to its own cluster.
            if frozenset({claim_argument}) not in self.arguments:
                raise ValueError(
                    f"Claim argument '{claim_argument}' does not map to any cluster in the Clustered QBAF.")

            # If it exists as a singleton cluster, it will be found below.

        # 2. Reconstruct Relations using atom maps
        reconstructed_attacks: Set[Tuple[str, str]] = set()
        for attacker_cluster, attacked_cluster in self.attacks:
            if attacker_cluster in cluster_to_atom_map and attacked_cluster in cluster_to_atom_map:
                attacker_atom = cluster_to_atom_map[attacker_cluster]
                attacked_atom = cluster_to_atom_map[attacked_cluster]
                reconstructed_attacks.add((attacker_atom, attacked_atom))

        reconstructed_supports: Set[Tuple[str, str]] = set()
        for supporter_cluster, supported_cluster in self.supports:
            if supporter_cluster in cluster_to_atom_map and supported_cluster in cluster_to_atom_map:
                supporter_atom = cluster_to_atom_map[supporter_cluster]
                supported_atom = cluster_to_atom_map[supported_cluster]
                reconstructed_supports.add((supporter_atom, supported_atom))

        # 3. Reconstruct Strengths (ordered list for QBAFramework constructor)

        # Get all atoms present in the reconstructed structure (excluding the claim itself if it was merged heavily)
        all_atoms = set(atom_to_cluster_map.values())

        # For QBAFramework reconstruction, we need a consistent ordered list of arguments
        # and a corresponding list of strengths. The paper implies that for a single QBAF,
        # arguments are often represented as a tree rooted at the claim.

        # Since we don't have the tree structure explicitly in the ClusteredQBAF output (only the graph structure X*),
        # we rely on reconstructing the argument set X and their initial strengths τ.

        # Gather all unique atoms that form the resulting structure X*
        reconstructed_atoms: Set[str] = set()
        for cluster in self.arguments:
            reconstructed_atoms.update(cluster)

        # Sort atoms for deterministic constructor input (Crucial for matching initial_strengths list)
        sorted_atoms = sorted(list(reconstructed_atoms))

        reconstructed_strengths_list: List[float] = []

        for atom in sorted_atoms:
            cluster = atom_to_cluster_map[atom]
            # Strength of the atom is the strength of its cluster in Q*
            strength = self.strengths.get(cluster, default_strength)
            reconstructed_strengths_list.append(strength)

        # 4. Create QBAFramework instance
        new_qba_framework = QBAFramework(
            arguments=sorted_atoms,
            initial_strengths=reconstructed_strengths_list,
            attack_relations=list(reconstructed_attacks),
            support_relations=list(reconstructed_supports),
            semantics=semantics
        )

        return new_qba_framework

def convert_qbaframework_to_clustered_qbaf(qba_framework_instance: QBAFramework) -> ClusteredQBAF:
    """
    Converts an instance of qbaf.QBAFramework into a ClusteredQBAF.
    Each atomic argument from QBAFramework becomes a singleton frozenset cluster.
    """
    clustered_arguments: Set[FrozenSet[str]] = {frozenset({arg}) for arg in qba_framework_instance.arguments}

    arg_to_cluster_map: Dict[str, FrozenSet[str]] = {arg: frozenset({arg}) for arg in
                                                     qba_framework_instance.arguments}

    clustered_attacks: Set[Tuple[FrozenSet[str], FrozenSet[str]]] = set()
    for attacker, attacked in qba_framework_instance.attack_relations:
        clustered_attacks.add((arg_to_cluster_map[attacker], arg_to_cluster_map[attacked]))

    clustered_supports: Set[Tuple[FrozenSet[str], FrozenSet[str]]] = set()
    for supporter, supported in qba_framework_instance.support_relations:
        clustered_supports.add((arg_to_cluster_map[supporter], arg_to_cluster_map[supported]))

    clustered_strengths: Dict[FrozenSet[str], float] = {
        arg_to_cluster_map[arg]: strength
        for arg, strength in qba_framework_instance.initial_strengths.items()
    }

    return ClusteredQBAF(
        arguments=clustered_arguments,
        attacks=clustered_attacks,
        supports=clustered_supports,
        strengths=clustered_strengths
    )

