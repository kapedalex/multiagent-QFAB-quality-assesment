from collections import defaultdict
from typing import Set, Tuple, Dict, FrozenSet, List
from common.src.ClusteredQBAF import ClusteredQBAF
from common.src.utils import aggregate_scores
from common.src.utils import calculate_cosine_similarity

class MultiAgentQBAFCombinator:
    """
    A class to combine multiple independently generated Quantitative Bipolar Argumentation Frameworks (QBAFs)
    into a single, more robust QBAF using a layer-by-layer clustering and aggregation approach.
    """

    def __init__(self,
                 qba_list: List[ClusteredQBAF],
                 claim_id_str: str,
                 similarity_threshold: float,
                 aggregation_method: str = 'mean'):

        self.qba_list = qba_list
        self.claim_id_str = claim_id_str
        self.similarity_threshold = similarity_threshold
        self.aggregation_method = aggregation_method

        # Initialize internal state for the combined QBAF
        self.all_unique_original_args: Set[str] = set()
        for qba in self.qba_list:
            # Extract original string arguments from frozenset({string})
            self.all_unique_original_args.update({next(iter(arg_fset)) for arg_fset in qba.arguments})
        # Ensure the central claim_id_str is always considered, even if not in any input QBAF
        if self.claim_id_str not in self.all_unique_original_args:
            self.all_unique_original_args.add(self.claim_id_str)

        # Algorithm 1, Line 1: Initialize Q_star_clusters with singleton clusters for every argument.
        self.Q_star_clusters: Set[FrozenSet[str]] = {frozenset({arg_str}) for arg_str in self.all_unique_original_args}

        # Map from original argument string to its current cluster (frozenset).
        # This map is crucial for quickly finding which cluster an individual argument belongs to.
        self.arg_to_current_cluster_map: Dict[str, FrozenSet[str]] = {arg_str: frozenset({arg_str}) for arg_str in
                                                                      self.all_unique_original_args}
        self.Q_star_attacks: Set[Tuple[FrozenSet[str], FrozenSet[str]]] = set()
        self.Q_star_supports: Set[Tuple[FrozenSet[str], FrozenSet[str]]] = set()
        self.Q_star_strengths: Dict[FrozenSet[str], float] = {}

        self.claim_cluster: FrozenSet[str] = frozenset() # Will be set by _initialize_claim_strength

        self._initialize_claim_strength()

    def _initialize_claim_strength(self):
        """
        Initializes the strength for the central claim cluster by merging all semantically similar
        original argument strings related to self.claim_id_str.
        This forms the Layer 0 cluster for the combined QBAF.
        """
        related_claim_strs: Set[str] = set()
        # Find all original string arguments that are semantically similar to self.claim_id_str
        for arg_str in self.all_unique_original_args:
            if calculate_cosine_similarity(self.claim_id_str, arg_str) >= self.similarity_threshold:
                related_claim_strs.add(arg_str)

        if not related_claim_strs:
            # This case should ideally not happen if self.claim_id_str is in all_unique_original_args
            # as similarity with itself is 1.0. But as a fallback:
            related_claim_strs.add(self.claim_id_str)

        self.claim_cluster = frozenset(related_claim_strs)

        # Aggregate strengths for the newly formed claim_cluster
        relevant_claim_strengths = []
        for arg_str_in_cluster in self.claim_cluster:
            for qba in self.qba_list:
                # Retrieve strength for the singleton frozenset in original QBAFs
                singleton_fset = frozenset({arg_str_in_cluster})
                if singleton_fset in qba.arguments and singleton_fset in qba.strengths:
                    relevant_claim_strengths.append(qba.strengths[singleton_fset])

        if relevant_claim_strengths:
            self.Q_star_strengths[self.claim_cluster] = aggregate_scores(relevant_claim_strengths, self.aggregation_method)
        else:
            self.Q_star_strengths[self.claim_cluster] = 0.5 # Default if no strengths found

        # Update Q_star_clusters and arg_to_current_cluster_map
        # Remove all singleton clusters that are now part of the merged claim_cluster
        for arg_str_in_cluster in self.claim_cluster:
            singleton_fset_to_remove = frozenset({arg_str_in_cluster})
            if singleton_fset_to_remove in self.Q_star_clusters:
                self.Q_star_clusters.remove(singleton_fset_to_remove)
            self.arg_to_current_cluster_map[arg_str_in_cluster] = self.claim_cluster

        # Add the new merged claim_cluster to Q_star_clusters
        self.Q_star_clusters.add(self.claim_cluster)

    def _get_max_depth_from_input_qbafs(self) -> int:
        """
        Calculates the maximum depth across all input QBAFs, considering any original
        argument that is part of the combined claim_cluster as a potential root.
        This ensures that the iteration goes deep enough for all relevant branches.
        """
        max_d_overall = 0
        for original_claim_str in self.claim_cluster:
            original_claim_fset = frozenset({original_claim_str})
            for qba in self.qba_list:
                if original_claim_fset in qba.arguments:
                    max_d_overall = max(max_d_overall, qba.get_max_depth(original_claim_fset))
        return max_d_overall

    def combine_qbafs(self) -> ClusteredQBAF:
        """
        Combines the list of individual QBAFs into a single, merged QBAF.
        This implements Algorithm 1 from the paper.
        """
        max_qba_depth = self._get_max_depth_from_input_qbafs()

        # Algorithm 1, Line 2: `previous_layer` holds clusters that acted as parents in the previous depth.
        # Initially, it contains only the cluster of the central claim.
        previous_layer_clusters: Set[FrozenSet[str]] = {self.claim_cluster}

        # Algorithm 1, Lines 3-19: Iterate layer by layer (depth `d` from 1 up to max_qba_depth)
        # Note: 'depth' here refers to the depth in the combined QBAF relative to the combined claim_cluster.
        for d in range(1, max_qba_depth + 1): # Start from depth 1 for children of the claim

            # Collect all *original string arguments* that appear at this depth across all QBAFs
            # and are children of arguments from the previous_layer_clusters.
            args_to_cluster_at_this_depth: Set[str] = set()

            for parent_cluster_from_prev_layer in previous_layer_clusters:
                for parent_arg_str_in_cluster in parent_cluster_from_prev_layer:
                    # Reconstruct the singleton frozenset for the parent in its original QBAF context
                    parent_arg_as_singleton_fset = frozenset({parent_arg_str_in_cluster})

                    for qba in self.qba_list:
                        if parent_arg_as_singleton_fset in qba.arguments:
                            for child_arg_as_singleton_fset, _ in qba.get_children(parent_arg_as_singleton_fset):
                                child_arg_str = next(iter(child_arg_as_singleton_fset))

                                # IMPORTANT: Only consider children that have not been processed yet (i.e.,
                                # are not part of the claim_cluster and are still in their original singleton clusters).
                                # This ensures strict layer-by-layer processing and prevents cycles in processing.
                                if self.arg_to_current_cluster_map.get(child_arg_str) == frozenset({child_arg_str}):
                                    args_to_cluster_at_this_depth.add(child_arg_str)

            if not args_to_cluster_at_this_depth:
                break # No more arguments found at this depth, end iteration

            # `current_depth_clusters_in_progress`: Stores clusters as they are being formed at this depth.
            # Initially, each argument at this depth starts in its own singleton cluster.
            current_depth_clusters_in_progress: Set[FrozenSet[str]] = {
                frozenset({arg_str}) for arg_str in args_to_cluster_at_this_depth
            }

            # `arg_to_temp_cluster_map`: A temporary map for merging operations within this depth.
            # Maps original argument strings to their current temporary cluster frozensets at this depth.
            arg_to_temp_cluster_map: Dict[str, FrozenSet[str]] = {
                arg_str: frozenset({arg_str}) for arg_str in args_to_cluster_at_this_depth
            }

            # --- Algorithm 1, Lines 5-12: Merge arguments into clusters at the current depth ---
            arguments_at_this_depth_list = list(args_to_cluster_at_this_depth)
            for i in range(len(arguments_at_this_depth_list)):
                arg_x_str = arguments_at_this_depth_list[i]
                for j in range(i + 1, len(arguments_at_this_depth_list)):
                    arg_y_str = arguments_at_this_depth_list[j]

                    # If arg_x_str and arg_y_str are already in the same temporary cluster, skip.
                    if arg_to_temp_cluster_map[arg_x_str] == arg_to_temp_cluster_map[arg_y_str]:
                        continue

                    # Algorithm 1, Line 6 condition:
                    # Check if (arg_x_str, original_parent_arg_str_z) and (arg_y_str, original_parent_arg_str_z_prime)
                    # exist in some QBAFs, AND original_parent_arg_str_z and original_parent_arg_str_z_prime
                    # belong to the SAME merged parent cluster (z*) from previous_layer_clusters.
                    # Also, the relation type must be the same (attack/support).

                    meets_relation_condition = False
                    for parent_cluster_from_prev_layer in previous_layer_clusters:
                        x_relations_to_this_parent_cluster_members = defaultdict(
                            set) # {rel_type: {set_of_original_parent_arg_strings}}
                        y_relations_to_this_parent_cluster_members = defaultdict(
                            set) # {rel_type: {set_of_original_parent_arg_strings}}

                        arg_x_as_singleton_fset = frozenset({arg_x_str})
                        for qba in self.qba_list:
                            for _, original_parent_fset, rel_type in qba.get_relations_to_parents(arg_x_as_singleton_fset):
                                original_parent_arg_str = next(iter(original_parent_fset))
                                if original_parent_arg_str in parent_cluster_from_prev_layer:
                                    x_relations_to_this_parent_cluster_members[rel_type].add(original_parent_arg_str)

                        arg_y_as_singleton_fset = frozenset({arg_y_str})
                        for qba in self.qba_list:
                            for _, original_parent_fset, rel_type in qba.get_relations_to_parents(arg_y_as_singleton_fset):
                                original_parent_arg_str = next(iter(original_parent_fset))
                                if original_parent_arg_str in parent_cluster_from_prev_layer:
                                    y_relations_to_this_parent_cluster_members[rel_type].add(original_parent_arg_str)

                        # Check if x and y share a common relation type to *any* original parent argument
                        # that belongs to the *same* `parent_cluster_from_prev_layer`.
                        for rel_type_x, original_parents_x in x_relations_to_this_parent_cluster_members.items():
                            if rel_type_x in y_relations_to_this_parent_cluster_members:
                                original_parents_y = y_relations_to_this_parent_cluster_members[rel_type_x]
                                if original_parents_x.intersection(original_parents_y):
                                    meets_relation_condition = True
                                    break # Found a common parent in this previous_layer_cluster with same relation
                        if meets_relation_condition:
                            break # No need to check other previous_layer_clusters

                    if meets_relation_condition:
                        # Check semantic similarity based on original string content
                        similarity = calculate_cosine_similarity(arg_x_str, arg_y_str)
                        if similarity >= self.similarity_threshold:
                            # Algorithm 1, Line 8: Merge clusters
                            cluster1 = arg_to_temp_cluster_map[arg_x_str]
                            cluster2 = arg_to_temp_cluster_map[arg_y_str]

                            if cluster1 != cluster2: # Only merge if they are not already in the same cluster
                                merged_cluster = frozenset(cluster1.union(cluster2))

                                current_depth_clusters_in_progress.remove(cluster1)
                                current_depth_clusters_in_progress.remove(cluster2)
                                current_depth_clusters_in_progress.add(merged_cluster)

                                # Update the temporary map for all arguments in the new merged cluster
                                for arg_s_in_merged in merged_cluster:
                                    arg_to_temp_cluster_map[arg_s_in_merged] = merged_cluster
                    # else: # Algorithm 1, Lines 9-11 (implicitly handled as they remain separate clusters)
                        # pass

            # Update global state with the new clusters formed at this depth
            for arg_str in args_to_cluster_at_this_depth:
                # Remove old singleton cluster from Q_star_clusters if it was merged
                singleton_fset = frozenset({arg_str})
                if singleton_fset in self.Q_star_clusters:
                    self.Q_star_clusters.remove(singleton_fset)
                # Update the global map for this argument to its new (merged) cluster
                self.arg_to_current_cluster_map[arg_str] = arg_to_temp_cluster_map[arg_str]

            self.Q_star_clusters.update(current_depth_clusters_in_progress) # Add new merged clusters

            # --- Algorithm 1, Lines 13-18: Aggregate strengths and add relations for newly formed clusters ---
            new_previous_layer_clusters: Set[FrozenSet[str]] = set()

            for current_depth_cluster in current_depth_clusters_in_progress:
                # Algorithm 1, Line 14: Aggregate strengths for this new cluster
                relevant_strengths = []
                for arg_s_in_cluster in current_depth_cluster:
                    for qba in self.qba_list:
                        singleton_fset = frozenset({arg_s_in_cluster})
                        if singleton_fset in qba.arguments and singleton_fset in qba.strengths:
                            relevant_strengths.append(qba.strengths[singleton_fset])

                if relevant_strengths:
                    self.Q_star_strengths[current_depth_cluster] = aggregate_scores(relevant_strengths,
                                                                                    self.aggregation_method)
                else:
                    self.Q_star_strengths[current_depth_cluster] = 0.0 # Default strength

                new_previous_layer_clusters.add(current_depth_cluster) # These clusters become parents for the next layer

                # Algorithm 1, Lines 15-18: Add relations from this current_depth_cluster (child)
                # to previous_layer_clusters (parent)
                for arg_s_in_child_cluster in current_depth_cluster:
                    arg_as_singleton_fset = frozenset({arg_s_in_child_cluster})
                    for qba in self.qba_list:
                        for _, original_parent_fset_from_qba, rel_type in qba.get_relations_to_parents(arg_as_singleton_fset):
                            original_parent_arg_str = next(iter(original_parent_fset_from_qba))

                            # Find the corresponding merged parent cluster from the previous layer
                            target_parent_merged_cluster = self.arg_to_current_cluster_map.get(original_parent_arg_str)

                            if target_parent_merged_cluster and target_parent_merged_cluster in previous_layer_clusters:
                                if rel_type == 'attack':
                                    self.Q_star_attacks.add((current_depth_cluster, target_parent_merged_cluster))
                                elif rel_type == 'support':
                                    self.Q_star_supports.add((current_depth_cluster, target_parent_merged_cluster))

            previous_layer_clusters = new_previous_layer_clusters # Algorithm 1, Line 19

        # Algorithm 1, Line 20: Return Q*
        final_qbaf = ClusteredQBAF(
            arguments=self.Q_star_clusters,
            attacks=self.Q_star_attacks,
            supports=self.Q_star_supports,
            strengths=self.Q_star_strengths
        )
        return final_qbaf