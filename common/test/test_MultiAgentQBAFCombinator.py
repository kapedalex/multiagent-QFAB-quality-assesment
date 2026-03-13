import pytest
from common.src.ClusteredQBAF import ClusteredQBAF
from common.src.MultiAgentQBAFCombinator import MultiAgentQBAFCombinator

# Define argument clusters as frozensets of strings
CLAIM = frozenset({"claim"})
CLAIM_VARIANT = frozenset({"the_claim"})  # Semantically similar to CLAIM
ARG1 = frozenset({"arg1"})
ARG1_VARIANT = frozenset({"argument1_variant"}) # Semantically similar to ARG1
ARG_PARENT = frozenset({"parent_arg"})
ARG_CHILD = frozenset({"arg_child"})
CHILD_VARIANT = frozenset({"child_variant"}) # Semantically similar to ARG_CHILD

# --- MODIFIED: Make these unique arguments clearly dissimilar ---
UNIQUE_ARG_1 = frozenset({"apple"})
UNIQUE_ARG_2 = frozenset({"banana"})
# --- END MODIFIED ---

UNRELATED_ARG = frozenset({"unrelated"})


@pytest.fixture
def qbaf_single_agent_1():
    """A simple QBAF with CLAIM as root (depth 0). ARG1 attacks CLAIM."""
    # CLAIM (depth 0)
    # ARG1 (depth 1) attacks CLAIM
    arguments = {CLAIM, ARG1}
    attacks = {(ARG1, CLAIM)}
    supports = set()
    strengths = {CLAIM: 0.7, ARG1: 0.9}
    return ClusteredQBAF(arguments, attacks, supports, strengths)


@pytest.fixture
def qbaf_single_agent_2():
    """Another simple QBAF with CLAIM_VARIANT as root. ARG1_VARIANT attacks CLAIM_VARIANT."""
    # CLAIM_VARIANT (depth 0)
    # ARG1_VARIANT (depth 1) attacks CLAIM_VARIANT
    arguments = {CLAIM_VARIANT, ARG1_VARIANT}
    attacks = {(ARG1_VARIANT, CLAIM_VARIANT)}
    supports = set()
    strengths = {CLAIM_VARIANT: 0.8, ARG1_VARIANT: 0.85}
    return ClusteredQBAF(arguments, attacks, supports, strengths)


@pytest.fixture
def qbaf_layered_1():
    """QBAF for multi-layer test with CLAIM as root."""
    # CLAIM (depth 0)
    # ARG_PARENT (depth 1) attacks CLAIM
    # ARG_CHILD (depth 2) attacks ARG_PARENT
    # UNIQUE_ARG_1 (depth 2) supports ARG_PARENT
    arguments = {CLAIM, ARG_PARENT, ARG_CHILD, UNIQUE_ARG_1}
    attacks = {(ARG_PARENT, CLAIM), (ARG_CHILD, ARG_PARENT)}
    supports = {(UNIQUE_ARG_1, ARG_PARENT)}
    strengths = {CLAIM: 0.6, ARG_PARENT: 0.7, ARG_CHILD: 0.8, UNIQUE_ARG_1: 0.5}
    return ClusteredQBAF(arguments, attacks, supports, strengths)


@pytest.fixture
def qbaf_layered_2():
    """Another QBAF for multi-layer test, with variants and CLAIM_VARIANT as root."""
    # CLAIM_VARIANT (depth 0)
    # ARG_PARENT (depth 1) attacks CLAIM_VARIANT
    # CHILD_VARIANT (depth 2) attacks ARG_PARENT
    # UNIQUE_ARG_2 (depth 2) supports ARG_PARENT
    arguments = {CLAIM_VARIANT, ARG_PARENT, CHILD_VARIANT, UNIQUE_ARG_2}
    attacks = {(ARG_PARENT, CLAIM_VARIANT), (CHILD_VARIANT, ARG_PARENT)}
    supports = {(UNIQUE_ARG_2, ARG_PARENT)}
    strengths = {CLAIM_VARIANT: 0.7, ARG_PARENT: 0.65, CHILD_VARIANT: 0.75, UNIQUE_ARG_2: 0.4}
    return ClusteredQBAF(arguments, attacks, supports, strengths)


@pytest.fixture
def qbaf_no_claim():
    """QBAF without the main claim string or its variants."""
    arguments = {UNRELATED_ARG, ARG1}
    attacks = {(ARG1, UNRELATED_ARG)}
    supports = set()
    strengths = {UNRELATED_ARG: 0.1, ARG1: 0.2}
    return ClusteredQBAF(arguments, attacks, supports, strengths)


# --- Tests for MultiAgentQBAFCombinator ---

def test_combinator_init(qbaf_single_agent_1, qbaf_single_agent_2):
    claim_str = "claim"
    combinator = MultiAgentQBAFCombinator(
        qba_list=[qbaf_single_agent_1, qbaf_single_agent_2],
        claim_id_str=claim_str,
        similarity_threshold=0.85, # Set similarity high enough to merge "claim" and "the_claim"
        aggregation_method='mean'
    )

    # all_unique_original_args should contain all unique argument strings
    expected_original_args = {"claim", "arg1", "the_claim", "argument1_variant"}
    assert combinator.all_unique_original_args == expected_original_args

    # After _initialize_claim_strength, "claim" and "the_claim" should be merged
    combined_claim_cluster = frozenset({"claim", "the_claim"})
    expected_q_star_clusters = {
        combined_claim_cluster,
        frozenset({"arg1"}),
        frozenset({"argument1_variant"})
    }
    assert combinator.Q_star_clusters == expected_q_star_clusters

    # arg_to_current_cluster_map should reflect the merged claim cluster
    expected_arg_to_cluster_map = {
        "arg1": frozenset({"arg1"}),
        "the_claim": combined_claim_cluster,
        "argument1_variant": frozenset({"argument1_variant"}),
        "claim": combined_claim_cluster
    }
    assert combinator.arg_to_current_cluster_map == expected_arg_to_cluster_map

    # The claim_cluster should be the merged one
    assert combinator.claim_cluster == combined_claim_cluster
    # Aggregated strength for the combined claim: (0.7 + 0.8) / 2 = 0.75
    assert combinator.Q_star_strengths[combined_claim_cluster] == pytest.approx(0.75)
    assert combinator.Q_star_attacks == set()
    assert combinator.Q_star_supports == set()


def test_initialize_claim_strength_no_claim_in_input(qbaf_no_claim):
    claim_str = "non_existent_claim"
    combinator = MultiAgentQBAFCombinator(
        qba_list=[qbaf_no_claim],
        claim_id_str=claim_str,
        similarity_threshold=0.85
    )

    # The non_existent_claim should be added as a singleton cluster
    expected_claim_cluster = frozenset({claim_str})
    assert expected_claim_cluster in combinator.Q_star_clusters
    # Default strength should be 0.5 if no original strengths are found
    assert combinator.Q_star_strengths[expected_claim_cluster] == 0.5
    assert combinator.arg_to_current_cluster_map[claim_str] == expected_claim_cluster
    assert combinator.claim_cluster == expected_claim_cluster


def test_get_max_depth_from_input_qbafs(qbaf_layered_1, qbaf_layered_2, qbaf_no_claim):
    claim_str = "claim"  # This will merge with "the_claim"
    combinator = MultiAgentQBAFCombinator(
        qba_list=[qbaf_layered_1, qbaf_layered_2, qbaf_no_claim],
        claim_id_str=claim_str,
        similarity_threshold=0.85
    )
    # qbaf_layered_1: CLAIM(0) -> ARG_PARENT(1) -> ARG_CHILD(2) or UNIQUE_ARG_1(2). Max depth = 2.
    # qbaf_layered_2: CLAIM_VARIANT(0) -> ARG_PARENT(1) -> CHILD_VARIANT(2) or UNIQUE_ARG_2(2). Max depth = 2.
    # qbaf_no_claim does not contain 'claim' or 'the_claim' related arguments.

    # The _get_max_depth_from_input_qbafs should check max depth from both 'claim' and 'the_claim'
    # in their respective original QBAFs. Both yield a max depth of 2.
    assert combinator._get_max_depth_from_input_qbafs() == 2


def test_combine_qbafs_basic_merging(qbaf_single_agent_1, qbaf_single_agent_2):
    claim_str = "claim"
    combinator = MultiAgentQBAFCombinator(
        qba_list=[qbaf_single_agent_1, qbaf_single_agent_2],
        claim_id_str=claim_str,
        similarity_threshold=0.85  # Sufficiently high to merge 'claim' and 'the_claim', but not 'arg1' variants
    )

    combined_qbaf = combinator.combine_qbafs()

    combined_claim_cluster = frozenset({"claim", "the_claim"})
    assert combined_claim_cluster in combined_qbaf.arguments

    # No merging between ARG1 and ARG1_VARIANT because they are not considered related
    # by common parent in the previous layer (they directly attack the root) and similarity threshold.
    expected_clusters = {
        combined_claim_cluster,
        frozenset({"arg1"}),
        frozenset({"argument1_variant"})
    }
    assert combined_qbaf.arguments == expected_clusters
    assert len(combined_qbaf.arguments) == 3 # 1 combined claim + 2 individual arg1 variants

    assert combined_qbaf.strengths[combined_claim_cluster] == pytest.approx(0.75) # (0.7 + 0.8) / 2
    assert combined_qbaf.strengths[frozenset({"arg1"})] == pytest.approx(0.9)
    assert combined_qbaf.strengths[frozenset({"argument1_variant"})] == pytest.approx(0.85)

    # Relations should lift to the combined_claim_cluster
    expected_attacks = {
        (frozenset({"arg1"}), combined_claim_cluster),
        (frozenset({"argument1_variant"}), combined_claim_cluster)
    }
    assert combined_qbaf.attacks == expected_attacks
    assert combined_qbaf.supports == set()


def test_combine_qbafs_multi_layer_merging(qbaf_layered_1, qbaf_layered_2):
    claim_str = "claim"
    combinator = MultiAgentQBAFCombinator(
        qba_list=[qbaf_layered_1, qbaf_layered_2],
        claim_id_str=claim_str,
        similarity_threshold=0.85 # Sufficiently high for merging variants
    )

    combined_qbaf = combinator.combine_qbafs()

    # Expected clusters and their strengths:
    # Layer 0: combined_claim = {"claim", "the_claim"}
    combined_claim = frozenset({"claim", "the_claim"})
    assert combined_claim in combined_qbaf.arguments
    assert combined_qbaf.strengths[combined_claim] == pytest.approx((0.6 + 0.7) / 2) # 0.65

    # Layer 1: ARG_PARENT = {"parent_arg"} (same string, no variant, so stays singleton)
    parent_cluster = frozenset({"parent_arg"})
    assert parent_cluster in combined_qbaf.arguments
    assert combined_qbaf.strengths[parent_cluster] == pytest.approx((0.7 + 0.65) / 2) # 0.675

    # Layer 2: combined_child = {"arg_child", "child_variant"}
    combined_child = frozenset({"arg_child", "child_variant"})
    assert combined_child in combined_qbaf.arguments
    assert combined_qbaf.strengths[combined_child] == pytest.approx((0.8 + 0.75) / 2) # 0.775

    # --- MODIFIED: Assertions for the distinct unique arguments ---
    unique_arg_1_cluster = frozenset({"apple"})
    unique_arg_2_cluster = frozenset({"banana"})

    assert unique_arg_1_cluster in combined_qbaf.arguments
    assert combined_qbaf.strengths[unique_arg_1_cluster] == pytest.approx(0.5)

    assert unique_arg_2_cluster in combined_qbaf.arguments
    assert combined_qbaf.strengths[unique_arg_2_cluster] == pytest.approx(0.4)
    # --- END MODIFIED ---

    # Total clusters should now correctly be 5:
    # combined_claim, parent_cluster, combined_child, unique_arg_1_cluster, unique_arg_2_cluster
    assert len(combined_qbaf.arguments) == 5

    # Expected Attacks
    expected_attacks = {
        (parent_cluster, combined_claim), # ARG_PARENT attacks CLAIM/CLAIM_VARIANT
        (combined_child, parent_cluster)  # ARG_CHILD/CHILD_VARIANT attack ARG_PARENT
    }
    assert combined_qbaf.attacks == expected_attacks

    # --- MODIFIED: Expected Supports for the distinct unique arguments ---
    expected_supports = {
        (unique_arg_1_cluster, parent_cluster), # UNIQUE_ARG_1 (apple) supports ARG_PARENT
        (unique_arg_2_cluster, parent_cluster)  # UNIQUE_ARG_2 (banana) supports ARG_PARENT
    }
    assert combined_qbaf.supports == expected_supports
    # --- END MODIFIED ---