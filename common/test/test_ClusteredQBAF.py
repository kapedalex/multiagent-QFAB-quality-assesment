import pytest
from common.src.ClusteredQBAF import ClusteredQBAF

A = frozenset({"A"})
B = frozenset({"B"})
C = frozenset({"C"}) # This will be the root in the updated sample_qbaf
D = frozenset({"D"})

# Sample QBAF for testing, now an acyclic tree
@pytest.fixture
def sample_qbaf():
    # C is the root (depth 0)
    # B attacks C (depth 1)
    # A attacks B (depth 2)
    # D supports B (depth 2)
    arguments = {A, B, C, D}
    attacks = {(B, C), (A, B)}
    supports = {(D, B)}
    strengths = {A: 0.8, B: 0.6, C: 0.7, D: 0.9}
    return ClusteredQBAF(arguments, attacks, supports, strengths)

def test_get_children(sample_qbaf):
    # Children of C: B (attacks C)
    children_of_C = set(sample_qbaf.get_children(C))
    expected_children_of_C = set([(B, 'attack')])
    assert children_of_C == expected_children_of_C

    # Children of B: A (attacks B), D (supports B)
    children_of_B = set(sample_qbaf.get_children(B))
    expected_children_of_B = set([(A, 'attack'), (D, 'support')])
    assert children_of_B == expected_children_of_B

    # Children of A: None
    assert sample_qbaf.get_children(A) == []

    # Children of D: None
    assert sample_qbaf.get_children(D) == []

def test_get_relations_to_parents(sample_qbaf):
    # Relations from B: B attacks C, B has D supporting it (but D is child, not parent of B)
    # Relations from B means B is the child, looking for its parents
    relations_from_B = set(sample_qbaf.get_relations_to_parents(B))
    expected_relations_from_B = set([(B, C, 'attack')])
    assert relations_from_B == expected_relations_from_B

    relations_from_A = set(sample_qbaf.get_relations_to_parents(A))
    expected_relations_from_A = set([(A, B, 'attack')])
    assert relations_from_A == expected_relations_from_A

    relations_from_D = set(sample_qbaf.get_relations_to_parents(D))
    expected_relations_from_D = set([(D, B, 'support')])
    assert relations_from_D == expected_relations_from_D

    assert sample_qbaf.get_relations_to_parents(C) == [] # C is a root, no parents

def test_get_max_depth(sample_qbaf):
    # C is root at depth 0
    assert sample_qbaf.get_max_depth(C) == 2 # C(0) -> B(1) -> A/D(2)
    assert sample_qbaf.get_max_depth(B) == 1 # B(0) -> A/D(1)
    assert sample_qbaf.get_max_depth(A) == 0 # A is a leaf, depth 0
    assert sample_qbaf.get_max_depth(D) == 0 # D is a leaf, depth 0
    non_existent = frozenset({"Z"})
    assert sample_qbaf.get_max_depth(non_existent) == -1