import numpy as np

from src.method.alignment import align, align_all_to_first, get_permutation


class TestAlignAllToFirst:
    """Test suite for align_all_to_first function."""

    def test_single_matrix_no_alignment_needed(self):
        """When C_list has only one matrix (the reference), should return it unchanged."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        C_list_out, p_list_out = align_all_to_first([C_ref])

        assert len(C_list_out) == 1
        np.testing.assert_array_equal(C_list_out[0], C_ref)

    def test_two_matrices_alignment(self):
        """Aligning two matrices (reference + one to align) should work correctly."""
        n = 5
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        # Create a permuted version of C_ref
        perm = np.random.permutation(n)
        C_permuted = C_ref[perm][:, perm]

        C_list_out, p_list_out = align_all_to_first([C_ref, C_permuted])

        # The aligned matrix should be close to C_ref
        assert len(C_list_out) == 2
        np.testing.assert_array_almost_equal(C_list_out[0], C_ref, decimal=5)
        np.testing.assert_array_almost_equal(C_list_out[1], C_ref, decimal=5)

    def test_multiple_matrices_alignment(self):
        """Aligning multiple matrices should align each one to the first."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        # Create multiple permuted versions
        perms = [np.random.permutation(n) for _ in range(3)]
        C_list = [C_ref] + [C_ref[p][:, p] for p in perms]

        C_list_out, p_list_out = align_all_to_first(C_list)

        assert len(C_list_out) == 4
        for C_aligned in C_list_out:
            np.testing.assert_array_almost_equal(C_aligned, C_ref, decimal=5)

    def test_preserves_reference_matrix(self):
        """The reference matrix (first element) should not be modified."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2
        C_ref_original = C_ref.copy()

        perm = np.random.permutation(n)
        C_permuted = C_ref[perm][:, perm]

        C_list_out, _ = align_all_to_first([C_ref, C_permuted])

        np.testing.assert_array_equal(C_list_out[0], C_ref_original)

    def test_with_custom_distributions(self):
        """Should work correctly with custom node distributions."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        perm = np.random.permutation(n)
        C_permuted = C_ref[perm][:, perm]

        p_ref = np.array([0.1, 0.2, 0.3, 0.4])
        p_permuted = p_ref[perm]

        C_list_out, p_list_out = align_all_to_first([C_ref, C_permuted], p_list=[p_ref, p_permuted])

        # Check that distributions sum to 1
        assert len(p_list_out) == 2
        assert np.isclose(p_list_out[0].sum(), 1.0)
        assert np.isclose(p_list_out[1].sum(), 1.0)

    def test_uniform_distributions_when_none(self):
        """When no distributions are provided, should default to uniform."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        perm = np.random.permutation(n)
        C_permuted = C_ref[perm][:, perm]

        C_list_out, p_list_out = align_all_to_first([C_ref, C_permuted])

        expected_uniform = np.ones(n) / n
        np.testing.assert_array_almost_equal(p_list_out[0], expected_uniform)
        np.testing.assert_array_almost_equal(p_list_out[1], expected_uniform)

    def test_output_shapes(self):
        """Output matrices should have the same shape as inputs."""
        n = 6
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        C_others = [np.random.rand(n, n) for _ in range(3)]
        C_others = [(C + C.T) / 2 for C in C_others]
        C_list = [C_ref] + C_others

        C_list_out, p_list_out = align_all_to_first(C_list)

        assert len(C_list_out) == 4
        for C_aligned in C_list_out:
            assert C_aligned.shape == (n, n)
        assert len(p_list_out) == 4
        for p in p_list_out:
            assert p.shape == (n,)

    def test_identical_matrices(self):
        """Aligning identical matrices should return the same matrices."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        C_list = [C_ref.copy() for _ in range(3)]

        C_list_out, _ = align_all_to_first(C_list)

        for C_aligned in C_list_out:
            np.testing.assert_array_almost_equal(C_aligned, C_ref, decimal=5)

    def test_symmetry_preserved(self):
        """Aligned matrices should preserve symmetry if inputs are symmetric."""
        n = 5
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        C_others = [np.random.rand(n, n) for _ in range(2)]
        C_others = [(C + C.T) / 2 for C in C_others]
        C_list = [C_ref] + C_others

        C_list_out, _ = align_all_to_first(C_list)

        for C_aligned in C_list_out:
            np.testing.assert_array_almost_equal(C_aligned, C_aligned.T, decimal=10, err_msg="Symmetry not preserved")

    def test_consistency_with_align_function(self):
        """Results should be consistent with calling align() on each matrix."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        C_others = [np.random.rand(n, n) for _ in range(2)]
        C_others = [(C + C.T) / 2 for C in C_others]
        C_list = [C_ref] + C_others

        C_list_out, p_list_out = align_all_to_first(C_list)

        # Compare with individual align calls (for matrices after the first)
        for i, C in enumerate(C_others):
            C_expected, p_expected = align(C_ref, C)
            np.testing.assert_array_almost_equal(C_list_out[i + 1], C_expected)
            np.testing.assert_array_almost_equal(p_list_out[i + 1], p_expected)

    def test_returns_two_values_without_plans(self):
        """Should return 2 values when plans is None."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        result = align_all_to_first([C_ref])

        assert len(result) == 2


class TestGetPermutation:
    """Tests for the get_permutation helper function."""

    def test_identity_permutation_for_same_matrix(self):
        """Same matrix should give close to identity permutation."""
        n = 4
        C = np.random.rand(n, n)
        C = (C + C.T) / 2

        perm = get_permutation(C, C)
        # The permutation should be close to identity (or equivalent under symmetry)
        assert len(perm) == n
        assert set(perm) == set(range(n))

    def test_recovers_known_permutation(self):
        """Should recover a known permutation."""
        n = 5
        C = np.random.rand(n, n)
        C = (C + C.T) / 2

        known_perm = np.array([2, 0, 4, 1, 3])
        C_permuted = C[known_perm][:, known_perm]

        recovered_perm = get_permutation(C, C_permuted)

        # Apply recovered permutation and check if we get back C
        C_recovered = C_permuted[recovered_perm][:, recovered_perm]
        np.testing.assert_array_almost_equal(C_recovered, C, decimal=5)


class TestAlign:
    """Tests for the align function."""

    def test_align_permuted_matrix(self):
        """Should correctly align a permuted matrix."""
        n = 5
        C1 = np.random.rand(n, n)
        C1 = (C1 + C1.T) / 2

        perm = np.random.permutation(n)
        C2 = C1[perm][:, perm]

        C2_aligned, _ = align(C1, C2)
        np.testing.assert_array_almost_equal(C2_aligned, C1, decimal=5)

    def test_align_returns_permuted_distribution(self):
        """Should return the correctly permuted distribution."""
        n = 4
        C1 = np.random.rand(n, n)
        C1 = (C1 + C1.T) / 2

        perm = np.random.permutation(n)
        C2 = C1[perm][:, perm]
        q = np.array([0.1, 0.2, 0.3, 0.4])

        _, q_aligned = align(C1, C2, q=q)

        assert np.isclose(q_aligned.sum(), 1.0)
        assert len(q_aligned) == n


class TestAlignAllToFirstWithPlans:
    """Test suite for align_all_to_first with plans parameter."""

    def test_plans_are_permuted_single_alignment(self):
        """Plans should be correctly permuted when aligning one matrix."""
        n = 5
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        # Create a known permutation
        perm = np.array([2, 0, 4, 1, 3])
        C_permuted = C_ref[perm][:, perm]

        # Create plans: first plan is for reference (not permuted), second for the aligned matrix
        m = 3
        plan_ref = np.outer(np.ones(m), np.arange(n))
        plan_permuted = plan_ref[:, perm]
        plans = [plan_ref, plan_permuted]

        C_list_out, p_list_out, plans_out = align_all_to_first([C_ref, C_permuted], plans=plans)

        # Check that plans_out has the correct shape
        assert len(plans_out) == 2
        # Reference plan should be unchanged
        np.testing.assert_array_equal(plans_out[0], plan_ref)
        # Second plan should be permuted
        assert plans_out[1].shape == (m, n)

    def test_plans_are_permuted_multiple_matrices(self):
        """Plans should be correctly permuted for multiple matrices."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        # Create multiple permuted versions
        perms = [np.random.permutation(n) for _ in range(3)]
        C_list = [C_ref] + [C_ref[p][:, p] for p in perms]

        # Create plans for each matrix
        m = 5
        plans = [np.outer(np.ones(m), np.arange(n)) for _ in range(4)]

        _, _, plans_out = align_all_to_first(C_list, plans=plans)

        assert len(plans_out) == 4
        for plan_out in plans_out:
            assert plan_out.shape == (m, n)

    def test_plans_none_returns_two_values(self):
        """When plans is None, should return 2 values."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        perm = np.random.permutation(n)
        C_permuted = C_ref[perm][:, perm]

        result = align_all_to_first([C_ref, C_permuted], plans=None)

        assert len(result) == 2

    def test_plans_permutation_consistency_with_matrix(self):
        """The permutation applied to plans should be consistent with matrix alignment."""
        n = 5
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        # Create a permuted version
        perm = np.random.permutation(n)
        C_permuted = C_ref[perm][:, perm]

        # Create a plan that mirrors the matrix diagonal (to verify consistency)
        plan_ref = np.diag(C_ref).reshape(1, n)
        plan_permuted = np.diag(C_permuted).reshape(1, n)
        plans = [plan_ref, plan_permuted]

        C_list_out, _, plans_out = align_all_to_first([C_ref, C_permuted], plans=plans)

        # After alignment, the plan should match the diagonal of the aligned matrix
        expected_plan = np.diag(C_list_out[1]).reshape(1, n)
        np.testing.assert_array_almost_equal(plans_out[1], expected_plan, decimal=5)

    def test_reference_plan_unchanged(self):
        """The first plan (for reference matrix) should not be permuted."""
        n = 4
        C_ref = np.random.rand(n, n)
        C_ref = (C_ref + C_ref.T) / 2

        perm = np.random.permutation(n)
        C_permuted = C_ref[perm][:, perm]

        mu = np.random.rand(n)
        mu = mu / mu.sum()
        plan_ref = np.outer(np.ones(n) / n, mu)
        plan_permuted = plan_ref[:, perm].copy()
        plans = [plan_ref.copy(), plan_permuted]

        _, _, plans_out = align_all_to_first([C_ref, C_permuted], plans=plans)

        # Reference plan should be unchanged
        np.testing.assert_array_equal(plans_out[0], plan_ref)
