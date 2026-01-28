import torch
import pytest
from src.method.srgw_barycenter import convert_plan_to_map, drop_zero_mass


class TestConvertPlanToMap:
    """Tests for convert_plan_to_map function."""

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        T_plan = torch.rand(10, 5)
        T_map = convert_plan_to_map(T_plan)
        assert T_map.shape == T_plan.shape

    def test_each_row_has_exactly_one_nonzero(self):
        """Each row should have exactly one non-zero entry (deterministic)."""
        T_plan = torch.rand(10, 5)
        T_map = convert_plan_to_map(T_plan)
        nonzeros_per_row = (T_map != 0).sum(dim=1)
        assert torch.all(nonzeros_per_row == 1)

    def test_row_marginal_is_uniform(self):
        """Row marginals should be uniform (1/n for each row)."""
        n, k = 10, 5
        T_plan = torch.rand(n, k)
        T_map = convert_plan_to_map(T_plan)
        row_sums = T_map.sum(dim=1)
        expected = torch.full((n,), 1.0 / n)
        assert torch.allclose(row_sums, expected)

    def test_total_mass_is_one(self):
        """Total mass of output should sum to 1."""
        T_plan = torch.rand(10, 5)
        T_map = convert_plan_to_map(T_plan)
        assert torch.isclose(T_map.sum(), torch.tensor(1.0))

    def test_argmax_selects_correct_column(self):
        """Argmax method should place mass in column with max value."""
        T_plan = torch.tensor(
            [
                [0.1, 0.9, 0.0],  # max at col 1
                [0.5, 0.3, 0.2],  # max at col 0
                [0.1, 0.2, 0.7],  # max at col 2
            ]
        )
        T_map = convert_plan_to_map(T_plan, method="argmax")
        nonzero_cols = T_map.argmax(dim=1)
        expected_cols = torch.tensor([1, 0, 2])
        assert torch.equal(nonzero_cols, expected_cols)

    def test_preserves_dtype(self):
        """Output should preserve input dtype."""
        T_plan = torch.rand(5, 3, dtype=torch.float64)
        T_map = convert_plan_to_map(T_plan)
        assert T_map.dtype == T_plan.dtype

    def test_preserves_device(self):
        """Output should be on same device as input."""
        T_plan = torch.rand(5, 3)
        T_map = convert_plan_to_map(T_plan)
        assert T_map.device == T_plan.device

    def test_invalid_method_raises_error(self):
        """Unknown method should raise ValueError."""
        T_plan = torch.rand(5, 3)
        with pytest.raises(ValueError, match="not implemented"):
            convert_plan_to_map(T_plan, method="unknown")

    def test_single_row(self):
        """Should work with a single row."""
        T_plan = torch.tensor([[0.2, 0.8, 0.0]])
        T_map = convert_plan_to_map(T_plan)
        assert T_map.shape == (1, 3)
        assert torch.isclose(T_map.sum(), torch.tensor(1.0))
        assert T_map[0, 1] == 1.0

    def test_single_column(self):
        """Should work with a single column (all rows map to it)."""
        T_plan = torch.rand(5, 1)
        T_map = convert_plan_to_map(T_plan)
        expected = torch.full((5, 1), 1.0 / 5)
        assert torch.allclose(T_map, expected)

    def test_ties_broken_consistently(self):
        """When row has ties, argmax should pick first occurrence."""
        T_plan = torch.tensor(
            [
                [0.5, 0.5, 0.0],  # tie between col 0 and 1
            ]
        )
        T_map = convert_plan_to_map(T_plan)
        # torch.argmax returns first index on tie
        assert T_map[0, 0] == 1.0


class TestDropZeroMass:
    """Tests for drop_zero_mass function."""

    def test_removes_zero_columns(self):
        """Columns with zero mass should be removed."""
        # Plan with col 1 having zero mass
        plan = torch.tensor(
            [
                [0.5, 0.0, 0.3],
                [0.2, 0.0, 0.0],
            ]
        )
        new_plan, new_mu = drop_zero_mass(plan)
        assert new_plan.shape == (2, 2)  # col 1 removed
        expected = torch.tensor([[0.5, 0.3], [0.2, 0.0]])
        assert torch.allclose(new_plan, expected)

    def test_keeps_nonzero_columns(self):
        """Columns with nonzero mass should be kept."""
        plan = torch.tensor(
            [
                [0.3, 0.2, 0.1],
                [0.1, 0.2, 0.1],
            ]
        )
        new_plan, new_mu = drop_zero_mass(plan)
        assert new_plan.shape == plan.shape
        assert torch.allclose(new_plan, plan)

    def test_returns_correct_marginals(self):
        """Returned marginals should be column sums of new plans."""
        plan = torch.tensor(
            [
                [0.5, 0.0, 0.3],
                [0.2, 0.0, 0.1],
            ]
        )
        new_plan, new_mu = drop_zero_mass(plan)
        expected_mu = torch.tensor([0.7, 0.4])  # col sums after removing col 1
        assert torch.allclose(new_mu, expected_mu)

    def test_uses_provided_mus(self):
        """Should use provided mus for determining zero columns."""
        plan = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.1, 0.2],
            ]
        )
        # Override: pretend col 0 has zero mass
        mu = torch.tensor([0.0, 0.3, 0.5])
        new_plan, new_mu = drop_zero_mass(plan, mu=mu)
        assert new_plan.shape == (2, 2)  # col 0 removed

    def test_threshold_respected(self):
        """Values below threshold should be treated as zero."""
        plan = torch.tensor(
            [
                [0.5, 1e-16, 0.3],
                [0.2, 1e-16, 0.1],
            ]
        )
        new_plan, new_mu = drop_zero_mass(plan, thresh=1e-15)
        assert new_plan.shape == (2, 2)  # col 1 removed (below thresh)

    def test_threshold_boundary(self):
        """Values at exactly threshold should be kept (> thresh)."""
        plan = torch.tensor(
            [
                [0.5, 1e-15, 0.3],
                [0.2, 1e-15, 0.1],
            ]
        )
        # Sum of col 1 = 2e-15, which is > 1e-15, so kept
        new_plan, new_mu = drop_zero_mass(plan, thresh=1e-15)
        assert new_plan.shape == (2, 3)

    def test_all_columns_nonzero(self):
        """When no columns are zero, plan unchanged."""
        plan = torch.rand(5, 3) + 0.1  # ensure all positive
        new_plan, new_mu = drop_zero_mass(plan)
        assert new_plan.shape == plan.shape
        assert torch.allclose(new_plan, plan)

    def test_preserves_dtype(self):
        """Output should preserve input dtype."""
        plan = torch.tensor([[0.5, 0.0], [0.3, 0.2]], dtype=torch.float64)
        new_plan, new_mu = drop_zero_mass(plan)
        assert new_plan.dtype == plan.dtype
        assert new_mu.dtype == plan.dtype

    def test_preserves_device(self):
        """Output should be on same device as input."""
        plan = torch.tensor([[0.5, 0.0], [0.3, 0.2]])
        new_plan, new_mu = drop_zero_mass(plan)
        assert new_plan.device == plan.device

    def test_empty_after_drop(self):
        """If all columns are zero, result should be empty."""
        plan = torch.zeros(3, 2)
        new_plan, new_mu = drop_zero_mass(plan)
        assert new_plan.shape == (3, 0)
        assert new_mu.shape == (0,)

    def test_single_column_nonzero(self):
        """Single column with mass should be preserved."""
        plan = torch.tensor([[0.5], [0.3], [0.2]])
        new_plan, new_mu = drop_zero_mass(plan)
        assert new_plan.shape == (3, 1)
        assert torch.allclose(new_mu, torch.tensor([1.0]))
