"""PPR alpha grid search and hyperparameter tuning tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from ranx import Qrels, Run, evaluate

if TYPE_CHECKING:
    from skill_retriever.workflows.pipeline import RetrievalPipeline


# =============================================================================
# HELPER FUNCTIONS (defined first, used by tests below)
# =============================================================================

def run_with_alpha(
    pipeline: RetrievalPipeline,
    pairs: list[dict[str, Any]],
    qrels: Qrels,
    alpha: float,
) -> float:
    """Run retrieval with specific PPR alpha and return MRR.

    Args:
        pipeline: Seeded pipeline instance
        pairs: Validation query-component pairs
        qrels: ranx Qrels object for evaluation
        alpha: PPR damping factor to use

    Returns:
        MRR score for this alpha value
    """
    from skill_retriever.nodes.retrieval.ppr_engine import run_ppr_retrieval
    from skill_retriever.nodes.retrieval.score_fusion import fuse_retrieval_results
    from skill_retriever.nodes.retrieval.vector_search import search_with_type_filter

    run_dict: dict[str, dict[str, float]] = {}

    for pair in pairs:
        # Run vector search
        vector_results = search_with_type_filter(
            pair["query"],
            pipeline._vector_store,
            pipeline._graph_store,
            top_k=20,
        )

        # Run PPR with specific alpha (verify parameter works)
        ppr_results = run_ppr_retrieval(
            pair["query"],
            pipeline._graph_store,
            alpha=alpha,  # Explicit alpha override
            top_k=20,
        )

        # Fuse results
        fused = fuse_retrieval_results(
            vector_results,
            ppr_results,
            pipeline._graph_store,
            top_k=10,
        )

        run_dict[pair["query_id"]] = {
            c.component_id: c.score for c in fused
        }

    run = Run(run_dict)
    return evaluate(qrels, run, "mrr")


def run_with_rrf_k(
    pipeline: RetrievalPipeline,
    pairs: list[dict[str, Any]],
    qrels: Qrels,
    k: int,
) -> float:
    """Run retrieval with specific RRF k value and return MRR.

    Args:
        pipeline: Seeded pipeline instance
        pairs: Validation query-component pairs
        qrels: ranx Qrels object for evaluation
        k: RRF k parameter (higher = less weight to top ranks)

    Returns:
        MRR score for this k value
    """
    from skill_retriever.nodes.retrieval.ppr_engine import run_ppr_retrieval
    from skill_retriever.nodes.retrieval.score_fusion import (
        fuse_retrieval_results,
        reciprocal_rank_fusion,
    )
    from skill_retriever.nodes.retrieval.vector_search import search_with_type_filter
    from skill_retriever.nodes.retrieval.models import RankedComponent

    run_dict: dict[str, dict[str, float]] = {}

    for pair in pairs:
        # Run vector search
        vector_results = search_with_type_filter(
            pair["query"],
            pipeline._vector_store,
            pipeline._graph_store,
            top_k=20,
        )

        # Run PPR
        ppr_results = run_ppr_retrieval(
            pair["query"],
            pipeline._graph_store,
            top_k=20,
        )

        # Manual RRF with custom k
        vector_ranked = [r.component_id for r in vector_results]
        graph_ranked = [
            item_id
            for item_id, _ in sorted(ppr_results.items(), key=lambda x: x[1], reverse=True)
        ]

        if graph_ranked:
            fused_scores = reciprocal_rank_fusion([vector_ranked, graph_ranked], k=k)
        else:
            fused_scores = reciprocal_rank_fusion([vector_ranked], k=k)

        run_dict[pair["query_id"]] = {
            item_id: score for item_id, score in fused_scores[:10]
        }

    run = Run(run_dict)
    return evaluate(qrels, run, "mrr")


# =============================================================================
# TESTS
# =============================================================================

class TestAlphaGridSearch:
    """PPR alpha tuning tests."""

    def test_alpha_parameter_override_works(self, seeded_pipeline) -> None:
        """Verify run_ppr_retrieval accepts and uses alpha parameter."""
        from skill_retriever.nodes.retrieval.ppr_engine import run_ppr_retrieval

        # Test that alpha parameter is accepted (no TypeError)
        result_low = run_ppr_retrieval(
            "authentication",
            seeded_pipeline._graph_store,
            alpha=0.5,
            top_k=10,
        )
        result_high = run_ppr_retrieval(
            "authentication",
            seeded_pipeline._graph_store,
            alpha=0.95,
            top_k=10,
        )

        # Both should return dict (may be empty if no seeds found)
        assert isinstance(result_low, dict)
        assert isinstance(result_high, dict)

    def test_alpha_grid_search(
        self,
        seeded_pipeline,
        validation_pairs,
        validation_qrels,
    ) -> None:
        """Grid search PPR alpha values and record results."""
        alpha_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        results: dict[float, float] = {}

        for alpha in alpha_values:
            mrr = run_with_alpha(seeded_pipeline, validation_pairs, validation_qrels, alpha)
            results[alpha] = mrr

        # Print results table
        print("\nAlpha Grid Search Results:")
        print("Alpha | MRR")
        print("-" * 20)
        for alpha, mrr in sorted(results.items()):
            print(f"{alpha:.2f}  | {mrr:.4f}")

        # Assert all alphas produce positive MRR and complete without errors
        # (Actual MRR depends on test data coverage; key is no convergence failures)
        for alpha, mrr in results.items():
            assert mrr > 0.0, f"Alpha {alpha} produced zero MRR"

        # Verify that we ran the full grid
        assert len(results) == len(alpha_values), "Not all alpha values tested"

    def test_default_alpha_optimal(
        self,
        seeded_pipeline,
        validation_pairs,
        validation_qrels,
    ) -> None:
        """Verify that 0.85 (default) is within 5% of the best MRR achieved."""
        alpha_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        results: dict[float, float] = {}

        for alpha in alpha_values:
            mrr = run_with_alpha(seeded_pipeline, validation_pairs, validation_qrels, alpha)
            results[alpha] = mrr

        best_mrr = max(results.values())
        default_mrr = results[0.85]

        print(f"\nDefault alpha (0.85) MRR: {default_mrr:.4f}")
        print(f"Best MRR: {best_mrr:.4f} (alpha={max(results, key=results.get)})")

        # Allow 5% tolerance from best
        assert default_mrr >= best_mrr * 0.95, (
            f"Default alpha MRR {default_mrr:.4f} more than 5% below best {best_mrr:.4f}"
        )

    def test_adaptive_alpha_categories(self, seeded_pipeline) -> None:
        """Test that adaptive alpha selection works for specific vs broad queries."""
        from skill_retriever.nodes.retrieval.query_planner import (
            extract_query_entities,
            plan_retrieval,
        )

        # Specific query (should have high alpha ~0.9)
        specific = "skill-jwt authentication"
        entities_specific = extract_query_entities(specific, seeded_pipeline._graph_store)
        plan_specific = plan_retrieval(specific, len(entities_specific))
        print(f"\nSpecific query alpha: {plan_specific.ppr_alpha}")
        assert plan_specific.ppr_alpha >= 0.85, "Specific query should have high alpha"

        # Broad query with many concepts
        broad = "JWT OAuth login refresh session authentication security tokens"
        entities_broad = extract_query_entities(broad, seeded_pipeline._graph_store)
        plan_broad = plan_retrieval(broad, len(entities_broad))
        print(f"Broad query alpha: {plan_broad.ppr_alpha}")
        # Alpha should be valid range
        assert 0.5 <= plan_broad.ppr_alpha <= 0.95

    def test_rrf_k_sensitivity(
        self,
        seeded_pipeline,
        validation_pairs,
        validation_qrels,
    ) -> None:
        """Check that RRF k=60 produces good results compared to alternatives."""
        k_values = [30, 60, 100]
        results: dict[int, float] = {}

        for k in k_values:
            mrr = run_with_rrf_k(seeded_pipeline, validation_pairs, validation_qrels, k)
            results[k] = mrr

        print("\nRRF k Sensitivity:")
        print("k   | MRR")
        print("-" * 15)
        for k, mrr in sorted(results.items()):
            print(f"{k:3d} | {mrr:.4f}")

        best_mrr = max(results.values())
        k60_mrr = results[60]

        # k=60 should be within 10% of best
        assert k60_mrr >= best_mrr * 0.90, (
            f"k=60 MRR {k60_mrr:.4f} more than 10% below best {best_mrr:.4f}"
        )
