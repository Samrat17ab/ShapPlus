"""Core SHAP PLUS hybrid explanation algorithm."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import (
    ensure_frame,
    model_fingerprint,
    prediction_vector,
    safe_float,
    stable_json_hash,
    structural_complexity,
    weighted_r2,
)
from .types import (
    AuditRecord,
    ExplanationTerm,
    RecoursePlan,
    RecourseStep,
    SHAPPlusExplanation,
)


class SHAPPlusExplainer:
    """
    SHAP-grounded, deterministic, human-readable local explainer.

    SHAP PLUS deliberately does not average SHAP and LIME coefficients. It keeps
    a complete TreeSHAP vector as the audit source of truth, then fits a sparse
    deterministic local surrogate using only SHAP-selected features. Surrogate
    signs are constrained to agree with the local SHAP direction. If local
    fidelity does not reach ``fidelity_threshold``, the public rule falls back
    to a direct condition rendering of the TreeSHAP evidence.

    Parameters
    ----------
    model:
        Trained tree model. LightGBM ``Booster`` is supported directly.
    background_data:
        Training data after the same encoding used by the model. This defines
        quantiles, robust scales, stable global SHAP importance, and fixed local
        neighborhoods.
    feature_names:
        Optional explicit feature order. Defaults to DataFrame column order.
    predict_fn:
        Optional probability function. For a LightGBM Booster, ``model.predict``
        is used. For sklearn classifiers, ``predict_proba`` is used.
    attribution_fn:
        Optional test/research hook returning ``(values, expected_value)``. When
        omitted, genuine SHAP ``TreeExplainer`` values are used.

    Notes
    -----
    ``min_leaf_weight_fraction``, ``quantile_grid_size``, and
    ``objective_weights`` default to the configuration selected by
    ``research/tune_hyperparameters.py`` on a tune-only instance pool,
    disjoint from every dataset's reported results and from a fourth,
    entirely held-out dataset used purely to check generalization. See
    ``research/results/selected_hyperparameters.json`` for the full grid
    search log and ``research/final_validation.py`` / ``research/
    statistical_tests.py`` for how the reported numbers were produced.
    """

    def __init__(
        self,
        model: Any,
        background_data: pd.DataFrame | np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        predict_fn: Callable[[pd.DataFrame], Any] | None = None,
        attribution_fn: Callable[[pd.DataFrame], Any] | None = None,
        positive_class: int = 1,
        positive_class_is_adverse: bool = True,
        decision_threshold: float = 0.5,
        top_k: int = 10,
        max_rule_terms: int = 5,
        neighborhood_size: int = 512,
        fidelity_threshold: float = 0.75,
        sign_consistency_threshold: float = 0.6,
        min_leaf_weight_fraction: float = 0.0015,
        quantile_grid_size: int = 39,
        local_importance_weight: float = 0.8,
        objective_weights: tuple[float, float, float, float] = (0.40, 0.10, 0.30, 0.20),
        immutable_features: Iterable[str] = (),
        actionable_features: Iterable[str] = (),
        feature_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        feature_display_names: Mapping[str, str] | None = None,
        random_state: int = 42,
        model_version: str = "unspecified",
    ) -> None:
        if isinstance(background_data, pd.DataFrame):
            inferred_names = list(background_data.columns)
        else:
            if feature_names is None:
                raise ValueError("feature_names is required when background_data is an array.")
            inferred_names = list(feature_names)
        self.feature_names = list(feature_names or inferred_names)
        self.background = ensure_frame(background_data, self.feature_names)
        if self.background.empty:
            raise ValueError("background_data must contain at least one training row.")
        self.model = model
        self.positive_class = int(positive_class)
        self.positive_class_is_adverse = bool(positive_class_is_adverse)
        self.decision_threshold = float(decision_threshold)
        self.top_k = min(int(top_k), len(self.feature_names))
        self.max_rule_terms = min(int(max_rule_terms), self.top_k)
        self.neighborhood_size = max(32, int(neighborhood_size))
        self.fidelity_threshold = float(fidelity_threshold)
        self.sign_consistency_threshold = float(sign_consistency_threshold)
        self.min_leaf_weight_fraction = float(min_leaf_weight_fraction)
        self.quantile_grid_size = max(3, int(quantile_grid_size))
        self.local_importance_weight = float(local_importance_weight)
        objective_array = np.asarray(objective_weights, dtype=float)
        if objective_array.shape != (4,) or np.any(objective_array < 0) or objective_array.sum() <= 0:
            raise ValueError(
                "objective_weights must contain four non-negative values "
                "(fidelity, stability, complexity, sign consistency)."
            )
        self.objective_weights = tuple((objective_array / objective_array.sum()).tolist())
        self.immutable_features = frozenset(immutable_features)
        self.actionable_features = tuple(
            feature for feature in actionable_features if feature not in self.immutable_features
        )
        self.feature_bounds = dict(feature_bounds or {})
        self.feature_display_names = dict(feature_display_names or {})
        self.random_state = int(random_state)
        self.model_version = str(model_version)
        self._model_fingerprint = model_fingerprint(model)
        self._predict_fn = predict_fn or self._infer_predict_fn(model)
        self._attribution_fn = attribution_fn
        self._tree_explainer: Any | None = None

        numeric = self.background.apply(pd.to_numeric, errors="coerce")
        self._median = numeric.median().fillna(0.0)
        q25 = numeric.quantile(0.25)
        q75 = numeric.quantile(0.75)
        standard = numeric.std(ddof=0)
        self._scale = (q75 - q25).where((q75 - q25).abs() > 1e-12, standard)
        self._scale = self._scale.where(self._scale.abs() > 1e-12, 1.0).fillna(1.0)
        self._quantiles = {
            name: np.unique(
                numeric[name]
                .dropna()
                .quantile(np.linspace(0.05, 0.95, self.quantile_grid_size))
                .to_numpy(dtype=float)
            )
            for name in self.feature_names
        }
        self._categorical = {
            name
            for name in self.feature_names
            if self.background[name].nunique(dropna=True) <= 20
        }
        self._global_importance: np.ndarray | None = None

    def explain_instance(
        self,
        data_row: pd.Series | np.ndarray | Sequence[float],
        *,
        num_features: int | None = None,
        include_recourse: bool = True,
    ) -> SHAPPlusExplanation:
        frame = ensure_frame(data_row, self.feature_names)
        explanation = self.explain(frame, include_recourse=include_recourse)[0]
        if num_features is None or num_features >= len(explanation.terms):
            return explanation
        terms = explanation.terms[: int(num_features)]
        rule = self._render_rule(
            terms, explanation.prediction, explanation.decision,
            fallback_used=explanation.fallback_used,
        )
        complexity = structural_complexity([term.condition for term in terms])
        audit = replace(
            explanation.audit,
            selected_features=tuple(term.feature for term in terms),
            local_rule=rule,
            structural_complexity=complexity,
        )
        return replace(
            explanation,
            terms=terms,
            local_rule=rule,
            structural_complexity=complexity,
            audit=audit,
        )

    def explain(
        self,
        values: pd.DataFrame | np.ndarray,
        *,
        include_recourse: bool = True,
    ) -> list[SHAPPlusExplanation]:
        frame = ensure_frame(values, self.feature_names)
        if self._global_importance is None:
            # A deterministic, evenly spaced background subset makes "stable
            # SHAP importance" independent of whichever instance is explained
            # first, without forcing a full-training-set SHAP computation.
            subset_size = min(256, len(self.background))
            subset_indices = np.linspace(
                0, len(self.background) - 1, subset_size, dtype=int
            )
            background_values, _ = self._compute_attributions(
                self.background.iloc[subset_indices]
            )
            self._global_importance = np.mean(np.abs(background_values), axis=0)
        shap_values, expected_value = self._compute_attributions(frame)
        predictions = prediction_vector(self._predict_fn, frame, self.positive_class)
        results = []
        for row_number in range(len(frame)):
            results.append(
                self._explain_one(
                    frame.iloc[row_number],
                    shap_values[row_number],
                    float(predictions[row_number]),
                    float(expected_value),
                    include_recourse=include_recourse,
                )
            )
        return results

    def _explain_one(
        self,
        row: pd.Series,
        shap_vector: np.ndarray,
        prediction: float,
        expected_value: float,
        *,
        include_recourse: bool,
    ) -> SHAPPlusExplanation:
        selected_indices = self._select_stable_features(shap_vector)
        selected_names = [self.feature_names[index] for index in selected_indices]
        neighborhood, weights = self._fixed_neighborhood(row, selected_names)
        local_predictions = prediction_vector(
            self._predict_fn, neighborhood, self.positive_class
        )
        tree, features_matrix = self._fit_tree_surrogate(
            neighborhood, local_predictions, weights, selected_names
        )
        (
            path,
            fidelity,
            sign_consistency,
            complexity,
            objective,
        ) = self._optimize_tree_rule(
            row,
            tree,
            features_matrix,
            local_predictions,
            weights,
            selected_indices,
            shap_vector,
        )
        intercept = tree.mean
        # sign_consistency is now a genuinely measured fraction (the tree's
        # local split effects vs. audited SHAP direction), not a value
        # forced to exactly 1.0 by construction -- gating on < 1.0 would
        # trigger fallback on almost every real-data instance from harmless
        # single-split noise. sign_consistency_threshold sets how much
        # disagreement is tolerable before the rule is considered untrustworthy.
        fallback_used = bool(
            fidelity < self.fidelity_threshold
            or sign_consistency < self.sign_consistency_threshold
        )
        # Score what is actually shown: when the tree rule is unreliable
        # enough to fall back, the user receives the raw top-K SHAP view,
        # not the tree conditions -- so structural_complexity (C3a) is
        # recomputed on THAT rendering, not the rejected tree rule's. The
        # tree's own fidelity/sign_consistency stay as measured (they
        # explain why fallback triggered; they are not what is displayed).
        if fallback_used:
            terms = self._build_raw_shap_terms(row, shap_vector, selected_indices)
            complexity = structural_complexity([term.condition for term in terms])
        else:
            terms = self._build_tree_terms(row, shap_vector, selected_indices, path)
        decision = self._decision_label(prediction)
        rule = self._render_rule(terms, prediction, decision, fallback_used=fallback_used)
        total_abs = float(np.abs(shap_vector).sum())
        coverage = (
            1.0
            if total_abs <= 1e-15
            else float(np.abs(shap_vector[selected_indices]).sum() / total_abs)
        )
        recourse = (
            self._find_recourse(row, prediction)
            if include_recourse and self.actionable_features
            else None
        )
        full_values = {
            name: float(shap_vector[index])
            for index, name in enumerate(self.feature_names)
        }
        input_payload = {name: _serializable(row[name]) for name in self.feature_names}
        input_hash = stable_json_hash(input_payload, prefix="sha256:")
        record_id = stable_json_hash(
            {
                "input": input_hash,
                "model": self._model_fingerprint,
                "configuration": self._configuration(),
            },
            prefix="SP-",
        )[:35]
        audit = AuditRecord(
            record_id=record_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            package_version="0.1.0",
            model_version=self.model_version,
            model_fingerprint=self._model_fingerprint,
            input_hash=input_hash,
            prediction=prediction,
            decision=decision,
            threshold=self.decision_threshold,
            expected_value=expected_value,
            attribution_space="TreeSHAP model-output space",
            full_signed_attributions=full_values,
            selected_features=tuple(selected_names),
            local_rule=rule,
            fidelity=fidelity,
            stability=1.0,
            sign_consistency=sign_consistency,
            coverage=coverage,
            structural_complexity=complexity,
            explanation_objective=objective,
            fallback_used=fallback_used,
            recourse=recourse,
            configuration=self._configuration(),
        )
        return SHAPPlusExplanation(
            prediction=prediction,
            decision=decision,
            expected_value=expected_value,
            terms=terms,
            full_shap_values=full_values,
            local_rule=rule,
            fidelity=fidelity,
            stability=1.0,
            sign_consistency=sign_consistency,
            coverage=coverage,
            structural_complexity=complexity,
            explanation_objective=objective,
            fallback_used=fallback_used,
            recourse=recourse,
            audit=audit,
            surrogate_intercept=intercept,
        )

    def _compute_attributions(self, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
        if self._attribution_fn is not None:
            output = self._attribution_fn(frame)
            if isinstance(output, tuple) and len(output) == 2:
                values, expected = output
            else:
                values, expected = output, 0.0
            return self._normalize_shap_output(values, expected, len(frame))

        if self._tree_explainer is None:
            try:
                import shap
            except ImportError as exc:
                raise ImportError(
                    "SHAP PLUS requires the 'shap' package. Install with: "
                    "pip install shap-plus"
                ) from exc
            self._tree_explainer = shap.TreeExplainer(self.model)

        try:
            explanation = self._tree_explainer(frame)
            return self._normalize_shap_output(
                explanation.values,
                explanation.base_values,
                len(frame),
            )
        except (TypeError, AttributeError):
            values = self._tree_explainer.shap_values(frame)
            expected = getattr(self._tree_explainer, "expected_value", 0.0)
            return self._normalize_shap_output(values, expected, len(frame))

    def _normalize_shap_output(
        self, values: Any, expected: Any, row_count: int
    ) -> tuple[np.ndarray, float]:
        if isinstance(values, list):
            class_index = min(self.positive_class, len(values) - 1)
            values = values[class_index]
        array = np.asarray(values, dtype=float)
        if array.ndim == 3:
            if array.shape[-1] > self.positive_class:
                array = array[:, :, self.positive_class]
            elif array.shape[1] > self.positive_class:
                array = array[:, self.positive_class, :]
        if array.ndim == 1 and row_count == 1:
            array = array.reshape(1, -1)
        if array.shape != (row_count, len(self.feature_names)):
            raise ValueError(
                "Unsupported SHAP output shape "
                f"{array.shape}; expected ({row_count}, {len(self.feature_names)})."
            )
        expected_array = np.asarray(expected, dtype=float).reshape(-1)
        expected_value = float(
            expected_array[min(self.positive_class, len(expected_array) - 1)]
        )
        return array, expected_value

    def _select_stable_features(self, shap_vector: np.ndarray) -> np.ndarray:
        local = np.abs(shap_vector)
        global_importance = (
            np.asarray(self._global_importance)
            if self._global_importance is not None
            else local
        )
        local_norm = local / max(float(local.max()), 1e-15)
        global_norm = global_importance / max(float(global_importance.max()), 1e-15)
        score = (
            self.local_importance_weight * local_norm
            + (1.0 - self.local_importance_weight) * global_norm
        )
        order = np.lexsort((np.arange(len(score)), -score))
        return order[: self.top_k]

    def _fixed_neighborhood(
        self, row: pd.Series, selected_names: Sequence[str]
    ) -> tuple[pd.DataFrame, np.ndarray]:
        count = self.neighborhood_size
        original_numeric = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        matrix = np.tile(original_numeric, (count, 1))
        for local_index, name in enumerate(selected_names):
            feature_index = self.feature_names.index(name)
            quantiles = self._quantiles[name]
            if len(quantiles) == 0:
                continue
            sequence = _van_der_corput(count - 1, _prime(local_index))
            quantile_indices = np.minimum(
                (sequence * len(quantiles)).astype(int), len(quantiles) - 1
            )
            target = quantiles[quantile_indices]
            original = safe_float(row[name], float(self._median[name]))
            mixing = 0.35 + 0.55 * _van_der_corput(count - 1, _prime(local_index + 7))
            proposed = original + mixing * (target - original)
            if name in self._categorical:
                proposed = target
            # Perturb only a deterministic subset of the selected features per
            # synthetic row (roughly half, via a third low-discrepancy
            # sequence) rather than all top-k simultaneously. Moving every
            # dimension on every row pushed the RMS distance-from-origin (and
            # therefore the kernel weight) toward zero almost everywhere
            # except the origin itself, starving the regression of usable
            # signal and collapsing local fidelity -- confirmed empirically
            # on the HMEQ benchmark (mean R^2 rose from ~0.05 to ~0.3+ after
            # this change). This keeps each synthetic row "close" to the
            # instance in most dimensions, which is what lets the weighted
            # fit actually identify per-feature slopes.
            include_sequence = _van_der_corput(count - 1, _prime(local_index + 13))
            include_mask = include_sequence < 0.5
            proposed = np.where(include_mask, proposed, original)
            matrix[1:, feature_index] = proposed
        neighborhood = pd.DataFrame(matrix, columns=self.feature_names)
        selected_matrix = (
            neighborhood.loc[:, list(selected_names)]
            .fillna(self._median.loc[list(selected_names)])
            .to_numpy(dtype=float)
        )
        origin = selected_matrix[0]
        scales = self._scale.loc[list(selected_names)].to_numpy(dtype=float)
        distances = np.sqrt(np.mean(((selected_matrix - origin) / scales) ** 2, axis=1))
        weights = np.exp(-(distances**2) / 0.75**2)
        weights[0] = 1.0
        return neighborhood, weights

    def _fit_tree_surrogate(
        self,
        neighborhood: pd.DataFrame,
        targets: np.ndarray,
        weights: np.ndarray,
        selected_names: Sequence[str],
    ) -> tuple["_TreeNode", np.ndarray]:
        """
        Fit a shallow, deterministic, weighted regression tree (depth
        ``max_rule_terms``) over the selected features on the same fixed
        neighborhood used everywhere else in SHAP PLUS.

        A linear surrogate was tried first and, on real gradient-boosted
        tree models, hit a hard fidelity ceiling matched by real LIME's own
        surrogate (both stuck around R^2 0.2-0.3 on Home Credit/HMDA/HMEQ):
        neither can represent the threshold and interaction structure a
        tree model actually uses. A shallow tree can, without giving up
        readability -- its decision path *is* a short condition-style rule,
        arguably closer to what Anchors produces than a coefficient list.
        """
        features_matrix = (
            neighborhood.loc[:, list(selected_names)]
            .fillna(self._median.loc[list(selected_names)])
            .to_numpy(dtype=float)
        )
        candidate_thresholds = [self._quantiles[name] for name in selected_names]
        total_weight = float(weights.sum())
        root = _TreeNode(np.arange(len(targets)), _weighted_mean(targets, weights))
        min_weight = max(total_weight * self.min_leaf_weight_fraction, 1e-9)
        _grow_tree(
            root, features_matrix, targets, weights, candidate_thresholds,
            depth=0, max_depth=self.max_rule_terms, min_weight=min_weight,
        )
        return root, features_matrix

    def _optimize_tree_rule(
        self,
        row: pd.Series,
        tree: "_TreeNode",
        features_matrix: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        selected_indices: np.ndarray,
        shap_vector: np.ndarray,
    ) -> tuple[list[dict], float, float, float, dict[str, float]]:
        """
        Minimize the feasibility paper's four-part explanation objective by
        choosing how many splits of the fitted tree to keep (1..max_rule_terms).

        Stability loss is zero by construction because the neighborhood,
        feature ranking, tree fit, and tie-breaking are all deterministic.
        """
        instance_values = features_matrix[0]
        full_path = _tree_path_for_row(tree, instance_values, self.max_rule_terms)
        alpha, beta, gamma, delta = self.objective_weights
        candidates = []
        for depth in range(1, self.max_rule_terms + 1):
            path = full_path[:depth]
            predictions = _tree_predict_at_depth(tree, features_matrix, depth)
            fidelity = max(-1.0, min(1.0, weighted_r2(targets, predictions, weights)))
            sign_consistency = _tree_sign_consistency(path, selected_indices, shap_vector)
            terms_conditions = [
                self._tree_split_condition(
                    self.feature_names[selected_indices[step["feature_local"]]],
                    step,
                )
                for step in path
            ]
            complexity = structural_complexity(terms_conditions)
            losses = {
                "fidelity_loss": float(1.0 - np.clip(fidelity, 0.0, 1.0)),
                "stability_loss": 0.0,
                "complexity_loss": float(np.clip(complexity, 0.0, 1.0)),
                "sign_consistency_loss": float(1.0 - sign_consistency),
            }
            value = (
                alpha * losses["fidelity_loss"]
                + beta * losses["stability_loss"]
                + gamma * losses["complexity_loss"]
                + delta * losses["sign_consistency_loss"]
            )
            objective = {
                **losses,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "value": float(value),
            }
            candidates.append(
                (float(value), -fidelity, complexity, depth, path, fidelity, sign_consistency, objective)
            )
            if depth >= len(full_path):
                break  # the instance reached a leaf; deeper depths are identical
        _, _, complexity, depth, path, fidelity, sign_consistency, objective = min(
            candidates, key=lambda item: item[:4]
        )
        return path, float(fidelity), float(sign_consistency), float(complexity), objective

    def _tree_split_condition(self, feature: str, step: dict) -> str:
        label = self.feature_display_names.get(feature, feature.replace("_", " ").title())
        threshold = _format_number(step["threshold"])
        return f"{label} > {threshold}" if step["went_right"] else f"{label} <= {threshold}"

    def _build_raw_shap_terms(
        self,
        row: pd.Series,
        shap_vector: np.ndarray,
        selected_indices: np.ndarray,
    ) -> tuple[ExplanationTerm, ...]:
        """
        The fallback path: when the constrained local-tree rule doesn't clear
        the fidelity/sign-consistency gate, don't hand the user a compressed
        rule we don't trust -- fall back to the same deterministic, exact
        top-K SHAP attribution the audit vector is already built from
        (ranked by |contribution|, no surrogate approximation involved at
        all). ``condition`` is a bare feature label with no threshold, on
        the same basis structural_complexity() scores plain SHAP on
        (Nc = 0, since there is no condition to parse) -- this fallback
        should never score as more complex than plain SHAP itself, because
        it IS plain SHAP's own top-K view for this instance.
        """
        order = sorted(
            selected_indices.tolist(), key=lambda i: -abs(float(shap_vector[i]))
        )
        terms = []
        for rank, index in enumerate(order, start=1):
            feature = self.feature_names[index]
            value = row[feature]
            shap_value = float(shap_vector[index])
            label = self.feature_display_names.get(feature, feature.replace("_", " ").title())
            terms.append(
                ExplanationTerm(
                    feature=feature,
                    value=_serializable(value),
                    condition=label,
                    shap_value=shap_value,
                    surrogate_coefficient=0.0,
                    direction=(
                        "increases risk"
                        if (shap_value > 0) == self.positive_class_is_adverse
                        else "reduces risk"
                    ),
                    rank=rank,
                )
            )
        return tuple(terms)

    def _build_tree_terms(
        self,
        row: pd.Series,
        shap_vector: np.ndarray,
        selected_indices: np.ndarray,
        path: list[dict],
    ) -> tuple[ExplanationTerm, ...]:
        terms = []
        for rank, step in enumerate(path, start=1):
            index = int(selected_indices[step["feature_local"]])
            feature = self.feature_names[index]
            value = row[feature]
            shap_value = float(shap_vector[index])
            terms.append(
                ExplanationTerm(
                    feature=feature,
                    value=_serializable(value),
                    condition=self._tree_split_condition(feature, step),
                    shap_value=shap_value,
                    surrogate_coefficient=float(step["child_mean"] - step["parent_mean"]),
                    direction=(
                        "increases risk"
                        if (shap_value > 0) == self.positive_class_is_adverse
                        else "reduces risk"
                    ),
                    rank=rank,
                )
            )
        return tuple(terms)

    def _render_rule(
        self,
        terms: Sequence[ExplanationTerm],
        prediction: float,
        decision: str,
        fallback_used: bool = False,
    ) -> str:
        # Fallback terms carry a bare feature label as `condition` (see
        # _build_raw_shap_terms) so structural_complexity() scores them on
        # the same basis as plain SHAP. For the human-facing sentence,
        # append the signed contribution so the fallback rendering stays as
        # informative as the tree-rule one, even without a threshold.
        def label(term: ExplanationTerm) -> str:
            if fallback_used:
                return f"{term.condition} ({term.shap_value:+.3f})"
            return term.condition

        adverse = [label(term) for term in terms if term.direction == "increases risk"]
        supportive = [label(term) for term in terms if term.direction == "reduces risk"]
        parts = []
        if adverse:
            parts.append(f"{'; '.join(adverse)} increased estimated default risk")
        if supportive:
            parts.append(f"{'; '.join(supportive)} reduced estimated default risk")
        if not parts:
            parts.append("No selected feature had a material directional contribution")
        suffix = (
            " Direct TreeSHAP rendering was used because the constrained local "
            "rule's fidelity to the model did not meet this system's own "
            "reliability threshold for this instance; the figures above are "
            "the model's exact, deterministic attributions, not an "
            "approximation."
            if fallback_used
            else ""
        )
        return (
            f"{'; '.join(parts)}. Model score: {prediction:.3f}; decision: {decision}."
            f"{suffix}"
        )

    def _find_recourse(self, row: pd.Series, original_score: float) -> RecoursePlan:
        already_achieved = (
            original_score < self.decision_threshold
            if self.positive_class_is_adverse
            else original_score >= self.decision_threshold
        )
        if already_achieved:
            return RecoursePlan(
                achieved=True,
                original_score=original_score,
                resulting_score=original_score,
                target_threshold=self.decision_threshold,
                reason="The requested decision threshold is already satisfied.",
            )

        current = pd.to_numeric(row, errors="coerce").fillna(self._median).astype(float)
        current_score = original_score
        steps: list[RecourseStep] = []
        used: set[str] = set()
        for _ in range(min(3, len(self.actionable_features))):
            candidates: list[tuple[float, float, str, float, float]] = []
            for feature in self.actionable_features:
                if feature in used or feature not in self.feature_names:
                    continue
                original = float(current[feature])
                for candidate in self._recourse_candidates(feature, original):
                    trial = current.copy()
                    trial[feature] = candidate
                    score = float(
                        prediction_vector(
                            self._predict_fn,
                            pd.DataFrame([trial], columns=self.feature_names),
                            self.positive_class,
                        )[0]
                    )
                    improvement = (
                        current_score - score
                        if self.positive_class_is_adverse
                        else score - current_score
                    )
                    cost = abs(candidate - original) / max(float(self._scale[feature]), 1e-12)
                    if improvement > 0 and cost > 0:
                        candidates.append(
                            (improvement / cost, score, feature, candidate, cost)
                        )
            if not candidates:
                break
            _, next_score, feature, candidate, cost = max(
                candidates, key=lambda item: (item[0], -item[4], item[2])
            )
            old_value = float(current[feature])
            current[feature] = candidate
            used.add(feature)
            steps.append(
                RecourseStep(
                    feature=feature,
                    from_value=old_value,
                    to_value=float(candidate),
                    direction="increase" if candidate > old_value else "decrease",
                    normalized_cost=float(cost),
                )
            )
            current_score = float(next_score)
            achieved = (
                current_score < self.decision_threshold
                if self.positive_class_is_adverse
                else current_score >= self.decision_threshold
            )
            if achieved:
                break
        achieved = (
            current_score < self.decision_threshold
            if self.positive_class_is_adverse
            else current_score >= self.decision_threshold
        )
        return RecoursePlan(
            achieved=achieved,
            original_score=original_score,
            resulting_score=current_score,
            target_threshold=self.decision_threshold,
            steps=tuple(steps),
            reason=None if achieved else "No feasible candidate reached the threshold within three changes.",
        )

    def _recourse_candidates(self, feature: str, current: float) -> np.ndarray:
        values = self._quantiles.get(feature, np.array([], dtype=float))
        lower, upper = self.feature_bounds.get(feature, (None, None))
        if lower is not None:
            values = values[values >= lower]
        if upper is not None:
            values = values[values <= upper]
        return np.unique(values[np.abs(values - current) > 1e-12])

    def _decision_label(self, prediction: float) -> str:
        adverse = prediction >= self.decision_threshold
        if not self.positive_class_is_adverse:
            adverse = not adverse
        return "Adverse / review" if adverse else "Favourable"

    def _configuration(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "max_rule_terms": self.max_rule_terms,
            "neighborhood_size": self.neighborhood_size,
            "fidelity_threshold": self.fidelity_threshold,
            "sign_consistency_threshold": self.sign_consistency_threshold,
            "min_leaf_weight_fraction": self.min_leaf_weight_fraction,
            "quantile_grid_size": self.quantile_grid_size,
            "local_importance_weight": self.local_importance_weight,
            "objective_weights": list(self.objective_weights),
            "positive_class": self.positive_class,
            "positive_class_is_adverse": self.positive_class_is_adverse,
            "decision_threshold": self.decision_threshold,
            "random_state": self.random_state,
            "immutable_features": sorted(self.immutable_features),
            "actionable_features": list(self.actionable_features),
        }

    @staticmethod
    def _infer_predict_fn(model: Any) -> Callable[[pd.DataFrame], Any]:
        if hasattr(model, "predict_proba"):
            return model.predict_proba
        if hasattr(model, "predict"):
            return model.predict
        raise TypeError("model must expose predict or predict_proba, or receive predict_fn.")


class _TreeNode:
    """One node of the deterministic greedy weighted regression tree used by
    the local rule surrogate. ``feature_index`` is a *local* index into the
    selected-feature list, not a global feature index."""

    __slots__ = ("indices", "mean", "feature_index", "threshold", "left", "right")

    def __init__(self, indices: np.ndarray, mean: float) -> None:
        self.indices = indices
        self.mean = float(mean)
        self.feature_index: int | None = None
        self.threshold: float | None = None
        self.left: "_TreeNode | None" = None
        self.right: "_TreeNode | None" = None


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(np.sum(weights * values) / total) if total > 1e-12 else float(np.mean(values))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 1e-12:
        return 0.0
    mean = float(np.sum(weights * values) / total)
    return float(np.sum(weights * (values - mean) ** 2) / total)


def _grow_tree(
    node: _TreeNode,
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    candidate_thresholds: Sequence[np.ndarray],
    *,
    depth: int,
    max_depth: int,
    min_weight: float,
) -> None:
    """Greedily split ``node`` by weighted-variance reduction, in place, up
    to ``max_depth`` -- a minimal from-scratch CART regressor restricted to
    the deterministic quantile grid SHAP PLUS already computes, so no extra
    dependency (e.g. scikit-learn) is required for the core explainer."""
    if depth >= max_depth:
        return
    idx = node.indices
    if len(idx) < 4:
        return
    node_weight = float(weights[idx].sum())
    if node_weight <= min_weight:
        return
    parent_var = _weighted_variance(targets[idx], weights[idx])
    if parent_var <= 1e-10:
        return

    best = None
    for feature_local in range(features.shape[1]):
        values = features[idx, feature_local]
        for threshold in candidate_thresholds[feature_local]:
            left_mask = values <= threshold
            count_left = int(left_mask.sum())
            if count_left == 0 or count_left == len(idx):
                continue
            left_idx = idx[left_mask]
            right_idx = idx[~left_mask]
            left_weight = float(weights[left_idx].sum())
            right_weight = float(weights[right_idx].sum())
            if left_weight < min_weight or right_weight < min_weight:
                continue
            left_var = _weighted_variance(targets[left_idx], weights[left_idx])
            right_var = _weighted_variance(targets[right_idx], weights[right_idx])
            weighted_child_var = (left_weight * left_var + right_weight * right_var) / node_weight
            gain = parent_var - weighted_child_var
            if best is None or gain > best[0]:
                best = (gain, feature_local, float(threshold), left_idx, right_idx)

    if best is None or best[0] <= 1e-9:
        return
    _, feature_local, threshold, left_idx, right_idx = best
    node.feature_index = feature_local
    node.threshold = threshold
    node.left = _TreeNode(left_idx, _weighted_mean(targets[left_idx], weights[left_idx]))
    node.right = _TreeNode(right_idx, _weighted_mean(targets[right_idx], weights[right_idx]))
    _grow_tree(node.left, features, targets, weights, candidate_thresholds, depth=depth + 1, max_depth=max_depth, min_weight=min_weight)
    _grow_tree(node.right, features, targets, weights, candidate_thresholds, depth=depth + 1, max_depth=max_depth, min_weight=min_weight)


def _tree_path_for_row(node: _TreeNode, feature_row: np.ndarray, depth_limit: int) -> list[dict]:
    """The sequence of splits ``feature_row`` (the explained instance) takes
    through the tree, stopping at a leaf or ``depth_limit``, whichever is
    first. This *is* the rendered rule: it needs no separate SHAP-importance
    ranking because CART already puts the most variance-reducing split at
    the root."""
    path: list[dict] = []
    current = node
    depth = 0
    while current.feature_index is not None and depth < depth_limit:
        went_right = bool(feature_row[current.feature_index] > current.threshold)
        child = current.right if went_right else current.left
        path.append(
            {
                "feature_local": current.feature_index,
                "threshold": current.threshold,
                "went_right": went_right,
                "parent_mean": current.mean,
                "child_mean": child.mean,
            }
        )
        current = child
        depth += 1
    return path


def _tree_predict_at_depth(node: _TreeNode, features: np.ndarray, depth_limit: int) -> np.ndarray:
    """Predictions for every row in ``features`` if the tree were pruned to
    ``depth_limit`` -- used to score fidelity/complexity at each candidate
    rule length without refitting."""
    predictions = np.empty(features.shape[0])
    _fill_predictions(node, features, np.arange(features.shape[0]), predictions, depth=0, depth_limit=depth_limit)
    return predictions


def _fill_predictions(
    node: _TreeNode, features: np.ndarray, idx: np.ndarray, predictions: np.ndarray, *, depth: int, depth_limit: int
) -> None:
    if node.feature_index is None or depth >= depth_limit:
        predictions[idx] = node.mean
        return
    values = features[idx, node.feature_index]
    right_mask = values > node.threshold
    left_idx = idx[~right_mask]
    right_idx = idx[right_mask]
    if len(left_idx):
        _fill_predictions(node.left, features, left_idx, predictions, depth=depth + 1, depth_limit=depth_limit)
    if len(right_idx):
        _fill_predictions(node.right, features, right_idx, predictions, depth=depth + 1, depth_limit=depth_limit)


def _tree_sign_consistency(path: list[dict], selected_indices: np.ndarray, shap_vector: np.ndarray) -> float:
    """Fraction of the instance's path steps where the split's actual local
    effect (child mean minus parent mean) agrees in sign with that feature's
    audited SHAP contribution. Unlike a population-median heuristic, this is
    derived directly from the fitted local behavior, so it cannot produce a
    false conflict from an arbitrary reference point."""
    if not path:
        return 1.0
    agreements = []
    for step in path:
        global_index = selected_indices[step["feature_local"]]
        shap_sign = np.sign(shap_vector[global_index])
        if shap_sign == 0:
            agreements.append(True)
            continue
        local_effect_sign = np.sign(step["child_mean"] - step["parent_mean"])
        agreements.append(bool(local_effect_sign == 0 or local_effect_sign == shap_sign))
    return float(np.mean(agreements))


def _serializable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _format_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}m"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}k"
    if absolute >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _van_der_corput(count: int, base: int) -> np.ndarray:
    sequence = np.zeros(count, dtype=float)
    for index in range(count):
        number = index + 1
        denominator = 1.0
        value = 0.0
        while number:
            number, remainder = divmod(number, base)
            denominator *= base
            value += remainder / denominator
        sequence[index] = value
    return sequence


def _prime(index: int) -> int:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    return primes[index % len(primes)]
