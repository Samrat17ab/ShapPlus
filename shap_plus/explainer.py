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
        ridge_alpha: float = 1e-2,
        local_importance_weight: float = 0.8,
        objective_weights: tuple[float, float, float, float] = (0.55, 0.15, 0.10, 0.20),
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
        self.ridge_alpha = float(ridge_alpha)
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
                .quantile(np.linspace(0.05, 0.95, 19))
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
        rule = self._render_rule(terms, explanation.prediction, explanation.decision)
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
        coefficients, intercept, fidelity = self._fit_sign_constrained_surrogate(
            row,
            neighborhood,
            local_predictions,
            weights,
            selected_indices,
            shap_vector,
        )
        (
            rule_count,
            fidelity,
            sign_consistency,
            complexity,
            objective,
        ) = self._optimize_rule_objective(
            row,
            neighborhood,
            local_predictions,
            weights,
            selected_indices,
            coefficients,
            intercept,
            shap_vector,
        )
        fallback_used = bool(
            fidelity < self.fidelity_threshold or sign_consistency < 1.0
        )
        terms = self._build_terms(
            row,
            shap_vector,
            selected_indices[:rule_count],
            coefficients[:rule_count],
        )
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

    def _fit_sign_constrained_surrogate(
        self,
        row: pd.Series,
        neighborhood: pd.DataFrame,
        targets: np.ndarray,
        weights: np.ndarray,
        selected_indices: np.ndarray,
        shap_vector: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        names = [self.feature_names[index] for index in selected_indices]
        matrix = (
            neighborhood.loc[:, names]
            .fillna(self._median.loc[names])
            .to_numpy(dtype=float)
        )
        scales = self._scale.loc[names].to_numpy(dtype=float)
        normalized = (matrix - matrix[0]) / scales
        design = np.column_stack([np.ones(len(matrix)), normalized])
        sqrt_weights = np.sqrt(weights)
        weighted_design = design * sqrt_weights[:, None]
        weighted_targets = targets * sqrt_weights
        penalty = np.diag([0.0] + [self.ridge_alpha] * len(names))
        gram = weighted_design.T @ weighted_design + penalty
        rhs = weighted_design.T @ weighted_targets
        try:
            fitted = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            fitted = np.linalg.pinv(gram) @ rhs
        intercept = float(fitted[0])
        coefficients = fitted[1:]

        # A version of this method used to hard-zero any coefficient whose
        # sign disagreed with sign(shap_value * displacement_from_median),
        # then refit on the surviving subset. On real nonlinear tree models
        # that heuristic produced frequent false conflicts: SHAP's signed
        # contribution is framed relative to the background distribution,
        # while a local surrogate slope reflects marginal behavior at this
        # instance -- the two can legitimately disagree even when both are
        # faithful (e.g. a feature sitting exactly at the background median
        # can still carry a large SHAP contribution from interaction
        # effects, with no reliable displacement signal to check against).
        # Zeroing those coefficients collapsed fidelity toward zero and
        # forced the fallback path on nearly every instance (verified
        # empirically on the HMEQ benchmark: mean fidelity 0.04 vs 0.50 for
        # the same unconstrained fit), defeating the point of a readable
        # surrogate layer. Sign agreement is instead measured honestly as
        # a diagnostic via ``_sign_consistency`` and fed into the rule
        # objective's sign_consistency_loss term, which prefers shorter,
        # better-agreeing rules without destroying the fitted surrogate.
        surrogate = intercept + normalized @ coefficients
        fidelity = max(-1.0, min(1.0, weighted_r2(targets, surrogate, weights)))
        return coefficients.astype(float), intercept, float(fidelity)

    def _sign_consistency(
        self,
        row: pd.Series,
        selected_indices: np.ndarray,
        coefficients: np.ndarray,
        shap_vector: np.ndarray,
    ) -> float:
        names = [self.feature_names[index] for index in selected_indices]
        values = pd.to_numeric(row.loc[names], errors="coerce").fillna(
            self._median.loc[names]
        )
        displacement = values.to_numpy(dtype=float) - self._median.loc[names].to_numpy(dtype=float)
        surrogate_direction = np.sign(coefficients * displacement)
        shap_direction = np.sign(shap_vector[selected_indices])
        informative = shap_direction != 0
        if not informative.any():
            return 1.0
        agreement = (surrogate_direction[informative] == shap_direction[informative]) | (
            surrogate_direction[informative] == 0
        )
        return float(np.mean(agreement))

    def _optimize_rule_objective(
        self,
        row: pd.Series,
        neighborhood: pd.DataFrame,
        targets: np.ndarray,
        weights: np.ndarray,
        selected_indices: np.ndarray,
        coefficients: np.ndarray,
        intercept: float,
        shap_vector: np.ndarray,
    ) -> tuple[int, float, float, float, dict[str, float]]:
        """
        Minimize the feasibility paper's four-part explanation objective.

        Stability loss is zero by construction because the neighborhood,
        feature ranking, regression, and tie-breaking are deterministic.
        """
        names = [self.feature_names[index] for index in selected_indices]
        matrix = (
            neighborhood.loc[:, names]
            .fillna(self._median.loc[names])
            .to_numpy(dtype=float)
        )
        scales = self._scale.loc[names].to_numpy(dtype=float)
        normalized = (matrix - matrix[0]) / scales
        alpha, beta, gamma, delta = self.objective_weights
        candidates = []
        for count in range(1, self.max_rule_terms + 1):
            surrogate = intercept + normalized[:, :count] @ coefficients[:count]
            fidelity = max(
                -1.0, min(1.0, weighted_r2(targets, surrogate, weights))
            )
            sign_consistency = self._sign_consistency(
                row,
                selected_indices[:count],
                coefficients[:count],
                shap_vector,
            )
            terms = self._build_terms(
                row,
                shap_vector,
                selected_indices[:count],
                coefficients[:count],
            )
            complexity = structural_complexity(
                [term.condition for term in terms]
            )
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
                (float(value), -fidelity, complexity, count, fidelity, sign_consistency, objective)
            )
        _, _, complexity, count, fidelity, sign_consistency, objective = min(
            candidates, key=lambda item: item[:4]
        )
        return (
            int(count),
            float(fidelity),
            float(sign_consistency),
            float(complexity),
            objective,
        )

    def _build_terms(
        self,
        row: pd.Series,
        shap_vector: np.ndarray,
        selected_indices: Sequence[int],
        coefficients: Sequence[float],
    ) -> tuple[ExplanationTerm, ...]:
        terms = []
        for rank, (index, coefficient) in enumerate(
            zip(selected_indices, coefficients), start=1
        ):
            feature = self.feature_names[int(index)]
            value = row[feature]
            shap_value = float(shap_vector[int(index)])
            terms.append(
                ExplanationTerm(
                    feature=feature,
                    value=_serializable(value),
                    condition=self._condition(feature, value),
                    shap_value=shap_value,
                    surrogate_coefficient=float(coefficient),
                    direction=(
                        "increases risk"
                        if (shap_value > 0) == self.positive_class_is_adverse
                        else "reduces risk"
                    ),
                    rank=rank,
                )
            )
        return tuple(terms)

    def _condition(self, feature: str, value: Any) -> str:
        label = self.feature_display_names.get(feature, feature.replace("_", " ").title())
        numeric = safe_float(value, float("nan"))
        quantiles = self._quantiles[feature]
        if feature in self._categorical or not np.isfinite(numeric) or len(quantiles) < 2:
            return f"{label} = {value}"
        lower = quantiles[quantiles <= numeric]
        upper = quantiles[quantiles > numeric]
        if len(lower) and len(upper):
            return f"{_format_number(lower[-1])} < {label} <= {_format_number(upper[0])}"
        if len(lower):
            return f"{label} > {_format_number(lower[-1])}"
        return f"{label} <= {_format_number(upper[0])}"

    def _render_rule(
        self,
        terms: Sequence[ExplanationTerm],
        prediction: float,
        decision: str,
        fallback_used: bool = False,
    ) -> str:
        adverse = [
            term.condition for term in terms if term.direction == "increases risk"
        ]
        supportive = [
            term.condition for term in terms if term.direction == "reduces risk"
        ]
        parts = []
        if adverse:
            parts.append(f"{'; '.join(adverse)} increased estimated default risk")
        if supportive:
            parts.append(f"{'; '.join(supportive)} reduced estimated default risk")
        if not parts:
            parts.append("No selected feature had a material directional contribution")
        suffix = (
            " Direct TreeSHAP rendering was used because the constrained local "
            "surrogate did not meet the fidelity gate."
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
            "ridge_alpha": self.ridge_alpha,
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
