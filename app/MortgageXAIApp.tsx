"use client";

import { useMemo, useState } from "react";
import { MortgageApplication, scoreApplication } from "./hybrid-xai";

const initial: MortgageApplication = {
  creditScore: 704,
  annualIncome: 96000,
  propertyValue: 425000,
  loanAmount: 340000,
  monthlyDebt: 780,
  delinquencies: 0,
  employmentYears: 6,
  cashReserves: 26000,
};

const fields: Array<{
  key: keyof MortgageApplication;
  label: string;
  prefix?: string;
  suffix?: string;
  min: number;
  max: number;
  step: number;
}> = [
  { key: "creditScore", label: "Credit score", min: 500, max: 850, step: 1 },
  { key: "annualIncome", label: "Annual income", prefix: "$", min: 24000, max: 300000, step: 1000 },
  { key: "propertyValue", label: "Property value", prefix: "$", min: 80000, max: 1500000, step: 5000 },
  { key: "loanAmount", label: "Requested loan", prefix: "$", min: 50000, max: 1200000, step: 5000 },
  { key: "monthlyDebt", label: "Monthly obligations", prefix: "$", min: 0, max: 8000, step: 50 },
  { key: "delinquencies", label: "Recent delinquencies", min: 0, max: 6, step: 1 },
  { key: "employmentYears", label: "Employment history", suffix: " yrs", min: 0, max: 30, step: 1 },
  { key: "cashReserves", label: "Verified cash reserves", prefix: "$", min: 0, max: 250000, step: 1000 },
];

const pct = (n: number) => `${Math.round(n * 100)}%`;
const money = (n: number) => `$${Math.round(n).toLocaleString()}`;

export default function MortgageXAIApp() {
  const [input, setInput] = useState(initial);
  const [tab, setTab] = useState<"decision" | "architecture">("decision");
  const [copied, setCopied] = useState(false);
  const result = useMemo(() => scoreApplication(input), [input]);
  const scoreDegrees = Math.round(result.probability * 360);

  const update = (key: keyof MortgageApplication, value: number) =>
    setInput((current) => ({ ...current, [key]: value }));

  const exportLog = () => {
    const payload = {
      applicationHash: result.applicationHash,
      modelVersion: result.modelVersion,
      generatedAt: result.generatedAt,
      score: result.probability,
      decision: result.decision,
      calibratedThreshold: result.threshold,
      signedAttributionVector: result.evidence.map(({ key, contribution }) => ({ feature: key, contribution })),
      localRule: result.localRule,
      fidelity: result.fidelity,
      stability: result.stability,
      counterfactualRecourse: result.recourse,
      explanationPipeline: "deterministic-attribution > constrained-local-rule > recourse > compliance-log",
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${result.applicationHash}-compliance-log.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const copyRule = async () => {
    await navigator.clipboard.writeText(result.localRule);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <main className="studio">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Mortgage XAI Studio home">
          <span className="brand-mark">M</span>
          <span><b>Mortgage XAI</b><small>STUDIO</small></span>
        </a>
        <nav aria-label="Primary navigation">
          <button className={tab === "decision" ? "active" : ""} onClick={() => setTab("decision")}>Decision workbench</button>
          <button className={tab === "architecture" ? "active" : ""} onClick={() => setTab("architecture")}>Method & controls</button>
        </nav>
        <div className="model-chip"><i /> HLHXAI v0.1 · deterministic</div>
      </header>

      {tab === "decision" ? (
        <>
          <section className="intro" id="top">
            <div>
              <p className="eyebrow">MORTGAGE-SPECIFIC HYBRID EXPLAINABILITY</p>
              <h1>Evidence you can <em>act on.</em><br />Decisions you can audit.</h1>
            </div>
            <p className="intro-copy">A working research demonstrator that separates the internal evidence trail, the frontline rule summary, and applicant recourse—without blending unstable LIME coefficients into the result.</p>
          </section>

          <section className="pipeline" aria-label="Explanation pipeline">
            {[
              ["01", "Predictive core", "Calibrated risk score"],
              ["02", "Evidence backbone", "Signed contributions"],
              ["03", "Local rule", "Constrained & readable"],
              ["04", "Recourse", "Feasible next actions"],
              ["05", "Compliance log", "Versioned audit record"],
            ].map((item, index) => (
              <div className={index === 0 ? "current" : ""} key={item[0]}>
                <span>{item[0]}</span><b>{item[1]}</b><small>{item[2]}</small>
              </div>
            ))}
          </section>

          <section className="workbench">
            <aside className="application-panel">
              <div className="section-title">
                <div><p>APPLICATION INPUT</p><h2>Case HMX-2048</h2></div>
                <button onClick={() => setInput(initial)}>Reset</button>
              </div>
              <p className="panel-note">Protected attributes are intentionally excluded from decision and recourse calculations.</p>
              <div className="input-grid">
                {fields.map((field) => (
                  <label key={field.key}>
                    <span>{field.label}</span>
                    <div className="number-input">
                      {field.prefix && <i>{field.prefix}</i>}
                      <input
                        aria-label={field.label}
                        type="number"
                        min={field.min}
                        max={field.max}
                        step={field.step}
                        value={input[field.key]}
                        onChange={(event) => update(field.key, Number(event.target.value))}
                      />
                      {field.suffix && <i>{field.suffix}</i>}
                    </div>
                    <input
                      className="range"
                      aria-label={`${field.label} slider`}
                      type="range"
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      value={input[field.key]}
                      onChange={(event) => update(field.key, Number(event.target.value))}
                    />
                  </label>
                ))}
              </div>
              <div className="derived">
                <span><small>Calculated DTI</small><b>{pct(result.dti)}</b></span>
                <span><small>Calculated LTV</small><b>{pct(result.ltv)}</b></span>
              </div>
            </aside>

            <section className="decision-panel">
              <div className="decision-top">
                <div className="gauge" style={{ "--score-deg": `${scoreDegrees}deg` } as React.CSSProperties}>
                  <div><strong>{pct(result.probability)}</strong><span>approval likelihood</span></div>
                </div>
                <div className="decision-copy">
                  <p>CALIBRATED DECISION</p>
                  <h2>{result.decision}</h2>
                  <span className={`decision-badge ${result.decision.toLowerCase().replace(" ", "-")}`}>{result.decision === "Approve" ? "Policy threshold met" : result.decision === "Decline" ? "Threshold not met" : "Human judgment required"}</span>
                  <small>Threshold {pct(result.threshold)} · Score generated from mutable underwriting factors only.</small>
                </div>
              </div>

              <div className="evidence-head">
                <div><p>LAYER 01 · EVIDENCE BACKBONE</p><h3>Signed contribution vector</h3></div>
                <span>Source of truth</span>
              </div>
              <div className="evidence-list">
                {result.evidence.map((item) => {
                  const width = Math.min(100, Math.abs(item.contribution) / 1.8 * 100);
                  return (
                    <div className="evidence-row" key={item.key}>
                      <div><b>{item.label}</b><small>{item.value}</small></div>
                      <div className="contribution-track">
                        <span className="axis" />
                        <i className={item.direction} style={{ width: `${width / 2}%` }} />
                      </div>
                      <strong className={item.direction}>{item.contribution > 0 ? "+" : ""}{item.contribution.toFixed(2)}</strong>
                    </div>
                  );
                })}
              </div>
            </section>

            <aside className="explanation-panel">
              <div className="layer-card rule-card">
                <div className="layer-label"><span>02</span><p>CONSTRAINED LOCAL RULE</p><i>Stable</i></div>
                <blockquote>“{result.localRule}”</blockquote>
                <div className="metrics">
                  <span><small>Local fidelity</small><b>{pct(result.fidelity)}</b></span>
                  <span><small>Run stability</small><b>{pct(result.stability)}</b></span>
                  <span><small>Sign agreement</small><b>{pct(result.signConsistency)}</b></span>
                </div>
                <button className="text-action" onClick={copyRule}>{copied ? "Copied" : "Copy reviewer summary"} <span>↗</span></button>
              </div>

              <div className="layer-card recourse-card">
                <div className="layer-label"><span>03</span><p>COUNTERFACTUAL RECOURSE</p><i>Actionable</i></div>
                {result.recourse.length ? (
                  <div className="recourse-list">
                    {result.recourse.map((item, index) => (
                      <div key={item.action}>
                        <span>{index + 1}</span>
                        <p><b>{item.action}</b><small>{item.value}</small></p>
                        <strong>{pct(item.probability)}</strong>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="approved-copy">No adverse recourse is required. The applicant-facing artifact can explain which factors supported this result.</p>
                )}
                <p className="guardrail">Immutable and protected traits are locked. Suggestions are directional and require policy validation.</p>
              </div>

              <div className="layer-card audit-card">
                <div className="layer-label"><span>04</span><p>COMPLIANCE LOG</p><i>Ready</i></div>
                <dl>
                  <div><dt>Application hash</dt><dd>{result.applicationHash}</dd></div>
                  <div><dt>Model version</dt><dd>{result.modelVersion}</dd></div>
                  <div><dt>Coverage</dt><dd>{pct(result.coverage)}</dd></div>
                </dl>
                <button className="export" onClick={exportLog}>Export audit record <span>↓</span></button>
              </div>
            </aside>
          </section>
        </>
      ) : (
        <MethodView />
      )}

      <footer>
        <span>Mortgage XAI Studio · Research demonstrator</span>
        <p>Decision support, not autonomous underwriting. Production use requires a trained and validated mortgage model, genuine TreeSHAP values, calibration, fairness testing, and human governance.</p>
      </footer>
    </main>
  );
}

function MethodView() {
  const layers = [
    ["01", "TreeSHAP backbone", "The production design uses a complete signed attribution vector from the trained tree model as the internal source of truth.", "Deterministic · complete coverage · portfolio monitoring"],
    ["02", "Constrained local surrogate", "Top stable features are rendered as a short mortgage rule. Fixed neighborhoods, sign consistency and a fidelity floor replace stochastic LIME sampling.", "Readable · reproducible · fidelity-gated"],
    ["03", "Rule-format rendering", "Underwriting language replaces raw coefficient lists. High-precision sufficient conditions can be used when an Anchors-style precision gate passes.", "Audience-specific · fallback-aware"],
    ["04", "Counterfactual recourse", "Feasible changes are searched only over mutable features, ranked by score improvement relative to cost, and kept separate from the reason for the decision.", "Actionable · immutable traits locked"],
    ["05", "Compliance log package", "The score, threshold, version, input hash, signed vector, rule, fidelity, stability and recourse are captured in one exportable record.", "Traceable · reviewable · monitorable"],
  ];
  return (
    <section className="method">
      <div className="method-hero">
        <p className="eyebrow">METHOD & CONTROL SURFACE</p>
        <h1>One predictor.<br /><em>Four explanation layers.</em></h1>
        <p>This is a SHAP-grounded and LIME-inspired stack—not a weighted average of two explanation methods.</p>
      </div>
      <div className="objective">
        <span>EXPLANATION OBJECTIVE</span>
        <p><i>L</i><sub>explain</sub> = α · <i>L</i><sub>fidelity</sub> + β · <i>L</i><sub>stability</sub> + γ · <i>L</i><sub>complexity</sub> + δ · <i>L</i><sub>sign consistency</sub></p>
      </div>
      <div className="method-layers">
        {layers.map((layer) => (
          <article key={layer[0]}>
            <span>{layer[0]}</span>
            <div><h2>{layer[1]}</h2><p>{layer[2]}</p></div>
            <small>{layer[3]}</small>
          </article>
        ))}
      </div>
      <div className="validation-grid">
        <article><p>VALIDATION PORTFOLIO</p><h3>Three datasets, three roles</h3><ul><li>Home Credit · continuity benchmark</li><li>HMEQ · mortgage-adjacent ablation</li><li>HMDA · origination and group analysis</li></ul></article>
        <article><p>EXPLANATION METRICS</p><h3>Multi-property evaluation</h3><ul><li>Fidelity, stability and sparsity</li><li>Coverage, runtime and reproducibility</li><li>Fairness visibility and recourse validity</li></ul></article>
        <article><p>BASELINES</p><h3>Beyond SHAP and LIME</h3><ul><li>SHAP · LIME · Anchors</li><li>Counterfactual recourse method</li><li>Explainable Boosting Machine</li></ul></article>
      </div>
    </section>
  );
}

