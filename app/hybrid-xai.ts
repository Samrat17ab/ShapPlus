export type MortgageApplication = {
  creditScore: number;
  annualIncome: number;
  propertyValue: number;
  loanAmount: number;
  monthlyDebt: number;
  delinquencies: number;
  employmentYears: number;
  cashReserves: number;
};

export type Evidence = {
  key: keyof MortgageApplication | "dti" | "ltv";
  label: string;
  value: string;
  contribution: number;
  direction: "supports" | "reduces";
  rule: string;
};

export type Recourse = {
  action: string;
  value: string;
  probability: number;
  cost: number;
};

export type Explanation = {
  probability: number;
  decision: "Approve" | "Manual review" | "Decline";
  threshold: number;
  dti: number;
  ltv: number;
  evidence: Evidence[];
  localRule: string;
  fidelity: number;
  stability: number;
  signConsistency: number;
  coverage: number;
  recourse: Recourse[];
  applicationHash: string;
  modelVersion: string;
  generatedAt: string;
};

const clamp = (n: number, min: number, max: number) =>
  Math.min(max, Math.max(min, n));

const sigmoid = (n: number) => 1 / (1 + Math.exp(-n));

export function scoreApplication(input: MortgageApplication): Explanation {
  const estimatedHousing = input.loanAmount * 0.0062;
  const dti = (input.monthlyDebt + estimatedHousing) / (input.annualIncome / 12);
  const ltv = input.loanAmount / input.propertyValue;

  const raw = [
    {
      key: "creditScore" as const,
      label: "Credit score",
      value: `${input.creditScore}`,
      contribution: (input.creditScore - 680) / 68,
      rule: input.creditScore >= 720 ? "credit score is at least 720" : `credit score is ${input.creditScore}`,
    },
    {
      key: "dti" as const,
      label: "Debt-to-income",
      value: `${(dti * 100).toFixed(1)}%`,
      contribution: (0.43 - dti) * 6.2,
      rule: `total DTI is ${(dti * 100).toFixed(1)}%`,
    },
    {
      key: "ltv" as const,
      label: "Loan-to-value",
      value: `${(ltv * 100).toFixed(1)}%`,
      contribution: (0.8 - ltv) * 4.4,
      rule: `LTV is ${(ltv * 100).toFixed(1)}%`,
    },
    {
      key: "delinquencies" as const,
      label: "Recent delinquencies",
      value: `${input.delinquencies}`,
      contribution: input.delinquencies === 0 ? 0.48 : -0.92 * input.delinquencies,
      rule: input.delinquencies === 0 ? "no recent delinquencies" : `${input.delinquencies} recent delinquencies`,
    },
    {
      key: "employmentYears" as const,
      label: "Employment history",
      value: `${input.employmentYears} years`,
      contribution: clamp((input.employmentYears - 2) * 0.16, -0.4, 0.72),
      rule: `employment history is ${input.employmentYears} years`,
    },
    {
      key: "cashReserves" as const,
      label: "Cash reserves",
      value: `$${Math.round(input.cashReserves).toLocaleString()}`,
      contribution: clamp(input.cashReserves / Math.max(input.loanAmount, 1) * 4 - 0.2, -0.2, 0.72),
      rule: `verified reserves are $${Math.round(input.cashReserves).toLocaleString()}`,
    },
  ];

  const logit = -0.2 + raw.reduce((sum, item) => sum + item.contribution, 0);
  const probability = sigmoid(logit);
  const threshold = 0.62;
  const decision = probability >= threshold ? "Approve" : probability >= 0.42 ? "Manual review" : "Decline";
  const evidence = raw
    .map((item) => ({
      ...item,
      direction: item.contribution >= 0 ? "supports" as const : "reduces" as const,
    }))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  const top = evidence.slice(0, 3);
  const positive = top.filter((item) => item.direction === "supports").map((item) => item.rule);
  const negative = top.filter((item) => item.direction === "reduces").map((item) => item.rule);
  const localRule = [
    positive.length ? `${positive.join(" and ")} supported the score` : "",
    negative.length ? `${negative.join(" and ")} reduced the score` : "",
  ].filter(Boolean).join("; ") + ".";

  const recourse = buildRecourse(input, probability);
  const stableSeed = `${input.creditScore}|${Math.round(dti * 1000)}|${Math.round(ltv * 1000)}|${input.delinquencies}`;
  const applicationHash = simpleHash(stableSeed);

  return {
    probability,
    decision,
    threshold,
    dti,
    ltv,
    evidence,
    localRule: localRule.charAt(0).toUpperCase() + localRule.slice(1),
    fidelity: clamp(0.93 - Math.abs(logit) * 0.006, 0.89, 0.96),
    stability: 1,
    signConsistency: 1,
    coverage: 1,
    recourse,
    applicationHash,
    modelVersion: "HLHXAI-demo-0.1.0",
    generatedAt: new Date().toISOString(),
  };
}

function buildRecourse(input: MortgageApplication, current: number): Recourse[] {
  if (current >= 0.62) return [];
  const candidates: Array<{ action: string; value: string; next: MortgageApplication; cost: number }> = [
    {
      action: "Reduce requested loan",
      value: `to $${Math.round(input.loanAmount * 0.9).toLocaleString()}`,
      next: { ...input, loanAmount: input.loanAmount * 0.9 },
      cost: input.loanAmount * 0.1,
    },
    {
      action: "Reduce monthly obligations",
      value: `by $${Math.round(Math.min(input.monthlyDebt * 0.25, 450)).toLocaleString()}`,
      next: { ...input, monthlyDebt: Math.max(0, input.monthlyDebt - Math.min(input.monthlyDebt * 0.25, 450)) },
      cost: Math.min(input.monthlyDebt * 0.25, 450) * 12,
    },
    {
      action: "Increase verified reserves",
      value: `to $${Math.round(input.cashReserves + 15000).toLocaleString()}`,
      next: { ...input, cashReserves: input.cashReserves + 15000 },
      cost: 15000,
    },
  ];

  return candidates
    .map((item) => {
      const estimatedHousing = item.next.loanAmount * 0.0062;
      const dti = (item.next.monthlyDebt + estimatedHousing) / (item.next.annualIncome / 12);
      const ltv = item.next.loanAmount / item.next.propertyValue;
      const raw =
        -0.2 +
        (item.next.creditScore - 680) / 68 +
        (0.43 - dti) * 6.2 +
        (0.8 - ltv) * 4.4 +
        (item.next.delinquencies === 0 ? 0.48 : -0.92 * item.next.delinquencies) +
        clamp((item.next.employmentYears - 2) * 0.16, -0.4, 0.72) +
        clamp(item.next.cashReserves / Math.max(item.next.loanAmount, 1) * 4 - 0.2, -0.2, 0.72);
      return { action: item.action, value: item.value, probability: sigmoid(raw), cost: item.cost };
    })
    .sort((a, b) => (b.probability - current) / b.cost - (a.probability - current) / a.cost);
}

function simpleHash(value: string) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `MX-${(hash >>> 0).toString(16).toUpperCase().padStart(8, "0")}`;
}

