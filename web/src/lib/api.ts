const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res.json()
}

export const api = {
  // Portfolio
  portfolioSummary:      () => get<any>('/api/portfolio/summary'),
  healthDistribution:    () => get<any[]>('/api/portfolio/health-distribution'),
  calibration:           () => get<any[]>('/api/portfolio/calibration'),
  bandSummary:           () => get<any[]>('/api/portfolio/band-summary'),

  // Customer
  customer:              (id: number) => get<any>(`/api/customer/${id}`),
  customerShap:          (id: number) => get<any[]>(`/api/customer/${id}/shap`),

  // EWS
  ewsSummary:            () => get<any>('/api/ews/summary'),
  ruleFrequency:         () => get<any[]>('/api/ews/rule-frequency'),
  ewsRegistry:           (params: Record<string,string>) => {
    const q = new URLSearchParams(params).toString()
    return get<any[]>(`/api/ews/registry?${q}`)
  },
  tierBandCross:         () => get<any[]>('/api/ews/tier-band-cross'),

  // Performance
  metrics:               () => get<any>('/api/performance/metrics'),
  shapGlobal:            () => get<any[]>('/api/performance/shap-global'),
}
