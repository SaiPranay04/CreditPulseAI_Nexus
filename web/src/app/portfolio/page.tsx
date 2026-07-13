'use client'
import useSWR from 'swr'
import { api } from '@/lib/api'
import { colors, bandColors, tierColors } from '@/lib/theme'
import KpiCard from '@/components/KpiCard'
import SectionHeader from '@/components/SectionHeader'
import {
  BarChart, Bar, Cell, XAxis, YAxis, Tooltip, Legend,
  PieChart, Pie, ResponsiveContainer,
} from 'recharts'

const BAND_ORDER = ['Excellent','Good','Fair','Watch','Critical']
const TIER_ORDER = ['Green','Yellow','Amber','Red']

// Convert flat [{score, count, band}] → [{score, Excellent: n, Good: n, ...}] for stacked bar
function pivotHist(data: any[] | undefined) {
  if (!data) return []
  const map = new Map<number, any>()
  for (const d of data) {
    if (!map.has(d.score)) map.set(d.score, { score: d.score })
    map.get(d.score)[d.band] = d.count
  }
  return Array.from(map.values()).sort((a, b) => a.score - b.score)
}

export default function PortfolioPage() {
  const { data: summary, isLoading: s1 } = useSWR('portfolio-summary', api.portfolioSummary)
  const { data: histData }                = useSWR('health-dist',       api.healthDistribution)
  const { data: calData }                 = useSWR('calibration',       api.calibration)
  const { data: bandData }                = useSWR('band-summary',      api.bandSummary)

  if (s1) return <LoadingState />

  const tierPie = TIER_ORDER.map(t => ({
    name: t, value: summary?.tier_counts?.[t] ?? 0
  }))

  return (
    <div style={{ padding: '2rem 2.5rem' }} className="fade-in">
      {/* Page Title */}
      <div style={{ marginBottom: '1.8rem' }}>
        <div className="mono" style={{ fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: colors.pulse, marginBottom: 4 }}>Overview</div>
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.02em' }}>Portfolio Risk Intelligence</h2>
      </div>

      {/* KPI Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
        <KpiCard label="Active Accounts"    value={summary?.total?.toLocaleString()}     note="Scored customers"               accent={colors.pulse} icon="🏛️" />
        <KpiCard label="Historical Default" value={`${summary?.default_rate?.toFixed(2)}%`} note="Portfolio baseline rate"     accent={colors.blue}  icon="📉" />
        <KpiCard label="Avg Health Score"   value={summary?.mean_health?.toFixed(0)}     note="Range 300 – 850"                accent={colors.gold}  icon="❤️" />
        <KpiCard label="Red-Tier Alerts"    value={summary?.red_tier_count?.toLocaleString()} note={`${summary?.red_tier_pct}% of portfolio`} accent={colors.red} icon="🚨" />
      </div>

      {/* Row 2: Histogram + Donut */}
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '1.5rem', marginBottom: '2rem' }}>
        {/* Histogram */}
        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title="Credit Health Score Distribution" kicker="Segmentation" />
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={pivotHist(histData)} barCategoryGap="2%" barGap={0}>
              <XAxis dataKey="score" tick={{ fill: colors.muted, fontSize: 10, fontFamily: 'JetBrains Mono' }} interval={4} />
              <YAxis tick={{ fill: colors.muted, fontSize: 10, fontFamily: 'JetBrains Mono' }} />
              <Tooltip contentStyle={{ background: colors.surface2, border: `1px solid ${colors.border2}`, borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: colors.muted }} />
              {BAND_ORDER.map(band => (
                <Bar key={band} dataKey={band} name={band} stackId="stack" fill={bandColors[band]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Donut */}
        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title="EWS Risk Tier Distribution" kicker="Alert levels" />
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={tierPie} cx="50%" cy="45%" innerRadius={75} outerRadius={115} paddingAngle={3} dataKey="value" label={({ name, percent }: any) => `${name} ${((percent ?? 0)*100).toFixed(1)}%`} labelLine={false}>
                {tierPie.map((entry) => (
                  <Cell key={entry.name} fill={tierColors[entry.name]} />
                ))}
              </Pie>
              <Tooltip formatter={(v: any) => v.toLocaleString()} contentStyle={{ background: colors.surface2, border: `1px solid ${colors.border2}`, borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: colors.border2, margin: '0 0 2rem 0' }}/>

      {/* Row 3: Calibration + Band Table */}
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '1.5rem' }}>
        {/* Calibration */}
        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title="PD Calibration — Predicted vs. Actual Defaults" kicker="Model quality" />
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={calData ?? []} barCategoryGap="25%" barGap={4}>
              <XAxis dataKey="decile" tick={{ fill: colors.muted, fontSize: 10, fontFamily: 'JetBrains Mono' }} label={{ value: 'PD Decile (1=Low → 10=High)', position: 'insideBottom', offset: -4, fill: colors.muted, fontSize: 10 }} />
              <YAxis tick={{ fill: colors.muted, fontSize: 10, fontFamily: 'JetBrains Mono' }} unit="%" />
              <Tooltip contentStyle={{ background: colors.surface2, border: `1px solid ${colors.border2}`, borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: colors.muted }} />
              <Bar dataKey="predicted_pd" name="Predicted PD" fill={colors.blue} radius={[3,3,0,0]} />
              <Bar dataKey="actual_dr"    name="Actual Default Rate" fill={colors.pulse} radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Band table */}
        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title="Credit Band Breakdown" kicker="Risk summary" />
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Band','Accounts','Avg Score','Avg PD','Actual DR'].map(h => (
                  <th key={h} className="mono" style={{ fontSize: '0.6rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: colors.muted, padding: '0 0 0.6rem 0', textAlign: h === 'Band' ? 'left' : 'right', borderBottom: `1px solid ${colors.border2}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(bandData ?? []).map((row: any) => (
                <tr key={row.band}>
                  <td style={{ padding: '0.5rem 0', borderBottom: `1px solid ${colors.border}` }}>
                    <span className="mono" style={{ fontSize: '0.75rem', color: bandColors[row.band], fontWeight: 600 }}>{row.band}</span>
                  </td>
                  {[row.count?.toLocaleString(), row.avg_health, `${row.avg_pd}%`, `${row.actual_dr}%`].map((v, i) => (
                    <td key={i} className="mono" style={{ textAlign: 'right', fontSize: '0.75rem', color: colors.text2, padding: '0.5rem 0', borderBottom: `1px solid ${colors.border}` }}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: '1rem', background: colors.surface2, border: `1px solid ${colors.border2}`, borderLeft: `3px solid ${colors.pulse}`, borderRadius: 8, padding: '0.75rem' }}>
            <p className="mono" style={{ fontSize: '0.72rem', color: colors.text2, lineHeight: 1.6 }}>
              💡 <b>Calibration note:</b> A well-calibrated model shows actual default rates rising sharply in decile 10.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function LoadingState() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '50vh' }}>
      <div className="mono" style={{ color: colors.pulse, fontSize: '0.85rem', letterSpacing: '0.1em' }}>Loading portfolio data…</div>
    </div>
  )
}
