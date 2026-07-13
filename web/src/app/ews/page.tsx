'use client'
import { useState } from 'react'
import useSWR from 'swr'
import { api } from '@/lib/api'
import { colors, tierColors, bandColors } from '@/lib/theme'
import KpiCard from '@/components/KpiCard'
import SectionHeader from '@/components/SectionHeader'
import TierBadge from '@/components/TierBadge'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from 'recharts'

const TIERS = ['Green','Yellow','Amber','Red']
const BANDS = ['Excellent','Good','Fair','Watch','Critical']

function hexToRgb(hex: string) {
  const h = hex.replace('#','')
  return [0,2,4].map(i => parseInt(h.slice(i,i+2),16)).join(',')
}

export default function EwsPage() {
  const [selTiers, setSelTiers] = useState(['Red','Amber'])
  const [selBands, setSelBands] = useState(['Critical','Watch','Fair'])
  const [minRules, setMinRules] = useState(1)

  const { data: summary }   = useSWR('ews-summary', api.ewsSummary)
  const { data: ruleFreq }  = useSWR('rule-freq',   api.ruleFrequency)
  const { data: registry }  = useSWR(
    ['registry', selTiers, selBands, minRules],
    () => api.ewsRegistry({ tiers: selTiers.join(','), bands: selBands.join(','), min_rules: String(minRules) })
  )

  const toggle = (arr: string[], setArr: (a: string[]) => void, val: string) =>
    setArr(arr.includes(val) ? arr.filter(x => x !== val) : [...arr, val])

  const tc = summary?.tier_counts ?? {}

  return (
    <div style={{ padding: '2rem 2.5rem' }} className="fade-in">
      <div style={{ marginBottom: '1.8rem' }}>
        <div className="mono" style={{ fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: colors.pulse, marginBottom: 4 }}>Registry</div>
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.02em' }}>Early Warning System — Risk Registry</h2>
      </div>

      {/* KPI Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
        <KpiCard label="Green — Clear"    value={tc.Green?.toLocaleString()  ?? '—'} note={`${summary?.tier_pcts?.Green ?? 0}% of portfolio`}  accent={colors.green}  icon="🟢" />
        <KpiCard label="Yellow — Watch"   value={tc.Yellow?.toLocaleString() ?? '—'} note={`${summary?.tier_pcts?.Yellow ?? 0}% · 1 breach`}    accent={colors.yellow} icon="🟡" />
        <KpiCard label="Amber — Elevated" value={tc.Amber?.toLocaleString()  ?? '—'} note={`${summary?.tier_pcts?.Amber ?? 0}% · 2 breaches`}   accent={colors.amber}  icon="🟠" />
        <KpiCard label="Red — Critical"   value={tc.Red?.toLocaleString()    ?? '—'} note={`${summary?.tier_pcts?.Red ?? 0}% · 3+ breaches`}     accent={colors.red}    icon="🔴" />
      </div>

      <div style={{ height: 1, background: colors.border2, margin: '0 0 1.5rem 0' }}/>

      {/* Filters */}
      <SectionHeader title="Screening Controls" kicker="Registry filters" />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
        <div>
          <div className="mono" style={{ fontSize: '0.62rem', color: colors.muted, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8 }}>EWS Risk Tier</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {TIERS.map(t => {
              const active = selTiers.includes(t)
              const c = tierColors[t]
              const rgb = hexToRgb(c)
              return <button key={t} onClick={() => toggle(selTiers, setSelTiers, t)} className="mono" style={{ padding: '4px 12px', borderRadius: 6, fontSize: '0.72rem', cursor: 'pointer', border: `1px solid rgba(${rgb},${active ? 0.5 : 0.2})`, background: active ? `rgba(${rgb},0.15)` : 'transparent', color: active ? c : colors.muted, fontWeight: active ? 600 : 400 }}>{t}</button>
            })}
          </div>
        </div>
        <div>
          <div className="mono" style={{ fontSize: '0.62rem', color: colors.muted, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8 }}>Score Band</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {BANDS.map(b => {
              const active = selBands.includes(b)
              const c = bandColors[b]
              const rgb = hexToRgb(c)
              return <button key={b} onClick={() => toggle(selBands, setSelBands, b)} className="mono" style={{ padding: '4px 10px', borderRadius: 6, fontSize: '0.7rem', cursor: 'pointer', border: `1px solid rgba(${rgb},${active ? 0.5 : 0.2})`, background: active ? `rgba(${rgb},0.12)` : 'transparent', color: active ? c : colors.muted, fontWeight: active ? 600 : 400 }}>{b}</button>
            })}
          </div>
        </div>
        <div>
          <div className="mono" style={{ fontSize: '0.62rem', color: colors.muted, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8 }}>Min. Breached Rules: <span style={{ color: colors.pulse }}>{minRules}</span></div>
          <input type="range" min={0} max={8} value={minRules} onChange={e => setMinRules(Number(e.target.value))} style={{ width: '100%', accentColor: colors.pulse }} />
        </div>
      </div>

      <div style={{ height: 1, background: colors.border2, margin: '0 0 1.5rem 0' }}/>

      {/* Charts + Table */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 3fr', gap: '1.5rem' }}>
        {/* Rule frequency */}
        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title="Rule Breach Frequency" kicker="Portfolio-wide" />
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={[...(ruleFreq ?? [])].sort((a,b) => a.count - b.count)} layout="vertical" margin={{ left: 10 }}>
              <XAxis type="number" tick={{ fill: colors.muted, fontSize: 9, fontFamily: 'JetBrains Mono' }} />
              <YAxis type="category" dataKey="rule" width={145} tick={{ fill: colors.text2, fontSize: 9, fontFamily: 'JetBrains Mono' }} />
              <Tooltip contentStyle={{ background: colors.surface2, border: `1px solid ${colors.border2}`, borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
              <Bar dataKey="count" fill={colors.pulse} radius={[0,4,4,0]} opacity={0.85} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Registry table */}
        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title={`Screened Registry — ${(registry ?? []).length} matched`} kicker="Top 100 · sorted by PD ↓" />
          <div style={{ overflowY: 'auto', maxHeight: 380 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead style={{ position: 'sticky', top: 0, background: colors.surface2, zIndex: 1 }}>
                <tr>
                  {['ID','PD','Score','Band','Tier','Rules','Breached'].map(h => (
                    <th key={h} className="mono" style={{ fontSize: '0.6rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: colors.muted, padding: '0.5rem 0.4rem', textAlign: h==='ID'||h==='Band'||h==='Tier'||h==='Breached' ? 'left' : 'right', borderBottom: `1px solid ${colors.border2}`, whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(registry ?? []).map((row: any) => (
                  <tr key={row.id} style={{ borderBottom: `1px solid ${colors.border}` }}>
                    <td className="mono" style={{ fontSize: '0.74rem', color: colors.text, padding: '0.42rem 0.4rem' }}>{row.id}</td>
                    <td className="mono" style={{ fontSize: '0.74rem', color: colors.red,  padding: '0.42rem 0.4rem', textAlign: 'right', fontWeight: 600 }}>{row.pd}%</td>
                    <td className="mono" style={{ fontSize: '0.74rem', color: colors.text2, padding: '0.42rem 0.4rem', textAlign: 'right' }}>{row.health}</td>
                    <td style={{ padding: '0.42rem 0.4rem' }}><span className="mono" style={{ fontSize: '0.7rem', color: bandColors[row.score_band], fontWeight: 600 }}>{row.score_band}</span></td>
                    <td style={{ padding: '0.42rem 0.4rem' }}><TierBadge tier={row.ews_tier} /></td>
                    <td className="mono" style={{ fontSize: '0.74rem', color: colors.text2, padding: '0.42rem 0.4rem', textAlign: 'right' }}>{row.rules_fired}</td>
                    <td className="mono" style={{ fontSize: '0.65rem', color: colors.muted, padding: '0.42rem 0.4rem', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.breached_rules.join(', ') || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: '0.75rem', background: `rgba(${hexToRgb(colors.pulse)},0.06)`, border: `1px solid rgba(${hexToRgb(colors.pulse)},0.20)`, borderLeft: `3px solid ${colors.pulse}`, borderRadius: 8, padding: '0.65rem 0.9rem' }}>
            <p className="mono" style={{ fontSize: '0.72rem', color: colors.text2 }}>💡 Copy a Customer ID and paste it in <b>Customer Lookup</b> for a full risk breakdown.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
