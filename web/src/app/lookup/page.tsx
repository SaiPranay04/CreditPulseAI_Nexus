'use client'
import { useState } from 'react'
import useSWR from 'swr'
import { api } from '@/lib/api'
import { colors, bandColors, tierColors, FEATURE_LABELS } from '@/lib/theme'
import KpiCard from '@/components/KpiCard'
import SectionHeader from '@/components/SectionHeader'
import TierBadge from '@/components/TierBadge'
import { BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts'

const EXAMPLE_HIGH = [100002, 100031, 100047, 100083]
const EXAMPLE_LOW  = [100003, 100004, 100006, 100009]

function hexToRgb(hex: string) {
  const h = hex.replace('#','')
  return [0,2,4].map(i => parseInt(h.slice(i,i+2),16)).join(',')
}

export default function LookupPage() {
  const [inputId, setInputId]   = useState('100002')
  const [searchId, setSearchId] = useState(100002)

  const { data: cust, error } = useSWR(`customer-${searchId}`, () => api.customer(searchId))
  const { data: shap }         = useSWR(`shap-${searchId}`,    () => api.customerShap(searchId))

  const handleSearch = () => {
    const n = parseInt(inputId)
    if (!isNaN(n)) setSearchId(n)
  }

  const color     = cust ? bandColors[cust.score_band] ?? colors.red : colors.muted
  const rgb       = hexToRgb(color)
  const tierColor = cust ? tierColors[cust.ews_tier] ?? colors.muted : colors.muted
  const tierRgb   = hexToRgb(tierColor)

  // Build SHAP chart data
  const shapData = (shap ?? []).map((d: any) => ({
    feature: FEATURE_LABELS[d.feature] ?? d.feature,
    value:   d.shap_value,
    fval:    d.feature_value,
    color:   d.shap_value >= 0 ? colors.red : colors.green,
  })).sort((a: any, b: any) => Math.abs(a.value) - Math.abs(b.value))

  return (
    <div style={{ padding: '2rem 2.5rem' }} className="fade-in">
      {/* Title */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div className="mono" style={{ fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: colors.pulse, marginBottom: 4 }}>Analysis</div>
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.02em' }}>Individual Customer Risk Profile</h2>
      </div>

      {/* Search */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 3fr', gap: '1rem', marginBottom: '1.5rem' }}>
        <div>
          <label className="mono" style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: colors.muted, display: 'block', marginBottom: 6 }}>Customer ID (SK_ID_CURR)</label>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              value={inputId} onChange={e => setInputId(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              className="mono"
              style={{ flex: 1, background: colors.surface3, border: `1px solid ${colors.border2}`, borderRadius: 8, padding: '0.6rem 0.85rem', color: colors.text, fontSize: '0.9rem', outline: 'none' }}
            />
            <button onClick={handleSearch} style={{ background: colors.pulse, color: '#04070F', border: 'none', borderRadius: 8, padding: '0.6rem 1rem', fontFamily: 'JetBrains Mono', fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer', letterSpacing: '0.06em' }}>
              SEARCH
            </button>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end' }}>
          <div>
            <div className="mono" style={{ fontSize: '0.62rem', color: colors.muted, marginBottom: 4 }}>HIGH RISK EXAMPLES</div>
            {EXAMPLE_HIGH.map(id => (
              <button key={id} onClick={() => { setInputId(String(id)); setSearchId(id) }} className="mono" style={{ marginRight: 6, background: `rgba(${hexToRgb(colors.red)},0.10)`, border: `1px solid rgba(${hexToRgb(colors.red)},0.30)`, color: colors.red, borderRadius: 5, padding: '2px 8px', fontSize: '0.7rem', cursor: 'pointer' }}>{id}</button>
            ))}
          </div>
          <div>
            <div className="mono" style={{ fontSize: '0.62rem', color: colors.muted, marginBottom: 4 }}>LOW RISK EXAMPLES</div>
            {EXAMPLE_LOW.map(id => (
              <button key={id} onClick={() => { setInputId(String(id)); setSearchId(id) }} className="mono" style={{ marginRight: 6, background: `rgba(${hexToRgb(colors.green)},0.10)`, border: `1px solid rgba(${hexToRgb(colors.green)},0.30)`, color: colors.green, borderRadius: 5, padding: '2px 8px', fontSize: '0.7rem', cursor: 'pointer' }}>{id}</button>
            ))}
          </div>
        </div>
      </div>

      {error && <div style={{ color: colors.red, fontFamily: 'JetBrains Mono', fontSize: '0.85rem', padding: '1rem', background: `rgba(${hexToRgb(colors.red)},0.08)`, borderRadius: 8, border: `1px solid rgba(${hexToRgb(colors.red)},0.25)` }}>Customer {searchId} not found. Try 100002 (high risk) or 100003 (low risk).</div>}

      {cust && (
        <>
          <div style={{ height: 1, background: colors.border2, margin: '0 0 1.5rem 0' }}/>

          {/* Main grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '5fr 7fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
            {/* Left: Score + EWS */}
            <div>
              <SectionHeader title="Credit Health Score" kicker="Risk state" />

              {/* Score circle */}
              <div style={{ textAlign: 'center', padding: '1.5rem', background: `linear-gradient(145deg, ${colors.surface2}, ${colors.surface})`, border: `1px solid rgba(${rgb},0.30)`, borderRadius: 14, marginBottom: '1rem', position: 'relative', overflow: 'hidden' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${color}, transparent)` }}/>
                <div style={{ fontSize: '5rem', fontFamily: 'JetBrains Mono', fontWeight: 700, color, lineHeight: 1 }}>{cust.health}</div>
                <div className="mono" style={{ fontSize: '0.75rem', color: colors.muted, marginTop: 4, letterSpacing: '0.1em' }}>OUT OF 850</div>
                <div style={{ display: 'inline-block', marginTop: 8, fontFamily: 'JetBrains Mono', fontSize: '0.78rem', fontWeight: 700, color, background: `rgba(${rgb},0.12)`, border: `1px solid rgba(${rgb},0.35)`, borderRadius: 6, padding: '3px 12px', letterSpacing: '0.08em', textTransform: 'uppercase' }}>{cust.score_band}</div>
              </div>

              {/* PD card */}
              <div style={{ textAlign: 'center', background: `linear-gradient(145deg, ${colors.surface2}, ${colors.surface})`, border: `1px solid rgba(${rgb},0.30)`, borderRadius: 14, padding: '1.2rem', marginBottom: '1rem', position: 'relative', overflow: 'hidden' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${color}, transparent)` }}/>
                <div className="mono" style={{ fontSize: '0.63rem', letterSpacing: '0.18em', textTransform: 'uppercase', color: colors.muted, marginBottom: 6 }}>Probability of Default</div>
                <div style={{ fontSize: '3rem', fontFamily: 'JetBrains Mono', fontWeight: 700, color, lineHeight: 1 }}>{cust.pd}%</div>
                <div className="mono" style={{ fontSize: '0.68rem', color: colors.muted, marginTop: 6 }}>Target threshold: 30%</div>
              </div>

              {/* EWS Alert */}
              <SectionHeader title="Early Warning Alerts" kicker="Breached limits" />
              <div style={{ background: `rgba(${tierRgb},0.07)`, border: `1px solid rgba(${tierRgb},0.25)`, borderLeft: `3px solid ${tierColor}`, borderRadius: 10, padding: '0.9rem 1rem', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ fontFamily: 'Inter', fontSize: '0.95rem', fontWeight: 600, color: colors.text }}>Alert Level</div>
                  <TierBadge tier={cust.ews_tier} />
                </div>
                <div className="mono" style={{ fontSize: '0.7rem', color: colors.muted, marginTop: 4 }}>
                  {cust.ews_rules_fired} of {Object.keys(cust.ews_details).length} limits breached
                </div>
              </div>
              {cust.rules_fired.length === 0
                ? <div className="mono" style={{ fontSize: '0.78rem', color: colors.green, padding: '0.5rem 0' }}>✅ Clean record — no limits breached.</div>
                : cust.rules_fired.map((r: string) => {
                    const d = cust.ews_details[r]
                    return (
                      <div key={r} className="mono" style={{ fontSize: '0.76rem', color: colors.amber, padding: '4px 0', borderBottom: `1px solid ${colors.border}` }}>
                        🔥 <b>{r}</b>&nbsp;<span style={{ color: colors.muted }}>{d.col} {d.op} {d.val}</span>
                      </div>
                    )
                  })
              }
            </div>

            {/* Right: SHAP */}
            <div>
              <SectionHeader title="Local SHAP — Top Credit Risk Drivers" kicker="Feature attribution" />
              <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
                <ResponsiveContainer width="100%" height={380}>
                  <BarChart data={shapData} layout="vertical" margin={{ left: 10, right: 20, top: 0, bottom: 0 }}>
                    <XAxis type="number" tick={{ fill: colors.muted, fontSize: 9, fontFamily: 'JetBrains Mono' }} label={{ value: '← Lowers risk · SHAP · Raises risk →', position: 'insideBottom', offset: -2, fill: colors.muted, fontSize: 9 }} />
                    <YAxis type="category" dataKey="feature" width={180} tick={{ fill: colors.text2, fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                    <ReferenceLine x={0} stroke={colors.border2} strokeWidth={1.5} />
                    <Tooltip formatter={(v: any, _: any, props: any) => [`${(+v).toFixed(5)} (val: ${props.payload.fval ?? 'N/A'})`, 'SHAP']} contentStyle={{ background: colors.surface2, border: `1px solid ${colors.border2}`, borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                    <Bar dataKey="value" radius={[0,3,3,0]}>
                      {shapData.map((d: any, i: number) => <Cell key={i} fill={d.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ marginTop: '0.75rem', background: `rgba(${hexToRgb(colors.red)},0.07)`, border: `1px solid rgba(${hexToRgb(colors.red)},0.20)`, borderLeft: `3px solid ${colors.red}`, borderRadius: 8, padding: '0.7rem 0.9rem' }}>
                  <p className="mono" style={{ fontSize: '0.72rem', color: colors.text2, lineHeight: 1.6 }}>
                    🔴 <b>Red bars</b> raise default risk · 🟢 <b>Green bars</b> lower risk. Hover bars for raw feature values.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Profile ledger */}
          <div style={{ height: 1, background: colors.border2, margin: '0 0 1.5rem 0' }}/>
          <SectionHeader title="Detailed Customer Ledger" kicker="Profile data" />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
            {[
              { title: 'Financials', accent: colors.pulse, rows: [
                ['Income', `₹${cust.profile.amt_income?.toLocaleString()}`],
                ['Credit Limit', `₹${cust.profile.amt_credit?.toLocaleString()}`],
                ['Annuity', `₹${cust.profile.amt_annuity?.toLocaleString()}`],
                ['Credit/Income', `${cust.profile.credit_income_ratio?.toFixed(2)}×`],
              ]},
              { title: 'Personal', accent: colors.blue, rows: [
                ['Age', cust.profile.days_birth ? `${Math.abs(Math.floor(cust.profile.days_birth / 365))} yrs` : 'N/A'],
                ['Employment', cust.profile.days_employed ? `${Math.abs(Math.floor(cust.profile.days_employed / 365))} yrs` : 'Unemployed'],
                ['Income/Person', `₹${cust.profile.income_per_person?.toLocaleString()}`],
                ['Annuity/Income', `${cust.profile.annuity_income_ratio?.toFixed(2)}×`],
              ]},
              { title: 'Credit History', accent: colors.gold, rows: [
                ['Bureau Score', cust.profile.ext_mean?.toFixed(4) ?? 'N/A'],
                ['Payment/Credit', cust.profile.payment_credit_ratio?.toFixed(4) ?? 'N/A'],
              ]},
              { title: 'EWS Flags', accent: colors.red, rows: [
                ['Rules Fired', `${cust.ews_rules_fired} / 8`],
                ['Late Payments', cust.profile.inst_late_ratio != null ? `${(cust.profile.inst_late_ratio * 100).toFixed(1)}%` : 'N/A'],
                ['Prior Refusals', cust.profile.prev_refused_ratio != null ? `${(cust.profile.prev_refused_ratio * 100).toFixed(1)}%` : 'N/A'],
                ['Peak Card Util', cust.profile.cc_utilization_max != null ? `${(cust.profile.cc_utilization_max * 100).toFixed(1)}%` : 'No Card'],
              ]},
            ].map(({ title, accent, rows }) => (
              <div key={title} style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 12, padding: '1rem 1.1rem' }}>
                <div className="mono" style={{ fontSize: '0.62rem', letterSpacing: '0.16em', textTransform: 'uppercase', color: accent, marginBottom: '0.7rem' }}>{title}</div>
                {rows.map(([lbl, val]) => (
                  <div key={lbl} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.38rem 0', borderBottom: `1px solid ${colors.border}` }}>
                    <span className="mono" style={{ fontSize: '0.74rem', color: colors.muted }}>{lbl}</span>
                    <span className="mono" style={{ fontSize: '0.74rem', color: colors.text, fontWeight: 600 }}>{val}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
