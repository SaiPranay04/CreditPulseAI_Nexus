'use client'
import useSWR from 'swr'
import { api } from '@/lib/api'
import { colors, FEATURE_LABELS } from '@/lib/theme'
import KpiCard from '@/components/KpiCard'
import SectionHeader from '@/components/SectionHeader'
import { BarChart, Bar, XAxis, YAxis, Tooltip, LineChart, Line, ReferenceLine, Cell, ResponsiveContainer } from 'recharts'

function hexToRgb(hex: string) {
  const h = hex.replace('#','')
  return [0,2,4].map(i=>parseInt(h.slice(i,i+2),16)).join(',')
}

// Gradient from blue → mint for global SHAP bars
function lerpColor(i: number, n: number) {
  const t = i / Math.max(n - 1, 1)
  const r = Math.round(59  + t * (0   - 59))
  const g = Math.round(130 + t * (229 - 130))
  const b = Math.round(246 + t * (160 - 246))
  return `rgb(${r},${g},${b})`
}

export default function PerformancePage() {
  const { data: metrics }   = useSWR('metrics',     api.metrics)
  const { data: shapGlobal} = useSWR('shap-global', api.shapGlobal)

  const oof   = metrics?.stacked_oof_auc ?? 0
  const gini  = ((2 * oof - 1) * 100).toFixed(1)
  const f1    = ((metrics?.f1_score ?? 0) * 100).toFixed(1)
  const prec  = ((metrics?.precision ?? 0) * 100).toFixed(1)
  const rec   = ((metrics?.recall ?? 0) * 100).toFixed(1)
  const thresh = metrics?.optimal_threshold?.toFixed(4) ?? '—'
  const trainM = ((metrics?.total_time_seconds ?? 0) / 60).toFixed(1)

  const cm = metrics?.confusion_matrix ?? {}
  const tn = cm.tn ?? 0; const fp = cm.fp ?? 0
  const fn_ = cm.fn ?? 0; const tp = cm.tp ?? 0

  // AUC comparison
  const aucData = [
    { name: 'LightGBM',       auc: metrics?.lgb_oof_auc ?? 0, color: colors.blue },
    { name: 'XGBoost',        auc: metrics?.xgb_oof_auc ?? 0, color: colors.amber },
    { name: 'CatBoost',       auc: metrics?.cat_oof_auc ?? 0, color: colors.gold },
    { name: 'Stacked Meta',   auc: oof,                        color: colors.pulse },
  ]

  // ROC curve
  const rocData = metrics?.roc_points?.slice(0, 200) ?? []

  // SHAP global
  const shapTop = [...(shapGlobal ?? [])].reverse().map((d: any, i: number, arr: any[]) => ({
    feature: FEATURE_LABELS[d.feature] ?? d.feature,
    importance: d.importance,
    color: lerpColor(i, arr.length),
  }))

  return (
    <div style={{ padding: '2rem 2.5rem' }} className="fade-in">
      <div style={{ marginBottom: '1.8rem' }}>
        <div className="mono" style={{ fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: colors.pulse, marginBottom: 4 }}>Evaluation</div>
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.02em' }}>Model Stacking & Validation Performance</h2>
      </div>

      {/* KPI Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
        <KpiCard label="Stacked OOF AUC"   value={oof.toFixed(4)}   note={`Gini: ${gini}%`}              accent={colors.pulse} icon="🎯" />
        <KpiCard label="Optimal Threshold"  value={thresh}           note="Maximises F1-Score"              accent={colors.blue}  icon="🎚️" />
        <KpiCard label="Peak F1-Score"      value={`${f1}%`}         note={`P: ${prec}% · R: ${rec}%`}     accent={colors.gold}  icon="📐" />
        <KpiCard label="Training Time"      value={`${trainM}m`}     note="GPU Accelerated"                 accent={colors.red}   icon="⚡" />
      </div>

      <div style={{ height: 1, background: colors.border2, margin: '0 0 1.5rem 0' }}/>

      {/* Row 2: AUC bars + ROC */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title="Ensemble Stacking Uplift — OOF AUC" kicker="Model comparison" />
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={aucData} barCategoryGap="40%">
              <XAxis dataKey="name" tick={{ fill: colors.muted, fontSize: 10, fontFamily: 'JetBrains Mono' }} />
              <YAxis domain={[0.68, 0.82]} tick={{ fill: colors.muted, fontSize: 10, fontFamily: 'JetBrains Mono' }} tickFormatter={(v: number) => v.toFixed(3)} />
              <Tooltip formatter={(v: any) => v.toFixed(4)} contentStyle={{ background: colors.surface2, border: `1px solid ${colors.border2}`, borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
              <ReferenceLine y={0.5} stroke={colors.muted} strokeDasharray="4 4" label={{ value: 'Random (0.5)', fill: colors.muted, fontSize: 9, fontFamily: 'JetBrains Mono' }} />
              <Bar dataKey="auc" radius={[4,4,0,0]} label={{ position: 'top', fill: colors.muted, fontSize: 9, fontFamily: 'JetBrains Mono', formatter: (v: any) => v.toFixed(4) }}>
                {aucData.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title="Stacked Ensemble ROC Curve" kicker="Classification power" />
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rocData} margin={{ left: 0, right: 10 }}>
              <XAxis dataKey="fpr" type="number" domain={[0,1]} tick={{ fill: colors.muted, fontSize: 9, fontFamily: 'JetBrains Mono' }} label={{ value: 'FPR', position: 'insideBottom', offset: -2, fill: colors.muted, fontSize: 9 }} />
              <YAxis domain={[0,1]} tick={{ fill: colors.muted, fontSize: 9, fontFamily: 'JetBrains Mono' }} label={{ value: 'TPR', angle: -90, position: 'insideLeft', fill: colors.muted, fontSize: 9 }} />
              <ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} stroke={colors.muted} strokeDasharray="4 4" />
              <Tooltip formatter={(v: any) => v.toFixed(4)} contentStyle={{ background: colors.surface2, border: `1px solid ${colors.border2}`, borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
              <Line dataKey="tpr" stroke={colors.pulse} strokeWidth={2.5} dot={false} name={`AUC = ${oof.toFixed(4)}`} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Row 3: CV Folds + Confusion Matrix */}
      <div style={{ display: 'grid', gridTemplateColumns: '5fr 4fr', gap: '1.5rem', marginBottom: '2rem' }}>
        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title="5-Fold Cross-Validation Breakdown" kicker="Fold scores" />
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>{['Fold','LGBM AUC','XGB AUC','CatBoost AUC','Time'].map((h, hi)=>(
                <th key={h} className="mono" style={{ fontSize: '0.6rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: colors.muted, padding: '0.4rem 0', textAlign: hi === 0 ? 'left' : 'right', borderBottom: `1px solid ${colors.border2}` }}>{h}</th>
              ))}</tr>
            </thead>
            <tbody>
              {(metrics?.folds ?? []).map((fold: any) => (
                <tr key={fold.fold}>
                  {[fold.fold, fold.lgb_auc?.toFixed(5), fold.xgb_auc?.toFixed(5), fold.cat_auc?.toFixed(5), `${fold.duration?.toFixed(1)}s`].map((v, i) => (
                    <td key={i} className="mono" style={{ textAlign: 'right', fontSize: '0.75rem', color: i===0 ? colors.pulse : colors.text2, fontWeight: i===0 ? 700 : 400, padding: '0.45rem 0', borderBottom: `1px solid ${colors.border}` }}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: '1rem', background: `rgba(${hexToRgb(colors.pulse)},0.07)`, border: `1px solid rgba(${hexToRgb(colors.pulse)},0.20)`, borderLeft: `3px solid ${colors.pulse}`, borderRadius: 8, padding: '0.75rem' }}>
            <p className="mono" style={{ fontSize: '0.72rem', color: colors.text2, lineHeight: 1.6 }}>
              🏆 Stacking gain: The logistic meta-learner achieves <b>OOF AUC {oof.toFixed(4)}</b> (Gini {gini}%) by combining orthogonal error signals from three tree architectures.
            </p>
          </div>
        </div>

        {/* Confusion Matrix */}
        <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
          <SectionHeader title="Confusion Matrix" kicker="Decision boundary" />
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'JetBrains Mono', fontSize: '0.76rem', border: `1px solid ${colors.border2}`, borderRadius: 10, overflow: 'hidden' }}>
            <thead>
              <tr>
                <td style={{ padding: '0.6rem', background: colors.surface2, border: `1px solid ${colors.border2}` }}/>
                <td style={{ padding: '0.6rem', background: colors.surface2, color: colors.green, fontWeight: 600, textAlign: 'center', border: `1px solid ${colors.border2}` }}>PRED: HEALTHY</td>
                <td style={{ padding: '0.6rem', background: colors.surface2, color: colors.red, fontWeight: 600, textAlign: 'center', border: `1px solid ${colors.border2}` }}>PRED: DEFAULT</td>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding: '0.65rem', background: colors.surface2, color: colors.green, fontWeight: 600, border: `1px solid ${colors.border2}` }}>ACTUAL: HEALTHY</td>
                <td style={{ padding: '1.1rem', background: `rgba(${hexToRgb(colors.green)},0.08)`, textAlign: 'center', border: `1px solid ${colors.border2}` }}>
                  <div style={{ fontSize: '1.35rem', fontWeight: 700, color: colors.green }}>{tn.toLocaleString()}</div>
                  <div style={{ fontSize: '0.65rem', color: colors.muted, marginTop: 2 }}>True Negative ✓</div>
                </td>
                <td style={{ padding: '1.1rem', background: `rgba(${hexToRgb(colors.red)},0.07)`, textAlign: 'center', border: `1px solid ${colors.border2}` }}>
                  <div style={{ fontSize: '1.35rem', fontWeight: 700, color: colors.red }}>{fp.toLocaleString()}</div>
                  <div style={{ fontSize: '0.65rem', color: colors.muted, marginTop: 2 }}>False Positive (I)</div>
                </td>
              </tr>
              <tr>
                <td style={{ padding: '0.65rem', background: colors.surface2, color: colors.red, fontWeight: 600, border: `1px solid ${colors.border2}` }}>ACTUAL: DEFAULT</td>
                <td style={{ padding: '1.1rem', background: `rgba(${hexToRgb(colors.red)},0.07)`, textAlign: 'center', border: `1px solid ${colors.border2}` }}>
                  <div style={{ fontSize: '1.35rem', fontWeight: 700, color: colors.red }}>{fn_.toLocaleString()}</div>
                  <div style={{ fontSize: '0.65rem', color: colors.muted, marginTop: 2 }}>False Negative (II)</div>
                </td>
                <td style={{ padding: '1.1rem', background: `rgba(${hexToRgb(colors.green)},0.08)`, textAlign: 'center', border: `1px solid ${colors.border2}` }}>
                  <div style={{ fontSize: '1.35rem', fontWeight: 700, color: colors.green }}>{tp.toLocaleString()}</div>
                  <div style={{ fontSize: '0.65rem', color: colors.muted, marginTop: 2 }}>True Positive ✓</div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Global SHAP */}
      <div style={{ height: 1, background: colors.border2, margin: '0 0 1.5rem 0' }}/>
      <SectionHeader title="Global SHAP — Top 20 Portfolio Risk Drivers" kicker="Feature importance" />
      <div style={{ background: colors.surface, border: `1px solid ${colors.border2}`, borderRadius: 14, padding: '1.25rem' }}>
        <ResponsiveContainer width="100%" height={480}>
          <BarChart data={shapTop} layout="vertical" margin={{ left: 10, right: 20 }}>
            <XAxis type="number" tick={{ fill: colors.muted, fontSize: 9, fontFamily: 'JetBrains Mono' }} label={{ value: 'Mean |SHAP| (Global Portfolio Impact)', position: 'insideBottom', offset: -4, fill: colors.muted, fontSize: 9 }} />
            <YAxis type="category" dataKey="feature" width={200} tick={{ fill: colors.text2, fontSize: 10, fontFamily: 'JetBrains Mono' }} />
            <Tooltip formatter={(v: any) => v.toFixed(6)} contentStyle={{ background: colors.surface2, border: `1px solid ${colors.border2}`, borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
            <Bar dataKey="importance" radius={[0,4,4,0]} opacity={0.9}>
              {shapTop.map((d: any, i: number) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
