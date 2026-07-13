'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { LayoutDashboard, Search, AlertTriangle, BarChart2 } from 'lucide-react'
import { colors } from '@/lib/theme'
import useSWR from 'swr'
import { api } from '@/lib/api'

const navItems = [
  { href: '/portfolio',    label: 'Portfolio',         icon: LayoutDashboard },
  { href: '/lookup',       label: 'Customer Lookup',   icon: Search },
  { href: '/ews',          label: 'Early Warning',     icon: AlertTriangle },
  { href: '/performance',  label: 'Model Performance', icon: BarChart2 },
]

export default function Sidebar() {
  const path = usePathname()
  const { data: summary } = useSWR('portfolio-summary', api.portfolioSummary)
  const { data: metrics } = useSWR('metrics', api.metrics)

  const oof = metrics?.stacked_oof_auc ?? 0.7844
  const gini = ((2 * oof - 1) * 100).toFixed(1)
  const redN = summary?.tier_counts?.Red ?? '—'
  const amberN = summary?.tier_counts?.Amber ?? '—'
  const total = summary?.total ?? '—'

  return (
    <aside style={{
      width: 228,
      minWidth: 228,
      background: 'linear-gradient(180deg, #04070F 0%, #060C18 100%)',
      borderRight: `1px solid ${colors.border2}`,
      display: 'flex',
      flexDirection: 'column',
      padding: '1.5rem 1rem',
      position: 'sticky',
      top: 0,
      height: '100vh',
      overflowY: 'auto',
    }}>
      {/* Brand */}
      <div style={{ marginBottom: '1.4rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 9,
            background: `linear-gradient(135deg, ${colors.pulse} 0%, ${colors.pulseDim} 100%)`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: `0 0 18px ${colors.glow}`, flexShrink: 0,
          }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M3 12 H7 L9 6 L12 18 L15 9 L17 12 H21"
                    stroke="#04070F" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div>
            <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '1.1rem', fontWeight: 800, color: colors.text, letterSpacing: '-0.02em', lineHeight: 1.1 }}>
              Credit<span style={{ color: colors.pulse }}>Pulse</span> AI
            </div>
            <div className="mono" style={{ fontSize: '0.58rem', letterSpacing: '0.16em', textTransform: 'uppercase', color: colors.muted, marginTop: 1 }}>
              IDBI Innovate 2026
            </div>
          </div>
        </div>
        <div style={{ height: 1, background: `linear-gradient(90deg, ${colors.pulse}, transparent)`, opacity: 0.35 }}/>
      </div>

      {/* Navigation */}
      <nav style={{ flex: 1 }}>
        {navItems.map(({ href, label, icon: Icon }) => {
          const active = path === href || path.startsWith(href + '/')
          return (
            <Link key={href} href={href} style={{ textDecoration: 'none' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '0.55rem 0.75rem', borderRadius: 8, marginBottom: 2,
                background: active ? `rgba(0,229,160,0.08)` : 'transparent',
                border: active ? `1px solid rgba(0,229,160,0.20)` : '1px solid transparent',
                transition: 'all 0.15s ease', cursor: 'pointer',
              }}
                onMouseEnter={e => { if (!active) (e.currentTarget as HTMLElement).style.background = `rgba(255,255,255,0.04)` }}
                onMouseLeave={e => { if (!active) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
              >
                <Icon size={15} color={active ? colors.pulse : colors.muted} />
                <span className="mono" style={{ fontSize: '0.75rem', letterSpacing: '0.04em', color: active ? colors.pulse : colors.text2, fontWeight: active ? 600 : 400 }}>
                  {label}
                </span>
              </div>
            </Link>
          )
        })}
      </nav>

      {/* System Readout */}
      <div style={{ borderTop: `1px solid ${colors.border}`, paddingTop: '1rem', marginTop: '1rem' }}>
        {[
          ['PORTFOLIO',  typeof total === 'number' ? total.toLocaleString() : total, colors.text],
          ['OOF AUC',    oof.toFixed(4), colors.pulse],
          ['GINI',       `${gini}%`, colors.pulse],
          ['STACK',      'LGBM·XGB·CAT', colors.text2],
          ['SHAP',       'PRECOMPUTED ✓', colors.green],
        ].map(([lbl, val, col]) => (
          <div key={lbl as string} style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '0.32rem 0', borderBottom: `1px solid ${colors.border}`,
          }}>
            <span className="mono" style={{ fontSize: '0.62rem', letterSpacing: '0.08em', textTransform: 'uppercase', color: colors.muted }}>{lbl}</span>
            <span className="mono" style={{ fontSize: '0.68rem', color: col as string, fontWeight: 600 }}>{val}</span>
          </div>
        ))}

        {/* Mini risk tiles */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: '0.9rem' }}>
          {[
            [redN,   'Red',   colors.red],
            [amberN, 'Amber', colors.amber],
          ].map(([n, tier, c]) => (
            <div key={tier as string} style={{
              background: colors.surface2, border: `1px solid rgba(${hexToRgb(c as string)},0.25)`,
              borderRadius: 8, padding: '0.55rem', textAlign: 'center',
            }}>
              <div className="mono" style={{ fontSize: '1rem', fontWeight: 700, color: c as string }}>{typeof n === 'number' ? n.toLocaleString() : n}</div>
              <div className="mono" style={{ fontSize: '0.56rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: colors.muted, marginTop: 2 }}>{tier}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="mono" style={{ fontSize: '0.58rem', textAlign: 'center', color: colors.muted, opacity: 0.5, marginTop: '1.2rem', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
        ⬡ Sai Pranay B.
      </div>
    </aside>
  )
}

function hexToRgb(hex: string) {
  const h = hex.replace('#', '')
  const [r, g, b] = [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16))
  return `${r},${g},${b}`
}
