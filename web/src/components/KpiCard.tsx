import { colors } from '@/lib/theme'

interface Props {
  label: string
  value: string
  note?: string
  accent?: string
  icon?: string
}

export default function KpiCard({ label, value, note, accent = colors.pulse, icon }: Props) {
  const r = parseInt(accent.slice(1, 3), 16)
  const g = parseInt(accent.slice(3, 5), 16)
  const b = parseInt(accent.slice(5, 7), 16)

  return (
    <div className="kpi-card fade-in" style={{
      background: `linear-gradient(145deg, ${colors.surface2} 0%, ${colors.surface} 100%)`,
      border: `1px solid ${colors.border2}`,
      borderTop: `1px solid rgba(${r},${g},${b},0.4)`,
      borderRadius: 14,
      padding: '1.25rem 1.4rem',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Glow bar */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 2,
        background: `linear-gradient(90deg, transparent, ${accent}, transparent)`,
        opacity: 0.7,
      }}/>
      {icon && <div style={{ fontSize: '1.3rem', marginBottom: '0.45rem' }}>{icon}</div>}
      <div className="mono" style={{ fontSize: '0.63rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: colors.muted, marginBottom: '0.35rem' }}>
        {label}
      </div>
      <div className="mono" style={{ fontSize: '1.85rem', fontWeight: 700, color: colors.text, letterSpacing: '-0.02em', lineHeight: 1.1 }}>
        {value}
      </div>
      {note && (
        <div className="mono" style={{ fontSize: '0.68rem', color: colors.muted, marginTop: '0.35rem', opacity: 0.85 }}>
          {note}
        </div>
      )}
    </div>
  )
}
