import { colors } from '@/lib/theme'

interface Props { tier: string }

const COLORS: Record<string, string> = {
  Green:  colors.green,
  Yellow: colors.yellow,
  Amber:  colors.amber,
  Red:    colors.red,
}

function hexToRgb(hex: string) {
  const h = hex.replace('#','')
  return [0,2,4].map(i => parseInt(h.slice(i,i+2),16)).join(',')
}

export default function TierBadge({ tier }: Props) {
  const c = COLORS[tier] ?? colors.muted
  return (
    <span className="mono" style={{
      fontSize: '0.7rem', letterSpacing: '0.1em', textTransform: 'uppercase',
      color: c, background: `rgba(${hexToRgb(c)},0.12)`,
      border: `1px solid rgba(${hexToRgb(c)},0.35)`,
      borderRadius: 6, padding: '3px 10px', whiteSpace: 'nowrap',
    }}>
      {tier}
    </span>
  )
}
