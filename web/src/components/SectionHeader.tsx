import { colors } from '@/lib/theme'

interface Props { title: string; kicker?: string; marginTop?: string }

export default function SectionHeader({ title, kicker, marginTop = '0' }: Props) {
  return (
    <div style={{ marginTop, marginBottom: '1rem' }}>
      {kicker && (
        <div className="mono" style={{ fontSize: '0.62rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: colors.pulse, marginBottom: 4, opacity: 0.9 }}>
          {kicker}
        </div>
      )}
      <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600, color: colors.text, letterSpacing: '-0.01em' }}>
        {title}
      </h3>
    </div>
  )
}
