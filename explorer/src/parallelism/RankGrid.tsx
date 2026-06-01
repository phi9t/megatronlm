interface RankGridProps {
  worldSize: number
  tp: number
  pp: number
  cp: number
  ep: number
}

export function inferredDp(
  worldSize: number,
  tp: number,
  pp: number,
  cp: number,
  ep: number,
): number | null {
  const factor = tp * pp * cp * ep
  if (factor <= 0 || worldSize % factor !== 0) {
    return null
  }
  return worldSize / factor
}

export function RankGrid({ worldSize, tp, pp, cp, ep }: RankGridProps) {
  const factor = tp * pp * cp * ep
  const dp = inferredDp(worldSize, tp, pp, cp, ep)
  const visibleRanks = Array.from({ length: Math.min(worldSize, 128) }, (_, rank) => rank)

  return (
    <section className="panel p-4">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-ink">Rank grid</h2>
          <p className="mt-1 text-sm text-ink-soft">
            Showing the first {visibleRanks.length} of {worldSize} ranks.
          </p>
        </div>
        <dl className="grid grid-cols-3 gap-2 text-xs sm:grid-cols-6">
          <Metric label="World" value={worldSize} />
          <Metric label="TP" value={tp} />
          <Metric label="PP" value={pp} />
          <Metric label="CP" value={cp} />
          <Metric label="EP" value={ep} />
          <Metric label="DP" value={dp ?? '—'} />
        </dl>
      </div>

      {dp === null && (
        <div className="mb-4 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
          world_size must be divisible by TP * PP * CP * EP = {factor}.
        </div>
      )}

      <div className="rank-grid" aria-label="Parallel rank grid">
        {visibleRanks.map((rank) => (
          <div key={rank} className="rank-cell">
            r{rank}
          </div>
        ))}
      </div>
    </section>
  )
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-md border border-panelborder bg-void/30 px-2 py-1.5 text-center">
      <dt className="font-mono text-[10px] uppercase text-ink-muted">{label}</dt>
      <dd className="mt-0.5 font-mono text-sm font-semibold text-ink">{value}</dd>
    </div>
  )
}
