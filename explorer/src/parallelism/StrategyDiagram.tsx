import { ExternalLink } from 'lucide-react'
import { sourceUrl } from '@/lib/assets'
import { cn } from '@/lib/utils'
import type { ParallelStrategy } from './types'

interface StrategyDiagramProps {
  strategies: ParallelStrategy[]
  selectedId: ParallelStrategy['id'] | null
  onSelect: (strategy: ParallelStrategy) => void
}

export function StrategyDiagram({ strategies, selectedId, onSelect }: StrategyDiagramProps) {
  return (
    <section className="panel p-4">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-ink">Strategies</h2>
        <p className="mt-1 text-sm text-ink-soft">Select a card to inspect the parallelism family.</p>
      </div>

      <div className="grid gap-3">
        {strategies.map((strategy) => {
          const active = strategy.id === selectedId
          return (
            <article
              key={strategy.id}
              className={cn(
                'rounded-lg border p-4 transition-colors',
                active
                  ? 'border-panelborder-active bg-panel-hover text-ink'
                  : 'border-panelborder bg-void/30 text-ink-soft',
              )}
            >
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="font-mono text-xs font-semibold uppercase text-cyan">{strategy.id}</div>
                  <h3 className="mt-1 text-base font-semibold text-ink">{strategy.label}</h3>
                </div>
                <button
                  type="button"
                  onClick={() => onSelect(strategy)}
                  aria-pressed={active}
                  className={cn(
                    'shrink-0 rounded-md border px-2.5 py-1 text-xs font-semibold transition-colors',
                    active
                      ? 'border-panelborder-active bg-primary/20 text-ink'
                      : 'border-panelborder bg-panel text-ink-soft hover:text-ink',
                  )}
                >
                  Inspect
                </button>
              </div>

              <div className="mb-3 inline-flex max-w-full rounded-md border border-panelborder bg-panel px-2 py-1 text-[11px] font-semibold text-ink-muted">
                <span className="break-all">{strategy.objective}</span>
              </div>

              <p className="text-sm leading-6 break-words">{strategy.desc}</p>
              <p className="mt-3 text-xs text-ink-muted break-words">{strategy.bestFor}</p>

              <div className="mt-4 flex flex-wrap gap-2">
                {strategy.flags.map((flag) => (
                  <code key={flag} className="code-ref max-w-full break-all">
                    {flag}
                  </code>
                ))}
              </div>

              <div className="mt-4 border-t border-panelborder pt-3">
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">Sources</div>
                <div className="space-y-2">
                  {strategy.sourceRefs.map((ref) => (
                    <a
                      key={ref}
                      href={sourceUrl(ref)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex min-w-0 items-start gap-1.5 break-all text-xs text-cyan hover:text-ink"
                    >
                      <span className="min-w-0 break-all">{ref}</span>
                      <ExternalLink size={12} aria-hidden="true" className="mt-0.5 shrink-0" />
                    </a>
                  ))}
                </div>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
