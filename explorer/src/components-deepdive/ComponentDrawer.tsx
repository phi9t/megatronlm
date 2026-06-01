import { Boxes, ExternalLink, Network, SplitSquareHorizontal } from 'lucide-react'
import { sourceUrl } from '@/lib/assets'
import type { ComponentNode } from './types'

interface Props {
  node: ComponentNode | null
  onNavigate: (id: string) => void
}

const GROUP_LABEL: Record<ComponentNode['group'], string> = {
  entry: 'Entry',
  training: 'Training',
  core: 'Core',
  parallel: 'Parallel',
  state: 'State',
}

export function ComponentDrawer({ node, onNavigate }: Props) {
  if (!node) {
    return (
      <aside className="panel p-5 text-sm text-ink-soft">
        <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg border border-panelborder bg-panel-hover text-cyan">
          <Network size={18} aria-hidden="true" />
        </div>
        <h2 className="mb-2 text-base font-semibold text-ink">Select a component</h2>
        <p>Choose a component in the graph to inspect its source location, symbol, docs, and related explorer views.</p>
      </aside>
    )
  }

  return (
    <aside className="panel p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <div className="mb-2 inline-flex rounded-md border border-panelborder bg-panel-hover px-2 py-1 text-xs font-semibold uppercase tracking-wide text-cyan">
            {GROUP_LABEL[node.group]}
          </div>
          <h2 className="text-xl font-semibold text-ink">{node.label}</h2>
        </div>
      </div>

      <p className="text-sm leading-6 text-ink-soft">{node.desc}</p>

      <dl className="mt-5 space-y-4 text-sm">
        {node.symbol && (
          <div>
            <dt className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-muted">Symbol</dt>
            <dd className="code-ref inline-block max-w-full break-words">{node.symbol}</dd>
          </div>
        )}

        <div>
          <dt className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-muted">Source</dt>
          <dd>
            <a
              href={sourceUrl(node.ref)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 break-all text-cyan hover:text-ink"
            >
              <span>{node.ref}</span>
              <ExternalLink size={13} aria-hidden="true" />
            </a>
          </dd>
        </div>

        {node.docs && node.docs.length > 0 && (
          <div>
            <dt className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">Docs</dt>
            <dd className="space-y-2">
              {node.docs.map((doc) => (
                <a
                  key={doc}
                  href={sourceUrl(doc)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 break-all text-cyan hover:text-ink"
                >
                  <span>{doc}</span>
                  <ExternalLink size={13} aria-hidden="true" />
                </a>
              ))}
            </dd>
          </div>
        )}
      </dl>

      <div className="mt-6 border-t border-panelborder pt-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-ink-muted">Related views</h3>
        <div className="grid gap-2">
          <button
            type="button"
            onClick={() => onNavigate('parallelism')}
            className="inline-flex items-center justify-center gap-2 rounded-btn border border-panelborder bg-panel-hover px-3 py-2 text-sm font-semibold text-ink-soft transition-colors hover:text-ink"
          >
            <SplitSquareHorizontal size={15} aria-hidden="true" />
            Parallelism
          </button>
          <button
            type="button"
            onClick={() => onNavigate('architecture')}
            className="inline-flex items-center justify-center gap-2 rounded-btn border border-panelborder bg-panel-hover px-3 py-2 text-sm font-semibold text-ink-soft transition-colors hover:text-ink"
          >
            <Boxes size={15} aria-hidden="true" />
            Architecture
          </button>
        </div>
      </div>
    </aside>
  )
}
