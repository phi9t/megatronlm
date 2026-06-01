import { useEffect, useMemo, useState } from 'react'
import { ExternalLink } from 'lucide-react'
import { fetchExplorerJson, errorMessage } from '@/lib/fetch'
import { sourceUrl } from '@/lib/assets'
import { cn } from '@/lib/utils'
import { SubjectSwitcher } from '@/explorer-kit/SubjectSwitcher'
import { ViewTabs } from '@/explorer-kit/ViewTabs'
import { DetailDrawer } from '@/explorer-kit/DetailDrawer'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { computeMetrics, fmtBytes, fmtCount, fmtFlops, lensMetric, summarize, type Lens } from './blockTypes'
import ModelCircuit, { layerSelectionId, preludeSelectionId } from './ModelCircuit'
import type { Block, ModelArch, ModelIndexEntry } from './modelArch'

const LENSES: { id: Lens; label: string }[] = [
  { id: 'flow', label: 'Flow' },
  { id: 'shapes', label: 'Shapes' },
  { id: 'compute', label: 'Compute' },
  { id: 'memory', label: 'Memory' },
]

const GENERATE_COPY = 'Run ./scripts/workflow.sh gen-data to generate model architecture manifests.'

export default function ArchitectureExplorer() {
  const [index, setIndex] = useState<ModelIndexEntry[] | null>(null)
  const [indexError, setIndexError] = useState<string | null>(null)

  const [slug, setSlug] = useState<string | null>(null)
  const [manifest, setManifest] = useState<ModelArch | null>(null)
  const [manifestError, setManifestError] = useState<string | null>(null)
  const [manifestLoading, setManifestLoading] = useState(false)

  const [lens, setLens] = useState<Lens>('shapes')
  const [tokens, setTokens] = useState(512)
  const [selectedId, setSelectedId] = useState<string>('')
  const [selectedBlock, setSelectedBlock] = useState<Block | null>(null)

  // Load the model index on mount
  useEffect(() => {
    fetchExplorerJson<ModelIndexEntry[]>('models/index.json')
      .then((entries) => {
        setIndex(entries)
        if (entries.length > 0) setSlug(entries[0].slug)
      })
      .catch((e) => setIndexError(errorMessage(e)))
  }, [])

  // Load the selected model manifest when slug changes
  useEffect(() => {
    if (!slug) return
    let ignore = false
    setManifest(null)
    setManifestError(null)
    setManifestLoading(true)
    setSelectedId('')
    setSelectedBlock(null)
    fetchExplorerJson<ModelArch>(`models/${slug}.json`)
      .then((m) => {
        if (ignore) return
        setManifest(m)
        setManifestLoading(false)
        // Default selection: first block in first branch's first step (or first prelude)
        const firstBranch = m.layers[0]?.branches[0]
        const firstLayerBlock = firstBranch?.steps[0] ?? firstBranch?.preNorm
        const firstPreludeBlock = m.prelude[0]
        if (firstBranch && firstLayerBlock) {
          const firstBlockIndex = firstBranch.steps[0] ? 1 : 0
          setSelectedId(layerSelectionId(0, 0, firstBranch.name, firstBlockIndex, firstLayerBlock))
          setSelectedBlock(firstLayerBlock)
        } else if (firstPreludeBlock) {
          setSelectedId(preludeSelectionId(0, firstPreludeBlock))
          setSelectedBlock(firstPreludeBlock)
        } else {
          setSelectedId('')
          setSelectedBlock(null)
        }
      })
      .catch((e) => {
        if (ignore) return
        setManifestError(errorMessage(e))
        setManifestLoading(false)
      })

    return () => {
      ignore = true
    }
  }, [slug])

  const metrics = useMemo(
    () => (manifest ? computeMetrics(manifest, tokens) : null),
    [manifest, tokens],
  )
  const summary = useMemo(
    () => (manifest ? summarize(manifest, tokens) : null),
    [manifest, tokens],
  )

  // Stage 1: index loading / error
  if (!index) {
    if (indexError) {
      return (
        <div className="panel p-6 text-danger">
          Failed to load models/index.json: {indexError}
          <p className="mt-2 text-sm text-ink-soft">{GENERATE_COPY}</p>
        </div>
      )
    }

    return (
      <div className="panel p-6 text-ink-soft">
        Loading model architecture manifests…
      </div>
    )
  }

  if (index.length === 0) {
    return (
      <div className="panel p-6 text-sm text-ink-soft">
        {GENERATE_COPY}
      </div>
    )
  }

  const lensValue = (id: string): string => {
    return metrics ? lensMetric(metrics[id], lens) : ''
  }

  const modelLabel =
    index.find((e) => e.slug === slug)?.label ?? slug ?? ''

  const handleSelectBlock = (id: string, block: Block) => {
    setSelectedId(id)
    setSelectedBlock(block)
  }

  return (
    <div className="grid items-start gap-5 lg:grid-cols-2">
      <Card className="flex flex-col">
        <CardHeader>
          <CardTitle>
            {manifest ? manifest.model.split('/').pop() ?? manifest.model : modelLabel} — Megatron Core forward pass
          </CardTitle>
          <p className="text-sm text-ink-soft">
            Qwen3 and DeepSeek-V3 decoder circuits grounded in Megatron Core.
            Residual mainline, input ↑ output.{manifest ? ` One decoder layer (×${manifest.config.num_hidden_layers});` : ''} click any block.
          </p>
        </CardHeader>
        <CardContent className="flex flex-1 justify-center">
          {/* Stage 2: per-model manifest loading / error — inline within the card */}
          {manifestLoading && (
            <div className="py-12 text-sm text-ink-soft">Loading {modelLabel}…</div>
          )}
          {manifestError && (
            <div className="py-6 text-sm text-danger">
              Failed to load {slug}.json: {manifestError}
              <p className="mt-2 text-sm text-ink-soft">{GENERATE_COPY}</p>
            </div>
          )}
          {manifest && metrics && (
            <ModelCircuit
              manifest={manifest}
              lens={lens}
              selectedId={selectedId}
              onSelect={handleSelectBlock}
              lensValue={lensValue}
            />
          )}
        </CardContent>
      </Card>

      <div className="flex flex-col gap-5 lg:sticky lg:top-6">
        {/* Model config selectors + summary */}
        <Card>
          <CardContent className="flex flex-col gap-4 p-5">
            {/* Model switcher */}
            {index.length > 1 && (
              <SubjectSwitcher
                label="Model"
                value={slug ?? ''}
                options={index.map((e) => ({ value: e.slug, label: e.label }))}
                onChange={setSlug}
                ariaLabel="Model"
              />
            )}

            <div className="flex flex-wrap items-center justify-between gap-4">
              {/* Lens tabs */}
              <ViewTabs
                ariaLabel="Lens"
                value={lens}
                onChange={setLens}
                options={LENSES.map((l) => ({ value: l.id, label: l.label }))}
              />
              {/* Token slider */}
              <div className="flex items-center gap-3 text-xs text-ink-soft">
                <span className="whitespace-nowrap">
                  tokens <span className="font-mono text-cyan">{tokens}</span>
                </span>
                <div className="w-32">
                  <input
                    type="range"
                    min={8}
                    max={4096}
                    step={8}
                    value={tokens}
                    onChange={(event) => setTokens(event.currentTarget.valueAsNumber)}
                    className="w-full"
                    aria-label="Token count"
                  />
                </div>
              </div>
            </div>

            {manifest && summary ? (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <Stat label="params" value={fmtCount(summary.totalParams)} />
                  <Stat label="active params/token" value={fmtCount(summary.activeParams)} />
                  <Stat label={`KV cache @ ${tokens}`} value={fmtBytes(summary.kvBytesTotal)} />
                  <Stat label={`forward FLOPs @ ${tokens}`} value={fmtFlops(summary.totalFlops)} />
                </div>
                <p className="text-xs text-ink-muted">
                  {manifest.config.num_hidden_layers} layers · d={manifest.config.hidden_size} ·
                  heads={manifest.config.num_attention_heads}/{manifest.config.num_key_value_heads}
                  {' '}(GQA {manifest.config.num_attention_heads / manifest.config.num_key_value_heads}×) ·
                  head_dim={manifest.config.head_dim} · Megatron Core source{' '}
                  <code className="code-ref">{manifest.source}</code>
                </p>
              </>
            ) : (
              <p className="text-xs text-ink-muted">—</p>
            )}
          </CardContent>
        </Card>

        <BlockDrawer
          block={selectedBlock}
          metricLine={selectedBlock ? lensValue(selectedBlock.id) : ''}
          lens={lens}
          sourceRevision={manifest?.source ?? null}
        />
      </div>
    </div>
  )
}

// --- Drawer -----------------------------------------------------------------
function BlockDrawer({
  block,
  metricLine,
  lens,
  sourceRevision,
}: {
  block: Block | null
  metricLine: string
  lens: Lens
  sourceRevision: string | null
}) {
  return (
    <DetailDrawer
      empty={!block}
      emptyText="Select a block to inspect it."
      eyebrow={block?.kind}
      title={block?.label}
      sourceRef={null}
    >
      {block && (
        <>
          <ArchitectureSourceLink refStr={block.ref} sourceRevision={sourceRevision} />
          <div className="text-sm">
            <span className="text-ink-muted">symbol </span>
            <code className="code-ref">{block.symbol}</code>
          </div>
          <p className="text-sm leading-relaxed text-ink-soft">{block.desc}</p>
          <p className="text-xs text-ink-muted">
            Linked block reference opens the corresponding Megatron Core source location.
          </p>
          {block.note && (
            <p className="rounded-lg border border-cyan/30 bg-cyan/5 px-3 py-2 text-xs text-cyan">
              {block.note}
            </p>
          )}
          {lens !== 'flow' && metricLine && (
            <div className="rounded-lg border border-panelborder bg-panel px-3 py-2">
              <div className="text-[11px] uppercase tracking-wide text-ink-muted">{lens}</div>
              <div className="font-mono text-sm text-ink">{metricLine}</div>
            </div>
          )}
        </>
      )}
    </DetailDrawer>
  )
}

function ArchitectureSourceLink({
  refStr,
  sourceRevision,
}: {
  refStr: string
  sourceRevision: string | null
}) {
  return (
    <a
      href={sourceUrl(refStr, sourceRevision ?? undefined)}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 text-sm text-cyan hover:underline"
    >
      <code className="code-ref">{refStr}</code>
      <ExternalLink size={13} aria-hidden="true" />
    </a>
  )
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-panelborder bg-panel px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-ink-muted">{label}</div>
      <div className={cn('text-lg font-bold text-ink', mono && 'font-mono text-base')}>{value}</div>
    </div>
  )
}
