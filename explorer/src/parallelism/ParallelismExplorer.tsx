import { useEffect, useMemo, useState } from 'react'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import type { ExplorerModeProps } from '@/explorer-kit/mode'
import { ViewTabs } from '@/explorer-kit/ViewTabs'
import { errorMessage, fetchExplorerJson } from '@/lib/fetch'
import { RankGrid } from './RankGrid'
import { StrategyDiagram } from './StrategyDiagram'
import type { ParallelismManifest, ParallelPreset, ParallelStrategy } from './types'

type ParallelismControls = Pick<ParallelPreset, 'worldSize' | 'tp' | 'pp' | 'cp' | 'ep'>
const CUSTOM_PRESET_ID = 'custom'

export default function ParallelismExplorer(_props: ExplorerModeProps) {
  const [manifest, setManifest] = useState<ParallelismManifest | null>(null)
  const [activePresetId, setActivePresetId] = useState<string>(CUSTOM_PRESET_ID)
  const [controls, setControls] = useState<ParallelismControls>({
    worldSize: 8,
    tp: 1,
    pp: 1,
    cp: 1,
    ep: 1,
  })
  const [selectedStrategyId, setSelectedStrategyId] = useState<ParallelStrategy['id'] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    setLoading(true)
    fetchExplorerJson<ParallelismManifest>('parallelism/strategies.json')
      .then((nextManifest) => {
        if (cancelled) {
          return
        }

        const firstPreset = nextManifest.presets[0]
        setManifest(nextManifest)
        setActivePresetId(firstPreset?.id ?? CUSTOM_PRESET_ID)
        if (firstPreset) {
          setControls(presetControls(firstPreset))
        }
        setSelectedStrategyId(nextManifest.strategies[0]?.id ?? null)
        setError(null)
      })
      .catch((err) => {
        if (!cancelled) {
          setError(errorMessage(err))
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const presetOptions = useMemo(
    () => [
      ...(manifest?.presets.map((preset) => ({ value: preset.id, label: preset.label })) ?? []),
      { value: CUSTOM_PRESET_ID, label: 'Custom' },
    ],
    [manifest],
  )

  function applyPreset(presetId: string) {
    if (presetId === CUSTOM_PRESET_ID) {
      setActivePresetId(CUSTOM_PRESET_ID)
      return
    }

    const preset = manifest?.presets.find((candidate) => candidate.id === presetId)
    setActivePresetId(presetId)
    if (preset) {
      setControls(presetControls(preset))
    }
  }

  function updateControl(key: keyof ParallelismControls, value: number) {
    setControls((current) => ({ ...current, [key]: value }))
    setActivePresetId(CUSTOM_PRESET_ID)
  }

  return (
    <div className="space-y-4">
      <AsyncBoundary
        loading={loading}
        error={error}
        loadingLabel="Loading parallelism strategies..."
        errorPrefix="Failed to load parallelism strategies"
      >
        {manifest && (
          <>
            <section className="panel space-y-4 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold text-ink">Parallelism layout</h2>
                  <p className="mt-1 text-sm text-ink-soft">
                    Tune the dimensions independently; invalid combinations stay editable.
                  </p>
                </div>
                <div className="text-xs text-ink-muted">Generated: {manifest.generated_at}</div>
              </div>

              {presetOptions.length > 1 && (
                <ViewTabs
                  value={activePresetId}
                  options={presetOptions}
                  onChange={applyPreset}
                  ariaLabel="Parallelism presets"
                />
              )}

              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
                <RangeControl
                  label="World"
                  min={1}
                  max={2048}
                  step={1}
                  value={controls.worldSize}
                  onChange={(value) => updateControl('worldSize', value)}
                />
                <RangeControl
                  label="TP"
                  min={1}
                  max={64}
                  step={1}
                  value={controls.tp}
                  onChange={(value) => updateControl('tp', value)}
                />
                <RangeControl
                  label="PP"
                  min={1}
                  max={64}
                  step={1}
                  value={controls.pp}
                  onChange={(value) => updateControl('pp', value)}
                />
                <RangeControl
                  label="CP"
                  min={1}
                  max={64}
                  step={1}
                  value={controls.cp}
                  onChange={(value) => updateControl('cp', value)}
                />
                <RangeControl
                  label="EP"
                  min={1}
                  max={128}
                  step={1}
                  value={controls.ep}
                  onChange={(value) => updateControl('ep', value)}
                />
              </div>

              {activePresetId !== CUSTOM_PRESET_ID && (
                <p className="text-sm text-ink-soft">
                  {manifest.presets.find((preset) => preset.id === activePresetId)?.note}
                </p>
              )}
            </section>

            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px] xl:items-start">
              <RankGrid
                worldSize={controls.worldSize}
                tp={controls.tp}
                pp={controls.pp}
                cp={controls.cp}
                ep={controls.ep}
              />
              <StrategyDiagram
                strategies={manifest.strategies}
                selectedId={selectedStrategyId}
                onSelect={(strategy) => setSelectedStrategyId(strategy.id)}
              />
            </div>
          </>
        )}
      </AsyncBoundary>
    </div>
  )
}

function presetControls(preset: ParallelPreset): ParallelismControls {
  return {
    worldSize: preset.worldSize,
    tp: preset.tp,
    pp: preset.pp,
    cp: preset.cp,
    ep: preset.ep,
  }
}

function RangeControl({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  onChange: (value: number) => void
}) {
  return (
    <label className="block rounded-lg border border-panelborder bg-void/30 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-ink">{label}</span>
        <span className="font-mono text-sm text-cyan">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.currentTarget.valueAsNumber)}
        className="w-full"
      />
    </label>
  )
}
