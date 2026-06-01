import { useEffect, useMemo, useState } from 'react'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import type { ExplorerModeProps } from '@/explorer-kit/mode'
import { SubjectSwitcher } from '@/explorer-kit/SubjectSwitcher'
import { errorMessage, fetchExplorerJson } from '@/lib/fetch'
import { ArchitectureGraph } from './ArchitectureGraph'
import { ComponentDrawer } from './ComponentDrawer'
import type { ComponentManifest, ComponentNode, GraphIndexEntry } from './types'

export default function ComponentExplorer({ navigate }: ExplorerModeProps) {
  const [subjects, setSubjects] = useState<GraphIndexEntry[]>([])
  const [activeSlug, setActiveSlug] = useState<string | null>(null)
  const [manifest, setManifest] = useState<ComponentManifest | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [loadingIndex, setLoadingIndex] = useState(true)
  const [loadingManifest, setLoadingManifest] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    setLoadingIndex(true)
    fetchExplorerJson<GraphIndexEntry[]>('graphs/index.json')
      .then((entries) => {
        if (cancelled) {
          return
        }
        setSubjects(entries)
        setActiveSlug((current) => current ?? entries[0]?.slug ?? null)
        setError(null)
      })
      .catch((err) => {
        if (!cancelled) {
          setError(errorMessage(err))
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingIndex(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!activeSlug || subjects.length === 0) {
      return
    }

    const subject = subjects.find((entry) => entry.slug === activeSlug)
    if (!subject) {
      setError(`Unknown component graph: ${activeSlug}`)
      return
    }

    let cancelled = false
    setLoadingManifest(true)
    setManifest(null)

    fetchExplorerJson<ComponentManifest>(subject.manifest)
      .then((nextManifest) => {
        if (cancelled) {
          return
        }
        setManifest(nextManifest)
        setSelectedId((current) => {
          if (current && nextManifest.nodes.some((node) => node.id === current)) {
            return current
          }
          return nextManifest.nodes[0]?.id ?? null
        })
        setError(null)
      })
      .catch((err) => {
        if (!cancelled) {
          setError(errorMessage(err))
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingManifest(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [activeSlug, subjects])

  const selectedNode = useMemo<ComponentNode | null>(() => {
    if (!manifest || !selectedId) {
      return null
    }
    return manifest.nodes.find((node) => node.id === selectedId) ?? null
  }, [manifest, selectedId])

  const options = subjects.map((subject) => ({ value: subject.slug, label: subject.label }))
  const loading = loadingIndex || loadingManifest

  return (
    <div className="space-y-4">
      {subjects.length > 0 && activeSlug && (
        <div className="panel flex flex-wrap items-center justify-between gap-3 p-4">
          <SubjectSwitcher
            label="Graph"
            value={activeSlug}
            options={options}
            onChange={(slug) => {
              setActiveSlug(slug)
              setSelectedId(null)
            }}
            ariaLabel="Component graph"
          />
          {manifest && (
            <div className="text-xs text-ink-muted">
              {manifest.nodes.length} components / {manifest.edges.length} edges
            </div>
          )}
        </div>
      )}

      <AsyncBoundary
        loading={loading}
        error={error}
        loadingLabel="Loading component graph..."
        errorPrefix="Failed to load component graph"
      >
        {manifest && (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-start">
            <section className="panel p-4">
              <div className="mb-4">
                <h2 className="text-lg font-semibold text-ink">{manifest.label}</h2>
                {manifest.warnings.length > 0 && (
                  <p className="mt-1 text-xs text-warning">{manifest.warnings.join(' ')}</p>
                )}
              </div>
              <ArchitectureGraph
                nodes={manifest.nodes}
                edges={manifest.edges}
                selectedId={selectedId}
                onSelect={(node) => setSelectedId(node.id)}
              />
            </section>
            <div className="lg:sticky lg:top-4">
              <ComponentDrawer node={selectedNode} onNavigate={navigate} />
            </div>
          </div>
        )}
      </AsyncBoundary>
    </div>
  )
}
