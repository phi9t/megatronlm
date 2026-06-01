export interface ComponentNode {
  id: string
  label: string
  group: 'entry' | 'training' | 'core' | 'parallel' | 'state'
  ref: string
  symbol: string | null
  desc: string
  docs?: string[]
}

export interface ComponentEdge {
  from: string
  to: string
  label?: string
}

export interface ComponentManifest {
  slug: string
  label: string
  generated_at: string
  warnings: string[]
  nodes: ComponentNode[]
  edges: ComponentEdge[]
}

export interface GraphIndexEntry {
  slug: string
  label: string
  manifest: string
}
