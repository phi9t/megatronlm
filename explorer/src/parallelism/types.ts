export interface ParallelStrategy {
  id: 'dp' | 'tp' | 'pp' | 'cp' | 'ep' | 'fsdp'
  label: string
  objective: string
  bestFor: string
  flags: string[]
  sourceRefs: string[]
  desc: string
}

export interface ParallelPreset {
  id: string
  label: string
  worldSize: number
  tp: number
  pp: number
  cp: number
  ep: number
  note: string
}

export interface ParallelismManifest {
  generated_at: string
  strategies: ParallelStrategy[]
  presets: ParallelPreset[]
}
