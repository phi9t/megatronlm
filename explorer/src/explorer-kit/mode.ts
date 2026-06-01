import type { ComponentType } from 'react'
import type { LucideIcon } from 'lucide-react'

export interface ExplorerModeProps {
  navigate: (id: string) => void
}

export interface ExplorerMode {
  id: string
  label: string
  icon: LucideIcon
  subtitle: string
  View: ComponentType<ExplorerModeProps>
}
