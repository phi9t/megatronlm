import { useState } from 'react'
import { ArrowLeft, BookOpen, Boxes, Network, SplitSquareHorizontal } from 'lucide-react'
import { REPO_HOME, logoMarkUrl } from './lib/assets'
import type { ExplorerMode } from './explorer-kit/mode'
import GuideExplorer from './guide/GuideExplorer'
import ComponentExplorer from './components-deepdive/ComponentExplorer'
import ParallelismExplorer from './parallelism/ParallelismExplorer'
import ArchitectureExplorer from './architecture/ArchitectureExplorer'

const MODES: ExplorerMode[] = [
  {
    id: 'guide',
    label: 'Guide',
    icon: BookOpen,
    subtitle: 'Megatron-LM and Megatron Core, code-first',
    View: GuideExplorer,
  },
  {
    id: 'components',
    label: 'Components',
    icon: Network,
    subtitle: 'Training loop and Megatron Core subsystem graphs',
    View: ComponentExplorer,
  },
  {
    id: 'parallelism',
    label: 'Parallelism',
    icon: SplitSquareHorizontal,
    subtitle: 'TP, PP, DP, CP, EP, and FSDP rank geometry',
    View: ParallelismExplorer,
  },
  {
    id: 'architecture',
    label: 'Model Architecture',
    icon: Boxes,
    subtitle: 'Qwen3 and DeepSeek-V3 circuits grounded in Megatron Core',
    View: ArchitectureExplorer,
  },
]

export default function App() {
  const [activeId, setActiveId] = useState<string>(MODES[0].id)
  const active = MODES.find((mode) => mode.id === activeId) ?? MODES[0]
  const ActiveView = active.View

  return (
    <div className="relative min-h-screen">
      <div className="observatory-bg" aria-hidden="true" />
      <div className="explorer-container">
        <header className="explorer-header">
          <div>
            <a href="#main-content" className="skip-link">
              Skip to main content
            </a>
            <a href={REPO_HOME} className="back-home-link" target="_blank" rel="noopener noreferrer">
              <ArrowLeft size={14} aria-hidden="true" />
              <span>NVIDIA/Megatron-LM</span>
            </a>
            <div className="header-title-row">
              <img src={logoMarkUrl()} alt="" className="header-logo" width={32} height={32} />
              <h1>Megatron Explorer</h1>
            </div>
            <p>{active.subtitle}</p>
          </div>
          <nav className="family-switch" aria-label="Explorer section">
            {MODES.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                className={`family-switch-btn ${activeId === id ? 'active' : ''}`}
                aria-pressed={activeId === id}
                onClick={() => setActiveId(id)}
              >
                <Icon size={14} aria-hidden="true" />
                {label}
              </button>
            ))}
          </nav>
        </header>
        <main id="main-content">
          <ActiveView navigate={setActiveId} />
        </main>
      </div>
    </div>
  )
}
