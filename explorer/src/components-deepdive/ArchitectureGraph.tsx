import type { KeyboardEvent } from 'react'
import type { ComponentEdge, ComponentNode } from './types'

interface Props {
  nodes: ComponentNode[]
  edges: ComponentEdge[]
  selectedId: string | null
  onSelect: (node: ComponentNode) => void
}

type Group = ComponentNode['group']

const GROUPS: Group[] = ['entry', 'training', 'core', 'parallel', 'state']

const GROUP_LABEL: Record<Group, string> = {
  entry: 'Entry',
  training: 'Training',
  core: 'Core',
  parallel: 'Parallel',
  state: 'State',
}

const GROUP_COLOR: Record<Group, { stroke: string; fill: string; accent: string }> = {
  entry: { stroke: '#38bdf8', fill: 'rgba(56, 189, 248, 0.14)', accent: '#7dd3fc' },
  training: { stroke: '#6366f1', fill: 'rgba(99, 102, 241, 0.16)', accent: '#a5b4fc' },
  core: { stroke: '#10b981', fill: 'rgba(16, 185, 129, 0.14)', accent: '#6ee7b7' },
  parallel: { stroke: '#f59e0b', fill: 'rgba(245, 158, 11, 0.14)', accent: '#fcd34d' },
  state: { stroke: '#8b5cf6', fill: 'rgba(139, 92, 246, 0.14)', accent: '#c4b5fd' },
}

const CARD_WIDTH = 180
const CARD_HEIGHT = 96
const COLUMN_GAP = 72
const ROW_GAP = 36
const PADDING_X = 32
const PADDING_TOP = 70
const PADDING_BOTTOM = 32
const LABEL_LINE_LENGTH = 20
const DESC_LINE_LENGTH = 34

interface PositionedNode {
  node: ComponentNode
  x: number
  y: number
}

function splitWords(text: string, maxLength: number, maxLines: number): string[] {
  const words = text.split(/\s+/).filter(Boolean)
  const lines: string[] = []
  let line = ''

  for (const word of words) {
    const next = line ? `${line} ${word}` : word
    if (next.length > maxLength && line) {
      lines.push(line)
      line = word
    } else {
      line = next
    }

    if (lines.length === maxLines) {
      break
    }
  }

  if (line && lines.length < maxLines) {
    lines.push(line)
  }

  if (lines.length === maxLines && words.join(' ').length > lines.join(' ').length) {
    lines[maxLines - 1] = `${lines[maxLines - 1].replace(/\s+\S*$/, '')}...`
  }

  return lines
}

function layoutNodes(nodes: ComponentNode[]): { positioned: PositionedNode[]; width: number; height: number } {
  const positioned: PositionedNode[] = []
  const counts = new Map<Group, number>()

  for (const group of GROUPS) {
    counts.set(group, 0)
  }

  for (const node of nodes) {
    const groupIndex = GROUPS.indexOf(node.group)
    const rowIndex = counts.get(node.group) ?? 0
    counts.set(node.group, rowIndex + 1)

    positioned.push({
      node,
      x: PADDING_X + groupIndex * (CARD_WIDTH + COLUMN_GAP),
      y: PADDING_TOP + rowIndex * (CARD_HEIGHT + ROW_GAP),
    })
  }

  const maxRows = Math.max(1, ...Array.from(counts.values()))
  return {
    positioned,
    width: PADDING_X * 2 + GROUPS.length * CARD_WIDTH + (GROUPS.length - 1) * COLUMN_GAP,
    height: PADDING_TOP + maxRows * CARD_HEIGHT + (maxRows - 1) * ROW_GAP + PADDING_BOTTOM,
  }
}

function edgePath(from: PositionedNode, to: PositionedNode): string {
  const startX = from.x + CARD_WIDTH
  const startY = from.y + CARD_HEIGHT / 2
  const endX = to.x
  const endY = to.y + CARD_HEIGHT / 2
  const delta = Math.max(48, Math.abs(endX - startX) / 2)
  const controlOffset = endX >= startX ? delta : -delta

  return `M ${startX} ${startY} C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`
}

export function ArchitectureGraph({ nodes, edges, selectedId, onSelect }: Props) {
  const { positioned, width, height } = layoutNodes(nodes)
  const byId = new Map(positioned.map((item) => [item.node.id, item]))

  function selectWithKeyboard(event: KeyboardEvent<SVGGElement>, node: ComponentNode) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onSelect(node)
    }
  }

  return (
    <div className="overflow-x-auto rounded-panel border border-panelborder bg-void/30">
      <svg
        role="img"
        aria-label="Megatron component graph"
        viewBox={`0 0 ${width} ${height}`}
        className="min-w-[1100px]"
        style={{ width: '100%', height: 'auto' }}
      >
        <defs>
          <marker id="component-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
          </marker>
        </defs>

        {GROUPS.map((group, index) => {
          const x = PADDING_X + index * (CARD_WIDTH + COLUMN_GAP)
          return (
            <g key={group}>
              <text x={x} y={32} fill={GROUP_COLOR[group].accent} fontSize={13} fontWeight={700}>
                {GROUP_LABEL[group]}
              </text>
              <line x1={x} x2={x + CARD_WIDTH} y1={44} y2={44} stroke={GROUP_COLOR[group].stroke} strokeOpacity={0.45} />
            </g>
          )
        })}

        <g fill="none" stroke="#94a3b8" strokeOpacity={0.55} strokeWidth={1.6}>
          {edges.map((edge, index) => {
            const from = byId.get(edge.from)
            const to = byId.get(edge.to)
            if (!from || !to) {
              return null
            }
            return (
              <path
                key={`${edge.from}-${edge.to}-${index}`}
                d={edgePath(from, to)}
                markerEnd="url(#component-arrow)"
                aria-label={edge.label ?? `${edge.from} to ${edge.to}`}
              />
            )
          })}
        </g>

        {positioned.map(({ node, x, y }) => {
          const colors = GROUP_COLOR[node.group]
          const selected = node.id === selectedId
          const labelLines = splitWords(node.label, LABEL_LINE_LENGTH, 2)
          const descLines = splitWords(node.desc, DESC_LINE_LENGTH, 2)

          return (
            <g
              key={node.id}
              role="button"
              tabIndex={0}
              aria-label={`${node.label}. ${node.desc}`}
              onClick={() => onSelect(node)}
              onKeyDown={(event) => selectWithKeyboard(event, node)}
              className="cursor-pointer"
            >
              <rect
                x={x}
                y={y}
                width={CARD_WIDTH}
                height={CARD_HEIGHT}
                rx={8}
                fill={colors.fill}
                stroke={selected ? '#f8fafc' : colors.stroke}
                strokeWidth={selected ? 2.6 : 1.4}
                filter={selected ? 'drop-shadow(0 0 10px rgba(248,250,252,0.22))' : undefined}
              />
              <rect x={x} y={y} width={5} height={CARD_HEIGHT} rx={3} fill={colors.stroke} />
              <text x={x + 16} y={y + 25} fill="#f8fafc" fontSize={14} fontWeight={700}>
                {labelLines.map((line, index) => (
                  <tspan key={line} x={x + 16} dy={index === 0 ? 0 : 17}>
                    {line}
                  </tspan>
                ))}
              </text>
              <text x={x + 16} y={y + 61} fill="#cbd5e1" fontSize={11}>
                {descLines.map((line, index) => (
                  <tspan key={line} x={x + 16} dy={index === 0 ? 0 : 14}>
                    {line}
                  </tspan>
                ))}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="border-t border-panelborder px-4 py-3 text-xs text-ink-muted">
        Select a component to inspect source, symbols, and related explorer views.
      </div>
    </div>
  )
}
