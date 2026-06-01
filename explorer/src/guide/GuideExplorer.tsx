import { isValidElement, useEffect, useId, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import mermaid from 'mermaid'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import { errorMessage, fetchExplorerText } from '@/lib/fetch'
import type { ExplorerModeProps } from '@/explorer-kit/mode'

mermaid.initialize({ startOnLoad: false })

function MermaidBlock({ chart }: { chart: string }) {
  const reactId = useId()
  const containerRef = useRef<HTMLDivElement>(null)
  const [svg, setSvg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const renderId = `guide-mermaid-${reactId.replace(/[^a-zA-Z0-9_-]/g, '')}`

  useEffect(() => {
    let cancelled = false

    mermaid
      .render(renderId, chart)
      .then((result) => {
        if (cancelled) {
          return
        }
        setSvg(result.svg)
        setError(null)
        requestAnimationFrame(() => {
          if (!cancelled && containerRef.current) {
            result.bindFunctions?.(containerRef.current)
          }
        })
      })
      .catch((err) => {
        if (!cancelled) {
          setSvg(null)
          setError(errorMessage(err))
        }
      })

    return () => {
      cancelled = true
    }
  }, [chart, renderId])

  if (error) {
    return (
      <pre>
        <code className="guide-mermaid-src">{`Mermaid render failed: ${error}\n\n${chart}`}</code>
      </pre>
    )
  }

  if (!svg) {
    return (
      <pre>
        <code className="guide-mermaid-src">{chart}</code>
      </pre>
    )
  }

  return <div ref={containerRef} className="guide-mermaid" dangerouslySetInnerHTML={{ __html: svg }} />
}

const markdownComponents: Components = {
  pre({ children }) {
    if (isValidElement<{ className?: string; children?: ReactNode }>(children)) {
      const language = /language-(\w+)/.exec(children.props.className ?? '')?.[1]

      if (language === 'mermaid') {
        const code = String(children.props.children).replace(/\n$/, '')
        return <MermaidBlock chart={code} />
      }
    }

    return <pre>{children}</pre>
  },
  code({ className, children, node, ...props }) {
    void node

    return (
      <code className={className} {...props}>
        {children}
      </code>
    )
  },
}

export default function GuideExplorer(_props: ExplorerModeProps) {
  const [markdown, setMarkdown] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchExplorerText('guide.md')
      .then(setMarkdown)
      .catch((err) => setError(errorMessage(err)))
  }, [])

  if (!markdown) {
    return (
      <AsyncBoundary
        loading={error === null}
        error={error}
        loadingLabel="Loading guide..."
        errorPrefix="Failed to load guide.md. Run ./scripts/workflow.sh gen-data"
      />
    )
  }

  return (
    <article className="panel guide p-6">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {markdown}
      </ReactMarkdown>
    </article>
  )
}
