export const REPO_HOME = import.meta.env.VITE_REPO_HOME ?? 'https://github.com/phi9t/megatronlm'
export const DEFAULT_SOURCE_REF = import.meta.env.VITE_SOURCE_REF ?? 'main'
export const REPO_LABEL = REPO_HOME.replace(/^https:\/\/github\.com\//, '')

export function dataUrl(path: string): string {
  const clean = path.replace(/^\//, '')
  return `${import.meta.env.BASE_URL}${clean}`
}

export function logoMarkUrl(): string {
  return dataUrl('logo-mark.svg')
}

export function sourceUrl(fileLine: string, ref = DEFAULT_SOURCE_REF): string {
  const [file, line] = fileLine.split(':')
  const anchor = line ? `#L${line}` : ''
  return `${REPO_HOME}/blob/${ref}/${file}${anchor}`
}
