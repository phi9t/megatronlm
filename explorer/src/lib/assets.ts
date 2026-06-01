export const REPO_HOME = 'https://github.com/NVIDIA/Megatron-LM'
export const DEFAULT_SOURCE_REF = 'main'

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
