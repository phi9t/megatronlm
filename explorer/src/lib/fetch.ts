import { dataUrl } from './assets'

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

export async function fetchExplorerJson<T>(path: string): Promise<T> {
  const url = dataUrl(path.startsWith('data/') ? path : `data/${path}`)
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export async function fetchExplorerText(path: string): Promise<string> {
  const url = dataUrl(path.startsWith('data/') ? path : `data/${path}`)
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status} ${response.statusText}`)
  }
  return response.text()
}
