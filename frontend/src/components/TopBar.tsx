import type { HealthResponse } from '../types'
import { KeyboardIcon, MoonIcon, PrintIcon, SunIcon } from './Icons'

interface TopBarProps {
  health: HealthResponse | null
  serverError: string | null
  theme: 'light' | 'dark'
  onToggleTheme: () => void
  onShortcuts: () => void
  onPrint: () => void
  canPrint: boolean
  onNewVisit: () => void
}

export function TopBar({
  health,
  serverError,
  theme,
  onToggleTheme,
  onShortcuts,
  onPrint,
  canPrint,
  onNewVisit,
}: TopBarProps) {
  const status = serverError
    ? { tone: 'fail', text: 'Server offline' }
    : !health
      ? { tone: 'idle', text: 'Connecting' }
      : health.demo
        ? { tone: 'warn', text: 'Demo mode, simulated results' }
        : { tone: 'ok', text: `${shortName(health.model?.name)} on ${health.device}` }

  return (
    <header className="topbar">
      <div className="brand">
        <svg viewBox="0 0 28 28" className="brand-mark" aria-hidden="true">
          <circle cx="14" cy="14" r="12.5" />
          <circle cx="14" cy="14" r="5" />
          <path d="M14 9V1.5M14 26.5V19M9 14H1.5M26.5 14H19" />
        </svg>
        <span className="brand-name">RetinaScan</span>
        <span className="brand-sub">Diabetic retinopathy screening</span>
      </div>

      <p className="model-status" data-tone={status.tone} role="status">
        <span className="dot" aria-hidden="true" />
        {status.text}
      </p>

      <nav className="topbar-actions" aria-label="Application">
        <button type="button" className="btn btn-quiet" onClick={onNewVisit}>
          New visit
        </button>
        <button type="button" className="btn btn-quiet" onClick={onPrint} disabled={!canPrint}>
          <PrintIcon />
          Print report
        </button>
        <button type="button" className="icon-btn" onClick={onShortcuts} title="Keyboard shortcuts (?)">
          <KeyboardIcon />
          <span className="visually-hidden">Keyboard shortcuts</span>
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={onToggleTheme}
          title={theme === 'dark' ? 'Switch to light theme' : 'Switch to reading-room theme'}
        >
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
          <span className="visually-hidden">
            {theme === 'dark' ? 'Switch to light theme' : 'Switch to dark reading-room theme'}
          </span>
        </button>
      </nav>
    </header>
  )
}

function shortName(name?: string): string {
  if (!name) return 'Model'
  return name.split('.')[0].replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}
