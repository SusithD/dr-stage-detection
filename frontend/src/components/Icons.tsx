/**
 * SVG icons: status (ok/warn/fail), eye, upload, sun/moon theme, print, keyboard, magnifier, fit.
 */
import type { CheckStatus } from '../types'

type IconProps = { className?: string }

const base = {
  viewBox: '0 0 20 20',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
}

/** Status indicator: check (ok), triangle (warn), or X (fail). */
export const StatusIcon = ({ status, className }: { status: CheckStatus } & IconProps) => (
  <svg {...base} className={`status-icon status-${status} ${className ?? ''}`}>
    {status === 'ok' && <path d="M5 10.5l3.2 3.2L15 6.8" />}
    {status === 'warn' && (
      <>
        <path d="M10 3.5l7 12.5H3z" />
        <path d="M10 8.5v3.2M10 14h.01" />
      </>
    )}
    {status === 'fail' && <path d="M5.5 5.5l9 9M14.5 5.5l-9 9" />}
  </svg>
)

export const EyeIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M1.8 10S4.8 4.5 10 4.5 18.2 10 18.2 10 15.2 15.5 10 15.5 1.8 10 1.8 10z" />
    <circle cx="10" cy="10" r="2.6" />
  </svg>
)

export const UploadIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M10 13V3.5M6.2 7.2 10 3.5l3.8 3.7M3.5 13v3h13v-3" />
  </svg>
)

export const SunIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <circle cx="10" cy="10" r="3.4" />
    <path d="M10 1.8v2M10 16.2v2M1.8 10h2M16.2 10h2M4.2 4.2l1.4 1.4M14.4 14.4l1.4 1.4M4.2 15.8l1.4-1.4M14.4 5.6l1.4-1.4" />
  </svg>
)

export const MoonIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M16.5 12.2A7 7 0 017.8 3.5a7 7 0 108.7 8.7z" />
  </svg>
)

export const PrintIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M5.5 7V2.8h9V7M5.5 14.5h-2V8.2a1.2 1.2 0 011.2-1.2h10.6a1.2 1.2 0 011.2 1.2v6.3h-2" />
    <path d="M5.5 11.5h9v5.7h-9z" />
  </svg>
)

export const KeyboardIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <rect x="1.8" y="5" width="16.4" height="10" rx="1.6" />
    <path d="M5 8.3h.01M8 8.3h.01M11 8.3h.01M14 8.3h.01M6 11.7h8" />
  </svg>
)

export const LoupeIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <circle cx="8.5" cy="8.5" r="5.2" />
    <path d="M12.4 12.4l4.6 4.6M8.5 6.3v4.4M6.3 8.5h4.4" />
  </svg>
)

export const FitIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M3 7V3h4M13 3h4v4M17 13v4h-4M7 17H3v-4" />
  </svg>
)

export const CloseIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M5 5l10 10M15 5L5 15" />
  </svg>
)
