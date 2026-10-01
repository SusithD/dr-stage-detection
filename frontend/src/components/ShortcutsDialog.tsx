/**
 * Modal dialog listing keyboard shortcuts: view modes, zoom, panning, opacity, eye selection.
 */
import { useEffect, useRef } from 'react'
import { CloseIcon } from './Icons'

const SHORTCUTS: [string, string][] = [
  ['1 2 3 4', 'Original, model input, compare and heatmap views'],
  ['+  −', 'Zoom in and out'],
  ['0', 'Fit the photograph to the viewer'],
  ['L', 'Magnifier on or off'],
  ['[  ]', 'Heatmap opacity down and up'],
  ['R  E', 'Select the right or left eye'],
  ['← →', 'Move the comparison divider (viewer focused)'],
  ['A', 'Open or close the screening assistant'],
  ['?', 'Show this list'],
]

export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const d = dialog.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])

  return (
    <dialog ref={dialog} className="dialog" onClose={onClose} aria-labelledby="shortcuts-title">
      <div className="dialog-head">
        <h2 id="shortcuts-title">Keyboard shortcuts</h2>
        <button type="button" className="icon-btn" onClick={onClose}>
          <CloseIcon />
          <span className="visually-hidden">Close</span>
        </button>
      </div>
      <dl className="shortcuts">
        {SHORTCUTS.map(([keys, what]) => (
          <div key={keys} className="shortcut">
            <dt>
              {keys.split(/\s+/).map((k) => (
                <kbd key={k}>{k}</kbd>
              ))}
            </dt>
            <dd>{what}</dd>
          </div>
        ))}
      </dl>
    </dialog>
  )
}
