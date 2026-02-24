import { useState, useEffect, useCallback, useRef, type FC } from 'react'

/* ================================================================
   Types
   ================================================================ */

interface CameraDevice {
  device: string
  width: number
  height: number
  fps: number
  fourcc: string
}

interface SerialPort {
  device: string
  description: string
  manufacturer: string
  hwid: string
}

interface CameraAssignment {
  role: string      // "front" | "wrist" | ""
  device: string
}

interface CurrentConfig {
  cameras: Record<string, string>
  cameras_raw: string
  robot_port: string
}

type Step = 'detect' | 'assign' | 'confirm'

/* ================================================================
   Camera Snapshot Component
   ================================================================ */

const SNAPSHOT_INTERVAL_MS = 2500  // slower poll — camera lock means sequential access

const CameraSnapshot: FC<{
  device: string
  role: string
  onRoleChange: (device: string, role: string) => void
  allAssignments: CameraAssignment[]
  index: number   // stagger index so cameras don't all poll at the same time
}> = ({ device, role, onRoleChange, allAssignments, index }) => {
  const imgRef = useRef<HTMLImageElement>(null)
  const [hasFrame, setHasFrame] = useState(false)
  const [loading, setLoading] = useState(true)
  const [errorMsg, setErrorMsg] = useState('')

  // Auto-refresh snapshot — staggered by index to prevent simultaneous requests
  useEffect(() => {
    let cancelled = false
    let timerId: ReturnType<typeof setTimeout> | null = null

    const refresh = () => {
      if (cancelled || !imgRef.current) return
      const loader = new Image()
      loader.onload = () => {
        if (!cancelled && imgRef.current) {
          imgRef.current.src = loader.src
          setHasFrame(true)
          setLoading(false)
          setErrorMsg('')
        }
        // Schedule next refresh after load completes
        if (!cancelled) timerId = setTimeout(refresh, SNAPSHOT_INTERVAL_MS)
      }
      loader.onerror = () => {
        if (!cancelled) {
          setHasFrame(false)
          setLoading(false)
          setErrorMsg('No signal')
        }
        // Retry slower on error
        if (!cancelled) timerId = setTimeout(refresh, SNAPSHOT_INTERVAL_MS * 2)
      }
      // Strip leading / from device path for URL
      const devPath = device.startsWith('/') ? device.slice(1) : device
      loader.src = `/api/preflight/snapshot/${devPath}?t=${Date.now()}`
    }

    // Stagger start: each camera waits (index × 600ms) before first poll
    timerId = setTimeout(refresh, index * 600)
    return () => { cancelled = true; if (timerId) clearTimeout(timerId) }
  }, [device, index])

  // Check if roles are already taken by other devices
  const frontTaken = allAssignments.some(a => a.role === 'front' && a.device !== device)
  const wristTaken = allAssignments.some(a => a.role === 'wrist' && a.device !== device)

  const roleColor = role === 'front'
    ? 'ring-blue-500 shadow-blue-500/30'
    : role === 'wrist'
      ? 'ring-purple-500 shadow-purple-500/30'
      : 'ring-gray-600'

  return (
    <div className={`relative rounded-2xl overflow-hidden bg-gray-800/70 transition-all duration-300
                    ring-2 shadow-lg ${roleColor}`}>
      {/* Camera preview */}
      <div className="relative">
        <img
          ref={imgRef}
          alt={device}
          className={`w-full h-auto block transition-opacity duration-300 ${hasFrame ? 'opacity-100' : 'opacity-0'}`}
        />
        {!hasFrame && (
          <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
            {loading ? (
              <div className="flex flex-col items-center gap-2">
                <div className="w-8 h-8 border-2 border-gray-600 border-t-cyan-400 rounded-full animate-spin" />
                <span>Loading camera...</span>
              </div>
            ) : (
              <span>📷 Camera unavailable</span>
            )}
          </div>
        )}

        {/* Device path label */}
        <div className="absolute top-2 left-2 px-2.5 py-1 bg-black/70 backdrop-blur-sm rounded-lg text-xs font-mono text-gray-300">
          {device}
        </div>

        {/* Live indicator */}
        {hasFrame && (
          <div className="absolute top-2 right-2 flex items-center gap-1.5 px-2 py-1 bg-black/60 backdrop-blur-sm rounded-lg">
            <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-[10px] font-semibold text-green-400 uppercase">Live</span>
          </div>
        )}

        {/* Role badge overlay */}
        {role && (
          <div className={`absolute bottom-2 left-2 px-3 py-1.5 rounded-lg text-sm font-bold uppercase tracking-wider backdrop-blur-sm
                          ${role === 'front'
                            ? 'bg-blue-600/90 text-blue-100'
                            : 'bg-purple-600/90 text-purple-100'}`}>
            {role === 'front' ? '📹 Front Camera' : '🤖 Wrist Camera'}
          </div>
        )}
      </div>

      {/* Role selector */}
      <div className="p-4 border-t border-gray-700/50">
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2 font-medium">Assign Role</p>
        <div className="flex gap-2">
          <button
            onClick={() => onRoleChange(device, role === 'front' ? '' : 'front')}
            disabled={frontTaken && role !== 'front'}
            className={`flex-1 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200
                       ${role === 'front'
                         ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                         : frontTaken
                           ? 'bg-gray-700/50 text-gray-600 cursor-not-allowed'
                           : 'bg-gray-700 text-gray-300 hover:bg-blue-600/30 hover:text-blue-300'}`}
          >
            📹 Front
          </button>
          <button
            onClick={() => onRoleChange(device, role === 'wrist' ? '' : 'wrist')}
            disabled={wristTaken && role !== 'wrist'}
            className={`flex-1 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200
                       ${role === 'wrist'
                         ? 'bg-purple-600 text-white shadow-lg shadow-purple-600/30'
                         : wristTaken
                           ? 'bg-gray-700/50 text-gray-600 cursor-not-allowed'
                           : 'bg-gray-700 text-gray-300 hover:bg-purple-600/30 hover:text-purple-300'}`}
          >
            🤖 Wrist
          </button>
          {role && (
            <button
              onClick={() => onRoleChange(device, '')}
              className="px-3 py-2.5 rounded-xl text-sm font-semibold bg-gray-700 text-gray-400 hover:bg-red-600/30 hover:text-red-300 transition-all duration-200"
              title="Clear assignment"
            >
              ✕
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/* ================================================================
   Step Indicator
   ================================================================ */

const StepIndicator: FC<{ current: Step }> = ({ current }) => {
  const steps: { key: Step; label: string; icon: string }[] = [
    { key: 'detect', label: 'Detect', icon: '🔍' },
    { key: 'assign', label: 'Assign', icon: '🎯' },
    { key: 'confirm', label: 'Confirm', icon: '✅' },
  ]
  const currentIdx = steps.findIndex(s => s.key === current)

  return (
    <div className="flex items-center justify-center gap-2 mb-8">
      {steps.map((s, i) => (
        <div key={s.key} className="flex items-center gap-2">
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-300
                          ${i <= currentIdx
                            ? 'bg-cyan-600/30 text-cyan-300 ring-1 ring-cyan-500/50'
                            : 'bg-gray-800/50 text-gray-600'}`}>
            <span>{s.icon}</span>
            <span className="text-sm font-semibold">{s.label}</span>
          </div>
          {i < steps.length - 1 && (
            <div className={`w-8 h-0.5 ${i < currentIdx ? 'bg-cyan-500' : 'bg-gray-700'}`} />
          )}
        </div>
      ))}
    </div>
  )
}

/* ================================================================
   Main Preflight Check Component
   ================================================================ */

interface PreflightCheckProps {
  onComplete: () => void
}

export default function PreflightCheck({ onComplete }: PreflightCheckProps) {
  const [step, setStep] = useState<Step>('detect')
  const [cameras, setCameras] = useState<CameraDevice[]>([])
  const [ports, setPorts] = useState<SerialPort[]>([])
  const [assignments, setAssignments] = useState<CameraAssignment[]>([])
  const [selectedPort, setSelectedPort] = useState('')
  const [currentConfig, setCurrentConfig] = useState<CurrentConfig | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [launching, setLaunching] = useState(false)
  const [launchStatus, setLaunchStatus] = useState('')
  const [error, setError] = useState('')

  /* ── Load current config on mount ── */
  useEffect(() => {
    fetch('/api/preflight/current-config')
      .then(r => r.json())
      .then(d => {
        if (d.config) {
          setCurrentConfig(d.config)
          if (d.config.robot_port) setSelectedPort(d.config.robot_port)
        }
      })
      .catch(() => {})
  }, [])

  /* ── Detect cameras and ports ── */
  const detectAll = useCallback(async () => {
    setDetecting(true)
    setError('')
    try {
      const [camRes, portRes] = await Promise.all([
        fetch('/api/preflight/detect-cameras'),
        fetch('/api/preflight/detect-ports'),
      ])
      const camData = await camRes.json()
      const portData = await portRes.json()

      const detectedCams: CameraDevice[] = camData.cameras || []
      setCameras(detectedCams)
      setPorts(portData.ports || [])

      // Initialize assignments - try to preserve current config
      const newAssignments: CameraAssignment[] = detectedCams.map(c => {
        let role = ''
        if (currentConfig?.cameras) {
          for (const [r, dev] of Object.entries(currentConfig.cameras)) {
            if (dev === c.device) role = r
          }
        }
        return { role, device: c.device }
      })
      setAssignments(newAssignments)

      if (detectedCams.length > 0) {
        setStep('assign')
      } else {
        setError('No cameras detected. Please check USB connections and try again.')
      }
    } catch (e) {
      setError('Failed to detect devices. Is the preflight server running?')
    } finally {
      setDetecting(false)
    }
  }, [currentConfig])

  /* ── Auto-detect on mount ── */
  useEffect(() => {
    const timer = setTimeout(detectAll, 500)
    return () => clearTimeout(timer)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  /* ── Handle role change ── */
  const handleRoleChange = useCallback((device: string, newRole: string) => {
    setAssignments(prev => prev.map(a => {
      // Clear same role from other devices
      if (newRole && a.device !== device && a.role === newRole) {
        return { ...a, role: '' }
      }
      // Set role for this device
      if (a.device === device) {
        return { ...a, role: newRole }
      }
      return a
    }))
  }, [])

  /* ── Save config ── */
  const saveConfig = useCallback(async () => {
    setSaving(true)
    setError('')
    try {
      const cameraAssignments = assignments.filter(a => a.role)
      const res = await fetch('/api/preflight/save-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cameras: cameraAssignments.map(a => ({ role: a.role, device: a.device })),
          robot_port: selectedPort || null,
        }),
      })
      const data = await res.json()
      if (res.ok && data.status === 'ok') {
        setSaved(true)
      } else {
        setError(data.detail || 'Failed to save configuration')
      }
    } catch {
      setError('Network error while saving configuration')
    } finally {
      setSaving(false)
    }
  }, [assignments, selectedPort])

  /* ── Derived state ── */
  const frontAssigned = assignments.find(a => a.role === 'front')
  const wristAssigned = assignments.find(a => a.role === 'wrist')
  const canProceedToConfirm = frontAssigned && wristAssigned
  const assignedCount = assignments.filter(a => a.role).length

  /* ================================================================
     Render
     ================================================================ */
  return (
    <div className="w-full min-h-screen bg-gradient-to-b from-gray-950 via-gray-900 to-gray-950 text-white flex flex-col select-none">

      {/* ── Header ── */}
      <header className="w-full px-6 md:px-10 py-5 flex items-center justify-between border-b border-gray-800/60">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-gray-100">
            🔧 Preflight Check
          </h1>
          <span className="text-sm text-gray-500 font-medium hidden md:inline">— Camera & Port Setup</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="px-3 py-1 rounded-full bg-cyan-600/20 text-cyan-400 text-xs font-semibold uppercase">
            Setup Mode
          </span>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="flex-1 flex flex-col items-center w-full px-6 md:px-10 py-6 md:py-10">

        <StepIndicator current={step} />

        {/* ── Error Banner ── */}
        {error && (
          <div className="w-full max-w-4xl mb-6 px-5 py-4 rounded-xl bg-red-600/20 ring-1 ring-red-500/50 text-red-300 text-sm md:text-base text-center">
            ⚠️ {error}
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════
            STEP: Detect
            ═══════════════════════════════════════════════════════════ */}
        {step === 'detect' && (
          <div className="w-full max-w-2xl flex flex-col items-center gap-6">
            <div className="text-center space-y-3">
              <div className="text-6xl mb-4">🔍</div>
              <h2 className="text-2xl md:text-3xl font-bold">Detecting Cameras</h2>
              <p className="text-gray-400 text-base md:text-lg max-w-md mx-auto">
                Scanning for connected USB cameras and serial ports...
              </p>
            </div>

            {detecting ? (
              <div className="flex flex-col items-center gap-4 py-8">
                <div className="w-16 h-16 border-4 border-gray-700 border-t-cyan-400 rounded-full animate-spin" />
                <p className="text-gray-400 animate-pulse">Scanning devices...</p>
              </div>
            ) : (
              <button
                onClick={detectAll}
                className="px-8 py-4 rounded-2xl text-lg font-bold bg-gradient-to-r from-cyan-600 to-cyan-500
                           hover:from-cyan-500 hover:to-cyan-400 shadow-xl shadow-cyan-500/25 transition-all duration-200"
              >
                🔄 Retry Detection
              </button>
            )}

            {/* Current config info */}
            {currentConfig && currentConfig.cameras_raw && (
              <div className="w-full bg-gray-800/60 backdrop-blur rounded-2xl p-6 space-y-3">
                <h3 className="text-sm uppercase tracking-wider text-gray-500 font-medium">Current Configuration</h3>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Cameras</span>
                  <code className="text-sm text-cyan-400 bg-gray-900/50 px-3 py-1 rounded-lg">{currentConfig.cameras_raw}</code>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Robot Port</span>
                  <code className="text-sm text-cyan-400 bg-gray-900/50 px-3 py-1 rounded-lg">{currentConfig.robot_port}</code>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════
            STEP: Assign
            ═══════════════════════════════════════════════════════════ */}
        {step === 'assign' && (
          <div className="w-full max-w-4xl flex flex-col items-center gap-6">
            <div className="text-center space-y-2 mb-2">
              <h2 className="text-2xl md:text-3xl font-bold">Assign Camera Roles</h2>
              <p className="text-gray-400 text-base md:text-lg">
                View each camera feed and assign it as <span className="text-blue-400 font-semibold">Front</span> or <span className="text-purple-400 font-semibold">Wrist</span>
              </p>
            </div>

            {/* Camera grid */}
            <div className={`w-full grid gap-4 md:gap-6 ${cameras.length === 1 ? 'grid-cols-1 max-w-lg mx-auto' : cameras.length <= 2 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'}`}>
              {cameras.map((cam, idx) => {
                const assignment = assignments.find(a => a.device === cam.device)
                return (
                  <CameraSnapshot
                    key={cam.device}
                    device={cam.device}
                    role={assignment?.role || ''}
                    onRoleChange={handleRoleChange}
                    allAssignments={assignments}
                    index={idx}
                  />
                )
              })}
            </div>

            {/* Camera info summary */}
            <div className="w-full bg-gray-800/40 rounded-xl p-4 flex flex-wrap gap-4 justify-center text-sm text-gray-400">
              <span>📷 {cameras.length} camera{cameras.length !== 1 ? 's' : ''} detected</span>
              <span>•</span>
              <span>🎯 {assignedCount}/2 assigned</span>
              {frontAssigned && <span>• 📹 Front: <code className="text-blue-400">{frontAssigned.device}</code></span>}
              {wristAssigned && <span>• 🤖 Wrist: <code className="text-purple-400">{wristAssigned.device}</code></span>}
            </div>

            {/* Robot Port Section */}
            <div className="w-full bg-gray-800/60 backdrop-blur rounded-2xl p-6 space-y-4">
              <h3 className="text-sm uppercase tracking-wider text-gray-500 font-medium">🔌 Robot Port</h3>
              {ports.length === 0 ? (
                <p className="text-gray-500 text-sm">No serial ports detected</p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {ports.map(p => (
                    <button
                      key={p.device}
                      onClick={() => setSelectedPort(p.device === selectedPort ? '' : p.device)}
                      className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 flex items-center gap-2
                                 ${selectedPort === p.device
                                   ? 'bg-emerald-600 text-white ring-2 ring-emerald-400/50 shadow-lg shadow-emerald-600/30'
                                   : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
                    >
                      <span className="font-mono">{p.device}</span>
                      {p.description && <span className="text-xs text-gray-400">({p.description})</span>}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Action buttons */}
            <div className="w-full flex gap-4">
              <button
                onClick={detectAll}
                className="flex-1 py-4 rounded-2xl text-base font-bold bg-gray-700 hover:bg-gray-600 transition-colors duration-200"
              >
                🔄 Re-scan
              </button>
              <button
                onClick={() => setStep('confirm')}
                disabled={!canProceedToConfirm}
                className={`flex-1 py-4 rounded-2xl text-base font-bold transition-all duration-200
                           ${canProceedToConfirm
                             ? 'bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 shadow-xl shadow-emerald-500/25'
                             : 'bg-gray-800 text-gray-600 cursor-not-allowed'}`}
              >
                {canProceedToConfirm ? '✅ Review & Save' : `Assign both cameras (${assignedCount}/2)`}
              </button>
            </div>
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════
            STEP: Confirm
            ═══════════════════════════════════════════════════════════ */}
        {step === 'confirm' && !saved && (
          <div className="w-full max-w-2xl flex flex-col items-center gap-6">
            <div className="text-center space-y-2">
              <div className="text-6xl mb-2">✅</div>
              <h2 className="text-2xl md:text-3xl font-bold">Confirm Configuration</h2>
              <p className="text-gray-400 text-base md:text-lg">Review your assignments before saving to config.py</p>
            </div>

            {/* Summary card */}
            <div className="w-full bg-gray-800/60 backdrop-blur rounded-2xl p-8 space-y-6">
              <h3 className="text-lg font-bold text-gray-200 text-center uppercase tracking-wider">Configuration Summary</h3>

              {/* Camera assignments */}
              <div className="space-y-3">
                {frontAssigned && (
                  <div className="flex items-center justify-between p-4 rounded-xl bg-blue-600/10 ring-1 ring-blue-500/30">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">📹</span>
                      <div>
                        <p className="font-bold text-blue-300">Front Camera</p>
                        <p className="text-sm text-gray-400">Primary workspace view</p>
                      </div>
                    </div>
                    <code className="text-blue-400 bg-gray-900/50 px-3 py-1.5 rounded-lg text-sm font-mono">{frontAssigned.device}</code>
                  </div>
                )}
                {wristAssigned && (
                  <div className="flex items-center justify-between p-4 rounded-xl bg-purple-600/10 ring-1 ring-purple-500/30">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">🤖</span>
                      <div>
                        <p className="font-bold text-purple-300">Wrist Camera</p>
                        <p className="text-sm text-gray-400">End-effector view</p>
                      </div>
                    </div>
                    <code className="text-purple-400 bg-gray-900/50 px-3 py-1.5 rounded-lg text-sm font-mono">{wristAssigned.device}</code>
                  </div>
                )}
                {selectedPort && (
                  <div className="flex items-center justify-between p-4 rounded-xl bg-emerald-600/10 ring-1 ring-emerald-500/30">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">🔌</span>
                      <div>
                        <p className="font-bold text-emerald-300">Robot Port</p>
                        <p className="text-sm text-gray-400">Motor bus connection</p>
                      </div>
                    </div>
                    <code className="text-emerald-400 bg-gray-900/50 px-3 py-1.5 rounded-lg text-sm font-mono">{selectedPort}</code>
                  </div>
                )}
              </div>

              {/* Config string preview */}
              <div className="bg-gray-900/70 rounded-xl p-4">
                <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">config.py cameras value</p>
                <code className="text-cyan-400 text-sm">
                  "{frontAssigned ? `front:${frontAssigned.device}` : ''}{frontAssigned && wristAssigned ? ',' : ''}{wristAssigned ? `wrist:${wristAssigned.device}` : ''}"
                </code>
              </div>
            </div>

            {/* Action buttons */}
            <div className="w-full flex gap-4">
              <button
                onClick={() => setStep('assign')}
                className="flex-1 py-4 rounded-2xl text-base font-bold bg-gray-700 hover:bg-gray-600 transition-colors duration-200"
              >
                ← Back
              </button>
              <button
                onClick={saveConfig}
                disabled={saving}
                className="flex-1 py-4 rounded-2xl text-base font-bold bg-gradient-to-r from-emerald-600 to-emerald-500
                           hover:from-emerald-500 hover:to-emerald-400 shadow-xl shadow-emerald-500/25
                           transition-all duration-200 disabled:opacity-50"
              >
                {saving ? '⏳ Saving...' : '💾 Save Configuration'}
              </button>
            </div>
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════
            STEP: Saved Successfully
            ═══════════════════════════════════════════════════════════ */}
        {saved && (
          <div className="w-full max-w-2xl flex flex-col items-center gap-6">
            <div className="text-center space-y-3">
              <div className="text-8xl mb-4 animate-bounce">🎉</div>
              <h2 className="text-3xl md:text-4xl font-bold text-emerald-400">Configuration Saved!</h2>
              <p className="text-gray-400 text-base md:text-lg max-w-md mx-auto">
                Camera assignments and port settings have been written to config.py.
                You can now launch the main control UI.
              </p>
            </div>

            {/* Summary */}
            <div className="w-full bg-emerald-600/10 ring-1 ring-emerald-500/30 rounded-2xl p-6 space-y-2">
              {frontAssigned && (
                <p className="text-sm text-gray-300">📹 Front → <code className="text-blue-400">{frontAssigned.device}</code></p>
              )}
              {wristAssigned && (
                <p className="text-sm text-gray-300">🤖 Wrist → <code className="text-purple-400">{wristAssigned.device}</code></p>
              )}
              {selectedPort && (
                <p className="text-sm text-gray-300">🔌 Port → <code className="text-emerald-400">{selectedPort}</code></p>
              )}
            </div>

            {/* Launching state — polls until main_robot is up, then reloads */}
            {launching ? (
              <div className="w-full bg-gray-800/60 rounded-2xl p-8 flex flex-col items-center gap-5">
                <div className="w-16 h-16 border-4 border-gray-700 border-t-blue-400 rounded-full animate-spin" />
                <p className="text-blue-300 font-semibold text-lg">{launchStatus || 'Starting robot control backend…'}</p>
                <p className="text-gray-500 text-sm text-center">
                  This page will reload automatically once the robot control server is ready (~30s).
                </p>
              </div>
            ) : (
              <div className="w-full flex gap-4">
                <button
                  onClick={() => { setSaved(false); setStep('assign') }}
                  className="flex-1 py-4 rounded-2xl text-base font-bold bg-gray-700 hover:bg-gray-600 transition-colors duration-200"
                >
                  🔄 Reconfigure
                </button>
                <button
                  onClick={async () => {
                    setLaunching(true)
                    setLaunchStatus('Stopping preflight server…')
                    try {
                      // Ask preflight server to switch to main_robot
                      await fetch('/api/preflight/launch-control', { method: 'POST' })
                    } catch {
                      // Expected — preflight server will shut down mid-request
                    }
                    // Poll /api/config (main_robot endpoint) until it responds
                    setLaunchStatus('Waiting for robot control server to start…')
                    let attempts = 0
                    const maxAttempts = 60   // 60 × 1s = 60s timeout
                    const pollTimer = setInterval(async () => {
                      attempts++
                      try {
                        const res = await fetch('/api/config', { signal: AbortSignal.timeout(2000) })
                        if (res.ok) {
                          clearInterval(pollTimer)
                          setLaunchStatus('Robot control server ready! Reloading…')
                          setTimeout(() => window.location.reload(), 800)
                          return
                        }
                      } catch { /* still starting */ }
                      if (attempts >= maxAttempts) {
                        clearInterval(pollTimer)
                        setLaunching(false)
                        setLaunchStatus('')
                        alert('Timed out waiting for control server.\nRun manually:\nbash start.sh stop && bash start.sh')
                      } else {
                        setLaunchStatus(`Waiting for robot control server… (${attempts}s)`)
                      }
                    }, 1000)
                  }}
                  className="flex-1 py-5 rounded-2xl text-lg font-bold bg-gradient-to-r from-blue-600 to-blue-500
                             hover:from-blue-500 hover:to-blue-400 shadow-xl shadow-blue-500/25
                             transition-all duration-200"
                >
                  🚀 Launch Control UI
                </button>
              </div>
            )}
          </div>
        )}

      </main>

      {/* ── Footer ── */}
      <footer className="w-full px-6 py-4 border-t border-gray-800/60 text-center text-xs text-gray-600">
        LeRobot Preflight Check — Ensure cameras and ports are correctly assigned before operation
      </footer>
    </div>
  )
}
