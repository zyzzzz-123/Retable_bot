import { useState, useEffect, useCallback, useRef, type FC } from 'react'

/* ================================================================
   Types
   ================================================================ */

type RobotState = 'WARMUP' | 'READY' | 'WORKING' | 'PAUSED' | 'HOMED' | 'DONE' | 'ERROR'

interface PipelineStageInfo {
  name: string
  llm_status: 'done' | 'todo' | 'not_found' | ''   // from LLM planner
  exec_status: 'pending' | 'active' | 'done' | 'skipped'  // execution state
}

interface StatusMessage {
  state: RobotState
  step?: string
  progress?: number
  message?: string
  hand_detect?: boolean
  hand_detected?: boolean
  auto_stopped?: boolean
  pipeline_stage?: string
  pipeline_stage_idx?: number
  pipeline_total?: number
  pipeline_status?: string
  pipeline_stages_info?: PipelineStageInfo[]
  llm_planning?: boolean
  llm_plan_error?: string
}

interface Toast {
  id: number
  message: string
  type: 'success' | 'error' | 'info'
}

/* ================================================================
   State config — warm palette
   ================================================================ */

const STATE_META: Record<RobotState, {
  label: string; color: string; bg: string; border: string
}> = {
  WARMUP:  { label: 'Warming Up',  color: '#5b8fd9', bg: '#e3f0fc', border: '#b3d4f0' },
  READY:   { label: 'Ready',    color: '#e8793a', bg: '#fef3e2', border: '#f5ddb5' },
  WORKING: { label: 'Running',  color: '#3b8f7e', bg: '#e8f5ee', border: '#c3e6d1' },
  PAUSED:  { label: 'Paused',  color: '#d9a03a', bg: '#fef8e8', border: '#f0dfa0' },
  HOMED:   { label: 'Homed',    color: '#8b7ec8', bg: '#f0edf8', border: '#d4cde8' },
  DONE:    { label: 'Done',    color: '#4caf7d', bg: '#e8f5ee', border: '#c3e6d1' },
  ERROR:   { label: 'Error',    color: '#d94f4f', bg: '#fdeaea', border: '#f0b8b8' },
}

/* ================================================================
   SVG Icons — clean, rounded style
   ================================================================ */
const IconPlay: FC<{ size?: number; className?: string }> = ({ size = 20, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M8 5.14v14.72a1 1 0 0 0 1.5.86l11.5-7.36a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z"/>
  </svg>
)
const IconStop: FC<{ size?: number; className?: string }> = ({ size = 20, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
  </svg>
)
const IconHome: FC<{ size?: number; className?: string }> = ({ size = 18, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>
  </svg>
)
const IconX: FC<{ size?: number; className?: string }> = ({ size = 16, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className={className}>
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
)
const IconHand: FC<{ size?: number; color?: string }> = ({ size = 18, color = '#9e978f' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v0M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8" />
    <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
  </svg>
)
const IconRefresh: FC<{ size?: number; className?: string }> = ({ size = 18, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
  </svg>
)
const IconZap: FC<{ size?: number; className?: string }> = ({ size = 14, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </svg>
)

/* ================================================================
   SVG Icons — Debug overlay
   ================================================================ */
const IconLayers: FC<{ size?: number; className?: string }> = ({ size = 16, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" />
  </svg>
)
const IconUpload: FC<{ size?: number; className?: string }> = ({ size = 14, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
  </svg>
)
const IconTrash: FC<{ size?: number; className?: string }> = ({ size = 14, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
)

/* ================================================================
   Camera Feed
   ================================================================ */
const CameraFeed: FC<{
  name: string; active: boolean; handDetected?: boolean; isHandCamera?: boolean;
  overlayUrl?: string | null; overlayOpacity?: number;
}> = ({ name, active, handDetected = false, isHandCamera = false, overlayUrl = null, overlayOpacity = 0.2 }) => {
  const imgRef = useRef<HTMLImageElement>(null)
  const [hasFrame, setHasFrame] = useState(false)

  useEffect(() => {
    if (!active) return
    let cancelled = false
    const refresh = () => {
      if (cancelled || !imgRef.current) return
      const loader = new Image()
      loader.onload = () => { if (!cancelled && imgRef.current) { imgRef.current.src = loader.src; setHasFrame(true) } }
      loader.onerror = () => { if (!cancelled) setHasFrame(false) }
      loader.src = `/api/frame/${name}?t=${Date.now()}`
    }
    refresh()
    const id = setInterval(refresh, 200)
    return () => { cancelled = true; clearInterval(id) }
  }, [name, active])

  return (
    <div className={`relative rounded-xl overflow-hidden warm-card transition-all duration-300
                    ${isHandCamera && handDetected ? 'border-red-400 glow-red' : ''}`}>
      <img ref={imgRef} alt={name}
        className={`w-full h-auto block transition-opacity duration-500 ${hasFrame ? 'opacity-100' : 'opacity-0'}`} />
      {/* Debug overlay image */}
      {overlayUrl && hasFrame && (
        <img src={overlayUrl} alt="debug overlay"
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
          style={{ opacity: overlayOpacity }} />
      )}
      {!hasFrame && (
        <div className="flex items-center justify-center h-32" style={{ color: '#9e978f' }}>
          <div className="w-5 h-5 border-2 border-[#d0cbc4] border-t-[#e8793a] rounded-full animate-smooth-spin" />
        </div>
      )}
      {/* Camera label */}
      <div className="absolute top-2.5 left-2.5 px-2.5 py-1 bg-white/90 backdrop-blur-sm rounded-lg text-xs font-heading tracking-wide"
        style={{ color: '#6b6560' }}>
        {name}
      </div>
      {/* Live indicator */}
      {hasFrame && (
        <div className="absolute top-2.5 right-2.5 flex items-center gap-1.5 px-2 py-1 bg-white/90 backdrop-blur-sm rounded-lg">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-60" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
          </span>
          <span className="text-xs font-heading tracking-wide text-red-500">LIVE</span>
        </div>
      )}
      {/* Overlay indicator badge */}
      {overlayUrl && hasFrame && (
        <div className="absolute bottom-2.5 left-2.5 flex items-center gap-1.5 px-2 py-1 bg-purple-500/90 backdrop-blur-sm rounded-lg">
          <IconLayers size={12} className="text-white" />
          <span className="text-[10px] font-heading tracking-wide text-white">OVERLAY {Math.round(overlayOpacity * 100)}%</span>
        </div>
      )}
      {/* Hand detection overlay */}
      {isHandCamera && handDetected && hasFrame && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="absolute inset-0 bg-red-500/15" />
          <div className="relative z-10 flex items-center gap-2 px-3 py-1.5 bg-white/95 backdrop-blur rounded-xl shadow-lg border border-red-200">
            <span className="text-lg">🖐️</span>
            <span className="text-sm font-heading font-bold text-red-600">Hand Detected</span>
          </div>
        </div>
      )}
      {isHandCamera && !handDetected && hasFrame && (
        <div className="absolute bottom-2.5 right-2.5 px-2.5 py-1 bg-white/90 backdrop-blur-sm rounded-lg text-xs font-heading"
          style={{ color: '#4caf7d' }}>
          ✓ Safe
        </div>
      )}
    </div>
  )
}

/* ================================================================
   Debug Overlay Panel
   ================================================================ */
const DebugOverlayPanel: FC<{
  overlays: Record<string, string | null>
  overlayOpacity: number
  onUpload: (camName: string, url: string) => void
  onRemove: (camName: string) => void
  onOpacityChange: (val: number) => void
  cameras: string[]
}> = ({ overlays, overlayOpacity, onUpload, onRemove, onOpacityChange, cameras }) => {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [targetCam, setTargetCam] = useState<string>('')

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !targetCam) return
    const url = URL.createObjectURL(file)
    onUpload(targetCam, url)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const triggerUpload = (camName: string) => {
    setTargetCam(camName)
    setTimeout(() => fileInputRef.current?.click(), 0)
  }

  return (
    <div className="flex flex-col gap-2.5 px-3 py-3 rounded-xl"
      style={{ background: '#faf8f6', border: '1px solid #e0dbd4' }}>
      <div className="flex items-center gap-2">
        <IconLayers size={14} className="text-[#8b7ec8]" />
        <span className="text-xs font-heading tracking-wide" style={{ color: '#8b7ec8' }}>Debug Overlay</span>
      </div>

      {/* Per-camera upload/remove buttons */}
      <div className="flex flex-col gap-1.5">
        {cameras.map(cam => (
          <div key={cam} className="flex items-center gap-2">
            <span className="text-[11px] font-heading flex-1 truncate" style={{ color: '#6b6560' }}>{cam}</span>
            {overlays[cam] ? (
              <button onClick={() => onRemove(cam)}
                className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-heading transition-colors"
                style={{ background: '#fdeaea', border: '1px solid #f0b8b8', color: '#d94f4f' }}>
                <IconTrash size={10} />
                <span>Remove</span>
              </button>
            ) : (
              <button onClick={() => triggerUpload(cam)}
                className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-heading transition-colors"
                style={{ background: '#f0edf8', border: '1px solid #d4cde8', color: '#8b7ec8' }}>
                <IconUpload size={10} />
                <span>Upload</span>
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Opacity slider */}
      {Object.values(overlays).some(v => v) && (
        <div className="flex items-center gap-2 mt-1">
          <span className="text-[10px] font-heading" style={{ color: '#9e978f' }}>Opacity</span>
          <input type="range" min="0" max="100" value={Math.round(overlayOpacity * 100)}
            onChange={e => onOpacityChange(Number(e.target.value) / 100)}
            className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer"
            style={{ accentColor: '#8b7ec8' }} />
          <span className="text-[10px] font-heading w-8 text-right" style={{ color: '#8b7ec8' }}>
            {Math.round(overlayOpacity * 100)}%
          </span>
        </div>
      )}

      <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
    </div>
  )
}

/* ================================================================
   Sidebar / Mobile Camera Feeds
   ================================================================ */
const SidebarCameraFeeds: FC<{ active: boolean; handDetected: boolean; handDetectEnabled: boolean }> = ({ active, handDetected, handDetectEnabled }) => {
  const [cameras, setCameras] = useState<string[]>([])
  const [overlays, setOverlays] = useState<Record<string, string | null>>({})
  const [overlayOpacity, setOverlayOpacity] = useState(0.2)
  const [showDebug, setShowDebug] = useState(false)

  useEffect(() => { fetch('/api/cameras').then(r => r.json()).then(d => setCameras(d.cameras || [])).catch(() => {}) }, [])

  const handleUpload = useCallback((cam: string, url: string) => {
    setOverlays(prev => ({ ...prev, [cam]: url }))
  }, [])
  const handleRemove = useCallback((cam: string) => {
    setOverlays(prev => {
      const old = prev[cam]
      if (old) URL.revokeObjectURL(old)
      return { ...prev, [cam]: null }
    })
  }, [])

  if (cameras.length === 0) return null
  return (
    <div className="flex flex-col gap-3 w-full">
      <div className="flex items-center gap-2">
        <div className="h-px flex-1" style={{ background: '#e0dbd4' }} />
        <span className="text-xs font-heading tracking-widest" style={{ color: '#9e978f' }}>Cameras</span>
        <div className="h-px flex-1" style={{ background: '#e0dbd4' }} />
        <button onClick={() => setShowDebug(p => !p)}
          className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-heading transition-colors"
          style={{
            background: showDebug ? '#f0edf8' : '#faf8f6',
            border: `1px solid ${showDebug ? '#d4cde8' : '#e0dbd4'}`,
            color: showDebug ? '#8b7ec8' : '#9e978f',
          }}>
          <IconLayers size={10} />
          <span>Debug</span>
        </button>
      </div>
      {showDebug && (
        <DebugOverlayPanel
          overlays={overlays}
          overlayOpacity={overlayOpacity}
          onUpload={handleUpload}
          onRemove={handleRemove}
          onOpacityChange={setOverlayOpacity}
          cameras={cameras}
        />
      )}
      {cameras.map(name => (
        <CameraFeed key={name} name={name} active={active} handDetected={handDetected}
          isHandCamera={handDetectEnabled && name === 'front'}
          overlayUrl={overlays[name] || null}
          overlayOpacity={overlayOpacity} />
      ))}
    </div>
  )
}

const MobileCameraFeeds: FC<{ active: boolean; handDetected: boolean; handDetectEnabled: boolean }> = ({ active, handDetected, handDetectEnabled }) => {
  const [cameras, setCameras] = useState<string[]>([])
  const [overlays, setOverlays] = useState<Record<string, string | null>>({})
  const [overlayOpacity, setOverlayOpacity] = useState(0.2)
  const [showDebug, setShowDebug] = useState(false)

  useEffect(() => { fetch('/api/cameras').then(r => r.json()).then(d => setCameras(d.cameras || [])).catch(() => {}) }, [])

  const handleUpload = useCallback((cam: string, url: string) => {
    setOverlays(prev => ({ ...prev, [cam]: url }))
  }, [])
  const handleRemove = useCallback((cam: string) => {
    setOverlays(prev => {
      const old = prev[cam]
      if (old) URL.revokeObjectURL(old)
      return { ...prev, [cam]: null }
    })
  }, [])

  if (cameras.length === 0) return null
  return (
    <div className="w-full mb-3">
      <div className="flex items-center gap-2 mb-2">
        <button onClick={() => setShowDebug(p => !p)}
          className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-heading transition-colors"
          style={{
            background: showDebug ? '#f0edf8' : '#faf8f6',
            border: `1px solid ${showDebug ? '#d4cde8' : '#e0dbd4'}`,
            color: showDebug ? '#8b7ec8' : '#9e978f',
          }}>
          <IconLayers size={10} />
          <span>Debug Overlay</span>
        </button>
      </div>
      {showDebug && (
        <div className="mb-2">
          <DebugOverlayPanel
            overlays={overlays}
            overlayOpacity={overlayOpacity}
            onUpload={handleUpload}
            onRemove={handleRemove}
            onOpacityChange={setOverlayOpacity}
            cameras={cameras}
          />
        </div>
      )}
      <div className={`grid gap-3 ${cameras.length === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
        {cameras.map(name => (
          <CameraFeed key={name} name={name} active={active} handDetected={handDetected}
            isHandCamera={handDetectEnabled && name === 'front'}
            overlayUrl={overlays[name] || null}
            overlayOpacity={overlayOpacity} />
        ))}
      </div>
    </div>
  )
}

/* ================================================================
   Object Icons
   ================================================================ */

const OBJECT_ICONS: Record<string, string> = {
  Lemon: '🍋',
  Tissue: '🧻',
  Cup: '🥤',
  Cloth: '🧹',
}

/* ================================================================
   App
   ================================================================ */

function App() {
  const [state, setState]       = useState<RobotState>('WARMUP')
  const [, setStep]             = useState('')
  const [progress, setProgress] = useState(0)
  const [message, setMessage]   = useState('Starting up...')
  const [connected, setConnected]       = useState(false)
  const [reconnecting, setReconnecting] = useState(false)
  const [toasts, setToasts]     = useState<Toast[]>([])
  const [handDetect, setHandDetect]       = useState(true)
  const [handDetected, setHandDetected]   = useState(false)
  const [, setAutoStopped]     = useState(false)
  const [, setPipelineStage]     = useState('')
  const [, setPipelineStageIdx] = useState(0)
  const [, setPipelineTotal]     = useState(0)
  const [pipelineStatus, setPipelineStatus]   = useState('')
  const [stagesInfo, setStagesInfo]     = useState<PipelineStageInfo[]>([])
  const [llmPlanning, setLlmPlanning]   = useState(false)
  const [llmPlanError, setLlmPlanError] = useState('')
  const [mounted, setMounted] = useState(false)

  const wsRef     = useRef<WebSocket | null>(null)
  const reconnRef = useRef<number | null>(null)
  const toastId   = useRef(0)

  useEffect(() => { const t = setTimeout(() => setMounted(true), 50); return () => clearTimeout(t) }, [])

  const toast = useCallback((msg: string, type: Toast['type']) => {
    const id = ++toastId.current
    setToasts(p => [...p, { id, message: msg, type }])
    setTimeout(() => setToasts(p => p.filter(t => t.id !== id)), 3500)
  }, [])

  const api = useCallback(async (endpoint: string, label: string) => {
    try {
      const r = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' } })
      const data = await r.json()
      if (r.ok && data.status === 'ok') toast(label, 'success')
      else toast(data.message || `${label} failed`, 'error')
    } catch { toast(`${label}: network error`, 'error') }
  }, [toast])

  const doStart   = useCallback(() => api('/api/start',   'Planning and starting...'),  [api])
  const doStop    = useCallback(() => api('/api/stop',   'Emergency Stop'),     [api])
  const doHome    = useCallback(() => api('/api/reset',  'Return to Home'),      [api])
  const doResume  = useCallback(() => api('/api/resume', 'Resumed'),            [api])
  const doRestart = useCallback(() => api('/api/restart','Replanning and starting...'), [api])
  const doQuit    = useCallback(() => api('/api/quit',   'Quit and re-warmup'),  [api])
  const doToggleHand = useCallback(() => api('/api/hand-detect', handDetect ? 'Hand safety disabled' : 'Hand safety enabled'), [api, handDetect])
  const doRunStage = useCallback(async (stageName: string) => {
    try {
      const r = await fetch('/api/run-stage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stage: stageName }),
      })
      const data = await r.json()
      if (r.ok && data.status === 'ok') toast(`Running ${stageName}`, 'success')
      else toast(data.message || `Failed to run ${stageName}`, 'error')
    } catch { toast(`Run ${stageName}: network error`, 'error') }
  }, [toast])

  useEffect(() => {
    let backoff = 1000
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      wsRef.current = ws
      ws.onopen = () => { setConnected(true); setReconnecting(false); backoff = 1000 }
      ws.onmessage = (ev) => {
        try {
          const d: StatusMessage = JSON.parse(ev.data)
          if (d.state) setState(d.state)
          if (d.step !== undefined)     setStep(d.step)
          if (d.progress !== undefined) setProgress(Math.max(0, Math.min(100, d.progress)))
          if (d.message !== undefined)  setMessage(d.message)
          if (d.hand_detect !== undefined)  setHandDetect(d.hand_detect)
          if (d.hand_detected !== undefined) setHandDetected(d.hand_detected)
          if (d.auto_stopped !== undefined) setAutoStopped(d.auto_stopped)
          if (d.pipeline_stage !== undefined)     setPipelineStage(d.pipeline_stage)
          if (d.pipeline_stage_idx !== undefined) setPipelineStageIdx(d.pipeline_stage_idx)
          if (d.pipeline_total !== undefined)     setPipelineTotal(d.pipeline_total)
          if (d.pipeline_status !== undefined)    setPipelineStatus(d.pipeline_status)
          if (d.pipeline_stages_info !== undefined) setStagesInfo(d.pipeline_stages_info)
          if (d.llm_planning !== undefined)   setLlmPlanning(d.llm_planning)
          if (d.llm_plan_error !== undefined) setLlmPlanError(d.llm_plan_error)
        } catch { /* ignore */ }
      }
      ws.onclose = () => {
        setConnected(false); setReconnecting(true)
        reconnRef.current = window.setTimeout(() => { backoff = Math.min(backoff * 1.5, 5000); connect() }, backoff)
      }
      ws.onerror = () => ws.close()
    }
    connect()
    return () => { reconnRef.current && clearTimeout(reconnRef.current); wsRef.current?.close() }
  }, [])

  const meta       = STATE_META[state]
  const isWarmup   = state === 'WARMUP'
  const canStart   = state === 'READY' || state === 'DONE' || state === 'ERROR'
  const isRunning  = state === 'WORKING'
  const isPaused   = state === 'PAUSED' || state === 'HOMED'
  const hasProcess = isRunning || isPaused

  /* ================================================================
     Render
     ================================================================ */
  return (
    <div className="w-full h-screen overflow-hidden flex flex-col select-none"
      style={{ background: '#f5f2ee', color: '#2d2a26' }}>

      {/* ═══ HEADER ═══ */}
      <header className={`relative z-10 w-full px-5 lg:px-8 h-14 flex items-center justify-between
                          flex-shrink-0 bg-white border-b
                          transition-all duration-500 ${mounted ? 'opacity-100' : 'opacity-0 -translate-y-2'}`}
        style={{ borderColor: '#e0dbd4' }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: '#e8793a' }}>
            <span className="text-sm font-bold text-white leading-none">R</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-heading text-lg" style={{ color: '#2d2a26' }}>Retable</span>
            <span className="font-heading text-lg" style={{ color: '#e8793a' }}>Bot</span>
          </div>
        </div>

        {/* Status badge */}
        <div className="flex items-center gap-3">
          <div className="px-3 py-1 rounded-full text-xs font-heading"
            style={{ background: meta.bg, color: meta.color, border: `1px solid ${meta.border}` }}>
            {meta.label}
          </div>
          <div className={`flex items-center gap-1.5 text-xs font-heading ${
            connected ? '' : ''
          }`} style={{ color: connected ? '#4caf7d' : '#d94f4f' }}>
            <span className="h-2 w-2 rounded-full"
              style={{ background: connected ? '#4caf7d' : '#d94f4f' }} />
            {connected ? 'Online' : reconnecting ? 'Reconnecting' : 'Offline'}
          </div>
        </div>
      </header>

      {/* ═══ MAIN ═══ */}
      <main className={`relative z-10 flex-1 flex flex-col lg:flex-row w-full px-5 lg:px-8 py-4 gap-4 lg:gap-6
                        overflow-hidden min-h-0
                        transition-all duration-500 ${mounted ? 'opacity-100' : 'opacity-0'}`}>

        {/* ═══ LEFT COLUMN ═══ */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0 lg:mx-0 w-full overflow-y-auto custom-scroll">

          {/* ─── STATUS MESSAGE ─── */}
          {(isWarmup || state === 'ERROR') && message && (
            <div className="mb-3 px-4 py-3 rounded-xl flex-shrink-0"
              style={{ background: meta.bg, border: `1px solid ${meta.border}` }}>
              <p className="text-sm font-mono leading-relaxed" style={{ color: meta.color }}>{message}</p>
            </div>
          )}

          {/* ─── PIPELINE (integrated with LLM plan) ─── */}
          {(stagesInfo.length > 0 || llmPlanning) && (
            <div className="mb-4 flex-shrink-0">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-xs font-heading tracking-wide flex-shrink-0" style={{ color: '#9e978f' }}>Execution Pipeline</span>
                <div className="flex-1 h-px" style={{ background: '#e0dbd4' }} />
                {llmPlanning && (
                  <div className="flex items-center gap-2">
                    <div className="w-3.5 h-3.5 border-2 border-[#8b7ec8] border-t-transparent rounded-full animate-smooth-spin" />
                    <span className="text-xs font-heading" style={{ color: '#8b7ec8' }}>🧠 Analyzing…</span>
                  </div>
                )}
                {!llmPlanning && pipelineStatus && (
                  <span className="text-xs font-heading px-2.5 py-1 rounded-full"
                    style={{
                      background: pipelineStatus === 'inference' ? '#e3f0fc' :
                                  pipelineStatus === 'waypoints' ? '#f0edf8' : '#e8f5ee',
                      color: pipelineStatus === 'inference' ? '#5b8fd9' :
                             pipelineStatus === 'waypoints' ? '#8b7ec8' : '#3b8f7e',
                      border: `1px solid ${pipelineStatus === 'inference' ? '#b3d4f0' :
                               pipelineStatus === 'waypoints' ? '#d4cde8' : '#c3e6d1'}`,
                    }}>
                    {pipelineStatus.toUpperCase()}
                  </span>
                )}
              </div>

              {llmPlanError && (
                <div className="px-3 py-2.5 rounded-xl mb-3"
                  style={{ background: '#fdeaea', border: '1px solid #f0b8b8' }}>
                  <span className="text-xs font-mono" style={{ color: '#d94f4f' }}>{llmPlanError}</span>
                </div>
              )}

              {/* Stage cards */}
              <div className="flex flex-col gap-2">
                {stagesInfo.map((stage) => {
                  const isActive = stage.exec_status === 'active'
                  const execDone = stage.exec_status === 'done'
                  const llmDone = stage.llm_status === 'done'
                  const isDone = execDone || llmDone
                  const icon = OBJECT_ICONS[stage.name] || '📦'
                  const canRunSingle = !isWarmup && !isActive

                  return (
                    <div key={stage.name}
                      className="relative flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300"
                      style={{
                        background: isDone ? '#e8f5ee' : isActive ? '#e3f0fc' : '#ffffff',
                        border: `1px solid ${isDone ? '#c3e6d1' : isActive ? '#b3d4f0' : '#e0dbd4'}`,
                        boxShadow: isActive ? '0 2px 8px rgba(59,143,126,0.08)' : '0 1px 3px rgba(0,0,0,0.04)',
                      }}>
                      <span className="text-xl flex-shrink-0">{icon}</span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-heading" style={{
                            color: isDone ? '#2e7d4f' : isActive ? '#2b6cb0' : '#6b6560'
                          }}>{stage.name}</span>

                          {isDone && (
                            <span className="badge-done text-[10px] px-2 py-0.5 rounded-full font-heading">
                              ✓ Done
                            </span>
                          )}
                          {isActive && (
                            <span className="badge-running text-[10px] px-2 py-0.5 rounded-full font-heading animate-gentle-pulse">
                              ▶ Running
                            </span>
                          )}
                          {!isDone && !isActive && (
                            <span className="badge-todo text-[10px] px-2 py-0.5 rounded-full font-heading">
                              ⏳ Pending
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex-shrink-0">
                        {canRunSingle && (
                          <button onClick={() => doRunStage(stage.name)}
                            title={`Run ${stage.name} alone`}
                            className="px-2.5 py-1.5 rounded-lg transition-all duration-200 group"
                            style={{
                              background: '#faf8f6',
                              border: '1px solid #e0dbd4',
                            }}
                            onMouseEnter={e => {
                              e.currentTarget.style.background = '#fef3e2'
                              e.currentTarget.style.borderColor = '#f5ddb5'
                            }}
                            onMouseLeave={e => {
                              e.currentTarget.style.background = '#faf8f6'
                              e.currentTarget.style.borderColor = '#e0dbd4'
                            }}>
                            <span className="flex items-center gap-1.5">
                              <IconPlay size={11} className="text-[#9e978f] group-hover:text-[#e8793a] transition-colors" />
                              <span className="text-[10px] font-heading text-[#9e978f] group-hover:text-[#e8793a] transition-colors">Run</span>
                            </span>
                          </button>
                        )}
                        {isDone && !canRunSingle && (
                          <span style={{ color: '#4caf7d' }} className="text-lg">✓</span>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Planning spinner when no stages info yet */}
              {llmPlanning && stagesInfo.length === 0 && (
                <div className="flex items-center justify-center py-6">
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-[#8b7ec8] border-t-transparent rounded-full animate-smooth-spin" />
                    <span className="text-sm font-heading" style={{ color: '#9e978f' }}>Analyzing scene…</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ─── QUICK RUN — test mode ─── */}
          <div className="mb-4 flex-shrink-0">
            <div className="flex items-center gap-2 mb-2.5">
              <IconZap size={12} className="text-[#e8793a]" />
              <span className="text-xs font-heading tracking-wide" style={{ color: '#9e978f' }}>Quick Run</span>
              <div className="flex-1 h-px" style={{ background: '#e0dbd4' }} />
            </div>
            <div className="flex flex-wrap gap-2">
              {['Lemon', 'Tissue', 'Cup', 'Cloth'].map(name => {
                const icon = OBJECT_ICONS[name] || '📦'
                const isCurrentlyRunning = stagesInfo.some(s => s.name === name && s.exec_status === 'active')
                return (
                  <button key={name} onClick={() => doRunStage(name)}
                    disabled={isCurrentlyRunning}
                    className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-heading
                               transition-all duration-200"
                    style={{
                      background: isCurrentlyRunning ? '#e3f0fc' : '#ffffff',
                      border: `1px solid ${isCurrentlyRunning ? '#b3d4f0' : '#e0dbd4'}`,
                      color: isCurrentlyRunning ? '#2b6cb0' : '#6b6560',
                      cursor: isCurrentlyRunning ? 'not-allowed' : 'pointer',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
                    }}
                    onMouseEnter={e => {
                      if (!isCurrentlyRunning) {
                        e.currentTarget.style.background = '#fef3e2'
                        e.currentTarget.style.borderColor = '#f5ddb5'
                        e.currentTarget.style.color = '#e8793a'
                      }
                    }}
                    onMouseLeave={e => {
                      if (!isCurrentlyRunning) {
                        e.currentTarget.style.background = '#ffffff'
                        e.currentTarget.style.borderColor = '#e0dbd4'
                        e.currentTarget.style.color = '#6b6560'
                      }
                    }}>
                    <span>{icon}</span>
                    <span>{name}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* ─── HAND SAFETY ─── */}
          <div className="flex items-center justify-between py-3 px-4 rounded-xl mb-4 flex-shrink-0 transition-all duration-300"
            style={{
              background: handDetected ? '#fdeaea' : handDetect ? '#e8f5ee' : '#ffffff',
              border: `1px solid ${handDetected ? '#f0b8b8' : handDetect ? '#c3e6d1' : '#e0dbd4'}`,
            }}>
            <div className="flex items-center gap-2.5">
              <IconHand size={20} color={handDetected ? '#d94f4f' : handDetect ? '#4caf7d' : '#9e978f'} />
              <span className="text-sm font-heading" style={{
                color: handDetected ? '#d94f4f' : handDetect ? '#3b8f7e' : '#9e978f'
              }}>
                {handDetected ? 'Hand detected — stopped' : handDetect ? 'Hand safety enabled' : 'Hand safety disabled'}
              </span>
              {handDetected && <span className="h-2 w-2 rounded-full bg-red-500 animate-ping" />}
            </div>
            <button onClick={doToggleHand} disabled={isWarmup}
              className="relative w-11 h-6 rounded-full transition-all duration-300 flex-shrink-0"
              style={{
                background: isWarmup ? '#e0dbd4' : handDetect ? '#4caf7d' : '#c8c1b8',
                opacity: isWarmup ? 0.4 : 1,
                cursor: isWarmup ? 'not-allowed' : 'pointer',
              }}>
              <div className="absolute top-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform duration-300"
                style={{ transform: handDetect ? 'translateX(22px)' : 'translateX(2px)' }} />
            </button>
          </div>

          {/* ─── Mobile Camera Feeds ─── */}
          <div className="lg:hidden flex-shrink-0">
            <MobileCameraFeeds active={!isWarmup && state !== 'ERROR'} handDetected={handDetected} handDetectEnabled={handDetect} />
          </div>

          {/* ═══════════════════════════════════════════════════════
             ACTION PANEL
             ═══════════════════════════════════════════════════════ */}
          <div className="flex-1 flex flex-col min-h-0">

            {/* ── Hero Button — changes based on state ── */}
            <div className="mb-3 flex-shrink-0">
              {isWarmup ? (
                /* Warmup: loading state with progress */
                <div className="flex flex-col gap-3 py-5 lg:py-6 px-5 rounded-xl neon-border-animated"
                  style={{ background: '#ffffff', border: '1px solid #b3d4f0' }}>
                  <div className="flex items-center justify-center gap-3">
                    <div className="w-5 h-5 border-2 border-[#5b8fd9] border-t-transparent rounded-full animate-smooth-spin" />
                    <span className="text-base font-heading" style={{ color: '#6b6560' }}>Loading model…</span>
                    <span className="text-base font-heading" style={{ color: '#5b8fd9' }}>{progress}%</span>
                  </div>
                  <div className="h-2 rounded-full overflow-hidden" style={{ background: '#eae6e0' }}>
                    <div
                      className="h-full rounded-full transition-all duration-700 ease-out progress-glow relative"
                      style={{
                        width: `${progress}%`,
                        background: 'linear-gradient(90deg, #7db8e0, #5b8fd9)',
                      }}
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
                    </div>
                  </div>
                </div>
              ) : canStart ? (
                /* Ready/Done/Error: START */
                <button onClick={doStart}
                  className="group w-full py-5 lg:py-6 rounded-xl font-heading font-bold text-xl lg:text-2xl tracking-wide
                             text-white
                             transition-all duration-200 btn-press relative overflow-hidden"
                  style={{
                    background: 'linear-gradient(135deg, #e8793a, #f59e5e)',
                    boxShadow: '0 4px 20px rgba(232,121,58,0.2)',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 6px 28px rgba(232,121,58,0.35)' }}
                  onMouseLeave={e => { e.currentTarget.style.boxShadow = '0 4px 20px rgba(232,121,58,0.2)' }}>
                  <span className="relative z-10 flex items-center justify-center gap-3">
                    <IconPlay size={24} />
                    Start
                  </span>
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                </button>
              ) : isRunning ? (
                /* Running: STOP */
                <button onClick={doStop}
                  className="group w-full py-5 lg:py-6 rounded-xl font-heading font-bold text-xl lg:text-2xl tracking-wide
                             text-white estop-active
                             transition-all duration-200 btn-press relative overflow-hidden select-none"
                  style={{
                    background: 'linear-gradient(135deg, #d94f4f, #e06b6b)',
                    boxShadow: '0 4px 20px rgba(217,79,79,0.25)',
                  }}>
                  <span className="relative z-10 flex items-center justify-center gap-3">
                    <IconStop size={28} />
                    Emergency Stop
                  </span>
                </button>
              ) : isPaused ? (
                /* Paused/Homed: RESUME */
                <button onClick={doResume}
                  className="group w-full py-5 lg:py-6 rounded-xl font-heading font-bold text-xl lg:text-2xl tracking-wide
                             text-white
                             transition-all duration-200 btn-press relative overflow-hidden"
                  style={{
                    background: 'linear-gradient(135deg, #5b8fd9, #7db8e0)',
                    boxShadow: '0 4px 20px rgba(91,143,217,0.2)',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 6px 28px rgba(91,143,217,0.35)' }}
                  onMouseLeave={e => { e.currentTarget.style.boxShadow = '0 4px 20px rgba(91,143,217,0.2)' }}>
                  <span className="relative z-10 flex items-center justify-center gap-3">
                    <IconPlay size={24} className="text-white" />
                    Resume
                  </span>
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                </button>
              ) : null}
            </div>

            {/* ── Secondary Actions ── */}
            {!isWarmup && (
              <div className="flex gap-3 mb-3">
                {/* Home */}
                <button onClick={doHome}
                  className="action-tile group flex-1">
                  <IconHome size={24} className="transition-colors text-[#9e978f] group-hover:text-[#e8793a]" />
                  <span className="text-xs font-heading text-[#9e978f] group-hover:text-[#e8793a] transition-colors">Return to Home</span>
                </button>
                {/* Restart (with LLM replan) */}
                <button onClick={doRestart}
                  className="action-tile group flex-1">
                  <IconRefresh size={24} className="transition-colors text-[#9e978f] group-hover:text-[#e8793a]" />
                  <span className="text-xs font-heading text-[#9e978f] group-hover:text-[#e8793a] transition-colors">Replan</span>
                </button>
              </div>
            )}

            {/* Quit — subtle, only when process active */}
            {hasProcess && (
              <button onClick={doQuit}
                className="w-full py-2 text-xs font-heading tracking-wide
                           transition-colors duration-200 mb-2 flex-shrink-0"
                style={{ color: '#9e978f' }}
                onMouseEnter={e => { e.currentTarget.style.color = '#6b6560' }}
                onMouseLeave={e => { e.currentTarget.style.color = '#9e978f' }}>
                <span className="flex items-center justify-center gap-2">
                  <IconX size={13} />
                  Quit Session
                </span>
              </button>
            )}

          </div>{/* end action panel */}
        </div>{/* end left column */}

        {/* ═══ RIGHT COLUMN ═══ */}
        <div className={`hidden lg:flex lg:w-[380px] xl:w-[440px] 2xl:w-[500px] flex-shrink-0 min-w-0 flex-col gap-3
                        overflow-y-auto transition-all duration-500 delay-150
                        ${mounted ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-6'}`}>
          <SidebarCameraFeeds active={!isWarmup && state !== 'ERROR'} handDetected={handDetected} handDetectEnabled={handDetect} />
        </div>

      </main>

      {/* ═══ TOASTS ═══ */}
      <div className="fixed top-3 left-1/2 -translate-x-1/2 flex flex-col gap-2 pointer-events-none z-50 w-[90vw] max-w-sm">
        {toasts.map(t => (
          <div key={t.id}
            className="px-5 py-3 rounded-xl text-sm font-heading text-center
                        shadow-lg animate-fadeInUp text-white"
            style={{
              background: t.type === 'success' ? '#4caf7d' :
                          t.type === 'error'   ? '#d94f4f' : '#5b8fd9',
            }}>
            {t.message}
          </div>
        ))}
      </div>

    </div>
  )
}

export default App
