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
   State config
   ================================================================ */

const STATE_META: Record<RobotState, {
  label: string; color: string; glow: string; textGlow: string
}> = {
  WARMUP:  { label: 'WARMING UP',  color: '#00f0ff', glow: 'glow-cyan',   textGlow: 'text-glow-cyan' },
  READY:   { label: 'READY',       color: '#d2ff00', glow: 'glow-neon',   textGlow: 'text-glow-neon' },
  WORKING: { label: 'RUNNING',     color: '#3b82f6', glow: 'glow-blue',   textGlow: 'text-glow-blue' },
  PAUSED:  { label: 'STOPPED',     color: '#f59e0b', glow: '',            textGlow: '' },
  HOMED:   { label: 'HOME',        color: '#a78bfa', glow: '',            textGlow: '' },
  DONE:    { label: 'COMPLETE',    color: '#10b981', glow: 'glow-emerald',textGlow: 'text-glow-emerald' },
  ERROR:   { label: 'ERROR',       color: '#ef4444', glow: 'glow-red',   textGlow: 'text-glow-red' },
}

/* ================================================================
   SVG Icons — inline for zero deps
   ================================================================ */
const IconPlay: FC<{ size?: number; className?: string }> = ({ size = 20, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <polygon points="6 3 20 12 6 21 6 3"/>
  </svg>
)
const IconStop: FC<{ size?: number; className?: string }> = ({ size = 20, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className={className}>
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
const IconHand: FC<{ size?: number; color?: string }> = ({ size = 18, color = '#555' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v0M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8" />
    <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
  </svg>
)

/* ================================================================
   Floating Particles
   ================================================================ */
const FloatingParticles: FC = () => {
  const particles = Array.from({ length: 12 }, (_, i) => ({
    id: i,
    left: `${Math.random() * 100}%`,
    top: `${Math.random() * 100}%`,
    delay: `${Math.random() * 6}s`,
    duration: `${5 + Math.random() * 5}s`,
    size: Math.random() > 0.5 ? 3 : 2,
    color: ['#d2ff00', '#00f0ff', '#ff00aa'][Math.floor(Math.random() * 3)],
  }))
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {particles.map(p => (
        <div key={p.id} className="particle"
          style={{ left: p.left, top: p.top, width: p.size, height: p.size,
                   background: p.color, animationDelay: p.delay, animationDuration: p.duration,
                   boxShadow: `0 0 8px ${p.color}50` }} />
      ))}
    </div>
  )
}

/* ================================================================
   Camera Feed
   ================================================================ */
const CameraFeed: FC<{ name: string; active: boolean; handDetected?: boolean; isHandCamera?: boolean }> = ({ name, active, handDetected = false, isHandCamera = false }) => {
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
    <div className={`relative rounded-lg overflow-hidden glass-card transition-all duration-300
                    ${isHandCamera && handDetected ? 'border-red-500/60 glow-red' : 'hover:border-[#d2ff0030]'}`}>
      <img ref={imgRef} alt={name}
        className={`w-full h-auto block transition-opacity duration-500 ${hasFrame ? 'opacity-100' : 'opacity-0'}`} />
      {!hasFrame && (
        <div className="flex items-center justify-center h-32 text-slate-700">
          <div className="w-5 h-5 border-2 border-slate-700 border-t-[#00f0ff] rounded-full animate-smooth-spin" />
        </div>
      )}
      <div className="absolute top-2.5 left-2.5 px-3 py-1 bg-black/80 rounded text-sm font-heading tracking-[0.3em] text-[#d2ff00]/80 border border-[#d2ff00]/15">
        {name}
      </div>
      {hasFrame && (
        <div className="absolute top-2.5 right-2.5 flex items-center gap-1.5 px-2.5 py-1 bg-black/80 rounded border border-red-500/20">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
          </span>
          <span className="text-sm font-heading tracking-[0.2em] text-red-400">LIVE</span>
        </div>
      )}
      {isHandCamera && handDetected && hasFrame && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="absolute inset-0 bg-red-600/25 animate-pulse" />
          <div className="relative z-10 flex items-center gap-2 px-3 py-1.5 bg-red-600/90 backdrop-blur rounded-lg border border-red-400/40 glow-red">
            <span className="text-lg">🖐️</span>
            <span className="text-sm font-heading font-bold text-white tracking-[0.2em]">DETECTED</span>
          </div>
        </div>
      )}
      {isHandCamera && !handDetected && hasFrame && (
        <div className="absolute bottom-2.5 right-2.5 px-3 py-1 bg-emerald-900/70 rounded text-sm font-heading tracking-[0.2em] text-emerald-300 border border-emerald-500/20">
          ✓ SAFE
        </div>
      )}
    </div>
  )
}

/* ================================================================
   Sidebar / Mobile Camera Feeds
   ================================================================ */
const SidebarCameraFeeds: FC<{ active: boolean; handDetected: boolean; handDetectEnabled: boolean }> = ({ active, handDetected, handDetectEnabled }) => {
  const [cameras, setCameras] = useState<string[]>([])
  useEffect(() => { fetch('/api/cameras').then(r => r.json()).then(d => setCameras(d.cameras || [])).catch(() => {}) }, [])
  if (cameras.length === 0) return null
  return (
    <div className="flex flex-col gap-3 w-full">
      <div className="flex items-center gap-2">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-[#d2ff0015] to-transparent" />
        <span className="text-sm font-heading tracking-[0.4em] text-[#d2ff00]/40">FEEDS</span>
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-[#d2ff0015] to-transparent" />
      </div>
      {cameras.map(name => (
        <CameraFeed key={name} name={name} active={active} handDetected={handDetected}
          isHandCamera={handDetectEnabled && name === 'front'} />
      ))}
    </div>
  )
}

const MobileCameraFeeds: FC<{ active: boolean; handDetected: boolean; handDetectEnabled: boolean }> = ({ active, handDetected, handDetectEnabled }) => {
  const [cameras, setCameras] = useState<string[]>([])
  useEffect(() => { fetch('/api/cameras').then(r => r.json()).then(d => setCameras(d.cameras || [])).catch(() => {}) }, [])
  if (cameras.length === 0) return null
  return (
    <div className="w-full mb-2">
      <div className={`grid gap-2 ${cameras.length === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
        {cameras.map(name => (
          <CameraFeed key={name} name={name} active={active} handDetected={handDetected}
            isHandCamera={handDetectEnabled && name === 'front'} />
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
  const [autoStopped, setAutoStopped]     = useState(false)
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

  const doStart   = useCallback(() => api('/api/start',   'Planning & starting...'),  [api])
  const doStop    = useCallback(() => api('/api/stop',   'Emergency stop'),     [api])
  const doHome    = useCallback(() => api('/api/reset',  'Going to home'),      [api])
  const doResume  = useCallback(() => api('/api/resume', 'Resumed'),            [api])
  const doRestart = useCallback(() => api('/api/restart','Replanning & restarting...'), [api])
  const doQuit    = useCallback(() => api('/api/quit',   'Quit & re-warming'),  [api])
  const doToggleHand = useCallback(() => api('/api/hand-detect', handDetect ? 'Hand safety OFF' : 'Hand safety ON'), [api, handDetect])

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
    <div className="w-full h-screen overflow-hidden bg-[#0a0a0a] text-white flex flex-col select-none bg-grid noise-overlay scanlines">
      <div className="fixed inset-0 bg-spotlight pointer-events-none" />
      <FloatingParticles />

      {/* ═══ HEADER ═══ */}
      <header className={`relative z-10 w-full px-6 lg:px-10 h-14 flex items-center justify-between
                          border-b border-white/[0.04] backdrop-blur-md flex-shrink-0
                          transition-all duration-500 ${mounted ? 'opacity-100' : 'opacity-0 -translate-y-2'}`}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-md bg-[#d2ff00] flex items-center justify-center">
            <span className="text-lg font-black text-black leading-none">R</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-heading text-xl tracking-[0.2em] text-white/90">RETABLE</span>
            <span className="font-heading text-xl tracking-[0.2em] text-[#d2ff00]">BOT</span>
          </div>
        </div>
        <div className={`flex items-center gap-2 text-sm font-heading tracking-[0.2em] ${
          connected ? 'text-emerald-400' : 'text-red-400'
        }`}>
          <span className={`h-2.5 w-2.5 rounded-full ${connected ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50' : 'bg-red-400'}`} />
          {connected ? 'ONLINE' : reconnecting ? 'RECONNECTING' : 'OFFLINE'}
        </div>
      </header>

      {/* ═══ MAIN ═══ */}
      <main className={`relative z-10 flex-1 flex flex-col lg:flex-row w-full px-6 lg:px-10 py-4 gap-5 lg:gap-8
                        overflow-hidden min-h-0
                        transition-all duration-500 ${mounted ? 'opacity-100' : 'opacity-0'}`}>

        {/* ═══ LEFT COLUMN ═══ */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0 lg:mx-0 w-full overflow-y-auto custom-scroll">

          {/* ─── HERO STATE ─── */}
          <div className="relative mb-4 lg:mb-6 flex-shrink-0">
            <div className="relative overflow-hidden">
              <h2 className={`font-heading leading-[0.85] tracking-[0.04em] transition-colors duration-500 ${meta.textGlow}`}
                style={{
                  color: meta.color,
                  fontSize: 'clamp(4.5rem, 10vw, 9rem)',
                  textShadow: `0 0 40px ${meta.color}40, 0 0 80px ${meta.color}15`,
                }}>
                {autoStopped && isPaused ? 'HAND\nSTOP' : meta.label}
              </h2>
              <div className="h-1 mt-2 rounded-full overflow-hidden" style={{ background: `${meta.color}15` }}>
                <div className={`h-full rounded-full transition-all duration-1000 ${
                  (isRunning || isWarmup) ? 'animate-shimmer-bar' : ''
                }`} style={{
                  width: (isRunning || isWarmup) ? '100%' : '40%',
                  background: `linear-gradient(90deg, transparent, ${meta.color}, transparent)`,
                }} />
              </div>
            </div>
            {(isWarmup || state === 'ERROR') && message && (
              <p className="text-base text-slate-500 mt-3 font-mono leading-relaxed">{message}</p>
            )}
          </div>

          {/* ─── PIPELINE (integrated with LLM plan) ─── */}
          {(stagesInfo.length > 0 || llmPlanning) && (
            <div className="mb-4 flex-shrink-0">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-sm font-heading tracking-[0.3em] text-slate-600 flex-shrink-0">PIPELINE</span>
                <div className="flex-1" />
                {llmPlanning && (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-smooth-spin" />
                    <span className="text-xs font-heading tracking-[0.2em] text-violet-400/60">🧠 ANALYZING…</span>
                  </div>
                )}
                {!llmPlanning && pipelineStatus && (
                  <span className={`text-sm font-heading tracking-[0.2em] px-2.5 py-1 rounded border flex-shrink-0 ${
                    pipelineStatus === 'inference' ? 'border-blue-500/30 text-blue-400' :
                    pipelineStatus === 'waypoints' ? 'border-violet-500/30 text-violet-400' :
                    'border-cyan-500/30 text-cyan-400'
                  }`}>
                    {pipelineStatus.toUpperCase()}
                  </span>
                )}
              </div>

              {llmPlanError && (
                <div className="px-3 py-2 rounded bg-red-500/10 border border-red-500/20 mb-3">
                  <span className="text-xs font-mono text-red-400">{llmPlanError}</span>
                </div>
              )}

              {/* Stage cards — only visible objects (not_found filtered by backend) */}
              <div className="flex flex-col gap-2">
                {stagesInfo.map((stage) => {
                  const isActive = stage.exec_status === 'active'
                  const execDone = stage.exec_status === 'done'
                  const llmDone = stage.llm_status === 'done'
                  const isDone = execDone || llmDone
                  const icon = OBJECT_ICONS[stage.name] || '📦'

                  return (
                    <div key={stage.name}
                      className={`relative flex items-center gap-3 px-4 py-3 rounded-lg border transition-all duration-300 ${
                        isDone
                          ? 'border-emerald-500/25 bg-emerald-500/[0.04]'
                          : isActive
                            ? 'border-[#00f0ff]/40 bg-[#00f0ff]/[0.04]'
                            : 'border-white/[0.06] bg-white/[0.02]'
                      }`}>
                      <span className="text-xl flex-shrink-0">{icon}</span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`text-sm font-heading tracking-[0.1em] ${
                            isDone ? 'text-emerald-400' :
                            isActive ? 'text-[#00f0ff]' :
                            'text-slate-500'
                          }`}>{stage.name}</span>
                          {isDone && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-heading tracking-wider">
                              ✓ DONE
                            </span>
                          )}
                          {isActive && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#00f0ff]/20 text-[#00f0ff] font-heading tracking-wider animate-pulse">
                              ▶ RUNNING
                            </span>
                          )}
                          {!isDone && !isActive && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-heading tracking-wider">
                              ⏳ TODO
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex-shrink-0 w-8 text-right">
                        {isDone && <span className="text-emerald-400 text-lg">✓</span>}
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Planning spinner when no stages info yet */}
              {llmPlanning && stagesInfo.length === 0 && (
                <div className="flex items-center justify-center py-6">
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-violet-400 border-t-transparent rounded-full animate-smooth-spin" />
                    <span className="text-sm font-heading tracking-[0.15em] text-slate-500">Analyzing scene…</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ─── HAND SAFETY ─── */}
          <div className={`flex items-center justify-between py-3 px-4 rounded-lg mb-4 flex-shrink-0 transition-all duration-300 border ${
            handDetected
              ? 'border-red-500/50 bg-red-500/[0.06] glow-red'
              : handDetect
                ? 'border-emerald-500/15 bg-emerald-500/[0.03]'
                : 'border-white/[0.04] bg-white/[0.01]'
          }`}>
            <div className="flex items-center gap-2">
              <IconHand size={22} color={handDetected ? '#ef4444' : handDetect ? '#10b981' : '#555'} />
              <span className={`text-base font-heading tracking-[0.15em] ${
                handDetected ? 'text-red-400' : handDetect ? 'text-emerald-400/80' : 'text-slate-600'
              }`}>
                {handDetected ? 'HAND DETECTED — STOPPED' : handDetect ? 'HAND SAFETY ACTIVE' : 'HAND SAFETY OFF'}
              </span>
              {handDetected && <span className="h-2 w-2 rounded-full bg-red-500 animate-ping" />}
            </div>
            <button onClick={doToggleHand} disabled={isWarmup}
              className={`relative w-12 h-6 rounded-full transition-all duration-300 flex-shrink-0
                         ${isWarmup ? 'opacity-30 cursor-not-allowed' : handDetect ? 'bg-emerald-500' : 'bg-slate-700'}`}>
              <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform duration-300
                              ${handDetect ? 'translate-x-[26px]' : 'translate-x-0.5'}`} />
            </button>
          </div>

          {/* ─── Mobile Camera Feeds ─── */}
          <div className="lg:hidden flex-shrink-0">
            <MobileCameraFeeds active={!isWarmup && state !== 'ERROR'} handDetected={handDetected} handDetectEnabled={handDetect} />
          </div>

          {/* ═══════════════════════════════════════════════════════
             ACTION PANEL — single morphing hero button
             ═══════════════════════════════════════════════════════ */}
          <div className="flex-1 flex flex-col min-h-0">

            {/* ── Morphing Hero Button — changes based on state ── */}
            <div className="mb-4 flex-shrink-0">
              {isWarmup ? (
                /* Warmup: loading state with progress */
                <div className="flex flex-col gap-3 py-6 lg:py-8 px-6 rounded-xl border border-[#00f0ff]/10 bg-[#00f0ff]/[0.02] neon-border-animated">
                  <div className="flex items-center justify-center gap-4">
                    <div className="w-6 h-6 border-2 border-[#00f0ff] border-t-transparent rounded-full animate-smooth-spin" />
                    <span className="text-lg font-heading tracking-[0.15em] text-slate-500">LOADING MODEL…</span>
                    <span className="text-lg font-heading tracking-wider text-[#00f0ff]/60">{progress}%</span>
                  </div>
                  <div className="h-2 bg-white/[0.04] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700 ease-out progress-glow relative"
                      style={{
                        width: `${progress}%`,
                        background: 'linear-gradient(90deg, #00f0ffcc, #00f0ff)',
                        boxShadow: '0 0 12px rgba(0,240,255,0.3)',
                      }}
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/25 to-transparent animate-shimmer" />
                    </div>
                  </div>
                </div>
              ) : canStart ? (
                /* Ready/Done/Error: START */
                <button onClick={doStart}
                  className="group w-full py-6 lg:py-8 rounded-xl font-heading font-black text-2xl lg:text-3xl tracking-[0.2em]
                             bg-[#d2ff00] text-black hover:bg-[#e5ff4d]
                             shadow-[0_0_40px_rgba(210,255,0,0.2)] hover:shadow-[0_0_60px_rgba(210,255,0,0.4)]
                             transition-all duration-200 btn-press relative overflow-hidden">
                  <span className="relative z-10 flex items-center justify-center gap-3">
                    <IconPlay size={28} />
                    START
                  </span>
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                </button>
              ) : isRunning ? (
                /* Running: STOP */
                <button onClick={doStop}
                  className="group w-full py-6 lg:py-8 rounded-xl font-heading font-black text-2xl lg:text-3xl tracking-[0.25em]
                             bg-gradient-to-b from-red-500 to-red-700 text-white
                             border-2 border-red-400/40 hover:border-red-300/60
                             estop-active
                             shadow-[0_0_50px_rgba(239,68,68,0.3),0_0_100px_rgba(239,68,68,0.1)]
                             hover:shadow-[0_0_70px_rgba(239,68,68,0.45),0_0_120px_rgba(239,68,68,0.15)]
                             hover:from-red-400 hover:to-red-600
                             active:from-red-600 active:to-red-800
                             transition-all duration-200 btn-press relative overflow-hidden select-none">
                  <span className="relative z-10 flex items-center justify-center gap-3">
                    <IconStop size={32} />
                    STOP
                  </span>
                </button>
              ) : isPaused ? (
                /* Paused/Homed: RESUME */
                <button onClick={doResume}
                  className="group w-full py-6 lg:py-8 rounded-xl font-heading font-black text-2xl lg:text-3xl tracking-[0.2em]
                             bg-blue-500 text-white hover:bg-blue-400
                             shadow-[0_0_40px_rgba(59,130,246,0.25)] hover:shadow-[0_0_60px_rgba(59,130,246,0.4)]
                             transition-all duration-200 btn-press relative overflow-hidden">
                  <span className="relative z-10 flex items-center justify-center gap-3">
                    <IconPlay size={28} className="text-white" />
                    RESUME
                  </span>
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                </button>
              ) : null}
            </div>

            {/* ── Secondary Actions ── */}
            {!isWarmup && (
              <div className="flex gap-3 mb-4">
                {/* Home */}
                <button onClick={doHome}
                  className="action-tile group flex-1"
                  style={{ '--tile-color': '#d2ff00' } as React.CSSProperties}>
                  <IconHome size={28} className="text-[#d2ff00]/60 group-hover:text-[#d2ff00] transition-colors" />
                  <span className="text-sm font-heading tracking-[0.2em] text-[#d2ff00]/50 group-hover:text-[#d2ff00]/90 transition-colors">HOME</span>
                </button>
                {/* Restart (with LLM replan) */}
                <button onClick={doRestart}
                  className="action-tile group flex-1"
                  style={{ '--tile-color': '#a78bfa' } as React.CSSProperties}>
                  <svg width={26} height={26} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    className="text-violet-400/60 group-hover:text-violet-400 transition-colors">
                    <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                  </svg>
                  <span className="text-sm font-heading tracking-[0.2em] text-violet-400/50 group-hover:text-violet-400/90 transition-colors">RESTART</span>
                </button>
              </div>
            )}

            {/* Quit — subtle, only when process active */}
            {hasProcess && (
              <button onClick={doQuit}
                className="w-full py-2 text-sm font-heading tracking-[0.25em] text-slate-700 hover:text-slate-500
                           transition-colors duration-200 mb-2 flex-shrink-0">
                <span className="flex items-center justify-center gap-2">
                  <IconX size={14} />
                  QUIT SESSION
                </span>
              </button>
            )}

          </div>{/* end action panel */}
        </div>{/* end left column */}

        {/* ═══ RIGHT COLUMN ═══ */}
        <div className={`hidden lg:flex lg:w-[400px] xl:w-[460px] 2xl:w-[520px] flex-shrink-0 min-w-0 flex-col gap-3
                        overflow-y-auto transition-all duration-500 delay-150
                        ${mounted ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-6'}`}>
          <SidebarCameraFeeds active={!isWarmup && state !== 'ERROR'} handDetected={handDetected} handDetectEnabled={handDetect} />
        </div>

      </main>

      {/* ═══ TOASTS ═══ */}
      <div className="fixed top-3 left-1/2 -translate-x-1/2 flex flex-col gap-2 pointer-events-none z-50 w-[90vw] max-w-sm">
        {toasts.map(t => (
          <div key={t.id}
            className={`px-5 py-3 rounded-lg text-sm font-heading tracking-[0.15em] text-center
                        shadow-xl backdrop-blur-md border animate-fadeInUp
              ${t.type === 'success' ? 'bg-emerald-600/90 border-emerald-400/20 text-white' :
                t.type === 'error'   ? 'bg-red-600/90 border-red-400/20 text-white' :
                                       'bg-blue-600/90 border-blue-400/20 text-white'}`}>
            {t.message}
          </div>
        ))}
      </div>

    </div>
  )
}

export default App
