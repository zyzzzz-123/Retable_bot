import { useState, useEffect, useCallback, useRef, type FC } from 'react'

/* ================================================================
   Types
   ================================================================ */

type RobotState = 'WARMUP' | 'READY' | 'WORKING' | 'PAUSED' | 'HOMED' | 'DONE' | 'ERROR'

interface StatusMessage {
  state: RobotState
  step?: string
  progress?: number
  message?: string
  hand_detect?: boolean
  hand_detected?: boolean
  auto_stopped?: boolean
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
  label: string; icon: string
  ring: string; bg: string; glow: string; text: string
}> = {
  WARMUP:  { label: 'Warming Up',  icon: '🔄', ring: 'ring-cyan-400',    bg: 'bg-cyan-500/20',    glow: 'shadow-cyan-500/30',    text: 'text-cyan-400' },
  READY:   { label: 'Ready',       icon: '🤖', ring: 'ring-emerald-400', bg: 'bg-emerald-500/20', glow: 'shadow-emerald-500/30', text: 'text-emerald-400' },
  WORKING: { label: 'Running',     icon: '⚡',  ring: 'ring-blue-400',    bg: 'bg-blue-500/20',    glow: 'shadow-blue-500/40',    text: 'text-blue-400' },
  PAUSED:  { label: 'Stopped',     icon: '⏸️',  ring: 'ring-amber-400',   bg: 'bg-amber-500/20',   glow: 'shadow-amber-500/30',   text: 'text-amber-400' },
  // Note: auto_stopped is shown via the hand safety bar, not a separate orb state
  HOMED:   { label: 'At Home',     icon: '🏠', ring: 'ring-violet-400',  bg: 'bg-violet-500/20',  glow: 'shadow-violet-500/30',  text: 'text-violet-400' },
  DONE:    { label: 'Complete',    icon: '✅',  ring: 'ring-emerald-400', bg: 'bg-emerald-500/20', glow: 'shadow-emerald-500/30', text: 'text-emerald-400' },
  ERROR:   { label: 'Error',       icon: '⚠️',  ring: 'ring-red-400',     bg: 'bg-red-500/20',     glow: 'shadow-red-500/30',     text: 'text-red-400' },
}

const FEEDBACK_TAGS = [
  'missed target', 'dropped object', 'wrong position',
  'collision', 'too slow', 'gripper issue',
]

/* ================================================================
   Camera Feed Component
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
      loader.onload = () => {
        if (!cancelled && imgRef.current) {
          imgRef.current.src = loader.src
          setHasFrame(true)
        }
      }
      loader.onerror = () => {
        if (!cancelled) setHasFrame(false)
      }
      loader.src = `/api/frame/${name}?t=${Date.now()}`
    }

    refresh()
    const id = setInterval(refresh, 200)
    return () => { cancelled = true; clearInterval(id) }
  }, [name, active])

  return (
    <div className={`relative rounded-xl overflow-hidden bg-gray-800/70 transition-all duration-300
                    ${isHandCamera && handDetected
                      ? 'ring-3 ring-red-500 shadow-lg shadow-red-500/40'
                      : 'border border-gray-700/50'}`}>
      <img
        ref={imgRef}
        alt={name}
        className={`w-full h-auto block transition-opacity duration-300 ${hasFrame ? 'opacity-100' : 'opacity-0'}`}
      />
      {!hasFrame && (
        <div className="flex items-center justify-center h-40 md:h-52 text-gray-500 text-sm">
          📷 Waiting for camera…
        </div>
      )}
      {/* Camera label */}
      <div className="absolute top-2 left-2 px-2.5 py-1 bg-black/60 backdrop-blur-sm rounded-lg text-xs md:text-sm font-mono uppercase tracking-wide text-gray-200">
        📷 {name}
      </div>
      {/* Live indicator */}
      {hasFrame && (
        <div className="absolute top-2 right-2 flex items-center gap-1.5 px-2 py-1 bg-black/60 backdrop-blur-sm rounded-lg">
          <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
          <span className="text-[10px] md:text-xs font-semibold text-red-400 uppercase">Live</span>
        </div>
      )}
      {/* ── Hand detection overlay on front camera ── */}
      {isHandCamera && handDetected && hasFrame && (
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          {/* Semi-transparent red overlay */}
          <div className="absolute inset-0 bg-red-600/25 animate-pulse" />
          {/* Hand detected badge */}
          <div className="relative z-10 flex flex-col items-center gap-1.5 px-4 py-3 bg-red-600/90 backdrop-blur-sm rounded-xl shadow-xl">
            <span className="text-3xl md:text-4xl">🖐️</span>
            <span className="text-sm md:text-base font-bold text-white uppercase tracking-wider">Hand Detected</span>
          </div>
        </div>
      )}
      {/* ── Hand safety active badge (no hand, shield icon) ── */}
      {isHandCamera && !handDetected && hasFrame && (
        <div className="absolute bottom-2 right-2 flex items-center gap-1.5 px-2 py-1 bg-emerald-600/70 backdrop-blur-sm rounded-lg">
          <span className="text-xs md:text-sm">🛡️</span>
          <span className="text-[10px] md:text-xs font-semibold text-emerald-200 uppercase">Safe</span>
        </div>
      )}
    </div>
  )
}

const CameraFeeds: FC<{ active: boolean; handDetected: boolean; handDetectEnabled: boolean }> = ({ active, handDetected, handDetectEnabled }) => {
  const [cameras, setCameras] = useState<string[]>([])

  useEffect(() => {
    fetch('/api/cameras')
      .then(r => r.json())
      .then(d => setCameras(d.cameras || []))
      .catch(() => {})
  }, [])

  if (cameras.length === 0) return null

  return (
    <div className="w-full max-w-2xl mb-6 md:mb-8">
      <h3 className="text-sm md:text-base uppercase tracking-wider text-gray-500 font-medium mb-3 text-center">
        Camera Feeds
      </h3>
      <div className={`grid gap-3 md:gap-4 ${cameras.length === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
        {cameras.map(name => (
          <CameraFeed
            key={name}
            name={name}
            active={active}
            handDetected={handDetected}
            isHandCamera={handDetectEnabled && name === 'front'}
          />
        ))}
      </div>
    </div>
  )
}

/* ================================================================
   App
   ================================================================ */

function App() {
  const [state, setState]       = useState<RobotState>('WARMUP')
  const [step, setStep]         = useState('')
  const [progress, setProgress] = useState(0)
  const [message, setMessage]   = useState('Starting up...')
  const [connected, setConnected]       = useState(false)
  const [reconnecting, setReconnecting] = useState(false)
  const [toasts, setToasts]     = useState<Toast[]>([])
  const [showFeedback, setShowFeedback] = useState(false)
  const [fbScore, setFbScore]   = useState<'up' | 'down' | null>(null)
  const [fbTags, setFbTags]     = useState<string[]>([])
  const [handDetect, setHandDetect]       = useState(true)
  const [handDetected, setHandDetected]   = useState(false)
  const [autoStopped, setAutoStopped]     = useState(false)

  const wsRef     = useRef<WebSocket | null>(null)
  const reconnRef = useRef<number | null>(null)
  const toastId   = useRef(0)

  /* ── helpers ── */

  const toast = useCallback((msg: string, type: Toast['type']) => {
    const id = ++toastId.current
    setToasts(p => [...p, { id, message: msg, type }])
    setTimeout(() => setToasts(p => p.filter(t => t.id !== id)), 3000)
  }, [])

  const api = useCallback(
    async (endpoint: string, label: string) => {
      try {
        const r = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' } })
        const data = await r.json()
        if (r.ok && data.status === 'ok') {
          toast(label, 'success')
        } else {
          toast(data.message || `${label} failed`, 'error')
        }
      } catch { toast(`${label}: network error`, 'error') }
    },
    [toast],
  )

  const doStart  = useCallback(() => api('/api/start',  'Inference started'),  [api])
  const doStop   = useCallback(() => api('/api/stop',   'Emergency stop'),     [api])
  const doHome   = useCallback(() => api('/api/reset',  'Going to home'),      [api])
  const doResume = useCallback(() => api('/api/resume', 'Resumed'),            [api])
  const doQuit   = useCallback(() => api('/api/quit',   'Quit & re-warming'),  [api])
  const doToggleHand = useCallback(() => api('/api/hand-detect', handDetect ? 'Hand safety OFF' : 'Hand safety ON'), [api, handDetect])

  const submitFeedback = useCallback(async () => {
    if (!fbScore) return
    try {
      const body: { score: string; tags?: string[] } = { score: fbScore }
      if (fbScore === 'down' && fbTags.length) body.tags = fbTags
      const r = await fetch('/api/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      })
      if (r.ok) toast('Feedback submitted', 'success')
    } catch { /* noop */ }
    setShowFeedback(false); setFbScore(null); setFbTags([])
  }, [fbScore, fbTags, toast])

  /* ── WebSocket ── */

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
          if (d.state) setState(prev => {
            if (d.state === 'DONE' && prev !== 'DONE') setTimeout(() => setShowFeedback(true), 500)
            return d.state
          })
          if (d.step !== undefined)     setStep(d.step)
          if (d.progress !== undefined) setProgress(Math.max(0, Math.min(100, d.progress)))
          if (d.message !== undefined)  setMessage(d.message)
          if (d.hand_detect !== undefined)  setHandDetect(d.hand_detect)
          if (d.hand_detected !== undefined) setHandDetected(d.hand_detected)
          if (d.auto_stopped !== undefined) setAutoStopped(d.auto_stopped)
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

  /* ── derived ── */
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
    <div className="w-full min-h-screen bg-gradient-to-b from-gray-950 via-gray-900 to-gray-950 text-white flex flex-col select-none">

      {/* ── Header ── */}
      <header className="w-full px-6 md:px-10 py-5 flex items-center justify-between border-b border-gray-800/60">
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-gray-100">🤖 LeRobot Control</h1>
        <div className="flex items-center gap-3">
          <span className={`h-3 w-3 md:h-4 md:w-4 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
          <span className="text-sm md:text-base text-gray-400 font-medium">
            {connected ? 'Online' : reconnecting ? 'Reconnecting…' : 'Offline'}
          </span>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="flex-1 flex flex-col items-center w-full px-6 md:px-10 py-6 md:py-10">

        {/* ─── State Orb ─── */}
        <div className="flex flex-col items-center mb-6 md:mb-10">
          <div className={`relative w-40 h-40 md:w-52 md:h-52 lg:w-60 lg:h-60 rounded-full flex items-center justify-center
                          ring-[5px] md:ring-[6px] ${meta.ring} ${meta.bg} shadow-2xl ${meta.glow}
                          transition-all duration-500
                          ${isRunning || isWarmup ? 'animate-pulse' : ''}`}>
            {/* spinning ring for WORKING or WARMUP */}
            {(isRunning || isWarmup) && (
              <div className={`absolute inset-[-8px] md:inset-[-10px] rounded-full border-[3px] md:border-4 border-transparent animate-spin
                ${isWarmup ? 'border-t-cyan-400' : 'border-t-blue-400'}`} />
            )}
            <span className="text-6xl md:text-7xl lg:text-8xl">{meta.icon}</span>
          </div>
          <p className={`mt-4 md:mt-6 text-2xl md:text-3xl lg:text-4xl font-extrabold tracking-wide ${meta.text}`}>
            {autoStopped && isPaused ? '🖐️ Hand Stop' : meta.label}
          </p>
        </div>

        {/* ─── Status Info Card ─── */}
        <div className="w-full max-w-2xl bg-gray-800/60 backdrop-blur rounded-2xl md:rounded-3xl p-6 md:p-8 space-y-4 mb-6 md:mb-10">
          {step && (
            <div className="flex items-center justify-between">
              <span className="text-sm md:text-base uppercase tracking-wider text-gray-500 font-medium">Step</span>
              <span className="text-base md:text-lg font-semibold text-gray-200">{step}</span>
            </div>
          )}
          {message && (
            <p className="text-base md:text-lg text-gray-300 text-center leading-relaxed">{message}</p>
          )}
          <div>
            <div className="flex justify-between text-sm md:text-base text-gray-500 mb-2 font-medium">
              <span>{isWarmup ? 'Loading' : 'Progress'}</span>
              <span>{progress}%</span>
            </div>
            <div className="h-3 md:h-4 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  state === 'WARMUP' ? 'bg-cyan-500' :
                  state === 'DONE'   ? 'bg-emerald-500' :
                  state === 'ERROR'  ? 'bg-red-500' :
                  state === 'PAUSED' ? 'bg-amber-500' :
                  'bg-blue-500'
                }`}
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>

        {/* ─── Hand Safety Bar ─── */}
        <div className="w-full max-w-2xl mb-4 md:mb-6">
          <div className={`flex items-center justify-between rounded-2xl px-5 py-4 md:px-6 md:py-5 transition-all duration-300
                          ${handDetected
                            ? 'bg-red-600/20 ring-2 ring-red-500/60'
                            : handDetect
                              ? 'bg-emerald-600/10 ring-1 ring-emerald-500/30'
                              : 'bg-gray-800/60 ring-1 ring-gray-700/50'}`}>
            <div className="flex items-center gap-3">
              {/* Status indicator */}
              <div className={`relative flex items-center justify-center w-10 h-10 md:w-12 md:h-12 rounded-full
                              ${handDetected
                                ? 'bg-red-500/30'
                                : handDetect
                                  ? 'bg-emerald-500/20'
                                  : 'bg-gray-700/50'}`}>
                <span className="text-xl md:text-2xl">{handDetected ? '🖐️' : handDetect ? '🛡️' : '🚫'}</span>
                {handDetected && (
                  <span className="absolute -top-0.5 -right-0.5 h-3 w-3 md:h-4 md:w-4 rounded-full bg-red-500 animate-ping" />
                )}
              </div>
              <div>
                <p className={`text-sm md:text-base font-bold ${
                  handDetected ? 'text-red-400' : handDetect ? 'text-emerald-400' : 'text-gray-500'
                }`}>
                  {handDetected ? '🖐️ Hand Detected — Auto Stopped!' :
                   handDetect ? 'Hand Safety Active' : 'Hand Safety Off'}
                </p>
                <p className="text-xs md:text-sm text-gray-500">
                  {handDetect
                    ? 'Auto e-stop if a human hand enters the front camera'
                    : 'Manual control only — tap to enable'}
                </p>
              </div>
            </div>
            {/* Toggle button */}
            <button
              onClick={doToggleHand}
              disabled={isWarmup}
              className={`relative w-14 h-8 md:w-16 md:h-9 rounded-full transition-all duration-300 flex-shrink-0
                         ${isWarmup ? 'opacity-40 cursor-not-allowed' :
                           handDetect
                             ? 'bg-emerald-500 hover:bg-emerald-400'
                             : 'bg-gray-600 hover:bg-gray-500'}`}
            >
              <div className={`absolute top-1 w-6 h-6 md:w-7 md:h-7 rounded-full bg-white shadow-md transition-transform duration-300
                              ${handDetect ? 'translate-x-7 md:translate-x-8' : 'translate-x-1'}`} />
            </button>
          </div>
        </div>

        {/* ─── Camera Feeds ─── */}
        <CameraFeeds active={!isWarmup} handDetected={handDetected} handDetectEnabled={handDetect} />

        {/* ─── Action Buttons ─── */}
        <div className="w-full max-w-2xl space-y-4 md:space-y-5">

          {/* Warmup indicator */}
          {isWarmup && (
            <div className="w-full py-6 md:py-8 rounded-2xl text-xl md:text-2xl lg:text-3xl font-bold text-center
                           bg-gray-800 text-gray-500 cursor-default ring-1 ring-gray-700">
              ⏳&ensp;Loading model & connecting robot...
            </div>
          )}

          {/* Primary: Start */}
          {canStart && (
            <button
              onClick={doStart}
              className="w-full py-6 md:py-8 rounded-2xl text-xl md:text-2xl lg:text-3xl font-bold
                         bg-gradient-to-r from-emerald-600 to-emerald-500
                         hover:from-emerald-500 hover:to-emerald-400
                         active:from-emerald-700 active:to-emerald-600
                         shadow-xl shadow-emerald-500/25 transition-all duration-200"
            >
              ▶&ensp;Start Inference
            </button>
          )}

          {/* Primary: Resume */}
          {isPaused && (
            <button
              onClick={doResume}
              className="w-full py-6 md:py-8 rounded-2xl text-xl md:text-2xl lg:text-3xl font-bold
                         bg-gradient-to-r from-blue-600 to-blue-500
                         hover:from-blue-500 hover:to-blue-400
                         active:from-blue-700 active:to-blue-600
                         shadow-xl shadow-blue-500/25 transition-all duration-200"
            >
              ▶&ensp;Resume Inference
            </button>
          )}

          {/* Secondary: Home + Quit */}
          {hasProcess && (
            <div className="flex gap-4">
              <button
                onClick={doHome}
                className="flex-1 py-5 md:py-6 rounded-2xl text-lg md:text-xl lg:text-2xl font-bold
                           bg-violet-600/80 hover:bg-violet-500 active:bg-violet-700
                           transition-colors duration-150"
              >
                🏠&ensp;Home
              </button>
              <button
                onClick={doQuit}
                className="flex-1 py-5 md:py-6 rounded-2xl text-lg md:text-xl lg:text-2xl font-bold
                           bg-gray-700 hover:bg-gray-600 active:bg-gray-800
                           transition-colors duration-150"
              >
                ✕&ensp;Quit
              </button>
            </div>
          )}
        </div>

        {/* Spacer */}
        <div className="flex-1 min-h-6 md:min-h-10" />

        {/* ─── EMERGENCY STOP ─── */}
        <div className="w-full max-w-2xl">
          <button
            onClick={hasProcess ? doStop : undefined}
            disabled={!hasProcess}
            className={`w-full rounded-2xl md:rounded-3xl uppercase tracking-[0.15em] font-black
                        transition-all duration-200 select-none
                        ${hasProcess
                          ? 'py-8 md:py-10 text-2xl md:text-3xl lg:text-4xl bg-red-600 hover:bg-red-500 active:bg-red-700 active:scale-[0.98] shadow-2xl shadow-red-600/40 ring-2 ring-red-400/50'
                          : 'py-6 md:py-8 text-xl md:text-2xl lg:text-3xl bg-gray-800 text-gray-600 cursor-default ring-1 ring-gray-700'
                        }`}
          >
            ⛔&ensp;Emergency Stop
          </button>
        </div>
      </main>

      {/* ── Toasts ── */}
      <div className="fixed top-5 left-1/2 -translate-x-1/2 flex flex-col gap-3 pointer-events-none z-50 w-[90vw] max-w-lg">
        {toasts.map(t => (
          <div key={t.id}
            className={`px-5 py-4 rounded-xl text-base md:text-lg font-medium text-center shadow-lg backdrop-blur
              ${t.type === 'success' ? 'bg-emerald-600/90' :
                t.type === 'error'   ? 'bg-red-600/90' :
                                       'bg-blue-600/90'}`}
          >
            {t.message}
          </div>
        ))}
      </div>

      {/* ── Feedback Modal ── */}
      {showFeedback && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-6 z-50">
          <div className="bg-gray-800 rounded-3xl p-8 w-full max-w-md space-y-6 shadow-2xl">
            <h2 className="text-2xl md:text-3xl font-bold text-center">How did it go?</h2>
            <div className="flex justify-center gap-8">
              <button
                onClick={() => setFbScore('up')}
                className={`text-6xl md:text-7xl p-5 md:p-6 rounded-2xl transition-all ${
                  fbScore === 'up' ? 'bg-emerald-600 scale-110' : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >👍</button>
              <button
                onClick={() => setFbScore('down')}
                className={`text-6xl md:text-7xl p-5 md:p-6 rounded-2xl transition-all ${
                  fbScore === 'down' ? 'bg-red-600 scale-110' : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >👎</button>
            </div>
            {fbScore === 'down' && (
              <div className="space-y-3">
                <p className="text-sm md:text-base text-gray-400 text-center uppercase tracking-wide">What went wrong?</p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {FEEDBACK_TAGS.map(tag => (
                    <button key={tag}
                      onClick={() => setFbTags(p => p.includes(tag) ? p.filter(t => t !== tag) : [...p, tag])}
                      className={`px-4 py-2 rounded-lg text-sm md:text-base font-medium transition-colors ${
                        fbTags.includes(tag) ? 'bg-red-600' : 'bg-gray-700 hover:bg-gray-600'
                      }`}
                    >{tag}</button>
                  ))}
                </div>
              </div>
            )}
            <div className="flex gap-4 pt-2">
              <button onClick={() => { setShowFeedback(false); setFbScore(null); setFbTags([]) }}
                className="flex-1 py-4 rounded-xl bg-gray-700 hover:bg-gray-600 font-semibold text-base md:text-lg">
                Skip
              </button>
              <button onClick={submitFeedback} disabled={!fbScore}
                className={`flex-1 py-4 rounded-xl font-semibold text-base md:text-lg transition-colors ${
                  fbScore ? 'bg-blue-600 hover:bg-blue-500' : 'bg-gray-700 opacity-40 cursor-not-allowed'
                }`}>
                Submit
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
