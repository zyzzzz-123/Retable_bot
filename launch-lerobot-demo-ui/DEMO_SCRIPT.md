# Demo Video Narration Script
## Mobile Command & Status Console (MWS) — Ray Lin

**Duration:** ~2 minutes

---

## Setup Before Recording

1. **Terminal 1** - Start the backend:
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```

2. **Terminal 2** - Start the frontend:
   ```bash
   cd ui
   npm run dev
   ```

3. Open browser to `http://localhost:5173` (or use your phone on same network)

---

## Narration Script

### [0:00-0:15] Introduction

> "Hi, I'm Ray Lin. I built the **Mobile Command & Status Console** — a phone-first web UI for our table-reset robot system. This Minimum Working System validates a key assumption: that a **simple phone UI can provide clear control and transparency** so users can safely start, stop, and monitor the robot without confusion or extra hardware."

### [0:15-0:30] Show the UI — Connected State

> "Here's the web interface. At the top, you can see the **connection status** — now showing 'Connected' in green, indicating our WebSocket connection to the backend is live. The **state badge** shows READY, and the control buttons are configured based on the current state."

### [0:30-0:50] Demonstrate Start — Watch Progress

> "Let me start the table reset routine."
> 
> *[Tap Start button]*
> 
> "Watch how the UI immediately transitions to WORKING. The current step updates in real-time — 'Scanning table', 'Identifying items', 'Clearing dishes' — and the progress bar fills smoothly. This is receiving live updates via WebSocket."

### [0:50-1:05] Demonstrate Pause/Resume

> *[Tap Pause while task is running]*
> 
> "I can pause at any time. The state changes to PAUSED, and notice how the Pause button is now disabled while Resume becomes active. This enforces correct state-based controls."
> 
> *[Tap Resume]*
> 
> "And resume continues exactly where we left off."

### [1:05-1:20] Demonstrate Stop (Start a New Run First)

> "The Stop button is always available — this is intentional for safety."
> 
> *[Let it run a bit, then tap Stop]*
> 
> "Stop immediately halts the task and returns to READY. The user always has control."

### [1:20-1:40] Full Run to Completion — Feedback Modal

> "Let me run a complete cycle to show the feedback system."
> 
> *[Tap Start, let it run to completion — about 14 seconds]*
> 
> "When the task completes, the state changes to DONE, and this **feedback modal** automatically appears. Users can give a thumbs up for success, or thumbs down and select issue tags like 'missed item', 'crumbs left', or 'took too long'. This captures per-run feedback for our measurability requirements."
> 
> *[Select thumbs up, tap Submit]*

### [1:40-1:55] Show Session Log (Evidence)

> "For measurability, every interaction is logged with timestamps."
> 
> *[Open new browser tab to http://localhost:8000/api/log]*
> 
> "Here's our session log showing: task start time, each step transition, pause/resume events, completion time, and the feedback submitted. This data validates our command-to-ack timing and captures user feedback per run."

### [1:55-2:00] Conclusion

> "This MWS de-risks our assumption that phone-based control is sufficient. The clean interface, always-available stop button, and real-time status updates provide transparency without needing voice commands or extra hardware. Next step: integrate with the real robot executor."

---

## Key Points to Hit for Rubric

✅ **Functional** — Does something real (live state transitions, command execution)  
✅ **Measurable** — Produces data (session log with timestamps, feedback)  
✅ **Isolated** — Can be tested independently (mock backend simulates robot)  
✅ **Foundational** — Unblocks future work (UI ready for real robot integration)  
✅ **Validates Assumption** — "Simple phone UI provides clear control + transparency"

---

## Quick Commands Reference

```bash
# Start backend (port 8000)
cd backend && python main.py

# Start frontend (port 5173)
cd ui && npm run dev

# View session log
curl http://localhost:8000/api/log | jq

# Test API directly
curl -X POST http://localhost:8000/api/start
curl -X POST http://localhost:8000/api/pause
curl -X POST http://localhost:8000/api/resume
curl -X POST http://localhost:8000/api/stop
```
