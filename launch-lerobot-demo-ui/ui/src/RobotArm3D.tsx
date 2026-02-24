/**
 * RobotArm3D — Real-time 3D visualization of SO-100/101 robot arm
 *
 * Uses joint angle data from /api/joints (written by eval_act_safe.py)
 * and renders a simplified kinematic model via React Three Fiber.
 *
 * Kinematic chain (6-DOF + gripper):
 *   Base → shoulder_pan(Y) → shoulder_lift(X) → upper_arm
 *        → elbow_flex(X) → forearm → wrist_flex(X) → wrist
 *        → wrist_roll(Z) → gripper
 *
 * All dimensions approximate the SO-101 physical arm.
 * Joint values: RANGE_M100_100 (-100..+100) for body, RANGE_0_100 (0..100) for gripper.
 */

import { useRef, useEffect, useState, type FC } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, Environment } from '@react-three/drei'
import * as THREE from 'three'

/* ================================================================
   Constants — SO-100/101 approximate dimensions (meters)
   ================================================================ */

const ARM = {
  BASE_RADIUS: 0.030,
  BASE_HEIGHT: 0.055,

  // Offset from top of base to shoulder pivot
  SHOULDER_UP: 0.040,
  SHOULDER_FWD: 0.000,

  UPPER_ARM_LEN: 0.1046,
  FOREARM_LEN: 0.1046,
  WRIST_LEN: 0.042,

  GRIPPER_BASE: 0.020,
  GRIPPER_FINGER_LEN: 0.042,
  GRIPPER_FINGER_THICK: 0.008,
  GRIPPER_MAX_GAP: 0.032,   // full-open distance between fingers

  LINK_RADIUS: 0.012,
  JOINT_RADIUS: 0.016,
} as const

/* ================================================================
   Joint angle mapping
   ================================================================ */

interface JointAngles {
  pan: number       // shoulder_pan   → rotation around Y
  lift: number      // shoulder_lift  → rotation around X
  elbow: number     // elbow_flex     → rotation around X
  wristFlex: number // wrist_flex     → rotation around X
  wristRoll: number // wrist_roll     → rotation around Z (along arm axis)
  gripper: number   // gripper        → 0 (closed) to 1 (open)
}

function mapJoints(raw: Record<string, number>): JointAngles {
  const get = (key: string, fallback = 0) => raw[key] ?? fallback

  // Convert normalized (-100..100) to radians
  // shoulder_pan: ±180° range → ±π
  const pan = -get('shoulder_pan.pos') / 100 * Math.PI

  // shoulder_lift: when val=-100 arm is vertical (up), val=0 arm horizontal, val=100 arm down
  // We map: angle = π/2 + val/100 * π/2
  // At val=-100 → angle=0 (pointing up from pivot)
  // At val=0 → angle=π/2 (horizontal)
  // At val=100 → angle=π (pointing down)
  const lift = Math.PI / 2 + get('shoulder_lift.pos') / 100 * (Math.PI / 2)

  // elbow_flex & wrist_flex: ±90° range
  const elbow = -get('elbow_flex.pos') / 100 * (Math.PI / 2)
  const wristFlex = -get('wrist_flex.pos') / 100 * (Math.PI / 2)

  // wrist_roll: ±180°
  const wristRoll = get('wrist_roll.pos') / 100 * Math.PI

  // gripper: 0..100 → 0..1
  const gripper = Math.max(0, Math.min(1, get('gripper.pos', 50) / 100))

  return { pan, lift, elbow, wristFlex, wristRoll, gripper }
}

/* ================================================================
   Materials (shared, created once)
   ================================================================ */

const baseMat = new THREE.MeshStandardMaterial({ color: '#3a3a3a', metalness: 0.7, roughness: 0.3 })
const jointMat = new THREE.MeshStandardMaterial({ color: '#1e88e5', metalness: 0.6, roughness: 0.25 })
const upperArmMat = new THREE.MeshStandardMaterial({ color: '#546e7a', metalness: 0.5, roughness: 0.3 })
const forearmMat = new THREE.MeshStandardMaterial({ color: '#78909c', metalness: 0.5, roughness: 0.3 })
const wristMat = new THREE.MeshStandardMaterial({ color: '#90a4ae', metalness: 0.5, roughness: 0.3 })
const gripperMat = new THREE.MeshStandardMaterial({ color: '#ff7043', metalness: 0.4, roughness: 0.4 })
const eeMarkerMat = new THREE.MeshStandardMaterial({ color: '#ffd740', metalness: 0.3, roughness: 0.5, emissive: '#ffd740', emissiveIntensity: 0.3 })

/* ================================================================
   Link component — cylinder between two points (along local +Y)
   ================================================================ */

const Link: FC<{ length: number; radius: number; material: THREE.Material }> = ({ length, radius, material }) => (
  <mesh position={[0, length / 2, 0]} material={material}>
    <cylinderGeometry args={[radius, radius, length, 16]} />
  </mesh>
)

const Joint: FC<{ material?: THREE.Material }> = ({ material = jointMat }) => (
  <mesh material={material}>
    <sphereGeometry args={[ARM.JOINT_RADIUS, 16, 16]} />
  </mesh>
)

/* ================================================================
   Gripper (two-finger parallel jaw)
   ================================================================ */

const GripperAssembly: FC<{ openness: number }> = ({ openness }) => {
  const halfGap = (openness * ARM.GRIPPER_MAX_GAP) / 2

  return (
    <group>
      {/* Gripper base */}
      <mesh position={[0, ARM.GRIPPER_BASE / 2, 0]} material={gripperMat}>
        <boxGeometry args={[ARM.GRIPPER_MAX_GAP + ARM.GRIPPER_FINGER_THICK * 2, ARM.GRIPPER_BASE, 0.015]} />
      </mesh>
      {/* Left finger */}
      <mesh position={[-halfGap - ARM.GRIPPER_FINGER_THICK / 2, ARM.GRIPPER_BASE + ARM.GRIPPER_FINGER_LEN / 2, 0]} material={gripperMat}>
        <boxGeometry args={[ARM.GRIPPER_FINGER_THICK, ARM.GRIPPER_FINGER_LEN, 0.012]} />
      </mesh>
      {/* Right finger */}
      <mesh position={[halfGap + ARM.GRIPPER_FINGER_THICK / 2, ARM.GRIPPER_BASE + ARM.GRIPPER_FINGER_LEN / 2, 0]} material={gripperMat}>
        <boxGeometry args={[ARM.GRIPPER_FINGER_THICK, ARM.GRIPPER_FINGER_LEN, 0.012]} />
      </mesh>
      {/* End-effector marker (tip between fingers) */}
      <mesh position={[0, ARM.GRIPPER_BASE + ARM.GRIPPER_FINGER_LEN, 0]} material={eeMarkerMat}>
        <sphereGeometry args={[0.006, 12, 12]} />
      </mesh>
    </group>
  )
}

/* ================================================================
   RobotArm — the full kinematic chain with smooth interpolation
   ================================================================ */

const RobotArm: FC<{ joints: JointAngles }> = ({ joints }) => {
  // Refs for smooth interpolation
  const panRef = useRef<THREE.Group>(null!)
  const liftRef = useRef<THREE.Group>(null!)
  const elbowRef = useRef<THREE.Group>(null!)
  const wristFlexRef = useRef<THREE.Group>(null!)
  const wristRollRef = useRef<THREE.Group>(null!)

  // Current interpolated values
  const current = useRef({ ...joints })

  useFrame((_state, delta) => {
    // Smooth lerp toward target
    const speed = 8 * delta
    const c = current.current
    c.pan += (joints.pan - c.pan) * speed
    c.lift += (joints.lift - c.lift) * speed
    c.elbow += (joints.elbow - c.elbow) * speed
    c.wristFlex += (joints.wristFlex - c.wristFlex) * speed
    c.wristRoll += (joints.wristRoll - c.wristRoll) * speed
    c.gripper += (joints.gripper - c.gripper) * speed

    if (panRef.current) panRef.current.rotation.y = c.pan
    if (liftRef.current) liftRef.current.rotation.x = c.lift
    if (elbowRef.current) elbowRef.current.rotation.x = c.elbow
    if (wristFlexRef.current) wristFlexRef.current.rotation.x = c.wristFlex
    if (wristRollRef.current) wristRollRef.current.rotation.z = c.wristRoll
  })

  return (
    <group>
      {/* ── Base (sits on ground plane) ── */}
      <mesh position={[0, ARM.BASE_HEIGHT / 2, 0]} material={baseMat}>
        <cylinderGeometry args={[ARM.BASE_RADIUS, ARM.BASE_RADIUS * 1.3, ARM.BASE_HEIGHT, 24]} />
      </mesh>

      {/* ── Shoulder Pan (rotates around Y) ── */}
      <group position={[0, ARM.BASE_HEIGHT + ARM.SHOULDER_UP, ARM.SHOULDER_FWD]}>
        <Joint />
        <group ref={panRef}>
          {/* ── Shoulder Lift (rotates arm pitch) ── */}
          <group ref={liftRef}>
            {/* Upper arm link */}
            <Link length={ARM.UPPER_ARM_LEN} radius={ARM.LINK_RADIUS} material={upperArmMat} />

            {/* ── Elbow joint ── */}
            <group position={[0, ARM.UPPER_ARM_LEN, 0]}>
              <Joint />
              <group ref={elbowRef}>
                {/* Forearm link */}
                <Link length={ARM.FOREARM_LEN} radius={ARM.LINK_RADIUS * 0.9} material={forearmMat} />

                {/* ── Wrist Flex joint ── */}
                <group position={[0, ARM.FOREARM_LEN, 0]}>
                  <Joint />
                  <group ref={wristFlexRef}>
                    {/* Wrist link */}
                    <Link length={ARM.WRIST_LEN} radius={ARM.LINK_RADIUS * 0.75} material={wristMat} />

                    {/* ── Wrist Roll joint ── */}
                    <group position={[0, ARM.WRIST_LEN, 0]}>
                      <Joint material={wristMat} />
                      <group ref={wristRollRef}>
                        {/* Gripper */}
                        <GripperAssembly openness={joints.gripper} />
                      </group>
                    </group>
                  </group>
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  )
}

/* ================================================================
   useJointData — fetch joint data from backend
   ================================================================ */

function useJointData(active: boolean, intervalMs = 100) {
  const [joints, setJoints] = useState<Record<string, number>>({})
  const [hasData, setHasData] = useState(false)

  useEffect(() => {
    if (!active) return
    let cancelled = false

    const poll = async () => {
      if (cancelled) return
      try {
        const r = await fetch('/api/joints')
        if (r.ok) {
          const data = await r.json()
          if (!cancelled && data.joints && Object.keys(data.joints).length > 0) {
            setJoints(data.joints)
            setHasData(true)
          }
        }
      } catch { /* ignore */ }
    }

    poll()
    const id = setInterval(poll, intervalMs)
    return () => { cancelled = true; clearInterval(id) }
  }, [active, intervalMs])

  return { joints, hasData }
}

/* ================================================================
   RobotArm3DViewer — complete scene with controls
   ================================================================ */

export const RobotArm3DViewer: FC<{ active: boolean }> = ({ active }) => {
  const { joints, hasData } = useJointData(active, 100) // ~10 fps
  const mapped = mapJoints(joints)

  return (
    <div className="w-full max-w-2xl mb-6 md:mb-8">
      <h3 className="text-sm md:text-base uppercase tracking-wider text-gray-500 font-medium mb-3 text-center">
        3D Robot Arm
      </h3>
      <div className="relative rounded-xl overflow-hidden bg-gray-900/80 border border-gray-700/50"
           style={{ height: 'min(50vw, 360px)' }}>
        {/* Placeholder when no data */}
        {!hasData && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm z-10">
            🦾 Waiting for joint data…
          </div>
        )}

        <Canvas
          camera={{ position: [0.25, 0.25, 0.35], fov: 45, near: 0.001, far: 10 }}
          gl={{ antialias: true, alpha: true }}
          style={{ background: 'transparent' }}
        >
          {/* Lighting */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[2, 4, 3]} intensity={1.2} castShadow />
          <directionalLight position={[-2, 2, -1]} intensity={0.4} />
          <Environment preset="city" background={false} />

          {/* Ground grid */}
          <Grid
            args={[1, 1]}
            cellSize={0.02}
            cellThickness={0.5}
            cellColor="#334155"
            sectionSize={0.1}
            sectionThickness={1}
            sectionColor="#475569"
            fadeDistance={0.8}
            fadeStrength={1}
            infiniteGrid
            position={[0, 0, 0]}
          />

          {/* Ground plane (subtle) */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.001, 0]} receiveShadow>
            <planeGeometry args={[2, 2]} />
            <meshStandardMaterial color="#1a1a2e" transparent opacity={0.3} />
          </mesh>

          {/* Robot arm */}
          <RobotArm joints={mapped} />

          {/* World axes helper (small, at origin) */}
          <axesHelper args={[0.05]} />

          {/* Camera controls */}
          <OrbitControls
            makeDefault
            target={[0, 0.15, 0]}
            minDistance={0.15}
            maxDistance={1.0}
            enablePan={false}
            maxPolarAngle={Math.PI * 0.85}
          />
        </Canvas>

        {/* Joint values overlay */}
        {hasData && (
          <div className="absolute bottom-2 left-2 right-2 flex flex-wrap gap-x-3 gap-y-1 px-2.5 py-1.5
                          bg-black/60 backdrop-blur-sm rounded-lg text-[10px] md:text-xs font-mono text-gray-300">
            {Object.entries(joints).map(([k, v]) => (
              <span key={k}>
                <span className="text-gray-500">{k.replace('.pos', '')}:</span>{' '}
                <span className={Math.abs(v as number) > 80 ? 'text-amber-400' : 'text-gray-200'}>
                  {(v as number).toFixed(1)}
                </span>
              </span>
            ))}
          </div>
        )}

        {/* 3D label */}
        <div className="absolute top-2 left-2 px-2.5 py-1 bg-black/60 backdrop-blur-sm rounded-lg text-xs md:text-sm font-mono uppercase tracking-wide text-gray-200">
          🦾 3D View
        </div>

        {/* Live indicator */}
        {hasData && (
          <div className="absolute top-2 right-2 flex items-center gap-1.5 px-2 py-1 bg-black/60 backdrop-blur-sm rounded-lg">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] md:text-xs font-semibold text-emerald-400 uppercase">Live</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default RobotArm3DViewer
