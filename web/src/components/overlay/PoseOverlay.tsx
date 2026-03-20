import { useEffect, useRef, useState } from "react";
import { useTrackedObjectUpdate } from "@/api/ws";

// COCO 17 keypoint skeleton connections (pairs of keypoint indices)
const SKELETON_CONNECTIONS: [number, number][] = [
  // Head
  [0, 1], [0, 2], [1, 3], [2, 4],
  // Torso
  [5, 6], [5, 11], [6, 12], [11, 12],
  // Left arm
  [5, 7], [7, 9],
  // Right arm
  [6, 8], [8, 10],
  // Left leg
  [11, 13], [13, 15],
  // Right leg
  [12, 14], [14, 16],
];

// Colors for different body parts
function getLimbColor(limbIdx: number): string {
  if (limbIdx < 4) return "#00ff88";   // head - green
  if (limbIdx < 8) return "#00ccff";   // torso - cyan
  if (limbIdx < 10) return "#ff6b6b";  // left arm - red
  if (limbIdx < 12) return "#ffa94d";  // right arm - orange
  if (limbIdx < 14) return "#cc5de8";  // left leg - purple
  return "#5c7cfa";                     // right leg - blue
}

function getKeypointColor(kpIdx: number): string {
  if (kpIdx < 5) return "#00ff88";
  if (kpIdx < 7) return "#00ccff";
  if (kpIdx < 9) return "#ff6b6b";
  if (kpIdx < 11) return "#ffa94d";
  if (kpIdx < 13) return "#cc5de8";
  return "#5c7cfa";
}

type Keypoint = { x: number; y: number; confidence: number };

type PoseEntry = {
  keypoints: Keypoint[];
  pose: string;
  score: number;
  box: [number, number, number, number];
  timestamp: number;
};

type PoseOverlayProps = {
  cameraName: string;
  containerRef: React.RefObject<HTMLDivElement | null>;
  enabled: boolean;
};

const MIN_KP_CONF = 0.3;
const POSE_EXPIRY_MS = 2000;

export default function PoseOverlay({
  cameraName,
  containerRef,
  enabled,
}: PoseOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const posesRef = useRef<Map<string, PoseEntry>>(new Map());
  const animFrameRef = useRef<number>(0);
  const [dims, setDims] = useState({ w: 0, h: 0 });

  // Track container size
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        setDims({ w: e.contentRect.width, h: e.contentRect.height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [containerRef]);

  // Listen for tracked_object_update with type=pose
  const { payload: wsUpdate } = useTrackedObjectUpdate();

  useEffect(() => {
    if (!enabled || !wsUpdate) return;
    if (wsUpdate.type !== "pose") return;
    if (wsUpdate.camera !== cameraName) return;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const data = wsUpdate as any;

    posesRef.current.set(data.id, {
      keypoints: data.keypoints || [],
      pose: data.pose || "",
      score: data.score || 0,
      box: data.box || [0, 0, 1, 1],
      timestamp: Date.now(),
    });
  }, [wsUpdate, cameraName, enabled]);

  // Render loop
  useEffect(() => {
    if (!enabled) {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        animFrameRef.current = requestAnimationFrame(render);
        return;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const { w, h } = dims;
      if (w === 0 || h === 0) {
        animFrameRef.current = requestAnimationFrame(render);
        return;
      }

      canvas.width = w;
      canvas.height = h;
      ctx.clearRect(0, 0, w, h);

      const now = Date.now();

      // Expire old poses
      for (const [id, p] of posesRef.current) {
        if (now - p.timestamp > POSE_EXPIRY_MS) posesRef.current.delete(id);
      }

      // Draw each pose
      for (const [, pose] of posesRef.current) {
        const kps = pose.keypoints;
        if (!kps || kps.length < 17) continue;

        const age = now - pose.timestamp;
        const alpha = Math.max(0.2, 1 - age / POSE_EXPIRY_MS);

        // Draw skeleton lines
        SKELETON_CONNECTIONS.forEach(([i, j], idx) => {
          const a = kps[i];
          const b = kps[j];
          if (!a || !b || a.confidence < MIN_KP_CONF || b.confidence < MIN_KP_CONF) return;

          ctx.beginPath();
          ctx.moveTo(a.x * w, a.y * h);
          ctx.lineTo(b.x * w, b.y * h);
          ctx.strokeStyle = getLimbColor(idx);
          ctx.globalAlpha = alpha * 0.85;
          ctx.lineWidth = 3;
          ctx.lineCap = "round";
          ctx.stroke();
        });

        // Draw keypoint dots
        kps.forEach((kp: Keypoint, idx: number) => {
          if (kp.confidence < MIN_KP_CONF) return;
          const x = kp.x * w;
          const y = kp.y * h;
          // Outer glow
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, Math.PI * 2);
          ctx.fillStyle = getKeypointColor(idx);
          ctx.globalAlpha = alpha * 0.4;
          ctx.fill();
          // Inner dot
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fillStyle = "#ffffff";
          ctx.globalAlpha = alpha;
          ctx.fill();
        });

        // Pose label
        if (pose.pose) {
          const cx = ((pose.box[1] + pose.box[3]) / 2) * w;
          const top = pose.box[0] * h;
          const label = `${pose.pose} ${Math.round(pose.score * 100)}%`;

          ctx.globalAlpha = alpha;
          ctx.font = "bold 13px Inter, system-ui, sans-serif";
          const tw = ctx.measureText(label).width;

          // Pill background
          ctx.fillStyle = "rgba(0,0,0,0.75)";
          ctx.beginPath();
          ctx.roundRect(cx - tw / 2 - 8, top - 26, tw + 16, 22, 6);
          ctx.fill();

          // Label text
          ctx.fillStyle = "#00ff88";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, cx, top - 15);
        }
      }

      ctx.globalAlpha = 1;
      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [enabled, dims]);

  if (!enabled) return null;

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 z-20"
      style={{ width: "100%", height: "100%" }}
    />
  );
}
