import { useEffect, useRef, useState} from "react";

// COCO 17 keypoint skeleton connections
const SKELETON_CONNECTIONS: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 11], [6, 12], [11, 12],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

function getLimbColor(limbIdx: number): string {
  if (limbIdx < 4) return "#00ff88";
  if (limbIdx < 8) return "#00ccff";
  if (limbIdx < 10) return "#ff6b6b";
  if (limbIdx < 12) return "#ffa94d";
  if (limbIdx < 14) return "#cc5de8";
  return "#5c7cfa";
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
type PoseResult = {
  keypoints: Keypoint[];
  pose: string;
  score: number;
  box: [number, number, number, number];
};

type Props = {
  cameraName: string;
  containerRef: React.RefObject<HTMLDivElement | null>;
  enabled: boolean;
  currentTime: number;
};


const MIN_KP_CONF = 0.3;

export default function RecordingPoseOverlay({
  cameraName,
  containerRef,
  enabled,
  currentTime,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [poses, setPoses] = useState<PoseResult[]>([]);
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const lastFetchedTime = useRef<number>(0);
  const fetchInFlight = useRef(false);

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

  // Poll API for pose data at current playback time
  useEffect(() => {
    if (!enabled || !cameraName || !currentTime) return;

    // Don't re-fetch if time hasn't moved enough
    if (Math.abs(currentTime - lastFetchedTime.current) < 0.3) return;
    if (fetchInFlight.current) return;

    fetchInFlight.current = true;
    lastFetchedTime.current = currentTime;

    fetch(`api/${cameraName}/pose/${currentTime}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.success && data.poses) {
          setPoses(data.poses);
        } else {
          setPoses([]);
        }
      })
      .catch(() => setPoses([]))
      .finally(() => {
        fetchInFlight.current = false;
      });
  }, [enabled, cameraName, currentTime]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { w, h } = dims;
    if (w === 0 || h === 0) return;

    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    if (!enabled || poses.length === 0) return;

    for (const pose of poses) {
      const kps = pose.keypoints;
      if (!kps || kps.length < 17) continue;

      // Draw skeleton lines
      SKELETON_CONNECTIONS.forEach(([i, j], idx) => {
        const a = kps[i];
        const b = kps[j];
        if (!a || !b || a.confidence < MIN_KP_CONF || b.confidence < MIN_KP_CONF) return;

        ctx.beginPath();
        ctx.moveTo(a.x * w, a.y * h);
        ctx.lineTo(b.x * w, b.y * h);
        ctx.strokeStyle = getLimbColor(idx);
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.stroke();
      });

      // Draw keypoints
      kps.forEach((kp: Keypoint, idx: number) => {
        if (kp.confidence < MIN_KP_CONF) return;
        const x = kp.x * w;
        const y = kp.y * h;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = getKeypointColor(idx);
        ctx.globalAlpha = 0.4;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.globalAlpha = 1;
        ctx.fill();
      });

      // Pose label
      if (pose.pose && pose.pose !== "unknown") {
        const cx = ((pose.box[1] + pose.box[3]) / 2) * w;
        const top = pose.box[0] * h;
        const label = `${pose.pose} ${Math.round(pose.score * 100)}%`;

        ctx.globalAlpha = 1;
        ctx.font = "bold 13px Inter, system-ui, sans-serif";
        const tw = ctx.measureText(label).width;

        ctx.fillStyle = "rgba(0,0,0,0.75)";
        ctx.beginPath();
        ctx.roundRect(cx - tw / 2 - 8, top - 26, tw + 16, 22, 6);
        ctx.fill();

        ctx.fillStyle = "#00ff88";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, cx, top - 15);
      }
    }

    ctx.globalAlpha = 1;
  }, [enabled, dims, poses]);

  if (!enabled) return null;

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 z-20"
      style={{ width: "100%", height: "100%" }}
    />
  );
}
