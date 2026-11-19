// Typed client for the label server's JSON API. The shapes mirror the pyserde records of `posekit.labeler.schema`;
// boxes are in native image pixels and times are segment video times in ns (durations, so they fit a JS number).

export type HandSide = 'left' | 'right';
export const SIDES: readonly HandSide[] = ['left', 'right'];

export type TrackOrigin =
  | 'reviewed'
  | 'derived'
  | 'chained'
  | 'held'
  | 'detected'
  | 'tracked'
  | 'fused'
  | 'disagree'
  | 'sphere'
  | 'none';

export type LabelStatus = 'accepted' | 'corrected' | 'no_hand' | 'rejected';
export type LabelSource =
  'person' | 'rig' | 'egohand' | 'hands_layer' | 'agent' | 'range' | 'tracker';
export type SolveMode = 'triangulated' | 'proposals' | 'prior';

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface ServerConfig {
  reviewer: string;
  dataset: string;
}

export interface SegmentSummary {
  segment_id: string;
  split: string;
  duration_s: number;
  n_labels: number;
}

export interface CameraInfo {
  cam: number;
  name: string;
  width: number;
  height: number;
}

export interface SegmentInfo {
  segment_id: string;
  split: string;
  cams: CameraInfo[];
  times_ns: number[];
  viewer_url: string;
}

export interface TrackBox {
  cam: number;
  side: HandSide;
  box: Box | null;
  origin: TrackOrigin;
  score: number | null;
}

export interface TrackFrame {
  t_ns: number;
  boxes: TrackBox[];
}

export interface CamBox {
  cam: number;
  box: Box | null;
}

export interface SolveRequest {
  segment_id: string;
  t_ns: number;
  side: HandSide;
  labeled: { cam: number; box: Box }[];
}

export interface SolveResult {
  side: HandSide;
  mode: SolveMode;
  confidence: number;
  radius_m: number;
  labeled: number[];
  boxes: CamBox[];
}

export interface ViewLabel {
  cam: number;
  side: HandSide;
  status: LabelStatus;
  box: Box | null;
  source: LabelSource;
  reviewer: string;
}

export interface FrameLabels {
  segment_id: string;
  t_ns: number;
  labels: ViewLabel[];
}

export interface AcceptResult {
  accepted: number;
  kept: number;
}

export interface CommitResult {
  path: string;
  n_labels: number;
}

/** One reviewer action; only `t_ns` and `kind` are read (to mark the saved fixes on the timeline). */
export interface ActionEvent {
  t_ns: number;
  kind: string;
}

const enc = encodeURIComponent;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`api/${path}`, init);
  if (!response.ok) {
    const detail = (await response.text()).slice(0, 300);
    throw new Error(`${init?.method ?? 'GET'} /api/${path}: ${response.status} ${detail}`);
  }
  return (await response.json()) as T;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
}

export const api = {
  config: () => request<ServerConfig>('config'),
  segments: () => request<SegmentSummary[]>('segments'),
  segment: (sid: string) => request<SegmentInfo>(`segment/${enc(sid)}`),
  track: (sid: string, t0_ns: number, n: number) =>
    request<TrackFrame[]>(`track/${enc(sid)}?t0=${t0_ns}&n=${n}`),
  actions: (sid: string) => request<ActionEvent[]>(`actions/${enc(sid)}`),
  solve: (body: SolveRequest) => post<SolveResult>('solve', body),
  labels: (body: FrameLabels) => post<{ stored: number }>('labels', body),
  acceptRange: (body: { segment_id: string; t0_ns: number; t1_ns: number; reviewer: string }) =>
    post<AcceptResult>('accept_range', body),
  commit: (sid: string) => post<CommitResult>(`commit/${enc(sid)}`, {})
};

/** URL of one camera frame as JPEG, downscaled so its longer side is at most `maxSide`. */
export function frameUrl(sid: string, cam: number, t_ns: number, maxSide: number): string {
  return `api/frame/${enc(sid)}/${cam}/${t_ns}.jpg?max_side=${maxSide}`;
}
