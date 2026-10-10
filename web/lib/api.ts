export interface ClassScore {
  key: string;
  label: string;
  score: number;
}

export interface Prediction {
  predicted_class: string;
  display_label: string;
  confidence: number;
  probs_verified: boolean;
  scores: ClassScore[];
  margin_pp: number;
  image_warnings: string[];
}

const BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

async function asError(res: Response, fallback: string): Promise<Error> {
  try {
    const data = await res.json();
    if (data && typeof data.detail === "string") return new Error(data.detail);
  } catch {
    /* fall through */
  }
  return new Error(`${fallback} (HTTP ${res.status})`);
}

export async function predictImage(file: File): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file, file.name);
  let res: Response;
  try {
    res = await fetch(`${BASE}/predict`, { method: "POST", body: form });
  } catch {
    throw new Error(
      "Could not reach the analysis server. Make sure the backend is running (uvicorn server/main.py on :8000) and try again."
    );
  }
  if (!res.ok) throw await asError(res, "Analysis failed");
  return (await res.json()) as Prediction;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`);
    if (!res.ok) return false;
    const data = await res.json();
    return data?.status === "ok" && data?.model_loaded === true;
  } catch {
    return false;
  }
}
