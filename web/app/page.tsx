"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { checkHealth, predictImage, type Prediction } from "../lib/api";

const ALLOWED = ["jpg", "jpeg", "png"];
const MAX_MB = 15;
const ACCENTS = ["#ffcf9e", "#babfff", "#bfe0f7", "#e3a2ef"];

type Phase = "idle" | "ready" | "analyzing" | "done";

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i + 1).toLowerCase() : "";
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [meta, setMeta] = useState<string>("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [dragging, setDragging] = useState(false);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    checkHealth().then(setBackendUp);
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const acceptFile = useCallback(
    (f: File | undefined | null) => {
      setError(null);
      setResult(null);
      if (!f) return;
      const ext = extOf(f.name);
      if (!ALLOWED.includes(ext)) {
        setError(
          `Unsupported file type “.${ext || "?"}”. Please upload a JPG, JPEG, or PNG image.`
        );
        return;
      }
      if (f.size === 0) {
        setError("The selected file is empty. Please choose a valid image.");
        return;
      }
      if (f.size > MAX_MB * 1024 * 1024) {
        setError(
          `File is ${(f.size / 1024 / 1024).toFixed(1)} MB — please choose an image under ${MAX_MB} MB.`
        );
        return;
      }
      const img = new Image();
      const url = URL.createObjectURL(f);
      img.onload = () => {
        URL.revokeObjectURL(url);
        setFile(f);
        setPreview((old) => {
          if (old) URL.revokeObjectURL(old);
          return URL.createObjectURL(f);
        });
        setMeta(
          `${f.name} · ${(f.size / 1024).toFixed(0)} KB · ${img.width}×${img.height} px`
        );
        setPhase("ready");
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        setError(
          "This file could not be read as an image. It may be corrupt — please try a different file."
        );
      };
      img.src = url;
    },
    []
  );

  const removeFile = useCallback(() => {
    setFile(null);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return null;
    });
    setMeta("");
    setResult(null);
    setError(null);
    setPhase("idle");
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const analyze = useCallback(async () => {
    if (!file || phase === "analyzing") return;
    setPhase("analyzing");
    setError(null);
    setResult(null);
    try {
      const pred = await predictImage(file);
      setResult(pred);
      setPhase("done");
      requestAnimationFrame(() =>
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" })
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed.");
      setPhase("ready");
    }
  }, [file, phase]);

  return (
    <main className="page">
      <div className="small-cap">Eye / Sight · Retinal screening aid</div>
      <h1 className="hero-title">Retinal image classification</h1>
      <p className="lede">
        Upload a retinal fundus photo to classify it into one of seven
        categories — AMD, Cataract, Diabetic Retinopathy, Glaucoma,
        Hypertensive Retinopathy, Normal Fundus, or Pathological Myopia. JPG,
        JPEG, or PNG, up to {MAX_MB} MB.
      </p>
      <p className="caption" aria-live="polite">
        <span
          className={`status-dot ${backendUp === true ? "ok" : backendUp === false ? "down" : ""}`}
          aria-hidden="true"
        />
        {backendUp === null
          ? "Checking analysis server…"
          : backendUp
            ? "Analysis server connected."
            : "Analysis server not reachable — start the backend, then analyze."}
      </p>
      <hr className="hairline" />

      <div className="grid">
        {/* Upload */}
        <section className="card card-lavender" aria-labelledby="upload-h">
          <h2 id="upload-h">1 · Upload image</h2>
          <p className="muted">
            Drag and drop or browse. The file is sent to the local analysis
            server only when you press “Analyze image”.
          </p>

          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            role="button"
            tabIndex={0}
            aria-label="Upload a retinal fundus image (JPG, JPEG, PNG)"
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              acceptFile(e.dataTransfer.files?.[0]);
            }}
          >
            <p>
              <strong>Drop your image here</strong> or{" "}
              <span className="browse-link">browse files</span>
            </p>
            <p className="caption">JPG, JPEG, PNG · max {MAX_MB} MB</p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            className="file-input-hidden"
            aria-label="Choose a retinal fundus image"
            onChange={(e) => acceptFile(e.target.files?.[0])}
          />

          {error && !result && (
            <div className="alert alert-error" role="alert">
              {error}
            </div>
          )}

          {preview && file && (
            <div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className="preview" src={preview} alt={`Preview of ${file.name}`} />
              <p className="file-meta">
                <strong>{file.name}</strong> · {meta.split("·").slice(1).join("·")}
              </p>
              <div className="row">
                <button type="button" className="btn" onClick={removeFile}>
                  Remove image
                </button>
              </div>
            </div>
          )}
          {!preview && <p className="caption">No image selected yet.</p>}
        </section>

        {/* Analyze + results */}
        <section className="card card-peach" aria-labelledby="analyze-h">
          <h2 id="analyze-h">2 · Analyze</h2>
          <button
            type="button"
            className="btn btn-block"
            disabled={!file || phase === "analyzing"}
            onClick={analyze}
            aria-busy={phase === "analyzing"}
          >
            {phase === "analyzing" ? (
              <>
                <span className="spinner" aria-hidden="true" /> Running model inference…
              </>
            ) : (
              "Analyze image"
            )}
          </button>
          {!file && <p className="caption">Select a valid image to enable analysis.</p>}

          {error && phase !== "analyzing" && result === null && (
            <div className="alert alert-error" role="alert">
              {error}
            </div>
          )}

          {result && (
            <div ref={resultRef} aria-live="polite">
              <hr className="hairline" />
              <div className="small-cap">Prediction</div>
              <div className="result-label">{result.display_label}</div>
              <p className="result-conf">
                Predicted class: <strong>{result.predicted_class}</strong> ·{" "}
                {result.probs_verified ? "Confidence" : "Model score"}:{" "}
                <strong>{(result.confidence * 100).toFixed(1)}%</strong>
              </p>
              {!result.probs_verified && (
                <p className="caption">
                  Note: raw model scores did not sum to 1, so values are shown
                  as relative scores, not calibrated probabilities.
                </p>
              )}
              {result.margin_pp < 10 && (
                <div className="alert alert-warn" role="note">
                  Uncertain result — the top two classes differ by only{" "}
                  {result.margin_pp.toFixed(1)} percentage points (
                  {result.scores[0]?.key} vs {result.scores[1]?.key}). Treat
                  this as a weak signal, not a decision.
                </div>
              )}

              <div className="small-cap" style={{ marginTop: 14 }}>
                All classes, ranked
              </div>
              <div>
                {result.scores.map((s, rank) => {
                  const pct = s.score * 100;
                  const border = rank === 0 ? ACCENTS[rank % ACCENTS.length] : "#3a3a3e";
                  return (
                    <div className="score-row" key={s.key}>
                      <div className="score-top">
                        <span>
                          <strong>
                            {rank + 1}. {s.key}
                          </strong>
                          {rank === 0 ? " — predicted" : ""}
                        </span>
                        <span>{pct.toFixed(1)}%</span>
                      </div>
                      <div
                        className="score-track"
                        style={{ borderColor: border }}
                        role="img"
                        aria-label={`${s.key}: ${pct.toFixed(1)} percent`}
                      >
                        <div
                          className={`score-fill ${rank === 0 ? "top" : ""}`}
                          style={{ width: `${Math.max(pct, 1.5)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              <details className="details">
                <summary>Show scores as a table</summary>
                <table className="score-table">
                  <thead>
                    <tr>
                      <th scope="col">Rank</th>
                      <th scope="col">Class</th>
                      <th scope="col">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.scores.map((s, i) => (
                      <tr key={s.key}>
                        <td>{i + 1}</td>
                        <td>{s.key}</td>
                        <td>{(s.score * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>

              {result.image_warnings.map((w) => (
                <div className="alert alert-warn" role="note" key={w}>
                  {w}
                </div>
              ))}
            </div>
          )}

          {!result && phase !== "analyzing" && (
            <p className="caption">Results will appear here after analysis.</p>
          )}
        </section>
      </div>

      <hr className="hairline" />
      <p className="disclaimer" role="note">
        <strong>Not a medical diagnosis.</strong> This is an AI classification
        result for informational purposes only. Fundus interpretation requires
        a qualified eye-care professional — please do not make medical
        decisions from this output alone.
      </p>

      <details className="details">
        <summary>About the model &amp; classes</summary>
        <ul>
          <li>Model: local Keras student model, 7-way softmax output.</li>
          <li>Preprocessing: RGB → resize 224×224 → normalize 0–1 → batch dimension.</li>
          <li>
            Class order: AMD (0), Cataract (1), DR (2), Glaucoma (3),
            Hypertensive_Retinopathy (4), Normal_Fundus (5),
            Pathological_Myopia (6).
          </li>
          <li>
            Limitations: no explainability heatmaps; accuracy depends on image
            quality; poor lighting, blur, or non-fundus photos give unreliable
            scores.
          </li>
        </ul>
      </details>
    </main>
  );
}
