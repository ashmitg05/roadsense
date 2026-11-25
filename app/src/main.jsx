import { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8080";
const SAMPLES = ["sample1.jpg", "sample2.jpg", "sample3.jpg", "sample4.jpg", "sample5.jpg"];
const SAMPLE_VIDEO = "sample.mp4";
const LABELS = {
  longitudinal_crack: "Longitudinal crack",
  transverse_crack: "Transverse crack",
  alligator_crack: "Alligator crack",
  pothole: "Pothole",
};

async function postFile(path, file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(API + path, { method: "POST", body: fd });
  const body = await r.json();
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

function Counts({ counts }) {
  return (
    <ul className="counts">
      {Object.entries(counts).map(([k, v]) => (
        <li key={k} className={v ? "hit" : ""}>
          <span>{LABELS[k]}</span>
          <b>{v}</b>
        </li>
      ))}
    </ul>
  );
}

function App() {
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [image, setImage] = useState(null);
  const [video, setVideo] = useState(null);

  async function run(file) {
    setError("");
    setImage(null);
    setVideo(null);
    try {
      if (file.type.startsWith("video/")) {
        setBusy("Uploading video…");
        const { job_id } = await postFile("/detect/video", file);
        setBusy("Tracking damage across frames…");
        for (;;) {
          await new Promise((r) => setTimeout(r, 1500));
          const s = await (await fetch(`${API}/jobs/${job_id}`)).json();
          if (s.status === "error") throw new Error(s.error);
          if (s.status === "done") {
            setVideo({ ...s.result, url: `${API}/jobs/${job_id}/video` });
            break;
          }
        }
      } else {
        setBusy("Detecting…");
        setImage(await postFile("/detect/image", file));
      }
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy("");
    }
  }

  async function sample(name) {
    const blob = await (await fetch(`/samples/${name}`)).blob();
    run(new File([blob], name, { type: blob.type }));
  }

  return (
    <main>
      <header>
        <h1>RoadSense</h1>
        <p>Road damage detection for Indian roads: YOLOv8 → INT8 ONNX, running on a CPU.</p>
      </header>

      <section className="input">
        <label className="drop">
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm"
            onChange={(e) => e.target.files[0] && run(e.target.files[0])}
            disabled={!!busy}
          />
          Upload an image, or a video up to 30 s / 50 MB
        </label>
        <div className="samples">
          {SAMPLES.map((s) => (
            <button key={s} onClick={() => sample(s)} disabled={!!busy} aria-label={`Try ${s}`}>
              <img src={`/samples/${s}`} alt="" />
            </button>
          ))}
          <button className="vid" onClick={() => sample(SAMPLE_VIDEO)} disabled={!!busy}>
            ▶ Sample video
          </button>
        </div>
      </section>

      {busy && <p className="status" role="status">{busy}</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {image && (
        <section className="result">
          <img src={`data:image/jpeg;base64,${image.image_jpeg_base64}`} alt="Detections" />
          <aside>
            <Counts counts={image.counts} />
            <p className="meta">Inference {image.inference_ms} ms (CPU, INT8)</p>
          </aside>
        </section>
      )}
      {video && (
        <section className="result">
          <video src={video.url} controls autoPlay muted />
          <aside>
            <h2>Unique damage (tracked)</h2>
            <Counts counts={video.unique_counts} />
            <p className="meta">
              {video.processed_frames}/{video.frames} frames processed (every {video.every_nth}) ·{" "}
              {video.inference_fps} inference FPS
            </p>
          </aside>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
