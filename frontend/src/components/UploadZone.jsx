import { useRef, useState } from "react";
import GuidedCameraCapture from "./GuidedCameraCapture.jsx";
import Icon from "./Icon.jsx";
import { Camera, UploadCloud, ArrowRight, ShieldCheck } from "lucide-react";

/** Two deliberate entry paths: a guided scan or one uploaded image. */
function UploadZone({ selectedFiles, previewUrls, onSelect, onValidateCapture, onClear, onAnalyseScan, onAnalyseSingle, fileError }) {
  const inputRef = useRef(null);
  const [mode, setMode] = useState("choice");
  const [scanReady, setScanReady] = useState(false);
  const [dragging, setDragging] = useState(false);
  const singleFile = selectedFiles.single;

  const reset = () => {
    onClear();
    setScanReady(false);
    setMode("choice");
  };

  const startScan = () => {
    onClear();
    setScanReady(false);
    setMode("scan");
  };

  const chooseSingle = () => inputRef.current?.click();
  const handleSingleFile = (event) => {
    const file = event.target.files?.[0];
    if (file) onSelect("single", file);
    event.target.value = "";
  };

  return (
    <div className="capture-shell">
      {mode === "choice" && (
        <section className="capture-landing">
          <div className="capture-intro">
            <p className="eyebrow">BLOOM / PHOTO REVIEW</p>
            <h1>A closer look,<br />at your pace.</h1>
            <p>See what your photo reveals about visible skin variation. Start with one clear image or capture three guided views.</p>
          </div>
          <div className="capture-options">
            <button type="button" onClick={startScan} className="capture-option capture-option-primary">
              <span className="capture-icon"><Camera size={25} strokeWidth={1.6} /></span>
              <span className="option-body"><strong>Guided camera scan</strong><small>A calm, step-by-step look from three angles.</small></span>
              <ArrowRight size={19} />
            </button>
            <button type="button" onClick={() => setMode("single")} className="capture-option">
              <span className="capture-icon"><UploadCloud size={25} strokeWidth={1.6} /></span>
              <span className="option-body"><strong>Upload one photo</strong><small>Choose a clear photo to begin.</small></span>
              <ArrowRight size={19} />
            </button>
          </div>
        </section>
      )}

      {mode === "scan" && !scanReady && (
        <GuidedCameraCapture
          onCapture={onSelect}
          onValidateCapture={onValidateCapture}
          onComplete={() => setScanReady(true)}
          onCancel={reset}
        />
      )}

      {mode === "scan" && scanReady && (
        <section className="scan-ready">
          <ShieldCheck size={30} />
          <h1>Your three scan angles are ready</h1>
          <p>They passed the capture checks and will be reviewed together.</p>
          <div className="button-row">
            <button type="button" onClick={startScan} className="quiet-button">Scan again</button>
            <button type="button" onClick={onAnalyseScan} className="primary-button">Analyse scan</button>
          </div>
        </section>
      )}

      {mode === "single" && (
        <section className="single-upload">
          <button type="button" onClick={reset} className="back-button">Back</button>
          <p className="eyebrow">Single photo review</p><h1>Upload one photo</h1>
          <p>Use an evenly lit, front-facing photo. Keep your face in frame.</p>
          {singleFile ? (
            <div className="photo-preview">
              <img src={previewUrls.single} alt="Selected photo preview" />
              <div><span>{singleFile.name}</span><div className="button-row"><button type="button" onClick={chooseSingle} className="quiet-button">Replace</button><button type="button" onClick={onAnalyseSingle} className="primary-button">Analyse photo</button></div></div>
            </div>
          ) : (
            <button type="button" onClick={chooseSingle} className={`drop-zone ${dragging ? "is-dragging" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files?.[0]; if (file) onSelect("single", file); }}>
              <Camera size={30} strokeWidth={1.5} /><strong>Drop a photo here, or browse</strong><span>JPG, PNG up to 5MB · Your photo is analyzed in memory and never stored</span>
            </button>
          )}
          <input ref={inputRef} type="file" accept="image/jpeg,image/png" className="hidden" onChange={handleSingleFile} />
        </section>
      )}

      {fileError && <p role="alert" className="form-error">{fileError}</p>}
    </div>
  );
}

export default UploadZone;
