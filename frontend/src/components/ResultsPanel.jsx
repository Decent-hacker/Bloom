import { useState } from "react";
import { Info, Sun, RotateCcw } from "lucide-react";
import ConcernBadges from "./ConcernBadges.jsx";
import ProgressTrend from "./ProgressTrend.jsx";
import RoutineCard from "./RoutineCard.jsx";
import RoutineQuestionnaire from "./RoutineQuestionnaire.jsx";
import { apiRequest } from "../api.js";
import ProcessingGrid from "./ProcessingGrid.jsx";

const dataUri = (value) => value ? `data:image/jpeg;base64,${value}` : null;
const views = [["original", "Original"], ["enhanced", "Enhanced"], ["imperfection_heatmap", "FFT detail"], ["imperfection_overlay", "Surface regions"], ["redness_map", "Redness map"]];

function FaceZones() {
  return <div className="bloom-zones"><svg viewBox="0 0 160 190" role="img" aria-label="Face diagram highlighting forehead, cheeks and chin">
    <path d="M80 10C42 10 23 41 27 91c2 42 22 75 53 88 31-13 51-46 53-88C137 41 118 10 80 10Z" fill="none" stroke="currentColor" strokeWidth="2" />
    <ellipse cx="80" cy="42" rx="30" ry="14" fill="var(--sage-soft)" /><ellipse cx="49" cy="101" rx="15" ry="25" fill="var(--sage-soft)" /><ellipse cx="111" cy="101" rx="15" ry="25" fill="var(--sage-soft)" /><ellipse cx="80" cy="150" rx="19" ry="11" fill="var(--sage-soft)" />
    <path d="M52 75h17m22 0h17M61 146c12 7 26 7 38 0" fill="none" stroke="currentColor" strokeWidth="1.5" />
  </svg><div><h3>Analysis zones</h3><p>Forehead, cheeks and chin. Eyes, lips, brows and nose are excluded from the surface review.</p></div></div>;
}

function AppearanceIndex({ tone }) {
  if (!tone || tone.ita_angle == null) return null;
  const position = Math.max(0, Math.min(100, ((55 - tone.ita_angle) / 85) * 100));
  return <section className="bloom-appearance"><div className="bloom-section-heading"><Sun size={19} /><div><h3>Appearance Index (ITA)</h3><p title="Individual Typology Angle, calculated from LAB color.">A color descriptor from the analyzed image area.</p></div></div><div className="bloom-ita-scale"><span className="bloom-ita-marker" style={{ left: `${position}%` }} /></div><div className="bloom-ita-ends"><span>Very light</span><strong>{tone.appearance_band} · {tone.ita_angle}°</strong><span>Dark</span></div><small>Appearance descriptor only. Lighting and camera settings affect this reading.</small></section>;
}

export default function ResultsPanel({ result, history = [], onDeleteHistory, onReset }) {
  const [view, setView] = useState("original");
  const [angle, setAngle] = useState("front");
  const [routine, setRoutine] = useState(null);
  const [showQuestions, setShowQuestions] = useState(false);
  const [routineBusy, setRoutineBusy] = useState(false);
  const [routineError, setRoutineError] = useState("");
  const requestRoutine = async (profile) => {
    setRoutineBusy(true);
    setRoutineError("");
    const response = await apiRequest("/api/routine", { method: "POST", body: JSON.stringify({ concerns: result.concerns, profile }) });
    setRoutineBusy(false);
    if (response.status === "error") { setRoutineError(response.message || "Could not build the routine. Try again."); return; }
    setRoutine(response);
    setShowQuestions(false);
  };
  const angleResults = result.angle_results || [];
  const shownAngle = angleResults.find((entry) => entry.angle === angle) || angleResults[0];
  const images = shownAngle?.images || result.images;
  const active = images[view] ? view : "original";
  return <div className="bloom-results">
    <div className="bloom-results-intro"><div><span className="bloom-kicker">YOUR PHOTO REVIEW</span><h1>What we can see.</h1><p>A visual reading of this image, with the photo beside every measurement.</p></div><button type="button" className="bloom-text-button" onClick={onReset}><RotateCcw size={16} />Analyze another</button></div>
    {angleResults.length > 0 && <div className="bloom-angle-tabs" aria-label="Photo angle">{angleResults.map((entry) => <button key={entry.angle} type="button" aria-pressed={shownAngle?.angle === entry.angle} onClick={() => setAngle(entry.angle)}>{entry.angle}</button>)}</div>}
    <ProcessingGrid images={images} stageTimes={shownAngle?.stage_times_sec || result.stage_times_sec} regionCount={(shownAngle?.imperfection_map || result.imperfection_map)?.region_count || 0} />
    <div className="bloom-results-grid"><section className="bloom-image-panel"><div className="bloom-image-tabs" role="tablist" aria-label="Image views">{views.filter(([key]) => images[key]).map(([key, label]) => <button role="tab" aria-selected={active === key} key={key} type="button" onClick={() => setView(key)}>{label}</button>)}</div><div className="bloom-image-frame"><img src={dataUri(images[active])} alt={`${views.find(([key]) => key === active)?.[1]} view of the analyzed photo`} /></div><p className="bloom-image-note">Areas shown are algorithmic image patterns, not diagnoses.</p></section>
      <div className="bloom-results-side"><div className="bloom-side-heading"><span>Visible variation</span><span className="bloom-mono">{Number(result.processing_time_sec || 0).toFixed(1)}s</span></div><ConcernBadges concerns={result.concerns} /><AppearanceIndex tone={result.skin_tone} /><FaceZones />{result.imperfection_map && <p className="bloom-surface-note"><strong>{result.imperfection_map.region_count}</strong> surface-detail regions · <strong>{result.imperfection_map.coverage_percent}%</strong> coverage</p>}</div></div>
    <aside className="bloom-disclaimer"><Info size={20} strokeWidth={1.7} /><p>Bloom provides educational visual analysis only. This is not a medical diagnosis. Consult a dermatologist for skin health concerns.</p></aside>
    {!routine && !showQuestions && <section className="routine-invite"><div><span className="bloom-kicker">OPTIONAL NEXT STEP</span><h2>A routine shaped by your preferences.</h2><p>Answer a few questions. Your photo is only one part of the picture.</p></div><button type="button" className="primary-button" onClick={() => setShowQuestions(true)}>Explore a routine</button></section>}
    {showQuestions && <RoutineQuestionnaire onSubmit={requestRoutine} onCancel={() => setShowQuestions(false)} busy={routineBusy} />}
    {routineError && <p role="alert" className="form-error">{routineError}</p>}
    {routine && <RoutineCard routine={routine} onEdit={() => { setRoutine(null); setShowQuestions(true); }} />}
    <ProgressTrend history={history} onDelete={onDeleteHistory} />
  </div>;
}
