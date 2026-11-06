import { useEffect, useRef, useState } from "react";
import ComparisonGrid from "./ComparisonGrid.jsx";
import ConcernBadges from "./ConcernBadges.jsx";
import RoutineCard from "./RoutineCard.jsx";
import ProgressTrend from "./ProgressTrend.jsx";
import ResetButton from "./ResetButton.jsx";
import RoutineQuestionnaire from "./RoutineQuestionnaire.jsx";
import { apiRequest } from "../api.js";

/**
 * Composes the full results view from the API result.
 * @param {object} result - the success AnalysisResult
 * @param {Array} history - private records loaded for the signed-in account
 * @param {function():void} onReset
 */
function ResultsPanel({ result, history = [], onDeleteHistory, onReset }) {
  const [routine, setRoutine] = useState(null);
  const [showRoutineQuestions, setShowRoutineQuestions] = useState(false);
  const [routineBusy, setRoutineBusy] = useState(false);
  const [routineError, setRoutineError] = useState("");
  const [selectedAngle, setSelectedAngle] = useState("front");
  const cancelled = useRef(false);

  const angleResults = result.angle_results || [];
  const shownAngle = angleResults.find((angle) => angle.angle === selectedAngle) || angleResults[0];
  const displayImages = shownAngle?.images || result.images;

  useEffect(() => {
    cancelled.current = false;
    return () => { cancelled.current = true; };
  }, []);

  const requestRoutine = async (profile) => {
    setRoutineBusy(true);
    setRoutineError("");
    const response = await apiRequest("/api/routine", {
      method: "POST",
      body: JSON.stringify({ concerns: result.concerns, profile }),
    });
    setRoutineBusy(false);
    if (response.status === "error") {
      setRoutineError(response.message || "We could not build your routine. Please try again.");
      return;
    }
    setRoutine(response);
    setShowRoutineQuestions(false);
  };

  return (
    <div className="mt-8 space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Analysis complete</p>
            <h1 className="mt-1 text-xl font-semibold text-slate-900">Your skin-image review</h1>
          </div>
          <p className="rounded-full bg-slate-100 px-3 py-1.5 font-mono text-xs text-slate-500">
            {Number(result.processing_time_sec || 0).toFixed(1)}s
          </p>
        </div>
      </section>
      {angleResults.length > 0 && (
        <section className="rounded-2xl border border-surface-border bg-white p-4 shadow-sm">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div>
              <h2 className="font-semibold text-slate-800">Review each angle</h2>
              <p className="mt-0.5 text-xs text-slate-500">Choose a photo to inspect its own marked findings.</p>
            </div>
            <p className="text-xs text-slate-400">{result.aggregation_note}</p>
          </div>
          <div className="mt-4 flex gap-2 overflow-x-auto pb-1">
            {angleResults.map((angle) => (
              <button
                key={angle.angle}
                type="button"
                onClick={() => setSelectedAngle(angle.angle)}
                className={`flex shrink-0 items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium capitalize transition-colors ${
                  shownAngle?.angle === angle.angle
                    ? "border-brand-500 bg-brand-50 text-brand-700"
                    : "border-surface-border bg-white text-slate-600 hover:border-brand-300"
                }`}
              >
                <span className={`h-2 w-2 rounded-full ${shownAngle?.angle === angle.angle ? "bg-brand-500" : "bg-slate-300"}`} />
                {angle.angle}
              </button>
            ))}
          </div>
        </section>
      )}

      <ComparisonGrid images={displayImages} />

      <ConcernBadges concerns={result.concerns} />

      {result.ml && (
        <section className="rounded-xl border border-surface-border bg-white p-5">
          <h2 className="text-lg font-semibold text-slate-800">Acne ML detector</h2>
          {result.ml.available ? (
            <p className="mt-1 text-sm text-slate-600">
              {result.ml.detections.length} lesion{result.ml.detections.length === 1 ? "" : "s"} detected by the
              acne-only model. Results are visual estimates, not a diagnosis.
            </p>
          ) : (
            <p className="mt-1 text-sm text-slate-500">
              The optional acne model is not installed yet. Classical visual analysis is shown above.
            </p>
          )}
        </section>
      )}

      {!routine && !showRoutineQuestions && <section className="routine-invite">
        <div><p className="eyebrow">Optional next step</p><h2>Want a routine that fits you?</h2><p>Answer a few questions about your skin and climate. Your scan is only one input.</p></div>
        <button type="button" className="primary-button" onClick={() => setShowRoutineQuestions(true)}>Get a routine</button>
      </section>}
      {showRoutineQuestions && <RoutineQuestionnaire onSubmit={requestRoutine} onCancel={() => setShowRoutineQuestions(false)} busy={routineBusy} />}
      {routineError && <p className="form-error" role="alert">{routineError}</p>}
      {routine && <RoutineCard routine={routine} onEdit={() => { setRoutine(null); setShowRoutineQuestions(true); }} />}

      <ProgressTrend history={history} onDelete={onDeleteHistory} />

      <p className="rounded-lg border border-surface-border bg-white px-4 py-3 text-sm text-slate-500">
        Detection results only. Accuracy metrics are available in the offline evaluation
        workflow, where both pipelines are scored against independent ground truth.
      </p>

      <ResetButton onReset={onReset} />
    </div>
  );
}

export default ResultsPanel;
