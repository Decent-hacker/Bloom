import { useCallback, useEffect, useRef, useState } from "react";

const REQUIRED_STEPS = [
  { id: "front", title: "Face forward", prompt: "Look directly at the camera" },
  { id: "left", title: "Turn left", prompt: "Turn 30–45° left and keep both eyes visible" },
  { id: "right", title: "Turn right", prompt: "Turn 30–45° right and keep both eyes visible" },
];

const CAPTURE_DELAY = 3;

/**
 * One uninterrupted camera session for the multi-angle scan. It guides the
 * person through each pose and captures the frame after a visible countdown.
 */
function GuidedCameraCapture({ onCapture, onValidateCapture, onComplete, onCancel }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const sideYawRef = useRef(null);
  const successTimerRef = useRef(null);
  const activeRef = useRef(true);
  const [stepIndex, setStepIndex] = useState(0);
  const [countdown, setCountdown] = useState(CAPTURE_DELAY);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [capturing, setCapturing] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const current = REQUIRED_STEPS[stepIndex];

  useEffect(() => {
    let active = true;
    activeRef.current = true;
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: "user" }, audio: false })
      .then((stream) => {
        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(() => setError("Could not access the camera. Check browser permission and try again."));

    return () => {
      active = false;
      activeRef.current = false;
      window.clearTimeout(successTimerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  const capture = useCallback(() => {
    const video = videoRef.current;
    if (!video?.videoWidth || !current || capturing) return;
    setCapturing(true);
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(async (blob) => {
      if (!blob) {
        if (activeRef.current) setCapturing(false);
        return;
      }
      const file = new File([blob], `${current.id}-angle.jpg`, { type: "image/jpeg" });
      let result;
      try {
        result = await onValidateCapture(file, current.id, current.id === "right" ? sideYawRef.current : null);
      } catch {
        result = { accepted: false, message: "We could not check this frame. Please try again." };
      }
      if (!activeRef.current) return;
      if (!result.accepted) {
        setFeedback({ type: "warning", message: result.message || "Adjust your position and try again." });
        setCountdown(CAPTURE_DELAY + 1);
        setCapturing(false);
        return;
      }

      if (current.id === "left") sideYawRef.current = result.yaw;
      onCapture(current.id, file);
      setFeedback({ type: "success", message: `${current.title} accepted` });
      successTimerRef.current = window.setTimeout(() => {
        if (!activeRef.current) return;
        if (stepIndex === REQUIRED_STEPS.length - 1) {
          onComplete();
        } else {
          setStepIndex((index) => index + 1);
          setCountdown(CAPTURE_DELAY);
        }
        setFeedback(null);
        setCapturing(false);
      }, 700);
    }, "image/jpeg", 0.92);
  }, [capturing, current, onCapture, onComplete, onValidateCapture, stepIndex]);

  useEffect(() => {
    if (!ready || error || capturing) return undefined;
    const timer = window.setTimeout(() => {
      if (countdown <= 1) capture();
      else setCountdown((value) => value - 1);
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [capture, capturing, countdown, error, ready]);

  return (
    <section className="mx-auto max-w-2xl overflow-hidden rounded-2xl border border-brand-200 bg-white shadow-sm" aria-live="polite">
      <div className="border-b border-brand-100 bg-brand-50 px-5 py-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Guided camera</p>
            <h2 className="mt-1 text-lg font-semibold text-slate-900">{current?.title}</h2>
          </div>
          <span className="rounded-full bg-white px-3 py-1 font-mono text-xs text-brand-700">
            {`${stepIndex + 1} / ${REQUIRED_STEPS.length}`}
          </span>
        </div>
        <p className="mt-1 text-sm text-slate-600">{current?.prompt}</p>
      </div>

      <div className="relative bg-black p-3">
        {error ? (
          <div className="flex aspect-video items-center justify-center bg-slate-900 p-6 text-center text-sm text-red-200">{error}</div>
        ) : (
          <>
            <video ref={videoRef} autoPlay playsInline muted onLoadedMetadata={() => setReady(true)} className="aspect-video w-full rounded-xl object-cover" />
            {ready && (
              <div className="absolute inset-3 flex items-center justify-center">
                <div className={`flex h-16 w-16 items-center justify-center rounded-full border-2 text-3xl font-semibold shadow-lg ${capturing ? "border-brand-200 bg-brand-500 text-white" : "border-white bg-black/35 text-white"}`}>
                  {capturing ? "…" : countdown}
                </div>
              </div>
            )}
            {feedback && (
              <div className={`absolute bottom-6 left-6 right-6 rounded-xl px-4 py-3 text-center text-sm font-medium shadow-lg ${feedback.type === "success" ? "bg-emerald-600 text-white" : "bg-amber-400 text-amber-950"}`}>
                {feedback.message}
              </div>
            )}
          </>
        )}
      </div>

      <div className="flex flex-wrap gap-3 p-4">
        <button type="button" disabled={!ready || !!error || capturing} onClick={capture} className="h-11 rounded-lg border border-surface-border px-4 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50">
          Capture now
        </button>
        <button type="button" onClick={onCancel} className="h-11 rounded-lg px-4 text-sm text-slate-500 hover:text-slate-800">Cancel session</button>
      </div>
    </section>
  );
}

export default GuidedCameraCapture;
