import { CameraOff, ImageOff, Lightbulb } from "lucide-react";

export default function ErrorView({ issues = [], isQuality, onReset }) {
  const message = issues?.[0] || "This photo could not be analyzed. Choose another image and try again.";
  const noFace = /face|landmark/i.test(message);
  const Icon = noFace ? CameraOff : ImageOff;
  return <section className="bloom-error" role="alert">
    <div className="bloom-error-figure" aria-hidden="true"><svg viewBox="0 0 160 160"><path d="M80 17c-35 0-55 26-55 62 0 38 23 64 55 69 32-5 55-31 55-69 0-36-20-62-55-62Z" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M53 74h14m26 0h14M64 108c11 7 21 7 32 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg></div>
    <span className="bloom-kicker">PHOTO NEEDS A RETAKE</span><h1>{noFace ? "We couldn't find a clear face." : isQuality ? "Let's try a clearer photo." : "This image didn't work."}</h1>
    <p className="bloom-error-message"><Icon size={18} />{message}</p>
    <div className="bloom-error-tip"><Lightbulb size={18} /><p>Face a window, keep your face centered, and hold your camera steady. Avoid filters and flash.</p></div>
    <button className="primary-button" type="button" onClick={onReset}>Try another photo</button>
  </section>;
}
