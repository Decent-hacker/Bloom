import { useState } from "react";

const INITIAL = { skin_type: "unsure", sensitive: "unsure", climate: "humid", goal: "balanced" };

function RoutineQuestionnaire({ onSubmit, onCancel, busy }) {
  const [answers, setAnswers] = useState(INITIAL);
  const choose = (field, value) => setAnswers((previous) => ({ ...previous, [field]: value }));
  const Choice = ({ field, value, children }) => (
    <button type="button" aria-pressed={answers[field] === value} className={`routine-choice ${answers[field] === value ? "selected" : ""}`} onClick={() => choose(field, value)}>{children}</button>
  );

  return <section className="routine-questionnaire" aria-labelledby="routine-questions-title">
    <div className="section-heading"><div><p className="eyebrow">Personalise your routine</p><h2 id="routine-questions-title">A few questions first</h2></div><button type="button" className="quiet-button" onClick={onCancel}>Close</button></div>
    <p className="routine-lead">Your answers guide the routine. Bloom does not decide your skin type from the photo.</p>
    <div className="question-block"><h3>How does your skin usually feel a few hours after washing?</h3><div className="routine-options"><Choice field="skin_type" value="oily">Shiny or oily</Choice><Choice field="skin_type" value="dry">Tight or flaky</Choice><Choice field="skin_type" value="combination">Oily in some areas, dry in others</Choice><Choice field="skin_type" value="normal">Mostly comfortable</Choice><Choice field="skin_type" value="unsure">Not sure</Choice></div></div>
    <div className="question-block"><h3>Do products often sting, burn or cause redness?</h3><div className="routine-options"><Choice field="sensitive" value="yes">Often</Choice><Choice field="sensitive" value="sometimes">Sometimes</Choice><Choice field="sensitive" value="no">Rarely</Choice><Choice field="sensitive" value="unsure">Not sure</Choice></div></div>
    <div className="question-block"><h3>What is the weather usually like where you live?</h3><div className="routine-options"><Choice field="climate" value="humid">Hot and humid</Choice><Choice field="climate" value="dry">Hot and dry</Choice><Choice field="climate" value="mixed">Changes by season</Choice></div></div>
    <div className="question-block"><h3>What would you like to focus on?</h3><div className="routine-options"><Choice field="goal" value="acne">Breakouts</Choice><Choice field="goal" value="marks">Dark marks</Choice><Choice field="goal" value="barrier">Calm, comfortable skin</Choice><Choice field="goal" value="balanced">A simple everyday routine</Choice></div></div>
    <div className="routine-actions"><button type="button" className="primary-button" disabled={busy} onClick={() => onSubmit(answers)}>{busy ? "Building your routine" : "Show my routine"}</button></div>
  </section>;
}

export default RoutineQuestionnaire;
