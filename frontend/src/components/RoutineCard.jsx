import { useState } from "react";

function RoutineCard({ routine, onEdit }) {
  const [open, setOpen] = useState({ am: true, pm: true, avoid: false, lifestyle: true });
  if (!routine) return null;
  const toggle = (name) => setOpen((current) => ({ ...current, [name]: !current[name] }));
  const groups = [
    ["am", "Morning"], ["pm", "Evening"], ["avoid", "Keep in mind"], ["lifestyle", "Why this may fit"]
  ];

  return <section className="routine-result" aria-live="polite">
    <div className="section-heading"><div><p className="eyebrow">Your answers, your routine</p><h2>A simple starting point</h2></div><button type="button" className="quiet-button" onClick={onEdit}>Edit answers</button></div>
    <p className="routine-lead">{routine.profile_summary}</p>
    {groups.map(([key, title]) => {
      const items = routine[key] || [];
      if (!items.length) return null;
      return <section className="routine-group" key={key}>
        <button type="button" className="routine-group-toggle" aria-expanded={open[key]} onClick={() => toggle(key)}><span>{title}</span><span>{open[key] ? "Hide" : "Show"}</span></button>
        {open[key] && <ul>{items.map((item, index) => <li key={`${key}-${index}`}>
          {typeof item === "string" ? item : <><strong>{item.step}</strong>: {item.ingredient}{item.note && <small>{item.note}</small>}</>}
        </li>)}</ul>}
      </section>;
    })}
    {routine.sunscreen && <p className="routine-footnote">{routine.sunscreen}</p>}
    <p className="routine-disclaimer">{routine.disclaimer}</p>
  </section>;
}

export default RoutineCard;
