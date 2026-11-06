import ConcernBadges from "./ConcernBadges.jsx";
import Icon from "./Icon.jsx";

function ProgressTrend({ history, onDelete }) {
  const formatDate = (value) => new Date(`${value}Z`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  return <section className="history-panel" aria-live="polite">
    <div className="section-heading"><div><p className="eyebrow">Your account</p><h2>Past reviews</h2></div><span>{history.length} saved</span></div>
    {history.length === 0 ? <div className="history-empty"><strong>No past reviews</strong><p>Your completed reviews will appear here. Original photos are never saved.</p></div> : <div className="history-list">
      {history.map((scan) => <article className="history-record" key={scan.id}>
        <div className="record-top"><div><strong>{scan.scan_type === "multi_angle" ? "Guided scan" : "Single photo review"}</strong><span>{formatDate(scan.created_at)}</span></div><button type="button" className="icon-button" title="Delete this record" onClick={() => onDelete(scan.id)}><Icon name="trash" size={17} /></button></div>
        <ConcernBadges concerns={scan.concerns} compact />
      </article>)}
    </div>}
  </section>;
}

export default ProgressTrend;
