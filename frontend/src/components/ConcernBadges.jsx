const SEVERITY_COLOURS = {
  Clear: "bg-slate-100 text-slate-600",
  Mild: "bg-green-100 text-green-800",
  Moderate: "bg-yellow-100 text-yellow-800",
  Severe: "bg-red-100 text-red-800",
};

const CONCERN_DOT = {
  Acne: "text-blue-500",
  Redness: "text-red-500",
  "Dark spots": "text-yellow-500",
};

/** One concern card: name, severity pill, count/area, relative extent. */
function ConcernCard({ name, concern, compact }) {
  const count = typeof concern.count === "number" ? `${concern.count} regions` : "—";
  const isClear = concern.severity === "Clear";

  if (compact) {
    return (
      <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-slate-50 border border-surface-border">
        <span className={`concern-dot ${CONCERN_DOT[name]}`} aria-hidden="true" />
        <span className="text-sm font-medium text-slate-700 w-20 truncate">{name}</span>
        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${SEVERITY_COLOURS[concern.severity]}`}>
          {concern.severity}
        </span>
        <span className="ml-auto text-xs text-slate-500">
          {isClear ? "Clear" : `${concern.area_pct}%`}
        </span>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-surface-border bg-slate-50/60 p-4">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 font-medium text-slate-700">
          <span className={`concern-dot ${CONCERN_DOT[name]}`} aria-hidden="true" />
          {name}
        </span>
        <span className={`rounded-full px-3 py-1 text-sm font-medium ${SEVERITY_COLOURS[concern.severity]}`}>
          {concern.severity}
        </span>
      </div>
      <p className="mt-2 text-sm text-slate-500">
        {isClear ? "No regions detected by this algorithm" : `${count} · ${concern.area_pct}% of skin area`}
      </p>
      <p className="text-sm text-slate-400">
        Relative extent: {Math.round((concern.extent_score || 0) * 100)}%
      </p>
    </div>
  );
}

/**
 * Severity pills for acne-like patterns, redness and dark spots.
 * @param {object} concerns - { acne, redness, blemishes }
 * @param {boolean} compact - compact inline style for history
 */
function ConcernBadges({ concerns, compact }) {
  const items = [
    { name: "Acne", concern: concerns.acne },
    { name: "Redness", concern: concerns.redness },
    { name: "Dark spots", concern: concerns.blemishes },
  ];

  if (compact) {
    return (
      <div className="flex flex-col gap-2">
        {items.map((item) => (
          <ConcernCard key={item.name} {...item} compact />
        ))}
      </div>
    );
  }

  return (
    <section className="rounded-xl border border-surface-border bg-white p-6">
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <h2 className="text-lg font-semibold text-slate-800">Visual summary</h2>
        <p className="text-xs text-slate-400">Area is relative to the analysed skin region</p>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {items.map((item) => (
          <ConcernCard key={item.name} {...item} />
        ))}
      </div>
    </section>
  );
}

export default ConcernBadges;
