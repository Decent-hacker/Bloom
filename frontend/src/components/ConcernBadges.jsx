import { Activity, Eye, Layers } from "lucide-react";

const cards = [
  { key: "surface_anomalies", legacy: "acne", title: "Surface detail", icon: Activity, description: "Small changes in fine surface texture." },
  { key: "redness_contrast", legacy: "redness", title: "Redness contrast", icon: Eye, description: "Areas with more red color than nearby skin." },
  { key: "dark_regions", legacy: "blemishes", title: "Dark regions", icon: Layers, description: "Areas darker than the surrounding skin." },
];

export default function ConcernBadges({ concerns = {}, compact = false }) {
  return <div className={`bloom-metrics ${compact ? "is-compact" : ""}`}>
    {cards.map(({ key, legacy, title, icon: Icon, description }) => {
      const item = concerns[key] || concerns[legacy];
      if (!item) return null;
      const extent = Math.max(0, Math.min(100, Math.round((item.extent_score || 0) * 100)));
      return <article className="bloom-metric" key={key}>
        <div className="bloom-metric-heading"><span className="bloom-metric-title"><Icon size={18} strokeWidth={1.7} />{title}</span><span className="bloom-band">{item.severity}</span></div>
        {!compact && <><p>{description}</p><div className="bloom-track" role="meter" aria-label={`${title} relative extent`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={extent}><span style={{ width: `${extent}%` }} /></div><div className="bloom-metric-foot"><span>{Number(item.area_pct || 0).toFixed(1)}% of analyzed area</span><span>Visual estimate only — not a medical assessment</span></div></>}
      </article>;
    })}
  </div>;
}
