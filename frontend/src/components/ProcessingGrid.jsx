const dataUri = (value) => `data:image/jpeg;base64,${value}`;

const stages = [
  { key: "original", title: "Original", description: "The photo before processing.", time: "preprocessing" },
  { key: "enhanced", title: "Enhanced", description: "Illumination correction, CLAHE and bilateral filtering.", time: "enhancement" },
  { key: "skin_mask", title: "Skin area", description: "The face region used for the visual measurements.", time: "skin_segmentation" },
  { key: "low_pass", title: "Low-pass", description: "Gaussian smoothing keeps broad brightness and shading.", time: "frequency_decomposition" },
  { key: "high_pass", title: "High-pass", description: "Fine changes after subtracting the smooth image; contrast stretched for display.", time: "frequency_decomposition" },
  { key: "dog", title: "DoG response", description: "Two Gaussian scales isolate smaller spot-like patterns.", time: "frequency_decomposition" },
  { key: "imperfection_heatmap", title: "FFT detail map", description: "Band-pass strength across the analysed skin area.", time: "imperfection_map" },
  { key: "imperfection_overlay", title: "Selected surface regions", description: "Adaptive threshold and morphology leave the green outlined regions.", time: "imperfection_map" },
  { key: "improved_annotated", title: "Final concern overlay", description: "The separate surface, redness and dark-region rules combined.", time: "postprocessing" },
];

export default function ProcessingGrid({ images = {}, stageTimes = {}, regionCount = 0 }) {
  const available = stages.filter(({ key }) => images[key]);
  if (!available.length) return null;

  return <section className="bloom-process" aria-labelledby="process-title">
    <div className="bloom-process-heading"><div><span className="bloom-kicker">IMAGE PROCESSING</span><h2 id="process-title">How this result was made</h2><p>Actual intermediate views from this scan. Spatial high-pass and DoG show fine detail; the FFT map is a separate frequency analysis used for the outlined surface regions.</p></div><span className="bloom-process-count">{regionCount} surface regions selected</span></div>
    <div className="bloom-process-grid">
      {available.map(({ key, title, description, time }, index) => <figure className="bloom-process-card" key={key}>
        <div className="bloom-process-image"><img src={dataUri(images[key])} alt={`${title} processing view`} loading={index > 2 ? "lazy" : "eager"} /></div>
        <figcaption><div className="bloom-process-card-title"><span className="bloom-process-number">{String(index + 1).padStart(2, "0")}</span><strong>{title}</strong>{stageTimes[time] != null && <span className="bloom-process-time">{Number(stageTimes[time]).toFixed(2)}s</span>}</div><p>{description}</p></figcaption>
      </figure>)}
    </div>
    <p className="bloom-process-note">Display contrast is adjusted for the high-pass and DoG views. Bright pixels and outlines are image patterns, not confirmed skin conditions.</p>
  </section>;
}
