import { useState } from "react";

/** Single image tile with loading placeholder, fade-in and error fallback. */
function ImageCell({ label, src, highlight, note }) {
  const [status, setStatus] = useState("loading");

  return (
    <div
      className={`min-w-[70%] overflow-hidden rounded-lg bg-white md:min-w-0 ${
        highlight ? "border-2 border-brand-500" : "border border-surface-border"
      }`}
    >
      {status === "error" ? (
        <div className="flex aspect-square items-center justify-center bg-slate-100 text-xs text-slate-400">
          Image unavailable
        </div>
      ) : (
        <div className="relative aspect-square">
          {status === "loading" && <div className="absolute inset-0 animate-pulse bg-slate-100" />}
          <img
            src={src}
            alt={label}
            onLoad={() => setStatus("loaded")}
            onError={() => setStatus("error")}
            className={`h-full w-full object-cover transition-opacity duration-300 ${
              status === "loaded" ? "opacity-100" : "opacity-0"
            }`}
          />
        </div>
      )}
      <div className="px-3 py-2.5">
        <p className={`text-xs ${highlight ? "font-semibold text-brand-700" : "font-medium text-slate-600"}`}>{label}</p>
        {note && <p className="mt-0.5 text-[11px] leading-4 text-slate-400">{note}</p>}
      </div>
    </div>
  );
}

/**
 * Side-by-side comparison of the pipeline stages.
 * @param {object} images - base64 strings keyed by stage name
 */
function ComparisonGrid({ images }) {
  // backend base64 bhejta hai; browser <img> ke liye data-URI prefix chahiye
  const dataUri = (b64) => `data:image/jpeg;base64,${b64}`;
  const cells = [
    { label: "Original", src: dataUri(images.original) },
    { label: "Enhanced", src: dataUri(images.enhanced) },
    { label: "Skin Mask", src: dataUri(images.skin_mask) },
    { label: "Fixed-threshold baseline", src: dataUri(images.baseline_annotated), note: "Reference comparison only; may mark clear skin." },
    { label: "Low-pass · grayscale shading", src: dataUri(images.low_pass), note: "Gaussian blur shows broad brightness changes." },
    { label: "High-pass · fine detail", src: dataUri(images.high_pass), note: "Signed detail is shifted to mid-grey for display; not a separate detection." },
  ];

  return (
    <section className="space-y-5">
      <div className="overflow-hidden rounded-2xl border border-brand-200 bg-white shadow-sm">
        <div className="flex flex-col gap-1 border-b border-brand-100 bg-brand-50 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-semibold text-slate-900">Marked findings</h2>
            <p className="mt-0.5 text-xs text-slate-600">Boxes mark visual patterns retained after image processing.</p>
          </div>
          <div className="flex gap-3 text-[11px] font-medium text-slate-600" aria-label="Finding colour legend">
            <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-blue-500" />Acne-like</span>
            <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-400" />Dark spots</span>
            <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-red-500" />Redness</span>
          </div>
        </div>
        <ImageCell
          label="Classical visual analysis"
          note="Pattern markers are visual estimates and not a diagnosis."
          src={dataUri(images.improved_annotated)}
          highlight
        />
      </div>

      <div>
        <div className="mb-3 flex items-end justify-between">
          <h2 className="text-base font-semibold text-slate-800">Processing views</h2>
          <p className="text-xs text-slate-400">For transparency</p>
        </div>
        <div className="flex gap-4 overflow-x-auto pb-2 md:grid md:grid-cols-4 md:overflow-visible">
        {cells.map((cell) => (
          <ImageCell key={cell.label} {...cell} />
        ))}
      </div>
      </div>
    </section>
  );
}

export default ComparisonGrid;
