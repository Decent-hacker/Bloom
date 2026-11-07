import { useEffect, useRef, useState } from "react";
import UploadZone from "./components/UploadZone.jsx";
import ProcessingView from "./components/ProcessingView.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";
import ErrorView from "./components/ErrorView.jsx";
import AuthScreen from "./components/AuthScreen.jsx";
import Icon from "./components/Icon.jsx";
import { Leaf } from "lucide-react";
import { analyseImage, analyseMultiAngle, checkGuidedCapture, deleteHistory, getCurrentUser, getHistory, login, register, saveHistory } from "./api.js";

const MAX_MB = 5;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];
const INITIAL_STATE = { view: "upload", selectedFiles: {}, previewUrls: {}, result: null, errorMessage: null, fileError: null, history: [] };

function App() {
  const [state, setState] = useState(INITIAL_STATE);
  const [user, setUser] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const controllerRef = useRef(null);
  const requestIdRef = useRef(0);
  const previewUrlsRef = useRef({});
  const update = (patch) => setState((previous) => ({ ...previous, ...(typeof patch === "function" ? patch(previous) : patch) }));
  const revokePreviews = () => { Object.values(previewUrlsRef.current).forEach(URL.revokeObjectURL); previewUrlsRef.current = {}; };
  const abortRequest = () => { controllerRef.current?.abort(); controllerRef.current = null; };

  useEffect(() => {
    const restore = async () => {
      if (sessionStorage.getItem("bloom_session_token")) {
        const account = await getCurrentUser();
        if (account.user) {
          setUser(account.user);
          const records = await getHistory();
          update({ history: records.records || [] });
        } else sessionStorage.removeItem("bloom_session_token");
      }
      setAuthReady(true);
    };
    restore();
    return () => { abortRequest(); revokePreviews(); };
  }, []);

  const handleAuthenticate = async (mode, form) => {
    const account = mode === "signin" ? await login(form) : await register(form);
    if (!account.token || !account.user) return account;
    sessionStorage.setItem("bloom_session_token", account.token);
    setUser(account.user);
    const records = await getHistory();
    update({ history: records.records || [] });
    return {};
  };
  const signOut = () => { abortRequest(); revokePreviews(); sessionStorage.removeItem("bloom_session_token"); setState(INITIAL_STATE); setUser(null); };
  const handleSelect = (angle, file) => {
    if (!ALLOWED_TYPES.includes(file.type)) return update({ fileError: "Choose a JPEG or PNG image." });
    if (file.size > MAX_MB * 1024 * 1024) return update({ fileError: `Choose an image smaller than ${MAX_MB} MB.` });
    abortRequest();
    const previewUrl = URL.createObjectURL(file);
    if (previewUrlsRef.current[angle]) URL.revokeObjectURL(previewUrlsRef.current[angle]);
    previewUrlsRef.current = { ...previewUrlsRef.current, [angle]: previewUrl };
    update((previous) => ({ selectedFiles: { ...previous.selectedFiles, [angle]: file }, previewUrls: { ...previous.previewUrls, [angle]: previewUrl }, fileError: null }));
  };
  const handleClear = () => { abortRequest(); revokePreviews(); update({ selectedFiles: {}, previewUrls: {}, fileError: null }); };
  const handleResult = async (data, requestId) => {
    if (requestId !== requestIdRef.current || data.status === "aborted") return;
    controllerRef.current = null;
    if (data.status === "success") {
      const saved = await saveHistory(data);
      update((previous) => ({ view: "results", result: data, history: saved.records || previous.history }));
    } else if (data.status === "quality_error") update({ view: "error", result: data, errorMessage: data.quality?.issues || [] });
    else update({ view: "error", result: data, errorMessage: [data.message || "Something went wrong."] });
  };
  const analyse = async (work) => {
    abortRequest(); const controller = new AbortController(); controllerRef.current = controller;
    const requestId = requestIdRef.current + 1; requestIdRef.current = requestId; update({ view: "processing", errorMessage: null });
    const data = await work(controller.signal); await handleResult(data, requestId);
  };
  const reset = () => { abortRequest(); revokePreviews(); setState((previous) => ({ ...INITIAL_STATE, history: previous.history })); };
  const removeRecord = async (recordId) => { const response = await deleteHistory(recordId); if (response.records) update({ history: response.records }); };

  if (!authReady) return <div className="app-loading">Opening Bloom…</div>;
  if (!user) return <AuthScreen onAuthenticate={handleAuthenticate} />;
  return <div className="app-shell">
    <header className="app-header bloom-header"><a className="wordmark" href="#top"><Leaf size={21} strokeWidth={1.6} aria-hidden="true" />Bloom</a><nav className="account-actions" aria-label="Main navigation"><a href="#about">About</a><span>{user.email}</span><button type="button" onClick={signOut} title="Sign out"><Icon name="logout" size={18} /><span>Sign out</span></button></nav></header>
    <main className="app-main">
      {state.view === "upload" && <UploadZone selectedFiles={state.selectedFiles} previewUrls={state.previewUrls} fileError={state.fileError} onSelect={handleSelect} onValidateCapture={checkGuidedCapture} onClear={handleClear} onAnalyseScan={() => analyse((signal) => analyseMultiAngle(state.selectedFiles, { signal }))} onAnalyseSingle={() => analyse((signal) => analyseImage(state.selectedFiles.single, { signal }))} />}
      {state.view === "processing" && <div className="bloom-processing">{state.previewUrls.single && <img src={state.previewUrls.single} alt="Photo being analyzed" />}<div className="bloom-processing-copy"><span className="bloom-kicker">PHOTO REVIEW</span><h1>Looking closely at your photo.</h1><p>Checking image quality, skin area, and visible variation.</p><div className="bloom-skeleton" /><div className="bloom-skeleton short" /></div></div>}
      {state.view === "results" && <ResultsPanel result={state.result} history={state.history} onDeleteHistory={removeRecord} onReset={reset} />}
      {state.view === "error" && <ErrorView issues={state.errorMessage} isQuality={state.result?.status === "quality_error"} onReset={reset} />}
    </main>
    <footer id="about">Bloom provides educational visual analysis only. It is not a medical diagnosis.</footer>
  </div>;
}

export default App;
