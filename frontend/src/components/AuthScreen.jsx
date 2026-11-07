import { useState } from "react";
import { Camera, Leaf, ShieldCheck } from "lucide-react";
import { getRecoveryQuestion, resetPassword } from "../api.js";

const QUESTIONS = [
  "What was the name of your first school?",
  "What is a memorable place from your childhood?",
  "What was the first concert or event you attended?",
];

function AuthScreen({ onAuthenticate }) {
  const [mode, setMode] = useState("signin");
  const [form, setForm] = useState({ email: "", password: "", security_question: QUESTIONS[0], security_answer: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const update = (event) => setForm((previous) => ({ ...previous, [event.target.name]: event.target.value }));
  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    if (mode === "recovery-email") {
      const result = await getRecoveryQuestion(form.email);
      setBusy(false);
      if (result.security_question) { setForm((previous) => ({ ...previous, security_question: result.security_question, password: "", security_answer: "" })); setMode("recovery-reset"); }
      else setError(result.message);
      return;
    }
    if (mode === "recovery-reset") {
      const result = await resetPassword({ email: form.email, security_answer: form.security_answer, new_password: form.password });
      setBusy(false);
      if (result.status === "success") { setMode("signin"); setError("Password updated. Sign in with your new password."); } else setError(result.message);
      return;
    }
    const result = await onAuthenticate(mode, form);
    setBusy(false);
    if (result?.message) setError(result.message);
  };

  return (
    <main className="auth-shell">
      <section className="auth-intro">
        <a className="wordmark auth-wordmark" href="#top" aria-label="Bloom home"><Leaf size={23} strokeWidth={1.6} aria-hidden="true" />Bloom</a>
        <div className="auth-copy">
          <p className="eyebrow">A PRIVATE PHOTO REVIEW</p>
          <h1>A clearer look<br />at your skin.</h1>
          <p>Explore visible patterns in one clear photo. Bloom keeps the image review in memory and gives you a simple way to revisit your measurements.</p>
        </div>
        <div className="principles" aria-label="Bloom principles">
          <div><Camera size={21} strokeWidth={1.6} /><span><strong>Start with a clear photo</strong><small>Use one image or a guided three-view scan.</small></span></div>
          <div><ShieldCheck size={21} strokeWidth={1.6} /><span><strong>Your photo stays private</strong><small>Images are analysed in memory, not saved to history.</small></span></div>
        </div>
      </section>

      <section className="auth-panel" aria-labelledby="auth-title">
        <div className="auth-card">
          <p className="eyebrow">Your private workspace</p>
          <h2 id="auth-title">{mode === "signin" ? "Welcome back" : mode === "signup" ? "Create your account" : mode === "recovery-email" ? "Reset your password" : "Confirm your answer"}</h2>
          <p className="muted">{mode === "signin" ? "Sign in to view only your own scan history." : mode === "signup" ? "Use an email, a strong password and a recovery answer." : "Your security question confirms it is your account."}</p>
          <form onSubmit={submit} className="auth-form">
            <label>Email address<input name="email" type="email" autoComplete="email" required value={form.email} onChange={update} /></label>
            {mode !== "recovery-email" && <label>{mode === "recovery-reset" ? "New password" : "Password"}<input name="password" type="password" autoComplete={mode === "signin" ? "current-password" : "new-password"} minLength={mode !== "signin" ? 10 : undefined} required value={form.password} onChange={update} /></label>}
            {mode === "signup" && <>
              <label>Security question<select name="security_question" value={form.security_question} onChange={update}>{QUESTIONS.map((question) => <option key={question}>{question}</option>)}</select></label>
              <label>Answer<input name="security_answer" type="text" autoComplete="off" required value={form.security_answer} onChange={update} /></label>
            </>}
            {mode === "recovery-reset" && <label>Security question<input value={form.security_question} readOnly aria-readonly="true" /><input name="security_answer" type="text" placeholder="Your answer" autoComplete="off" required value={form.security_answer} onChange={update} /></label>}
            {error && <p className="form-error" role="alert">{error}</p>}
            <button className="primary-button" disabled={busy}>{busy ? "Please wait" : mode === "signin" ? "Sign in" : mode === "signup" ? "Create account" : mode === "recovery-email" ? "Continue" : "Update password"}</button>
          </form>
          {mode === "signin" ? <p className="switch-copy">New to Bloom? <button type="button" onClick={() => { setMode("signup"); setError(""); }}>Create one</button> · <button type="button" onClick={() => { setMode("recovery-email"); setError(""); }}>Forgot password?</button></p> : <p className="switch-copy">Already have an account? <button type="button" onClick={() => { setMode("signin"); setError(""); }}>Sign in</button></p>}
        </div>
      </section>
    </main>
  );
}

export default AuthScreen;
