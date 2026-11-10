const REQUEST_TIMEOUT_MS = 60000;
// Always use Vite's /api proxy in development, even if a stale local .env
// contains an API URL. Hosted builds can set VITE_API_URL explicitly.
const API_BASE = import.meta.env.DEV ? "" : (import.meta.env.VITE_API_URL || "").trim().replace(/\/+$/, "");

const authHeaders = () => {
  const token = sessionStorage.getItem("bloom_session_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
};

async function jsonRequest(path, method, body, authenticated = false) {
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      method,
      headers: { "Content-Type": "application/json", ...(authenticated ? authHeaders() : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json();
    if (!response.ok) return { message: data.detail || data.message || "Something went wrong." };
    return data;
  } catch {
    return { message: "Could not reach the server. Please try again." };
  }
}

export const register = (form) => jsonRequest("/api/auth/register", "POST", form);
export const login = (form) => jsonRequest("/api/auth/login", "POST", form);
export const getRecoveryQuestion = (email) => jsonRequest(`/api/auth/recovery-question?email=${encodeURIComponent(email)}`, "GET");
export const resetPassword = (form) => jsonRequest("/api/auth/reset-password", "POST", form);
export const getCurrentUser = () => jsonRequest("/api/auth/me", "GET", null, true);
export const getHistory = () => jsonRequest("/api/history", "GET", null, true);
export const saveHistory = (result) => jsonRequest("/api/history", "POST", {
  scan_type: result.scan_type || "single_photo",
  concerns: result.concerns,
  processing_time_sec: result.processing_time_sec,
}, true);
export const deleteHistory = (recordId) => jsonRequest(`/api/history/${recordId}`, "DELETE", null, true);

/**
 * Generic JSON API request helper.
 */
export async function apiRequest(path, options = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      method: options.method || "GET",
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      body: options.body,
      signal: controller.signal,
    });

    if (response.status === 429) {
      return { status: "error", message: "Too many requests. Please wait a moment." };
    }

    const data = await response.json();
    if (!response.ok && !data.status) {
      return { status: "error", message: data.message || "Server error." };
    }
    return data;
  } catch (err) {
    if (controller.signal.aborted) {
      return { status: "aborted", message: "Request cancelled." };
    }
    if (timedOut) {
      return { status: "timeout", message: "The analysis timed out. Please try again." };
    }
    // network failure, CORS, or JSON parse failure
    return { status: "error", message: "Could not reach the server. Please try again." };
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Single API entry point for Bloom.
 * Returns a structured result object; never throws to the caller.
 * @param {File} file - the user-selected image
 * @param {{signal?: AbortSignal, pipeline?: string}} options - optional external cancellation
 * @param {string} options.pipeline - pipeline mode: "classical" | "ml" | "hybrid"
 * @returns {Promise<object>} success | quality_error | face_error | invalid_image |
 *                            timeout | aborted | error
 */
export async function analyseImage(file, { signal, pipeline } = {}) {
  const formData = new FormData();
  formData.append("file", file);

  // Add pipeline parameter if specified
  if (pipeline) {
    formData.append("pipeline", pipeline);
  }

  const controller = new AbortController();
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS);
  const onExternalAbort = () => controller.abort();
  signal?.addEventListener("abort", onExternalAbort);

  try {
    const response = await fetch(`${API_BASE}/api/analyse`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    // rate limited
    if (response.status === 429) {
      return { status: "error", message: "Too many requests. Please wait a moment." };
    }

    const data = await response.json();
    if (!response.ok && !data.status) {
      return { status: "error", message: data.message || "Server error." };
    }
    return data;
  } catch (err) {
    if (timedOut) {
      return { status: "timeout", message: "The analysis timed out. Please try again." };
    }
    if (signal?.aborted || controller.signal.aborted) {
      return { status: "aborted", message: "Request cancelled." };
    }
    // network failure, CORS, or JSON parse failure
    return { status: "error", message: "Could not reach the server. Please try again." };
  } finally {
    clearTimeout(timeoutId);
    signal?.removeEventListener("abort", onExternalAbort);
  }
}

/**
 * Analyse a guided set of face angles.  The API combines the strongest result
 * per concern and also returns each angle's annotated image for review.
 * @param {Record<string, File>} angleFiles - { front, left, right }
 */
export async function analyseMultiAngle(angleFiles, { signal } = {}) {
  const formData = new FormData();
  Object.entries(angleFiles).forEach(([angle, file]) => {
    if (file) {
      formData.append("files", file);
      formData.append("angles", angle);
    }
  });

  const controller = new AbortController();
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS * 3);
  const onExternalAbort = () => controller.abort();
  signal?.addEventListener("abort", onExternalAbort);

  try {
    const response = await fetch(`${API_BASE}/api/analyse/multi`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    if (response.status === 429) {
      return { status: "error", message: "Too many requests. Please wait a moment." };
    }
    const data = await response.json();
    if (!response.ok && !data.status) {
      return { status: "error", message: data.message || "Server error." };
    }
    return data;
  } catch (err) {
    if (timedOut) return { status: "timeout", message: "The multi-angle scan timed out. Please try again." };
    if (signal?.aborted || controller.signal.aborted) return { status: "aborted", message: "Request cancelled." };
    return { status: "error", message: "Could not reach the server. Please try again." };
  } finally {
    clearTimeout(timeoutId);
    signal?.removeEventListener("abort", onExternalAbort);
  }
}

/** Validate framing, quality, and requested pose before retaining a camera frame. */
export async function checkGuidedCapture(file, angle, referenceYaw = null) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("angle", angle);
  if (referenceYaw !== null) formData.append("reference_yaw", String(referenceYaw));

  try {
    const response = await fetch(`${API_BASE}/api/capture-check`, { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok && data.accepted === undefined) {
      return { accepted: false, message: data.message || "We could not check this frame. Please try again." };
    }
    return data;
  } catch {
    return { accepted: false, message: "We could not check this frame. Please try again." };
  }
}
