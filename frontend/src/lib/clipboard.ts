// Clipboard helpers that also work in insecure contexts.
//
// navigator.clipboard only exists on HTTPS or localhost, so when the app is
// served over plain HTTP on a remote host (e.g. a tailnet IP) the modern API
// is unavailable. These helpers prefer it when usable and otherwise fall back
// to the legacy document.execCommand("copy") path via a hidden element.

function legacyCopyText(text: string): boolean {
  if (typeof document === "undefined") return false;
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.top = "-9999px";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  ta.setSelectionRange(0, text.length);
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  document.body.removeChild(ta);
  return ok;
}

function legacyCopyHtml(html: string): boolean {
  if (typeof document === "undefined") return false;
  const el = document.createElement("div");
  el.contentEditable = "true";
  el.innerHTML = html;
  el.style.position = "fixed";
  el.style.top = "-9999px";
  el.style.opacity = "0";
  document.body.appendChild(el);
  const range = document.createRange();
  range.selectNodeContents(el);
  const sel = window.getSelection();
  sel?.removeAllRanges();
  sel?.addRange(range);
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  sel?.removeAllRanges();
  document.body.removeChild(el);
  return ok;
}

/** Copy plain text. Throws if both the modern and legacy paths fail. */
export async function copyText(text: string): Promise<void> {
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // fall through to legacy
    }
  }
  if (!legacyCopyText(text)) throw new Error("copy failed");
}

/** Copy rich HTML (with a plain-text alternative). Falls back to legacy
 *  rich copy, then to plain text. Throws only if everything fails. */
export async function copyRich(html: string, plain: string): Promise<void> {
  if (
    window.isSecureContext &&
    navigator.clipboard &&
    "write" in navigator.clipboard &&
    typeof ClipboardItem !== "undefined"
  ) {
    try {
      await navigator.clipboard.write([
        new ClipboardItem({
          "text/html": new Blob([html], { type: "text/html" }),
          "text/plain": new Blob([plain], { type: "text/plain" }),
        }),
      ]);
      return;
    } catch {
      // fall through to legacy
    }
  }
  if (legacyCopyHtml(html)) return;
  if (!legacyCopyText(plain)) throw new Error("copy failed");
}
