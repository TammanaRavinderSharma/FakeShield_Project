/* ═══════════════════════════════════════════════════════
   FakeShield — Frontend Logic
   ═══════════════════════════════════════════════════════ */
const API_BASE = "https://fakeshield-project-2.onrender.com";
let currentPlatform = "twitter";

/* ── Health check ──────────────────────────────────────── */
async function checkHealth() {
  const dot    = document.getElementById("apiDot");
  const status = document.getElementById("apiStatus");
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const d = await res.json();
      dot.className = "dot online";
      status.textContent = `API online${d.ml_available ? " · ML ✓" : ""}${d.nltk_available ? " · NLP ✓" : ""}`;
    } else {
      throw new Error("non-200");
    }
  } catch {
    dot.className = "dot offline";
    status.textContent = "API offline — start server";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

/* ── Platform selector ─────────────────────────────────── */
document.querySelectorAll(".platform-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".platform-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentPlatform = btn.dataset.p;
    applyPlatformVisibility(currentPlatform);
  });
});

function applyPlatformVisibility(platform) {
  const linkedinOnly = document.querySelectorAll(".linkedin-only");
  linkedinOnly.forEach(el => {
    el.classList.toggle("hidden", platform !== "linkedin");
  });

  // Hide followers/following for email-only check
  const socialFields = ["f-followers","f-following","f-posts_count","f-profile_pic_url","f-website","f-post_texts"];
  socialFields.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("hidden", platform === "email");
  });
}

/* ── Form submit ───────────────────────────────────────── */
document.getElementById("profileForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  await runAnalysis();
});

async function runAnalysis() {
  const btn     = document.getElementById("analyzeBtn");
  const label   = btn.querySelector(".btn-label");
  const spinner = btn.querySelector(".btn-spinner");

  btn.disabled = true;
  label.textContent = "Analyzing…";
  spinner.classList.remove("hidden");

  const formData = new FormData(document.getElementById("profileForm"));
  const payload  = buildPayload(formData);

  try {
    const res  = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Server error");
    }
    const report = await res.json();
    renderReport(report);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    label.textContent = "Analyze Profile";
    spinner.classList.add("hidden");
  }
}

function buildPayload(formData) {
  const postText = formData.get("post_texts") || "";
  const posts    = postText.split("\n").map(l => l.trim()).filter(Boolean);

  return {
    platform:         currentPlatform,
    username:         formData.get("username")         || "",
    display_name:     formData.get("display_name")     || "",
    bio:              formData.get("bio")               || "",
    email:            formData.get("email")             || "",
    followers:        parseInt(formData.get("followers"))         || 0,
    following:        parseInt(formData.get("following"))         || 0,
    posts_count:      parseInt(formData.get("posts_count"))       || 0,
    account_age_days: parseInt(formData.get("account_age_days"))  || 0,
    profile_pic_url:  formData.get("profile_pic_url")  || "",
    website:          formData.get("website")           || "",
    location:         formData.get("location")          || "",
    verified:         formData.get("verified") === "on",
    post_texts:       posts,
    connections:      parseInt(formData.get("connections"))       || 0,
    job_title:        formData.get("job_title")         || "",
    company:          formData.get("company")           || "",
    education:        formData.get("education")         || "",
    mutual_connections: 0,
    endorsements:     0,
  };
}

/* ── Render report ─────────────────────────────────────── */
function renderReport(r) {
  document.getElementById("emptyState").classList.add("hidden");
  const panel = document.getElementById("resultsPanel");
  panel.classList.remove("hidden");

  /* Verdict label */
  document.getElementById("verdictLabel").textContent = r.verdict;

  /* Score ring */
  const score   = r.overall_score;
  const ring    = document.getElementById("ringFill");
  const circumf = 314;
  const offset  = circumf - (score / 100) * circumf;

  // Color by risk
  ring.className = "ring-fill";
  const colorClass = { HIGH: "ring-high", MEDIUM: "ring-medium", LOW: "ring-low", SAFE: "ring-safe" }[r.risk_level] || "ring-safe";
  ring.classList.add(colorClass);
  setTimeout(() => { ring.style.strokeDashoffset = offset; }, 50);

  /* Score number — count up */
  animateNumber("scoreNum", 0, Math.round(score), 900);

  /* Risk badge */
  const badge = document.getElementById("riskBadge");
  badge.textContent = r.risk_level;
  badge.className = `risk-badge risk-${r.risk_level}`;

  /* Summary */
  document.getElementById("summaryText").textContent = r.summary;

  /* Signals */
  const list = document.getElementById("signalsList");
  list.innerHTML = "";
  r.signals.forEach((s, i) => {
    const color = signalColor(s.score);
    const item  = document.createElement("div");
    item.className = "signal-item";
    item.style.animationDelay = `${i * 40}ms`;
    item.innerHTML = `
      <span class="signal-name">${s.name}</span>
      <span class="signal-score" style="color:${color}">${s.score.toFixed(0)}%</span>
      <span class="signal-detail">${escHtml(s.detail)}</span>
      <div class="signal-bar-wrap" style="grid-column:span 2">
        <div class="signal-bar" style="width:0%;background:${color}" data-target="${s.score}%"></div>
      </div>
    `;
    list.appendChild(item);
  });

  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll(".signal-bar").forEach(bar => {
      bar.style.width = bar.dataset.target;
    });
  });

  /* Recommendations */
  const recsList = document.getElementById("recsList");
  recsList.innerHTML = "";
  r.recommendations.forEach(rec => {
    const li = document.createElement("li");
    li.textContent = rec;
    recsList.appendChild(li);
  });

  /* Meta */
  document.getElementById("metaTimestamp").textContent =
    "Analyzed " + new Date(r.timestamp).toLocaleString();
  document.getElementById("metaPlatform").textContent =
    r.platform.toUpperCase() + " · " + (r.confidence * 100).toFixed(0) + "% confidence";
}

/* ── Helpers ───────────────────────────────────────────── */
function signalColor(score) {
  if (score >= 70) return "var(--accent2)";
  if (score >= 45) return "var(--accent3)";
  if (score >= 25) return "#ff9800";
  return "#00e676";
}

function animateNumber(id, from, to, duration) {
  const el  = document.getElementById(id);
  const start = performance.now();
  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function showError(msg) {
  document.getElementById("emptyState").classList.remove("hidden");
  document.getElementById("emptyState").innerHTML = `
    <div class="empty-icon">⚠</div>
    <p style="color:var(--accent2)"><strong>Error</strong><br/>${escHtml(msg)}</p>
    <p style="font-size:12px;color:var(--muted)">Make sure the Python server is running:<br/>
    <code style="color:var(--accent)">python detector.py</code></p>
  `;
}

/* ── Modal ─────────────────────────────────────────────── */
function showDocs() { document.getElementById("docsModal").classList.remove("hidden"); }
function hideDocs(e) {
  if (!e || e.target === document.getElementById("docsModal")) {
    document.getElementById("docsModal").classList.add("hidden");
  }
}
document.addEventListener("keydown", e => { if (e.key === "Escape") hideDocs(); });

/* ── Load Demo ─────────────────────────────────────────── */
const DEMOS = {
  twitter: {
    username: "XYZ_crypto8847263",
    display_name: "Crypto Signals 💰",
    bio: "DM for crypto signals 💰 | Work from home | Earn $500/day | Follow for follow | F4F | Click here",
    followers: 48,
    following: 6200,
    posts_count: 4,
    account_age_days: 12,
    profile_pic_url: "",
    website: "earn-money-fast.xyz",
    location: "",
    verified: false,
    post_texts: [
      "Click here to earn $500 daily working from home 👇",
      "Get rich fast with our crypto signals DM me now",
      "100% guaranteed returns binary options trading",
      "Click here to earn $500 daily working from home 👇",
    ],
  },
  linkedin: {
    username: "johnsmith94827",
    display_name: "John Smith",
    bio: "CEO | Founder | Investor | Entrepreneur | Motivational Speaker | Digital Nomad",
    followers: 12,
    following: 0,
    posts_count: 0,
    account_age_days: 8,
    profile_pic_url: "",
    website: "",
    location: "",
    verified: false,
    job_title: "",
    company: "",
    education: "",
    connections: 3,
  },
  email: {
    username: "",
    display_name: "",
    email: "abc12345xyz@mailinator.com",
    bio: "",
    followers: 0, following: 0, posts_count: 0, account_age_days: 0,
    profile_pic_url: "", website: "", location: "", verified: false,
  },
};

function fillDemo() {
  const demo = DEMOS[currentPlatform] || DEMOS.twitter;
  const form = document.getElementById("profileForm");

  Object.entries(demo).forEach(([key, val]) => {
    const el = form.elements[key];
    if (!el) return;
    if (el.type === "checkbox") { el.checked = val; return; }
    if (key === "post_texts" && Array.isArray(val)) { el.value = val.join("\n"); return; }
    el.value = val;
  });
}
