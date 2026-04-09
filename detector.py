"""
FakeShield - Fake Profile Detection Engine
Advanced multi-platform fake profile analysis using ML, NLP, and heuristics.
"""

import re
import json
import math
import random
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urlparse

# ── Optional heavy deps (graceful fallback) ──────────────────────────────────
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─────────────────────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProfileInput:
    platform: str                         # "twitter" | "linkedin" | "instagram" | "email" | "facebook"
    username: str        = ""
    display_name: str    = ""
    bio: str             = ""
    email: str           = ""
    followers: int       = 0
    following: int       = 0
    posts_count: int     = 0
    account_age_days: int= 0
    profile_pic_url: str = ""
    website: str         = ""
    location: str        = ""
    verified: bool       = False
    post_texts: list     = field(default_factory=list)   # sample posts/messages
    connections: int     = 0                              # LinkedIn connections
    endorsements: int    = 0
    job_title: str       = ""
    company: str         = ""
    education: str       = ""
    mutual_connections: int = 0


@dataclass
class SignalResult:
    name: str
    score: float          # 0.0 (legit) → 1.0 (fake)
    weight: float
    detail: str
    category: str         # "identity" | "activity" | "network" | "content" | "metadata"


@dataclass
class DetectionReport:
    platform: str
    username: str
    overall_score: float          # 0–100 (fake probability %)
    verdict: str                  # "LIKELY FAKE" | "SUSPICIOUS" | "LIKELY REAL" | "CLEAN"
    confidence: float             # 0–1
    risk_level: str               # "HIGH" | "MEDIUM" | "LOW" | "SAFE"
    signals: list
    summary: str
    recommendations: list
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL ANALYZERS
# ─────────────────────────────────────────────────────────────────────────────

class IdentityAnalyzer:
    """Checks username patterns, display name entropy, email structure."""

    # Patterns strongly associated with fake / bot accounts
    BOT_USERNAME_PATTERNS = [
        r"^(user|account|profile)\d{4,}$",
        r"[a-z]{3,6}\d{6,}",          # random letters + long number string
        r"^[a-z]+_[a-z]+\d{3,}$",     # word_word123456
        r"^\w{1,3}\d{8,}$",            # very short prefix + many digits
    ]

    DISPOSABLE_DOMAINS = {
        "mailinator.com", "guerrillamail.com", "tempmail.com", "throwam.com",
        "yopmail.com", "sharklasers.com", "guerrillamailblock.com",
        "grr.la", "spam4.me", "trashmail.com", "dispostable.com",
        "maildrop.cc", "fakeinbox.com", "10minutemail.com", "tempr.email"
    }

    COMMON_NAME_WORDS = {"john", "jane", "mike", "sarah", "david", "emily", "chris", "kate"}

    def analyze(self, p: ProfileInput) -> list:
        signals = []

        # ── Username entropy ──────────────────────────────────────────────
        uname = p.username.lower().strip()
        bot_pattern_hit = any(re.search(pat, uname) for pat in self.BOT_USERNAME_PATTERNS)
        digit_ratio = sum(c.isdigit() for c in uname) / max(len(uname), 1)

        uname_score = 0.0
        if bot_pattern_hit:
            uname_score += 0.55
        if digit_ratio > 0.5:
            uname_score += 0.25
        if len(uname) > 20:
            uname_score += 0.10
        uname_score = min(uname_score, 1.0)

        signals.append(SignalResult(
            name="Username Pattern",
            score=uname_score,
            weight=0.15,
            detail=f"Digit ratio {digit_ratio:.0%}; bot pattern match: {bot_pattern_hit}",
            category="identity"
        ))

        # ── Display name analysis ─────────────────────────────────────────
        dname = p.display_name.strip()
        dname_score = 0.0
        if not dname:
            dname_score = 0.6
            detail = "No display name set"
        elif dname.lower() == uname:
            dname_score = 0.35
            detail = "Display name identical to username"
        elif re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", dname):
            dname_score = 0.05
            detail = "Normal-looking full name"
        else:
            entropy = self._string_entropy(dname)
            dname_score = min(max((entropy - 2.5) / 5, 0), 1.0)
            detail = f"Name entropy: {entropy:.2f}"

        signals.append(SignalResult(
            name="Display Name Quality",
            score=dname_score,
            weight=0.10,
            detail=detail,
            category="identity"
        ))

        # ── Email analysis ────────────────────────────────────────────────
        if p.email:
            email_score = self._analyze_email(p.email)
            signals.append(email_score)

        # ── Profile pic ───────────────────────────────────────────────────
        pic_score = 0.0
        pic_detail = ""
        if not p.profile_pic_url:
            pic_score = 0.65
            pic_detail = "No profile picture set"
        elif "default" in p.profile_pic_url.lower() or "placeholder" in p.profile_pic_url.lower():
            pic_score = 0.70
            pic_detail = "Default/placeholder profile picture"
        else:
            pic_score = 0.10
            pic_detail = "Profile picture present"

        signals.append(SignalResult(
            name="Profile Picture",
            score=pic_score,
            weight=0.12,
            detail=pic_detail,
            category="identity"
        ))

        return signals

    def _analyze_email(self, email: str) -> SignalResult:
        score = 0.0
        detail_parts = []

        if "@" not in email:
            return SignalResult("Email Format", 0.9, 0.20, "Invalid email format", "identity")

        local, domain = email.rsplit("@", 1)

        if domain.lower() in self.DISPOSABLE_DOMAINS:
            score += 0.80
            detail_parts.append("disposable email domain")

        # Random-looking local part
        if re.match(r"^[a-z0-9]{8,}$", local) and sum(c.isdigit() for c in local) > 3:
            score += 0.30
            detail_parts.append("random-looking local part")

        # Excessively long local part
        if len(local) > 20:
            score += 0.15
            detail_parts.append(f"long local part ({len(local)} chars)")

        score = min(score, 1.0)
        detail = "; ".join(detail_parts) if detail_parts else "Email looks normal"
        return SignalResult("Email Analysis", score, 0.20, detail, "identity")

    @staticmethod
    def _string_entropy(s: str) -> float:
        if not s:
            return 0.0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        n = len(s)
        return -sum((v / n) * math.log2(v / n) for v in freq.values())


class ActivityAnalyzer:
    """Analyses posting frequency, account age, follower/following ratio."""

    def analyze(self, p: ProfileInput) -> list:
        signals = []

        # ── Account age ───────────────────────────────────────────────────
        age = p.account_age_days
        if age == 0:
            age_score, age_detail = 0.50, "Account age unknown"
        elif age < 7:
            age_score, age_detail = 0.80, f"Very new account ({age} days)"
        elif age < 30:
            age_score, age_detail = 0.55, f"Young account ({age} days)"
        elif age < 180:
            age_score, age_detail = 0.25, f"Relatively new ({age} days)"
        else:
            age_score, age_detail = 0.05, f"Established account ({age} days)"

        signals.append(SignalResult("Account Age", age_score, 0.15, age_detail, "activity"))

        # ── Follower/Following ratio ───────────────────────────────────────
        if p.platform in ("twitter", "instagram", "facebook"):
            ratio_signal = self._follower_ratio(p)
            signals.append(ratio_signal)

        # ── Post frequency ────────────────────────────────────────────────
        if age > 0 and p.posts_count >= 0:
            posts_per_day = p.posts_count / max(age, 1)
            if posts_per_day > 50:
                freq_score, freq_detail = 0.90, f"Extremely high post rate: {posts_per_day:.1f}/day (bot-like)"
            elif posts_per_day > 20:
                freq_score, freq_detail = 0.65, f"Very high post rate: {posts_per_day:.1f}/day"
            elif posts_per_day == 0 and age > 30:
                freq_score, freq_detail = 0.50, "No posts on a mature account"
            else:
                freq_score, freq_detail = 0.10, f"Normal post rate: {posts_per_day:.1f}/day"

            signals.append(SignalResult("Post Frequency", freq_score, 0.12, freq_detail, "activity"))

        # ── LinkedIn-specific ─────────────────────────────────────────────
        if p.platform == "linkedin":
            conn_score = self._linkedin_connections(p)
            signals.append(conn_score)

        return signals

    def _follower_ratio(self, p: ProfileInput) -> SignalResult:
        followers = max(p.followers, 0)
        following = max(p.following, 1)
        ratio = followers / following

        if following > 5000 and followers < 100:
            return SignalResult("Follow Ratio", 0.85, 0.18, f"Mass following with tiny audience (F/F={ratio:.2f})", "activity")
        if ratio > 100 and followers > 10000:
            return SignalResult("Follow Ratio", 0.15, 0.18, f"High follower ratio suggests organic growth ({ratio:.1f}x)", "activity")
        if following > 2000 and ratio < 0.1:
            return SignalResult("Follow Ratio", 0.70, 0.18, f"Follow-for-follow pattern (F/F={ratio:.2f})", "activity")
        return SignalResult("Follow Ratio", 0.15, 0.18, f"Normal ratio ({ratio:.2f})", "activity")

    def _linkedin_connections(self, p: ProfileInput) -> SignalResult:
        if p.connections < 10:
            return SignalResult("LinkedIn Connections", 0.65, 0.15, f"Very few connections ({p.connections})", "network")
        if p.connections > 500:
            return SignalResult("LinkedIn Connections", 0.10, 0.15, f"500+ connections (established)", "network")
        return SignalResult("LinkedIn Connections", 0.20, 0.15, f"{p.connections} connections (moderate)", "network")


class ContentAnalyzer:
    """NLP analysis of bio and post texts."""

    SPAM_KEYWORDS = [
        "click here", "make money", "work from home", "earn $", "dm for",
        "onlyfans", "crypto signal", "forex signal", "investment opportunity",
        "100% guaranteed", "get rich", "passive income", "binary options",
        "follow back", "f4f", "follow for follow", "like for like",
        "giveaway", "win $", "limited offer", "act now", "free followers",
    ]

    GENERIC_BIO_PHRASES = [
        "lover of life", "living my best life", "entrepreneur", "motivational speaker",
        "social media influencer", "digital nomad", "crypto enthusiast",
        "investor | trader | mentor", "business owner", "ceo | founder",
    ]

    def analyze(self, p: ProfileInput) -> list:
        signals = []

        # ── Bio analysis ──────────────────────────────────────────────────
        bio = p.bio.lower().strip()
        if not bio:
            signals.append(SignalResult("Bio Completeness", 0.55, 0.10, "No bio provided", "content"))
        else:
            bio_score, bio_detail = self._score_bio(bio)
            signals.append(SignalResult("Bio Quality", bio_score, 0.12, bio_detail, "content"))

        # ── Post content ──────────────────────────────────────────────────
        if p.post_texts:
            post_signal = self._analyze_posts(p.post_texts)
            signals.append(post_signal)

        # ── LinkedIn completeness ─────────────────────────────────────────
        if p.platform == "linkedin":
            completeness = self._linkedin_completeness(p)
            signals.append(completeness)

        return signals

    def _score_bio(self, bio: str) -> tuple:
        score = 0.0
        details = []

        spam_hits = [kw for kw in self.SPAM_KEYWORDS if kw in bio]
        if spam_hits:
            score += min(len(spam_hits) * 0.20, 0.70)
            details.append(f"spam keywords: {', '.join(spam_hits[:3])}")

        generic_hits = [ph for ph in self.GENERIC_BIO_PHRASES if ph in bio]
        if generic_hits:
            score += min(len(generic_hits) * 0.10, 0.30)
            details.append(f"generic phrases: {', '.join(generic_hits[:2])}")

        # Excessive emoji (proxy for engagement-bait)
        emoji_count = sum(1 for c in bio if ord(c) > 0x1F300)
        if emoji_count > 10:
            score += 0.20
            details.append(f"excessive emoji ({emoji_count})")

        # Very short meaningful bio (not suspicious, just low signal)
        if len(bio) < 20 and not details:
            score += 0.20
            details.append("very short bio")

        score = min(score, 1.0)
        detail = "; ".join(details) if details else "Bio appears normal"
        return score, detail

    def _analyze_posts(self, posts: list) -> SignalResult:
        if not posts:
            return SignalResult("Post Content", 0.0, 0.15, "No posts to analyze", "content")

        spam_post_count = 0
        duplicate_count = 0
        seen = set()

        for post in posts:
            text = post.lower()
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                duplicate_count += 1
            seen.add(h)

            if any(kw in text for kw in self.SPAM_KEYWORDS):
                spam_post_count += 1

        spam_ratio = spam_post_count / len(posts)
        dup_ratio = duplicate_count / len(posts)

        score = min(spam_ratio * 0.7 + dup_ratio * 0.4, 1.0)

        detail = f"Spam posts: {spam_ratio:.0%}, duplicates: {dup_ratio:.0%}"

        # NLTK sentiment variance (flat sentiment → bot)
        if HAS_NLTK and len(posts) >= 5:
            sia = SentimentIntensityAnalyzer()
            scores = [sia.polarity_scores(p)["compound"] for p in posts]
            variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            if variance < 0.02:
                score = min(score + 0.30, 1.0)
                detail += "; near-zero sentiment variance (bot indicator)"

        return SignalResult("Post Content Analysis", score, 0.18, detail, "content")

    def _linkedin_completeness(self, p: ProfileInput) -> SignalResult:
        fields = [p.job_title, p.company, p.education, p.location, p.bio, p.profile_pic_url]
        filled = sum(1 for f in fields if f and f.strip())
        ratio = filled / len(fields)
        score = 1.0 - ratio
        detail = f"{filled}/{len(fields)} profile fields completed"
        return SignalResult("Profile Completeness", score, 0.14, detail, "content")


class MetadataAnalyzer:
    """Website, location, and cross-platform consistency checks."""

    SUSPICIOUS_TLDS = {".xyz", ".top", ".click", ".loan", ".work", ".gq", ".ml", ".cf", ".tk"}

    def analyze(self, p: ProfileInput) -> list:
        signals = []

        if p.website:
            signals.append(self._check_website(p.website))

        if p.verified:
            signals.append(SignalResult(
                "Verified Badge", 0.0, 0.08,
                "Account is platform-verified", "metadata"
            ))

        # Location consistency check
        if p.location and p.bio:
            loc_lower = p.location.lower()
            bio_lower = p.bio.lower()
            if loc_lower in bio_lower or any(word in bio_lower for word in loc_lower.split()):
                signals.append(SignalResult(
                    "Location Consistency", 0.05, 0.06,
                    "Location mentioned in bio — consistent", "metadata"
                ))

        return signals

    def _check_website(self, url: str) -> SignalResult:
        try:
            parsed = urlparse(url if "://" in url else "https://" + url)
            domain = parsed.netloc or parsed.path
            tld = "." + domain.rsplit(".", 1)[-1] if "." in domain else ""

            score = 0.0
            detail_parts = []

            if tld.lower() in self.SUSPICIOUS_TLDS:
                score += 0.55
                detail_parts.append(f"suspicious TLD ({tld})")

            if re.search(r"\d{4,}", domain):
                score += 0.25
                detail_parts.append("numbers in domain")

            if len(domain) > 40:
                score += 0.15
                detail_parts.append("abnormally long domain")

            score = min(score, 1.0)
            detail = "; ".join(detail_parts) if detail_parts else f"Website domain appears normal ({domain})"
            return SignalResult("Website/URL", score, 0.08, detail, "metadata")

        except Exception:
            return SignalResult("Website/URL", 0.40, 0.08, "Could not parse website URL", "metadata")


# ─────────────────────────────────────────────────────────────────────────────
#  ML SCORING LAYER (sklearn Random Forest, trained on synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

class MLScorer:
    """
    A lightweight Random Forest trained on synthetically generated profiles.
    Provides a secondary probability estimate to blend with heuristic signals.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        if HAS_SKLEARN:
            self._train()

    def _generate_synthetic_data(self, n=2000):
        """Generate synthetic labeled training data."""
        rng = random.Random(42)
        X, y = [], []

        for _ in range(n):
            is_fake = rng.random() < 0.5
            if is_fake:
                age   = rng.randint(0, 60)
                fwrs  = rng.randint(0, 300)
                fwing = rng.randint(500, 8000)
                posts = rng.randint(0, 50)
                bio_l = rng.randint(0, 30)
                digit_r = rng.uniform(0.3, 0.9)
                has_pic = rng.random() < 0.3
                conn  = rng.randint(0, 20)
            else:
                age   = rng.randint(90, 3000)
                fwrs  = rng.randint(50, 50000)
                fwing = rng.randint(10, 1000)
                posts = rng.randint(10, 5000)
                bio_l = rng.randint(30, 200)
                digit_r = rng.uniform(0.0, 0.2)
                has_pic = rng.random() < 0.9
                conn  = rng.randint(50, 800)

            ratio = fwrs / max(fwing, 1)
            X.append([age, fwrs, fwing, posts, bio_l, digit_r, int(has_pic), conn, ratio])
            y.append(int(is_fake))

        return X, y

    def _train(self):
        X, y = self._generate_synthetic_data()
        np_X = np.array(X, dtype=float)
        self.scaler = StandardScaler()
        np_X = self.scaler.fit_transform(np_X)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(np_X, y)

    def predict(self, p: ProfileInput) -> float:
        """Returns fake probability 0–1."""
        if not HAS_SKLEARN or self.model is None:
            return -1.0  # unavailable

        uname = p.username
        digit_r = sum(c.isdigit() for c in uname) / max(len(uname), 1)
        ratio = p.followers / max(p.following, 1)
        features = np.array([[
            p.account_age_days,
            p.followers,
            p.following,
            p.posts_count,
            len(p.bio),
            digit_r,
            int(bool(p.profile_pic_url)),
            p.connections,
            ratio,
        ]], dtype=float)
        features = self.scaler.transform(features)
        prob = self.model.predict_proba(features)[0][1]
        return float(prob)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class FakeProfileDetector:
    """Orchestrates all analyzers and produces the final DetectionReport."""

    def __init__(self):
        self.identity  = IdentityAnalyzer()
        self.activity  = ActivityAnalyzer()
        self.content   = ContentAnalyzer()
        self.metadata  = MetadataAnalyzer()
        self.ml        = MLScorer()

    def analyze(self, profile: ProfileInput) -> DetectionReport:
        all_signals = []
        all_signals += self.identity.analyze(profile)
        all_signals += self.activity.analyze(profile)
        all_signals += self.content.analyze(profile)
        all_signals += self.metadata.analyze(profile)

        # ── Weighted heuristic score ──────────────────────────────────────
        total_weight = sum(s.weight for s in all_signals)
        if total_weight > 0:
            heuristic_score = sum(s.score * s.weight for s in all_signals) / total_weight
        else:
            heuristic_score = 0.5

        # ── Blend with ML score ───────────────────────────────────────────
        ml_score = self.ml.predict(profile)
        if ml_score >= 0:
            final_score = 0.60 * heuristic_score + 0.40 * ml_score
        else:
            final_score = heuristic_score

        pct = round(final_score * 100, 1)

        # ── Verdict ───────────────────────────────────────────────────────
        if pct >= 75:
            verdict, risk = "LIKELY FAKE", "HIGH"
        elif pct >= 50:
            verdict, risk = "SUSPICIOUS", "MEDIUM"
        elif pct >= 25:
            verdict, risk = "POSSIBLY REAL", "LOW"
        else:
            verdict, risk = "LIKELY REAL", "SAFE"

        # ── Confidence ────────────────────────────────────────────────────
        # More signals = higher confidence; extreme scores = higher confidence
        n_signals = len(all_signals)
        base_conf  = min(n_signals / 12, 1.0)
        extreme    = abs(final_score - 0.5) * 2   # 0 at center, 1 at extremes
        confidence = round(0.4 * base_conf + 0.6 * extreme, 2)

        summary = self._build_summary(profile, pct, verdict, all_signals)
        recommendations = self._build_recommendations(verdict, all_signals)

        # Serialise signals
        signal_dicts = [
            {
                "name": s.name,
                "score": round(s.score * 100, 1),
                "weight": s.weight,
                "detail": s.detail,
                "category": s.category,
            }
            for s in sorted(all_signals, key=lambda x: x.score * x.weight, reverse=True)
        ]

        return DetectionReport(
            platform=profile.platform,
            username=profile.username or profile.email,
            overall_score=pct,
            verdict=verdict,
            confidence=confidence,
            risk_level=risk,
            signals=signal_dicts,
            summary=summary,
            recommendations=recommendations,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

    def _build_summary(self, p, pct, verdict, signals) -> str:
        top_risk = sorted(signals, key=lambda s: s.score * s.weight, reverse=True)[:3]
        top_names = ", ".join(s.name for s in top_risk if s.score > 0.4)
        platform_label = p.platform.title()
        base = f"{platform_label} profile '{p.username or p.email}' scored {pct}% fake probability — {verdict}."
        if top_names:
            base += f" Key risk signals: {top_names}."
        return base

    def _build_recommendations(self, verdict, signals) -> list:
        recs = []
        high_signals = [s for s in signals if s.score > 0.55]

        if verdict in ("LIKELY FAKE", "SUSPICIOUS"):
            recs.append("Do not share personal information with this account.")
            recs.append("Avoid clicking links sent by this account.")

        for s in high_signals:
            if s.name == "Email Analysis":
                recs.append("Verify the email via a secondary trusted channel before any transaction.")
            if s.name == "Account Age":
                recs.append("Exercise caution — account was created very recently.")
            if "Post Content" in s.name:
                recs.append("Review post history for spam or phishing content.")
            if s.name == "Follow Ratio":
                recs.append("Unbalanced follower ratio is a common indicator of fake or purchased followers.")
            if s.name == "Profile Picture":
                recs.append("Reverse image-search the profile picture to check for stolen photos.")
            if s.name == "Profile Completeness":
                recs.append("Incomplete LinkedIn profiles are a common fake indicator — request endorsements or referrals.")

        if not recs:
            recs.append("No immediate action required. Continue normal interactions.")

        return list(dict.fromkeys(recs))  # deduplicate, preserve order


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK WEB API
# ─────────────────────────────────────────────────────────────────────────────

def create_app():
    try:
        from flask import Flask, request, jsonify, render_template_string, send_from_directory
        from flask_cors import CORS
    except ImportError:
        print("Flask not installed. Run: pip install flask flask-cors")
        return None

    import os
    app = Flask(__name__, static_folder="static", template_folder=".")
    CORS(app)

    detector = FakeProfileDetector()

    @app.route("/")
    def index():
        return send_from_directory(".", "index.html")

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        data = request.get_json(force=True)
        try:
            profile = ProfileInput(
                platform        = data.get("platform", "unknown"),
                username        = data.get("username", ""),
                display_name    = data.get("display_name", ""),
                bio             = data.get("bio", ""),
                email           = data.get("email", ""),
                followers       = int(data.get("followers", 0)),
                following       = int(data.get("following", 0)),
                posts_count     = int(data.get("posts_count", 0)),
                account_age_days= int(data.get("account_age_days", 0)),
                profile_pic_url = data.get("profile_pic_url", ""),
                website         = data.get("website", ""),
                location        = data.get("location", ""),
                verified        = bool(data.get("verified", False)),
                post_texts      = data.get("post_texts", []),
                connections     = int(data.get("connections", 0)),
                endorsements    = int(data.get("endorsements", 0)),
                job_title       = data.get("job_title", ""),
                company         = data.get("company", ""),
                education       = data.get("education", ""),
                mutual_connections = int(data.get("mutual_connections", 0)),
            )
            report = detector.analyze(profile)
            return jsonify(asdict(report))
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "ml_available": HAS_SKLEARN,
            "nltk_available": HAS_NLTK,
        })

    return app


if __name__ == "__main__":
    import os
    app = create_app()
    if app:
        port = int(os.environ.get("PORT", 5000))
        print(f"FakeShield API running on port {port}")
        app.run(host="0.0.0.0", port=port, debug=False)
