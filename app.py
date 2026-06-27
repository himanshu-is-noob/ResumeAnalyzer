import streamlit as st
import matplotlib.pyplot as plt
import PyPDF2
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ================== PAGE CONFIG ==================

st.set_page_config(
    page_title="HireReady — Resume Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== CSS ==================

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("static/style.css")

# ================== NLTK ==================

for resource in ["punkt", "stopwords", "averaged_perceptron_tagger", "punkt_tab"]:
    nltk.download(resource, quiet=True)

# ================== MODEL ==================

@st.cache_resource(show_spinner="Loading AI models…")
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ================== CONSTANTS ==================

# Each entry is (display_label, [alias patterns to match in raw text])
# Patterns are matched case-insensitively against raw lowercased resume/JD text.
# Using aliases ensures "node.js", "nodejs", "node js" all map to the same skill.
TECH_SKILLS_MAP = {
    # --- Languages ---
    "python":       [r"\bpython\b"],
    "java":         [r"\bjava\b(?!script)"],           # avoid matching javascript
    "javascript":   [r"\bjavascript\b", r"\bjs\b"],
    "typescript":   [r"\btypescript\b", r"\bts\b"],
    "c++":          [r"\bc\+\+\b", r"\bc/c\+\+\b"],
    "c#":           [r"\bc#\b"],
    "c":            [r"\bc\b"],
    "go":           [r"\bgolang\b", r"\b(?<!\w)go(?!\w)language\b"],
    "rust":         [r"\brust\b"],
    "kotlin":       [r"\bkotlin\b"],
    "swift":        [r"\bswift\b"],
    "r":            [r"\blanguage r\b", r"\bprogramming in r\b"],
    "scala":        [r"\bscala\b"],
    "php":          [r"\bphp\b"],
    "ruby":         [r"\bruby\b"],
    "matlab":       [r"\bmatlab\b"],
    "bash":         [r"\bbash\b", r"\bshell script"],
    "sql":          [r"\bsql\b"],

    # --- Frontend / Web ---
    "react":        [r"\breact\b", r"\breactjs\b", r"\breact\.js\b"],
    "angular":      [r"\bangular\b", r"\bangularjs\b"],
    "vue":          [r"\bvue\b", r"\bvuejs\b", r"\bvue\.js\b"],
    "html":         [r"\bhtml\b"],
    "css":          [r"\bcss\b"],
    "sass":         [r"\bsass\b", r"\bscss\b"],
    "tailwind":     [r"\btailwind\b"],
    "bootstrap":    [r"\bbootstrap\b"],
    "nextjs":       [r"\bnext\.js\b", r"\bnextjs\b", r"\bnext js\b"],
    "nuxtjs":       [r"\bnuxt\.js\b", r"\bnuxtjs\b"],
    "webpack":      [r"\bwebpack\b"],
    "vite":         [r"\bvite\b"],
    "redux":        [r"\bredux\b"],

    # --- APIs / Architecture ---
    "graphql":      [r"\bgraphql\b"],
    "rest":         [r"\brest\b", r"\brestful\b", r"\brest api", r"\brestful api"],
    "microservices":[r"\bmicroservice"],
    "serverless":   [r"\bserverless\b"],

    # --- Backend Frameworks ---
    "node.js":      [r"\bnode\.js\b", r"\bnodejs\b", r"\bnode js\b"],
    "express.js":   [r"\bexpress\.js\b", r"\bexpressjs\b", r"\bexpress js\b", r"\bexpress\b"],
    "django":       [r"\bdjango\b"],
    "flask":        [r"\bflask\b"],
    "fastapi":      [r"\bfastapi\b"],
    "spring":       [r"\bspring\b"],
    "laravel":      [r"\blaravel\b"],
    ".net":         [r"\.net\b", r"\bdotnet\b", r"\b\.net framework\b"],

    # --- Cloud ---
    "aws":          [r"\baws\b", r"\bamazon web services\b"],
    "azure":        [r"\bazure\b", r"\bmicrosoft azure\b"],
    "gcp":          [r"\bgcp\b", r"\bgoogle cloud\b"],

    # --- DevOps / Infra ---
    "docker":       [r"\bdocker\b"],
    "kubernetes":   [r"\bkubernetes\b", r"\bk8s\b"],
    "terraform":    [r"\bterraform\b"],
    "ansible":      [r"\bansible\b"],
    "linux":        [r"\blinux\b", r"\bubuntu\b", r"\bdebian\b"],
    "nginx":        [r"\bnginx\b"],
    "apache":       [r"\bapache\b"],
    "ci/cd":        [r"\bci/cd\b", r"\bcontinuous integration\b", r"\bcontinuous deployment\b", r"\bjenkins\b", r"\bgithub actions\b"],

    # --- ML / AI ---
    "pandas":           [r"\bpandas\b"],
    "numpy":            [r"\bnumpy\b"],
    "scikit-learn":     [r"\bscikit.learn\b", r"\bsklearn\b"],
    "tensorflow":       [r"\btensorflow\b"],
    "pytorch":          [r"\bpytorch\b"],
    "keras":            [r"\bkeras\b"],
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep learning":    [r"\bdeep learning\b"],
    "nlp":              [r"\bnlp\b", r"\bnatural language processing\b"],
    "computer vision":  [r"\bcomputer vision\b"],
    "bert":             [r"\bbert\b"],
    "transformers":     [r"\btransformers\b"],
    "llm":              [r"\bllm\b", r"\blarge language model"],
    "openai":           [r"\bopenai\b"],
    "langchain":        [r"\blangchain\b"],
    "spark":            [r"\bapache spark\b", r"\bpyspark\b"],
    "hadoop":           [r"\bhadoop\b"],

    # --- BI / Analytics ---
    "tableau":      [r"\btableau\b"],
    "powerbi":      [r"\bpower\s*bi\b"],
    "excel":        [r"\bexcel\b"],
    "data analysis":[r"\bdata analysis\b", r"\bdata analytics\b"],
    "statistics":   [r"\bstatistics\b", r"\bstatistical\b"],

    # --- Databases ---
    "mysql":        [r"\bmysql\b"],
    "postgresql":   [r"\bpostgresql\b", r"\bpostgres\b"],
    "mongodb":      [r"\bmongodb\b", r"\bmongo\b"],
    "redis":        [r"\bredis\b"],
    "sqlite":       [r"\bsqlite\b"],
    "oracle":       [r"\boracle\b"],
    "elasticsearch":[r"\belasticsearch\b"],
    "cassandra":    [r"\bcassandra\b"],
    "dynamodb":     [r"\bdynamodb\b"],

    # --- Version Control / Collaboration ---
    "git":          [r"\bgit\b"],
    "github":       [r"\bgithub\b"],
    "gitlab":       [r"\bgitlab\b"],
    "jira":         [r"\bjira\b"],
    "confluence":   [r"\bconfluence\b"],

    # --- Methodologies ---
    "agile":        [r"\bagile\b"],
    "scrum":        [r"\bscrum\b"],
    "kanban":       [r"\bkanban\b"],

    # --- Soft Skills ---
    "communication":    [r"\bcommunication\b"],
    "leadership":       [r"\bleadership\b"],
    "teamwork":         [r"\bteamwork\b", r"\bteam player\b"],
    "problem-solving":  [r"\bproblem.solving\b", r"\bproblem solving\b"],
    "collaboration":    [r"\bcollaboration\b", r"\bcollaborat"],
    "management":       [r"\bmanagement\b"],
    "mentoring":        [r"\bmentor"],
    "presentation":     [r"\bpresentation\b"],
    "analytical":       [r"\banalytical\b", r"\banalysis skills\b"],
    "research":         [r"\bresearch\b"],
    "documentation":    [r"\bdocumentation\b"],
    "testing":          [r"\btesting\b", r"\bunit test", r"\bqa\b"],

    # --- CS Fundamentals ---
    "data structures":  [r"\bdata structures\b"],
    "algorithms":       [r"\balgorithm"],
    "software design":  [r"\bsoftware design\b", r"\bdesign patterns\b"],
}

SECTION_PATTERNS = {
    "education":    r"\b(education|academic|degree|university|college|school)\b",
    "experience":   r"\b(experience|employment|work history|professional background|career|professional experience)\b",
    "skills":       r"\b(skills|technical skills|competencies|expertise|technologies)\b",
    "projects":     r"\b(projects|portfolio|works|assignments|project work|major projects)\b",
    "achievements": r"\b(achievements|awards|honors|certifications|accomplishments)\b",
    "contact":      r"\b(contact|email|phone|linkedin|github|address)\b",
}

ACTION_VERBS = {
    "developed","built","designed","engineered","implemented","architected",
    "deployed","automated","optimized","refactored","migrated","integrated",
    "created","launched","delivered","shipped","coded","programmed",
    "led","managed","directed","coordinated","supervised","mentored",
    "trained","guided","oversaw","spearheaded","drove","established",
    "increased","decreased","reduced","improved","accelerated","boosted",
    "achieved","exceeded","generated","saved","cut","grew","scaled",
    "collaborated","partnered","contributed","supported","facilitated",
    "presented","communicated","negotiated","aligned",
    "analyzed","researched","evaluated","assessed","identified","diagnosed",
    "reviewed","audited","tested","validated","measured","tracked",
    "invented","detected","brainstormed","promoted","provided","assisted",
    "fixed","resolved","streamlined","standardized","documented",
}

QUANT_PATTERNS = [
    r'\b(increased|decreased|reduced|improved|boosted|grew|cut|saved|accelerated|optimized)\w*\s+\w*\s*\d+\s*%',
    r'\d+\s*%\s*(increase|decrease|reduction|improvement|growth|savings)',
    r'\d+(\.\d+)?\s*[xX]\s*(faster|more|better|improvement|increase|growth|reduction)',
    r'(by\s+)?\d+(\.\d+)?\s*times',
    r'\$\s*\d+[\d,\.]*\s*(million|billion|thousand|k|m|b)?',
    r'\b\d+[\d,]*\+?\s*(users|customers|clients|employees|teams|projects|systems|servers|records|transactions|requests|engineers|members|banks)',
    r'(saved|reduced|cut)\s+\w*\s*\d+\s*(hours?|days?|weeks?|months?)',
    r'\d+\s*(hours?|days?|weeks?|months?)\s*(saved|reduced|faster)',
    r'\b(handled|processed|managed|served|supported)\s+\w*\s*\d+[\d,k]+',
    r'\b\d{2,}[\d,]*\+?\s*(solutions|issues|bugs|features|components|modules)',
]

DEGREE_KEYWORDS = {
    "bachelor", "b.e", "b.tech", "b.sc", "b.s", "bs", "be", "btech",
    "master", "m.e", "m.tech", "m.sc", "m.s", "ms", "me", "mtech", "mba",
    "phd", "ph.d", "doctorate", "diploma", "degree", "associate",
}

# ================== CORE: RELIABLE SKILL EXTRACTOR ==================

def extract_skills_from_text(raw_text: str) -> set:
    """
    Match skills against raw lowercased text using per-skill regex aliases.
    This correctly handles: node.js, c++, .net, express.js, rest apis, etc.
    clean_text() is NOT used here — dots, plus signs, and spaces are preserved.
    """
    text_lower = raw_text.lower()
    found = set()
    for skill_label, patterns in TECH_SKILLS_MAP.items():
        for pattern in patterns:
            try:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found.add(skill_label)
                    break  # matched, no need to check other aliases
            except re.error:
                pass  # malformed pattern guard
    return found

# ================== HELPERS ==================

def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""

def clean_text(text: str) -> str:
    """Used ONLY for semantic similarity (sentence transformer needs clean input)."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def detect_contact_info(raw: str) -> dict:
    return {
        "Email":    bool(re.search(r'[\w.-]+@[\w.-]+\.\w+', raw)),
        "Phone":    bool(re.search(r'(\+?\d[\d\s\-().]{7,}\d)', raw)),
        "LinkedIn": bool(re.search(r'linkedin\.com', raw, re.IGNORECASE)),
    }

def detect_sections(raw: str) -> dict:
    low = raw.lower()
    return {s: bool(re.search(p, low)) for s, p in SECTION_PATTERNS.items()}

@st.cache_data(show_spinner=False)
def calculate_similarity(resume_text: str, job_text: str) -> float:
    emb = model.encode([resume_text, job_text])
    return round(float(cosine_similarity([emb[0]], [emb[1]])[0][0] * 100), 2)

def get_skill_gap(raw_resume: str, raw_job: str):
    """
    Both inputs should be raw (not cleaned) text.
    Returns (matched_skills_sorted, missing_skills_sorted).
    """
    job_skills    = extract_skills_from_text(raw_job)
    resume_skills = extract_skills_from_text(raw_resume)
    matched  = sorted(job_skills & resume_skills)
    missing  = sorted(job_skills - resume_skills)
    return matched, missing

# ================== QUANTIFICATION ANALYSIS ==================

def analyze_quantification(raw_resume: str) -> dict:
    lines = [l.strip() for l in raw_resume.split('\n') if l.strip()]
    bullet_lines = [l for l in lines if len(l) > 20]

    quantified_bullets   = []
    unquantified_bullets = []

    for line in bullet_lines:
        is_quantified = any(re.search(p, line, re.IGNORECASE) for p in QUANT_PATTERNS)
        if is_quantified:
            quantified_bullets.append(line)
        else:
            unquantified_bullets.append(line)

    total   = len(bullet_lines)
    q_count = len(quantified_bullets)
    q_ratio = (q_count / total * 100) if total > 0 else 0

    if q_ratio >= 20:   quant_score = 100
    elif q_ratio >= 12: quant_score = 80
    elif q_ratio >= 7:  quant_score = 60
    elif q_ratio >= 3:  quant_score = 40
    else:               quant_score = 20

    return {
        "score":               round(quant_score, 1),
        "quantified_count":    q_count,
        "total_bullets":       total,
        "ratio":               round(q_ratio, 1),
        "quantified_examples": quantified_bullets[:4],
        "unquantified_sample": unquantified_bullets[:4],
    }

# ================== ACTION VERB ANALYSIS ==================

def analyze_action_verbs(raw_resume: str) -> dict:
    lines = [l.strip() for l in raw_resume.split('\n') if len(l.strip()) > 15]

    strong_verb_lines = []
    weak_lines        = []

    for line in lines:
        first_word = line.split()[0].lower().rstrip('.,;:•●–—-') if line.split() else ""
        if first_word in ACTION_VERBS:
            strong_verb_lines.append((first_word, line))
        else:
            weak_lines.append(line)

    total      = len(lines)
    strong_n   = len(strong_verb_lines)
    verb_ratio = (strong_n / total * 100) if total > 0 else 0

    if verb_ratio >= 50:   verb_score = 100
    elif verb_ratio >= 35: verb_score = 80
    elif verb_ratio >= 20: verb_score = 55
    elif verb_ratio >= 10: verb_score = 30
    else:                  verb_score = 10

    unique_verbs = list({v for v, _ in strong_verb_lines})

    return {
        "score":        round(verb_score, 1),
        "strong_count": strong_n,
        "total_lines":  total,
        "ratio":        round(verb_ratio, 1),
        "verbs_used":   unique_verbs[:12],
        "weak_sample":  weak_lines[:3],
    }

# ================== EXPERIENCE YEARS DETECTION ==================

def analyze_experience(raw_resume: str, raw_job: str) -> dict:
    jd_exp_match = re.findall(
        r'(\d+)\+?\s*(?:to\s*\d+\s*)?years?\s*(?:of\s*)?(?:experience|exp)',
        raw_job, re.IGNORECASE
    )
    required_years = max([int(x) for x in jd_exp_match], default=0)

    date_ranges = re.findall(
        r'(20\d\d|19\d\d)\s*[-–—]\s*(20\d\d|present|current|now)',
        raw_resume, re.IGNORECASE
    )
    total_years = 0
    for start, end in date_ranges:
        try:
            s = int(start)
            e = 2025 if end.lower() in ('present', 'current', 'now') else int(end)
            total_years += max(0, e - s)
        except Exception:
            pass

    explicit = re.findall(
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        raw_resume, re.IGNORECASE
    )
    if explicit:
        total_years = max(total_years, max(int(x) for x in explicit))

    if required_years == 0:
        exp_score    = 80
        match_status = "Not specified in JD"
    elif total_years >= required_years:
        exp_score    = 100
        match_status = f"✓ Meets requirement ({total_years}y detected, {required_years}y required)"
    elif total_years >= required_years * 0.75:
        exp_score    = 70
        match_status = f"⚠ Close ({total_years}y detected, {required_years}y required)"
    elif total_years > 0:
        exp_score    = 40
        match_status = f"✗ Below requirement ({total_years}y detected, {required_years}y required)"
    else:
        exp_score    = 20
        match_status = f"✗ No experience dates found (JD requires {required_years}y)"

    return {
        "score":          round(exp_score, 1),
        "resume_years":   total_years,
        "required_years": required_years,
        "match_status":   match_status,
    }

# ================== JOB TITLE ALIGNMENT ==================

def analyze_job_title(raw_resume: str, raw_job: str) -> dict:
    jd_lower  = raw_job.lower()
    res_lower = raw_resume.lower()

    title_groups = {
        "software engineer":  ["software engineer","software developer","sde","swe","backend","frontend","fullstack","full stack","full-stack"],
        "data scientist":     ["data scientist","ml engineer","machine learning","ai engineer","data analyst","research scientist"],
        "devops / cloud":     ["devops","cloud engineer","site reliability","sre","platform engineer","infrastructure"],
        "product manager":    ["product manager","product owner","program manager"],
        "data engineer":      ["data engineer","etl","pipeline","spark","hadoop","big data"],
        "security":           ["security engineer","cybersecurity","penetration","infosec","soc analyst"],
        "mobile":             ["ios developer","android developer","mobile developer","flutter","react native"],
        "manager / lead":     ["engineering manager","tech lead","team lead","vp engineering","director"],
    }

    jd_role  = None
    res_role = None
    for role, keywords in title_groups.items():
        if any(k in jd_lower  for k in keywords): jd_role  = role
        if any(k in res_lower for k in keywords): res_role = role

    if jd_role and res_role and jd_role == res_role:
        title_score  = 100
        title_status = f"Strong match — both align to '{jd_role}'"
    elif jd_role and res_role:
        title_score  = 50
        title_status = f"Partial match — JD wants '{jd_role}', resume shows '{res_role}'"
    elif jd_role and not res_role:
        title_score  = 25
        title_status = f"JD targets '{jd_role}' but no matching title found in resume"
    else:
        title_score  = 60
        title_status = "Could not detect specific role family"

    return {
        "score":       round(title_score, 1),
        "jd_role":     jd_role or "Unknown",
        "resume_role": res_role or "Not detected",
        "status":      title_status,
    }

# ================== EDUCATION MATCH ==================

def analyze_education(raw_resume: str, raw_job: str) -> dict:
    res_lower = raw_resume.lower()
    jd_lower  = raw_job.lower()

    has_degree   = any(d in res_lower for d in DEGREE_KEYWORDS)
    jd_needs_deg = any(d in jd_lower  for d in DEGREE_KEYWORDS)

    cs_fields = [
        "computer science","information technology","software","electronics",
        "electrical","cse","ece","computer engineering","it ",
    ]
    has_cs_degree = any(f in res_lower for f in cs_fields)

    if has_degree and has_cs_degree:
        edu_score  = 100
        edu_status = "Relevant technical degree detected"
    elif has_degree:
        edu_score  = 75
        edu_status = "Degree detected (field may not be technical)"
    elif jd_needs_deg:
        edu_score  = 20
        edu_status = "JD requires degree — none clearly detected in resume"
    else:
        edu_score  = 60
        edu_status = "No degree requirement in JD"

    return {
        "score":      round(edu_score, 1),
        "has_degree": has_degree,
        "status":     edu_status,
    }

# ================== 9-COMPONENT ATS SCORE ==================

def calculate_ats_score(sim, raw_resume, raw_job, resume_clean) -> dict:
    """
    9-Component ATS Score:
      Semantic similarity      25%
      Keyword coverage         20%
      Quantified impact        15%
      Action verb strength     10%
      Section completeness      8%
      Contact completeness      7%
      Job title alignment       5%
      Experience years          5%
      Education match           5%

    NOTE: skill extraction uses raw text (not cleaned) to correctly handle
    node.js, c++, .net, express.js, rest, etc.
    """
    job_skills    = extract_skills_from_text(raw_job)
    resume_skills = extract_skills_from_text(raw_resume)
    matched       = job_skills & resume_skills
    kw_score      = (len(matched) / len(job_skills) * 100) if job_skills else 0

    sections  = detect_sections(raw_resume)
    sec_score = sum(sections[s] for s in ["education","experience","skills"]) / 3 * 100

    contact   = detect_contact_info(raw_resume)
    con_score = sum(contact.values()) / len(contact) * 100

    quant = analyze_quantification(raw_resume)
    verbs = analyze_action_verbs(raw_resume)
    exp   = analyze_experience(raw_resume, raw_job)
    title = analyze_job_title(raw_resume, raw_job)
    edu   = analyze_education(raw_resume, raw_job)

    total = (
        0.25 * sim            +
        0.35 * kw_score       +
        0.06 * quant["score"] +
        0.06 * verbs["score"] +
        0.07 * sec_score      +
        0.04 * con_score      +
        0.02 * title["score"] +
        0.05 * exp["score"]   +
        0.10 * edu["score"]
    )

    return {
        "total":        round(total, 1),
        "semantic":     round(sim, 1),
        "keyword":      round(kw_score, 1),
        "quantified":   quant["score"],
        "verbs":        verbs["score"],
        "title":        title["score"],
        "experience":   exp["score"],
        "education":    edu["score"],
        "sections":     round(sec_score, 1),
        "contact":      round(con_score, 1),
        "quant_detail": quant,
        "verb_detail":  verbs,
        "exp_detail":   exp,
        "title_detail": title,
        "edu_detail":   edu,
        "sections_map": sections,
        "contact_map":  contact,
        "word_count":   len(resume_clean.split()),
    }

# ================== FEEDBACK ==================

def generate_feedback(sim, ats, matched, missing) -> list:
    tips = []

    # Semantic
    if sim < 40:
        tips.append(("danger",  "Semantic match is very low. Rewrite your summary to mirror the job description language and responsibilities."))
    elif sim < 70:
        tips.append(("warning", "Semantic match is moderate. Align your experience bullets more closely with the job's stated responsibilities."))
    else:
        tips.append(("success", "Strong semantic alignment — your language closely mirrors what the role needs."))

    # Quantification
    q = ats["quant_detail"]
    if q["ratio"] < 7:
        tips.append(("danger",
            f"Only {q['quantified_count']} of your {q['total_bullets']} bullets contain measurable impact ({q['ratio']:.0f}%). "
            f"Add numbers like: 'Reduced API response time by 40%', 'Managed team of 8 engineers', 'Increased test coverage from 45% to 90%'."))
    elif q["ratio"] < 15:
        tips.append(("warning",
            f"{q['quantified_count']} quantified bullets detected ({q['ratio']:.0f}%). "
            f"Target 30–40% of your bullets having measurable impact. Add % improvements, team sizes, revenue figures, or time savings."))
    else:
        tips.append(("success",
            f"Good quantification — {q['quantified_count']} bullets with measurable impact ({q['ratio']:.0f}%)."))

    # Action verbs
    v = ats["verb_detail"]
    if v["ratio"] < 20:
        tips.append(("danger",
            f"Only {v['ratio']:.0f}% of your lines start with strong action verbs. "
            f"Start bullets with verbs like: Built, Designed, Led, Reduced, Deployed, Optimized."))
    elif v["ratio"] < 40:
        tips.append(("warning",
            f"Action verb usage is moderate ({v['ratio']:.0f}%). "
            f"Verbs found: {', '.join(v['verbs_used'][:6])}. Aim for 50%+ of bullets starting with strong action verbs."))
    else:
        tips.append(("success", f"Strong action verb usage ({v['ratio']:.0f}%). Verbs like {', '.join(v['verbs_used'][:5])} signal impact."))

    # Job title
    t = ats["title_detail"]
    if t["score"] < 50:
        tips.append(("warning", f"Job title alignment: {t['status']}. Consider adding the target role title in your summary section."))

    # Experience
    e = ats["exp_detail"]
    if e["score"] < 70:
        tips.append(("danger" if e["score"] < 40 else "warning", f"Experience: {e['match_status']}."))

    # Education
    ed = ats["edu_detail"]
    if ed["score"] < 50:
        tips.append(("warning", f"Education: {ed['status']}. Ensure your degree is clearly listed with institution and year."))

    # Missing skills
    if missing:
        top = ", ".join(f"`{s}`" for s in missing[:8])
        tips.append(("warning", f"Add these skills to your resume if you have them: {top}."))

    # Sections
    miss_sec = [s for s, v in ats["sections_map"].items() if not v and s in ["education","experience","skills"]]
    if miss_sec:
        tips.append(("danger", f"Missing sections: {', '.join(s.title() for s in miss_sec)}. Standard section headers are required for ATS parsing."))

    # Contact
    missing_contact = [f for f, v in ats["contact_map"].items() if not v]
    if missing_contact:
        tips.append(("warning", f"Missing contact info: {', '.join(missing_contact)}. Add these for recruiter reachability."))

    # Word count
    wc = ats["word_count"]
    if wc < 300:
        tips.append(("danger",  f"Resume is too short ({wc} words). Aim for 400–900 words."))
    elif wc > 1200:
        tips.append(("warning", f"Resume is too long ({wc} words). Trim to under 1000 words for ATS compliance."))

    return tips

# ================== CHARTS ==================

def score_color(v):
    if v >= 70: return "#22C55E"
    if v >= 40: return "#F59E0B"
    return "#EF4444"

def plot_ats_breakdown(ats):
    labels = ["Semantic\n(25%)", "Keywords\n(20%)", "Quantified\nImpact (15%)",
              "Action\nVerbs (10%)", "Sections\n(8%)", "Contact\n(7%)",
              "Education\n(5%)", "Experience\n(5%)", "Job Title\n(5%)"]
    scores = [ats["semantic"], ats["keyword"], ats["quantified"],
              ats["verbs"], ats["sections"], ats["contact"],
              ats["education"], ats["experience"], ats["title"]]
    colors = [score_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("#16161E")
    ax.set_facecolor("#16161E")

    bars = ax.barh(labels, scores, color=colors, height=0.55, zorder=3)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_tick_params(labelsize=8, colors="#8B8AA0")
    ax.yaxis.set_tick_params(labelsize=8, colors="#F0EFF8")
    ax.axvline(70, color=(1, 1, 1, 0.15), linewidth=0.8, linestyle="--", zorder=2)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#2A2A38", linewidth=0.5, zorder=0)
    for spine in ax.spines.values(): spine.set_visible(False)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8, color="#F0EFF8", fontweight="bold")

    plt.tight_layout(pad=1.0)
    return fig

def plot_skill_donut(matched, missing):
    m, n = len(matched), len(missing)
    if m + n == 0:
        return None

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor("#16161E")
    ax.set_facecolor("#16161E")

    wedges, _, autos = ax.pie(
        [m, n] if n else [m, 0.001],
        colors=["#22C55E", "#EF4444"],
        autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(width=0.52, edgecolor="#16161E", linewidth=2),
        pctdistance=0.75,
    )
    for auto in autos:
        auto.set_color("white")
        auto.set_fontsize(10)
        auto.set_fontweight("bold")

    ax.text(0, 0, f"{m+n}\nskills", ha="center", va="center",
            fontsize=11, color="#F0EFF8", fontweight="bold", linespacing=1.5)
    plt.tight_layout(pad=0)
    return fig

def plot_quant_gauge(ratio):
    fig, ax = plt.subplots(figsize=(5, 0.55))
    fig.patch.set_facecolor("#16161E")
    ax.set_facecolor("#16161E")

    ax.barh([0], [100], color="#2A2A38", height=0.5)
    ax.barh([0], [min(ratio, 100)], color=score_color(ratio * 2.5), height=0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 15, 30, 50, 100])
    ax.xaxis.set_tick_params(labelsize=7, colors="#8B8AA0")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.axvline(30, color=(1,1,1,0.3), linewidth=0.8, linestyle="--")
    plt.tight_layout(pad=0.2)
    return fig

# ================== MAIN ==================

def main():

    st.markdown('<p class="hero-title">HireReady</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Industry-grade ATS scoring · Quantification analysis · Semantic match · Skill gap detection</p>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown('<p class="section-label">Resume</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    with col_r:
        st.markdown('<p class="section-label">Job Description</p>', unsafe_allow_html=True)
        job_description = st.text_area("", height=180,
            placeholder="Paste the full job description here…",
            label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        analyze = st.button("⚡  Analyze Resume")

    if not analyze:
        return

    if not uploaded_file:
        st.warning("Please upload your resume PDF.")
        return
    if not job_description.strip():
        st.warning("Please paste the job description.")
        return

    with st.spinner("Running industry-grade ATS analysis…"):
        raw_resume = extract_text_from_pdf(uploaded_file)
        if not raw_resume:
            st.error("Could not extract text. Ensure the PDF is not image-only / scanned.")
            return

        # clean_text() ONLY used for semantic similarity model
        resume_clean = clean_text(raw_resume)
        job_clean    = clean_text(job_description)
        similarity   = calculate_similarity(resume_clean, job_clean)

        # Skill gap uses RAW text to correctly match node.js, c++, .net, etc.
        matched, missing = get_skill_gap(raw_resume, job_description)

        # ATS score also uses raw text internally for skill/keyword scoring
        ats      = calculate_ats_score(similarity, raw_resume, job_description, resume_clean)
        feedback = generate_feedback(similarity, ats, matched, missing)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── KPI Row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("ATS Score",          f"{ats['total']:.0f}%",
              "Strong" if ats['total'] >= 70 else ("Decent" if ats['total'] >= 50 else "Low"))
    k2.metric("Semantic Match",     f"{similarity:.0f}%",
              "Strong" if similarity >= 70 else ("Moderate" if similarity >= 40 else "Low"))
    k3.metric("Keyword Match",      f"{ats['keyword']:.0f}%",
              f"{len(matched)} of {len(matched)+len(missing)} skills")
    k4.metric("Quantified Bullets", f"{ats['quant_detail']['ratio']:.0f}%",
              f"{ats['quant_detail']['quantified_count']} found")
    k5.metric("Action Verbs",       f"{ats['verb_detail']['ratio']:.0f}%",
              f"{ats['verb_detail']['strong_count']} strong lines")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "  📊  ATS Scores  ",
        "  📈  Impact & Verbs  ",
        "  🛠  Skills  ",
        "  💡  Feedback  "
    ])

    # ─── TAB 1 — ATS SCORES ───
    with tab1:
        c1, c2 = st.columns([1.8, 1], gap="large")

        with c1:
            st.markdown('<p class="section-label">9-Component ATS Breakdown</p>', unsafe_allow_html=True)
            st.pyplot(plot_ats_breakdown(ats), use_container_width=True)

        with c2:
            st.markdown('<p class="section-label">Job Title Alignment</p>', unsafe_allow_html=True)
            t   = ats["title_detail"]
            cls = "badge-ok" if t["score"] >= 70 else "badge-bad"
            st.markdown(f'<span class="badge {cls}">{t["jd_role"]}</span>', unsafe_allow_html=True)
            st.caption(t["status"])

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">Experience</p>', unsafe_allow_html=True)
            e = ats["exp_detail"]
            st.markdown(
                f'<span class="badge {"badge-ok" if e["score"]>=70 else "badge-bad"}">{e["match_status"]}</span>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">Sections</p>', unsafe_allow_html=True)
            badges = ""
            for sec, present in ats["sections_map"].items():
                cls  = "badge-ok" if present else "badge-bad"
                icon = "✓" if present else "✗"
                badges += f'<span class="badge {cls}">{icon} {sec.title()}</span> '
            st.markdown(badges, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">Contact Info</p>', unsafe_allow_html=True)
            cbadges = ""
            for field, present in ats["contact_map"].items():
                cls  = "badge-ok" if present else "badge-bad"
                icon = "✓" if present else "✗"
                cbadges += f'<span class="badge {cls}">{icon} {field}</span> '
            st.markdown(cbadges, unsafe_allow_html=True)

    # ─── TAB 2 — IMPACT & VERBS ───
    with tab2:
        q = ats["quant_detail"]
        v = ats["verb_detail"]

        qa, va = st.columns(2, gap="large")

        with qa:
            st.markdown('<p class="section-label">Quantified Impact Analysis</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="tip-card">'
                f'<strong>{q["quantified_count"]}</strong> of <strong>{q["total_bullets"]}</strong> '
                f'bullets contain measurable impact — <strong>{q["ratio"]:.0f}%</strong><br>'
                f'<small>Industry target: 15–20% of bullets quantified</small>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.pyplot(plot_quant_gauge(q["ratio"]), use_container_width=True)

            if q["quantified_examples"]:
                st.markdown('<p class="section-label" style="margin-top:1rem">✓ Good Examples Found</p>', unsafe_allow_html=True)
                for ex in q["quantified_examples"]:
                    st.markdown(
                        f'<div class="tip-card" style="border-left-color:#22C55E;font-size:0.8rem">{ex[:120]}</div>',
                        unsafe_allow_html=True
                    )

            if q["unquantified_sample"]:
                st.markdown('<p class="section-label" style="margin-top:1rem">✗ Needs Numbers Added</p>', unsafe_allow_html=True)
                for ex in q["unquantified_sample"]:
                    st.markdown(
                        f'<div class="tip-card" style="border-left-color:#EF4444;font-size:0.8rem">{ex[:120]}<br>'
                        f'<small style="color:#8B8AA0">→ Try adding: "by X%", "for N users", "saving $X"</small></div>',
                        unsafe_allow_html=True
                    )

        with va:
            st.markdown('<p class="section-label">Action Verb Strength</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="tip-card">'
                f'<strong>{v["strong_count"]}</strong> lines start with strong action verbs — <strong>{v["ratio"]:.0f}%</strong><br>'
                f'<small>Industry target: 50%+ of bullets using action verbs</small>'
                f'</div>',
                unsafe_allow_html=True
            )

            if v["verbs_used"]:
                st.markdown('<p class="section-label" style="margin-top:1rem">Verbs Detected</p>', unsafe_allow_html=True)
                pills = "".join(f'<span class="skill-pill pill-match">{vb}</span>' for vb in v["verbs_used"])
                st.markdown(pills, unsafe_allow_html=True)

            st.markdown('<p class="section-label" style="margin-top:1rem">Suggested Power Verbs</p>', unsafe_allow_html=True)
            suggestions = ["Engineered","Optimized","Spearheaded","Architected",
                           "Accelerated","Reduced","Delivered","Scaled","Drove","Automated"]
            used_set  = set(v["verbs_used"])
            new_verbs = [s for s in suggestions if s.lower() not in used_set][:8]
            pills2 = "".join(f'<span class="skill-pill pill-miss">{vb}</span>' for vb in new_verbs)
            st.markdown(pills2, unsafe_allow_html=True)

    # ─── TAB 3 — SKILLS ───
    with tab3:
        s1, s2 = st.columns([1, 2], gap="large")
        with s1:
            st.markdown('<p class="section-label">Coverage</p>', unsafe_allow_html=True)
            donut = plot_skill_donut(matched, missing)
            if donut:
                st.pyplot(donut, use_container_width=True)
            else:
                st.info("No curated skills found in job description.")
        with s2:
            if matched:
                st.markdown('<p class="section-label">Matched Skills</p>', unsafe_allow_html=True)
                pills = "".join(f'<span class="skill-pill pill-match">{s}</span>' for s in matched)
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            if missing:
                st.markdown('<p class="section-label">Missing Skills</p>', unsafe_allow_html=True)
                pills = "".join(f'<span class="skill-pill pill-miss">{s}</span>' for s in missing)
                st.markdown(pills, unsafe_allow_html=True)

    # ─── TAB 4 — FEEDBACK ───
    with tab4:
        st.markdown('<p class="section-label">Prioritized Recommendations</p>', unsafe_allow_html=True)
        color_map = {
            "danger":  "#EF4444",
            "warning": "#F59E0B",
            "success": "#22C55E",
        }
        for level, tip in feedback:
            border_color = color_map.get(level, "#6C63FF")
            st.markdown(
                f'<div class="tip-card" style="border-left-color:{border_color}">{tip}</div>',
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()