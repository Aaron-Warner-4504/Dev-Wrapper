import requests
import random
import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import hashlib
from time import time
from functools import wraps

# LangChain imports
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
app = Flask(__name__)
CORS(app)
from dotenv import load_dotenv
load_dotenv()
# Configuration
GITHUB_API_URL = "https://api.github.com"
LEETCODE_API_URL = "https://leetcode.com/graphql"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Optional: Add 'Bearer YOUR_TOKEN'

# LLM Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "") 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# In-memory rate limiter
LLM_USAGE_TRACKER = {}
MAX_LLM_ATTEMPTS = 3

# Rate Limiting Configuration
RATE_LIMIT = 5             # max requests
RATE_WINDOW = 60           # per 60 seconds
RATE_STORE = {}            # in-memory tracker

def rate_limit(limit=RATE_LIMIT, window=RATE_WINDOW):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ip = request.remote_addr
            now = time()
            if ip not in RATE_STORE:
                RATE_STORE[ip] = []
            # Remove old timestamps
            RATE_STORE[ip] = [t for t in RATE_STORE[ip] if now - t < window]
            if len(RATE_STORE[ip]) >= limit:
                return jsonify({
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Allowed {limit} per {window} seconds."
                }), 429
            RATE_STORE[ip].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Initialize LangChain Models
groq_llm = None
gemini_llm = None

if GROQ_API_KEY:
    try:
        groq_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    except Exception as e:
        print(f"Failed to initialize Groq: {e}")

if GEMINI_API_KEY:
    try:
        gemini_llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")

# JSON Output Parser
json_parser = JsonOutputParser()

# --- DATA HELPERS ---

# Keep Synergies deterministic as they represent specific tech stacks
SYNERGIES = {
    "FULLSTACK_WIZARD": {"parts": {"JavaScript", "TypeScript", "HTML", "CSS", "Python"}, "icon": "🌐", "title": "Fullstack Wizard"},
    "DATA_ALCHEMIST": {"parts": {"Python", "Jupyter Notebook", "R", "SQL"}, "icon": "⚗️", "title": "Data Alchemist"},
    "SYSTEMS_ARCHITECT": {"parts": {"C++", "C", "Rust", "Go", "Assembly"}, "icon": "🏗️", "title": "Systems Architect"},
    "MOBILE_RANGER": {"parts": {"Swift", "Kotlin", "Dart", "Java", "Objective-C"}, "icon": "📱", "title": "Mobile Ranger"},
    "SCRIPT_KIDDIE": {"parts": {"Shell", "PowerShell", "Batchfile", "Lua"}, "icon": "📜", "title": "Automation Rogue"},
    "ENTERPRISE_TITAN": {"parts": {"Java", "C#", "TypeScript"}, "icon": "🏢", "title": "Enterprise Titan"}
}

# --- FALLBACK DATA (For when LLM fails) ---
FALLBACK_DESCRIPTIONS = {
    "Code Barbarian": "High STR. You solve problems by throwing code at them until they break. Volume over elegance.",
    "Syntax Rogue": "High DEX. You switch languages faster than a context switch. Adaptable and tricky.",
    "Algorithm Wizard": "High INT. You optimize for O(log n). Your code is efficient, but maybe unreadable.",
    "Legacy Druid": "High WIS. You respect the old ways. Your code is stable, documented, and boring (in a good way).",
    "Open Source Bard": "High CHA. You write code for humans. Documentation, READMEs, and community are your weapons."
}

# --- LLM ENGINE ---

def call_groq_llm(prompt):
    """Call Groq using LangChain"""
    if not groq_llm:
        return None
    try:
        messages = [HumanMessage(content=prompt)]
        response = groq_llm.invoke(messages)
        return json.loads(response.content)
    except Exception as e:
        print(f"Groq LangChain Error: {e}")
        return None

def call_gemini_llm(prompt):
    """Call Gemini using LangChain"""
    if not gemini_llm:
        return None
    try:
        # Add JSON instruction to prompt for Gemini
        json_prompt = f"{prompt}\n\nIMPORTANT: Return ONLY valid JSON with no markdown formatting or additional text."
        messages = [HumanMessage(content=json_prompt)]
        response = gemini_llm.invoke(messages)
        
        # Clean up response content (remove markdown if present)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        return json.loads(content)
    except Exception as e:
        print(f"Gemini LangChain Error: {e}")
        return None

def generate_ai_analysis(gh, lc, username):
    """
    The Core Brain: Uses LLM to generate RPG Stats, Aura, Prophecy, and Feedback all at once.
    """
    
    # 1. Prepare Data Context
    stats_summary = f"""
    User: {username}
    GitHub: {gh.get('stars', 0)} stars, {gh.get('repos', 0)} repos, {gh.get('forks', 0)} forks.
    Top Languages: {', '.join(list(gh.get('languages', {}).keys())[:4])}.
    LeetCode: {lc.get('total', 0)} solved ({lc.get('hard', 0)} Hard, {lc.get('medium', 0)} Medium).
    Account Age: {gh.get('created_at', '2020')}
    """
    
    # 2. The Mega Prompt
    prompt = f"""
    Analyze this developer profile deeply: {stats_summary}
    
    Generate a JSON object with these exact keys representing their "Developer Soul":
    
    1. "rpg": {{
        "class": "Creative Class Name (e.g. 'Spaghetti Cowboy', 'Recursion Monk', 'Div Centering Warlock')",
        "level": Integer (1-100 based on overall impressiveness),
        "stats": {{ "STR": (0-100, volume/output), "DEX": (0-100, versatility/langs), "INT": (0-100, algorithm/complexity), "WIS": (0-100, experience/age), "CHA": (0-100, stars/social) }},
        "class_desc": "A witty 2-sentence description of this class's playstyle."
    }}
    
    2. "aura": {{
        "type": "Creative Name (e.g. 'Neon Chaos', 'Velvet Void')",
        "color": "Hex Code (matching the vibe)",
        "desc": "Short, mystical description of their code energy.",
        "explanation": "Why they have this aura (based on their data)."
    }}
    
    3. "horoscope": {{
        "prophecy": "A specific, funny coding prophecy for 2026.",
        "lucky_item": "A random tech object (e.g. 'Mechanical Spacebar', 'Cold Brew')"
    }}
    
    4. "feedback": {{
        "doing_right": "Genuine compliment on their best metric.",
        "doing_wrong": "Constructive roast about a weakness.",
        "reality_check": "Brutal honesty about their standing in the industry.",
        "toast": "A short celebratory cheer.",
        "roast": "A savage burn."
    }}
    
    Return ONLY valid JSON.
    """

    # 3. Call LangChain APIs
    result = call_groq_llm(prompt)
    if not result:
        result = call_gemini_llm(prompt)
    
    return result

# --- FALLBACK CALCULATORS (Deterministic) ---

def fallback_rpg_stats(gh, lc):
    # Safely get values with defaults
    repos = gh.get('repos', 0)
    forks = gh.get('forks', 0)
    stars = gh.get('stars', 0)
    followers = gh.get('followers', 0)
    languages = gh.get('languages', {})
    
    total_problems = lc.get('total', 0)
    hard_problems = lc.get('hard', 0)
    
    str_score = min(100, (repos * 1.5) + (total_problems / 4))
    dex_score = min(100, len(languages) * 10)
    int_score = min(100, (hard_problems * 4) + (forks * 2))
    
    # Calculate account age
    created_at = gh.get('created_at')
    if created_at:
        try:
            created = datetime.strptime(created_at[:10], '%Y-%m-%d')
            years = (datetime.now() - created).days / 365
        except:
            years = 1
    else:
        years = 1
    
    wis_score = min(100, years * 20)
    cha_score = min(100, (stars * 2) + (followers * 4))
    
    # Ensure all scores are at least 1 and integers
    stats = {
        "STR": max(1, int(str_score)), 
        "DEX": max(1, int(dex_score)), 
        "INT": max(1, int(int_score)), 
        "WIS": max(1, int(wis_score)), 
        "CHA": max(1, int(cha_score))
    }
    
    highest = max(stats, key=stats.get)
    classes = { 
        "STR": "Code Barbarian", 
        "DEX": "Syntax Rogue", 
        "INT": "Algorithm Wizard", 
        "WIS": "Legacy Druid", 
        "CHA": "Open Source Bard" 
    }
    rpg_class = classes.get(highest, "Novice Adventurer")
    
    return {
        "class": rpg_class,
        "level": max(1, int(sum(stats.values()) / 5)),
        "stats": stats,
        "class_desc": FALLBACK_DESCRIPTIONS.get(rpg_class, "A balanced developer.")
    }

def fallback_aura(top_lang):
    # Simple map
    if top_lang in ["C++", "Rust"]: return {"type": "Crimson Fury", "color": "#ef4444", "desc": "Intense and low-level.", "explanation": "Powered by C++/Rust"}
    if top_lang in ["Python", "JS"]: return {"type": "Neon Flux", "color": "#facc15", "desc": "Flexible and bright.", "explanation": "Powered by Scripting"}
    return {"type": "Null Void", "color": "#ffffff", "desc": "Mysterious and unknown.", "explanation": "No top language detected."}

def fallback_feedback():
    return {
        "doing_right": "You are shipping code.", "doing_wrong": "You need more consistency.", 
        "reality_check": "Keep pushing to stand out.", "toast": "Cheers to your commits!", "roast": "Your git history is a ghost town."
    }

# --- DATA FETCHING ---

def get_opensource_contributions(username):
    """Fetch detailed open source contribution data"""
    if not username:
        return None
    
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if GITHUB_TOKEN:
        headers['Authorization'] = GITHUB_TOKEN
    
    try:
        contributions = {
            "merged_prs": 0,
            "open_prs": 0,
            "closed_prs": 0,
            "total_prs": 0,
            "issues_opened": 0,
            "issues_closed": 0,
            "commits_to_others": 0,
            "repos_contributed_to": [],
            "top_contributions": [],
            "contribution_streak": 0,
            "languages_contributed": set(),
            "total_contributions_last_year": 0
        }
        
        # 1. Get all events (limited to 300 for performance)
        events = []
        for page in range(1, 4):  # 3 pages = ~90 events
            events_url = f"{GITHUB_API_URL}/users/{username}/events/public"
            params = {"per_page": 100, "page": page}
            resp = requests.get(events_url, headers=headers, params=params, timeout=5)
            if resp.status_code == 200:
                page_events = resp.json()
                if not page_events:
                    break
                events.extend(page_events)
            else:
                break
        
        # 2. Process events
        contributed_repos = {}
        
        for event in events:
            event_type = event.get('type')
            repo = event.get('repo', {})
            repo_name = repo.get('name', '')
            
            # Skip own repos
            if repo_name.startswith(f"{username}/"):
                continue
            
            # Count PRs
            if event_type == 'PullRequestEvent':
                action = event.get('payload', {}).get('action')
                contributions['total_prs'] += 1
                
                if action == 'opened':
                    contributions['open_prs'] += 1
                elif action == 'closed':
                    pr = event.get('payload', {}).get('pull_request', {})
                    if pr.get('merged'):
                        contributions['merged_prs'] += 1
                    else:
                        contributions['closed_prs'] += 1
                
                # Track repo
                if repo_name:
                    contributed_repos[repo_name] = contributed_repos.get(repo_name, 0) + 1
            
            # Count Issues
            elif event_type == 'IssuesEvent':
                action = event.get('payload', {}).get('action')
                if action == 'opened':
                    contributions['issues_opened'] += 1
                elif action == 'closed':
                    contributions['issues_closed'] += 1
                
                if repo_name:
                    contributed_repos[repo_name] = contributed_repos.get(repo_name, 0) + 1
            
            # Count commits to other repos
            elif event_type == 'PushEvent':
                commits = event.get('payload', {}).get('commits', [])
                contributions['commits_to_others'] += len(commits)
                
                if repo_name:
                    contributed_repos[repo_name] = contributed_repos.get(repo_name, 0) + len(commits)
        
        # 3. Sort and get top contributions
        sorted_repos = sorted(contributed_repos.items(), key=lambda x: x[1], reverse=True)
        contributions['repos_contributed_to'] = [repo for repo, _ in sorted_repos]
        contributions['top_contributions'] = [
            {"repo": repo, "contributions": count} 
            for repo, count in sorted_repos[:5]
        ]
        
        # 4. Get additional stats from user profile
        user_url = f"{GITHUB_API_URL}/users/{username}"
        user_resp = requests.get(user_url, headers=headers, timeout=5)
        if user_resp.status_code == 200:
            user_data = user_resp.json()
            contributions['public_repos'] = user_data.get('public_repos', 0)
            contributions['public_gists'] = user_data.get('public_gists', 0)
        
        # 5. Try to get contribution calendar (requires scraping or GraphQL)
        # For now, estimate from events
        contributions['total_contributions_last_year'] = len(events)
        
        # Convert set to list for JSON serialization
        contributions['languages_contributed'] = list(contributions['languages_contributed'])
        
        return contributions
        
    except Exception as e:
        print(f"Error fetching open source data: {e}")
        return None

def get_github_data(username):
    if not username: return None
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if GITHUB_TOKEN: headers['Authorization'] = GITHUB_TOKEN
    try:
        u_resp = requests.get(f"{GITHUB_API_URL}/users/{username}", headers=headers, timeout=5)
        if u_resp.status_code != 200: return None
        user = u_resp.json()
        r_resp = requests.get(f"{GITHUB_API_URL}/users/{username}/repos", params={"per_page": 100, "sort": "updated"}, headers=headers, timeout=5)
        repos = r_resp.json() if r_resp.status_code == 200 else []
        
        stats = {
            "username": user.get('login'), "avatar": user.get('avatar_url'),
            "repos": len(repos), "stars": sum(r.get('stargazers_count', 0) for r in repos),
            "forks": sum(r.get('forks_count', 0) for r in repos),
            "followers": user.get('followers', 0), "created_at": user.get('created_at'),
            "languages": {}
        }
        for r in repos:
            if r.get('language'): stats["languages"][r['language']] = stats["languages"].get(r['language'], 0) + 1
        
        stats["sorted_langs"] = dict(sorted(stats["languages"].items(), key=lambda x: x[1], reverse=True))
        stats["top_lang"] = next(iter(stats["sorted_langs"])) if stats["sorted_langs"] else "Unknown"
        return stats
    except: return None

def get_leetcode_data(username):
    if not username: return None
    query = """query userProfile($username: String!) { matchedUser(username: $username) { submitStats: submitStatsGlobal { acSubmissionNum { difficulty count } } } }"""
    try:
        resp = requests.post(LEETCODE_API_URL, json={"query": query, "variables": {"username": username}}, headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}, timeout=5)
        data = resp.json()
        if "errors" in data or not data.get("data") or not data["data"].get("matchedUser"): return None
        stats = data["data"]["matchedUser"]["submitStats"]["acSubmissionNum"]
        def get_cnt(d): return next((x["count"] for x in stats if x["difficulty"] == d), 0)
        return {"total": get_cnt("All"), "easy": get_cnt("Easy"), "medium": get_cnt("Medium"), "hard": get_cnt("Hard")}
    except: return None

def detect_synergies(gh):
    if not gh: return []
    user_langs = set(gh['languages'].keys())
    unlocked = []
    for key, synergy in SYNERGIES.items():
        if len(user_langs.intersection(synergy["parts"])) >= 2:
            unlocked.append({"title": synergy["title"], "icon": synergy["icon"]})
    return unlocked[:3]

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/test")
def test():
    """Test endpoint to verify data structure"""
    test_gh = {"repos": 10, "stars": 50, "forks": 5, "languages": {"Python": 5, "JavaScript": 3}, "followers": 20, "created_at": "2020-01-01T00:00:00Z", "top_lang": "Python"}
    test_lc = {"total": 100, "hard": 10, "medium": 50, "easy": 40}
    
    rpg = fallback_rpg_stats(test_gh, test_lc)
    aura = fallback_aura("Python")
    feedback = fallback_feedback()
    
    return jsonify({
        "rpg": rpg,
        "aura": aura,
        "feedback": feedback,
        "message": "Test data - check if RPG stats structure is correct"
    })

@app.route("/api/profile")
@rate_limit(limit=10, window=60)  # 10 requests per minute
def profile():
    gh_user = request.args.get("gh_username")
    lc_user = request.args.get("lc_username")
    if not gh_user and not lc_user: return jsonify({"error": "Username required"}), 400

    gh = get_github_data(gh_user)
    lc = get_leetcode_data(lc_user)
    
    if not gh and not lc: return jsonify({"error": "Server Overload ! Try again after some time !"}), 404

    gh_safe = gh or {"repos":0, "stars":0, "forks":0, "languages":{}, "followers":0, "created_at":None, "top_lang": "Unknown"}
    lc_safe = lc or {"total":0, "hard":0, "medium":0, "easy":0}
    target_user = gh_user or lc_user

    # 1. Tech Stack Detection (Deterministic)
    synergies = detect_synergies(gh)

    # 2. Try LLM Generation for EVERYTHING else
    usage = LLM_USAGE_TRACKER.get(target_user, 0)
    ai_data = None
    
    if usage < MAX_LLM_ATTEMPTS:
        ai_data = generate_ai_analysis(gh_safe, lc_safe, target_user)
        if ai_data: 
            LLM_USAGE_TRACKER[target_user] = usage + 1
            # Validate that ai_data has the required structure
            if not isinstance(ai_data.get("rpg"), dict) or "stats" not in ai_data.get("rpg", {}):
                print("LLM returned invalid RPG data structure")
                ai_data = None

    # 3. Assemble Response (LLM vs Fallback)
    if ai_data and ai_data.get("rpg") and ai_data.get("aura") and ai_data.get("feedback"):
        response_data = {
            "rpg": ai_data.get("rpg"),
            "aura": ai_data.get("aura"),
            "horoscope": ai_data.get("horoscope", {"prophecy": "Your code will compile on the first try.", "lucky_item": "Rubber Duck"}),
            "feedback": ai_data.get("feedback")
        }
    else:
        # Fallback Calculation
        print(f"Using fallback for {target_user}")
        rpg = fallback_rpg_stats(gh_safe, lc_safe)
        aura = fallback_aura(gh_safe.get("top_lang", ""))
        feedback = fallback_feedback()
        horoscope = {"prophecy": "The server is overloaded, but your code still compiles.", "lucky_item": "Error Logs"}
        
        response_data = {
            "rpg": rpg,
            "aura": aura,
            "horoscope": horoscope,
            "feedback": feedback
        }

    return jsonify({
        "username": target_user,
        "gh": gh,
        "lc": lc,
        "synergies": synergies,
        **response_data
    })

@app.route("/api/timeline", methods=["POST"])
@rate_limit(limit=5, window=60)  # 5 requests per minute
def timeline():
    """Generate a developer's activity timeline with monthly data"""
    data = request.json
    gh_username = data.get("gh_username")
    
    if not gh_username:
        return jsonify({"error": "GitHub username required"}), 400
    
    # Fetch GitHub events to build timeline
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if GITHUB_TOKEN:
        headers['Authorization'] = GITHUB_TOKEN
    
    try:
        # Get recent events
        events_url = f"{GITHUB_API_URL}/users/{gh_username}/events"
        events_resp = requests.get(events_url, headers=headers, timeout=5)
        events = events_resp.json() if events_resp.status_code == 200 else []
        
        # Build monthly activity map
        monthly_activity = {}
        for event in events[:100]:  # Last 100 events
            created_at = event.get('created_at', '')
            if created_at:
                month = created_at[:7]  # YYYY-MM format
                monthly_activity[month] = monthly_activity.get(month, 0) + 1
        
        # Sort by month
        sorted_months = sorted(monthly_activity.keys())
        timeline_data = {month: monthly_activity[month] for month in sorted_months[-12:]}  # Last 12 months
        
        # Generate analysis
        if timeline_data:
            best_month = max(timeline_data, key=timeline_data.get)
            worst_month = min(timeline_data, key=timeline_data.get)
            avg_activity = sum(timeline_data.values()) / len(timeline_data)
            
            summary = f"Averaged {int(avg_activity)} commits per month with peaks in {best_month}"
        else:
            best_month = "N/A"
            worst_month = "N/A"
            summary = "Activity data unavailable"
            timeline_data = {"2025-01": 0}
        
        return jsonify({
            "timeline": timeline_data,
            "analysis": {
                "summary": summary,
                "best_month": best_month,
                "worst_month": worst_month
            }
        })
        
    except Exception as e:
        print(f"Timeline error: {e}")
        return jsonify({
            "timeline": {"2025-01": 0},
            "analysis": {"summary": "No data", "best_month": "N/A", "worst_month": "N/A"}
        })

@app.route("/api/personality", methods=["POST"])
@rate_limit(limit=5, window=60)  # 5 requests per minute
def personality():
    """Analyze developer personality traits"""
    data = request.json
    gh_username = data.get("gh_username")
    
    if not gh_username:
        return jsonify({"error": "GitHub username required"}), 400
    
    # Get fresh data
    gh = get_github_data(gh_username)
    lc = get_leetcode_data(data.get("lc_username", ""))
    
    gh_safe = gh or {"repos": 0, "stars": 0, "forks": 0, "languages": {}}
    lc_safe = lc or {"easy": 0, "medium": 0, "hard": 0, "total": 0}
    
    prompt = f"""
    Analyze this developer's coding personality:
    GitHub: {gh_safe.get('repos', 0)} repos, {gh_safe.get('stars', 0)} stars, {gh_safe.get('forks', 0)} forks
    Languages: {', '.join(list(gh_safe.get('languages', {}).keys())[:5])}
    LeetCode: {lc_safe.get('easy', 0)} Easy, {lc_safe.get('medium', 0)} Medium, {lc_safe.get('hard', 0)} Hard
    
    Generate a JSON object with personality analysis:
    {{
        "type": "A creative developer archetype (e.g., 'The Perfectionist Hacker', 'The Speed Demon')",
        "traits": ["Trait1", "Trait2", "Trait3", "Trait4"],
        "strengths": ["Strength 1 description", "Strength 2 description", "Strength 3 description"],
        "weaknesses": ["Weakness 1 description", "Weakness 2 description"]
    }}
    
    Make it insightful and specific to their coding style. Return ONLY valid JSON.
    """
    
    result = call_groq_llm(prompt) or call_gemini_llm(prompt)
    
    if not result:
        # Fallback
        result = {
            "type": "The Pragmatic Coder",
            "traits": ["Productive", "Curious", "Adaptable", "Methodical"],
            "strengths": [
                "Consistent code output",
                "Willingness to learn new technologies",
                "Problem-solving mindset"
            ],
            "weaknesses": [
                "Could improve documentation",
                "Needs more algorithm practice"
            ]
        }
    
    return jsonify(result)

@app.route("/api/badges", methods=["POST"])
@rate_limit(limit=5, window=60)  # 5 requests per minute
def badges():
    """Generate achievement badges"""
    data = request.json
    gh_username = data.get("gh_username")
    
    if not gh_username:
        return jsonify({"error": "GitHub username required"}), 400
    
    # Get fresh data
    gh = get_github_data(gh_username)
    lc = get_leetcode_data(data.get("lc_username", ""))
    
    gh_safe = gh or {"repos": 0, "stars": 0, "languages": {}}
    lc_safe = lc or {"total": 0}
    
    prompt = f"""
    Create achievement badges for this developer:
    GitHub: {gh_safe.get('repos', 0)} repos, {gh_safe.get('stars', 0)} stars
    Languages: {', '.join(list(gh_safe.get('languages', {}).keys())[:5])}
    LeetCode: {lc_safe.get('total', 0)} solved
    
    Generate a JSON object with badges:
    {{
        "badges": [
            {{
                "name": "Badge Name",
                "emoji": "emoji",
                "desc": "What they earned it for"
            }},
            (5-8 badges total)
        ]
    }}
    
    Make them specific, funny, and based on their actual stats. Return ONLY valid JSON.
    """
    
    result = call_groq_llm(prompt) or call_gemini_llm(prompt)
    
    if not result or "badges" not in result:
        result = {
            "badges": [
                {"name": "Code Warrior", "emoji": "⚔️", "desc": f"Written {gh_safe.get('repos', 0)} repositories"},
                {"name": "Polyglot", "emoji": "🗣️", "desc": f"Knows {len(gh_safe.get('languages', {}))} programming languages"},
                {"name": "Problem Solver", "emoji": "🧩", "desc": f"Solved {lc_safe.get('total', 0)} LeetCode problems"},
                {"name": "Star Collector", "emoji": "⭐", "desc": f"Earned {gh_safe.get('stars', 0)} stars"},
                {"name": "Git Master", "emoji": "🐙", "desc": "Active GitHub contributor"}
            ]
        }
    
    return jsonify(result)

@app.route("/api/skills", methods=["POST"])
@rate_limit(limit=5, window=60)  # 5 requests per minute
def skills():
    """Analyze skill gaps and improvement areas"""
    data = request.json
    gh_username = data.get("gh_username")
    
    if not gh_username:
        return jsonify({"error": "GitHub username required"}), 400
    
    # Get fresh data
    gh = get_github_data(gh_username)
    lc = get_leetcode_data(data.get("lc_username", ""))
    
    gh_safe = gh or {"repos": 0, "languages": {}}
    lc_safe = lc or {"total": 0, "hard": 0}
    
    prompt = f"""
    Analyze skill gaps for this developer:
    Languages: {json.dumps(gh_safe.get('languages', {}))}
    Total Repos: {gh_safe.get('repos', 0)}
    LeetCode: {lc_safe.get('total', 0)} problems ({lc_safe.get('hard', 0)} hard)
    
    Generate a JSON object identifying skill gaps:
    {{
        "gaps": [
            {{
                "skill": "Skill area name (e.g., System Design, Testing, DevOps)",
                "severity": "High/Medium/Low",
                "roadmap": "Brief advice on how to improve"
            }},
            (3-5 gaps)
        ]
    }}
    
    Be constructive and specific. Return ONLY valid JSON.
    """
    
    result = call_groq_llm(prompt) or call_gemini_llm(prompt)
    
    if not result or "gaps" not in result:
        top_lang = next(iter(gh_safe.get('languages', {})), 'Python')
        result = {
            "gaps": [
                {"skill": "Algorithm Complexity", "severity": "Medium", "roadmap": "Practice more LeetCode hard problems"},
                {"skill": "Testing & QA", "severity": "High", "roadmap": "Add unit tests to existing projects"},
                {"skill": "System Design", "severity": "Medium", "roadmap": "Study distributed systems patterns"}
            ]
        }
    
    return jsonify(result)

@app.route("/api/bug_predictor", methods=["POST"])
@rate_limit(limit=5, window=60)  # 5 requests per minute
def bug_predictor():
    """Predict future bugs and coding mishaps"""
    data = request.json
    gh_username = data.get("gh_username")
    
    if not gh_username:
        return jsonify({"error": "GitHub username required"}), 400
    
    # Get fresh data
    gh = get_github_data(gh_username)
    lc = get_leetcode_data(data.get("lc_username", ""))
    
    gh_safe = gh or {"repos": 0, "languages": {}}
    top_lang = next(iter(gh_safe.get('languages', {})), 'Unknown')
    
    prompt = f"""
    Be a mystical fortune teller predicting {gh_username}'s future coding mishaps.
    
    Based on their profile:
    - {gh_safe.get('repos', 0)} repos
    - Top language: {top_lang}
    - {lc.get('total', 0) if lc else 0} LeetCode problems solved
    
    Generate a JSON object with bug predictions:
    {{
        "predictions": [
            {{
                "bug": "Specific bug they'll encounter (e.g., 'Off-by-one error in loop')",
                "tip": "Sarcastic advice on how to avoid it"
            }},
            (3-5 predictions)
        ]
    }}
    
    Be creative, funny, and specific to their language/style. Return ONLY valid JSON.
    """
    
    result = call_groq_llm(prompt) or call_gemini_llm(prompt)
    
    if not result or "predictions" not in result:
        result = {
            "predictions": [
                {"bug": "Null pointer exception at 3 AM", "tip": "Maybe check if the object exists first?"},
                {"bug": "Merge conflict from hell", "tip": "Learn to pull before you push"},
                {"bug": "CSS alignment that defies physics", "tip": "Flexbox is your friend, not your enemy"},
                {"bug": "Infinite loop in production", "tip": "Test your exit conditions"}
            ]
        }
    
    return jsonify(result)

@app.route("/api/toxic_code", methods=["POST"])
@rate_limit(limit=3, window=60)  # 3 requests per minute (stricter for code review)
def toxic_code():
    """Roast user's code snippet"""
    data = request.json
    code = data.get("code", "")
    
    if not code or len(code.strip()) < 10:
        return jsonify({"error": "Code snippet too short"}), 400
    
    prompt = f"""
    You are a brutally honest code reviewer. Roast this code snippet with humor:
    
    ```
    {code}
    ```
    
    Generate a JSON object:
    {{
        "roast": "A savage but funny roast of the code (2-3 sentences)",
        "severity": Integer from 1-10 (how bad is it),
        "fix": "One actionable tip to improve it"
    }}
    
    Be harsh but constructive. Return ONLY valid JSON.
    """
    
    result = call_groq_llm(prompt) or call_gemini_llm(prompt)
    
    if not result:
        # Fallback roast
        result = {
            "roast": "This code looks like it was written during a power outage. I've seen more structure in a bowl of spaghetti.",
            "severity": 7,
            "fix": "Start by adding comments. Any comments. Please."
        }
    
    return jsonify(result)

@app.route("/api/opensource", methods=["POST"])
@rate_limit(limit=5, window=60)  # 5 requests per minute
def opensource():
    """Get detailed open source contribution statistics"""
    data = request.json
    gh_username = data.get("gh_username")
    
    if not gh_username:
        return jsonify({"error": "GitHub username required"}), 400
    
    # Fetch open source data
    os_data = get_opensource_contributions(gh_username)
    
    if not os_data:
        # Fallback data
        os_data = {
            "merged_prs": 0,
            "open_prs": 0,
            "total_prs": 0,
            "issues_opened": 0,
            "repos_contributed_to": [],
            "top_contributions": [],
            "message": "Could not fetch contribution data"
        }
    
    # Generate AI insights about their open source profile
    if os_data.get('merged_prs', 0) > 0 or os_data.get('total_prs', 0) > 0:
        prompt = f"""
        Analyze this developer's open source contributions:
        - Merged PRs: {os_data.get('merged_prs', 0)}
        - Open PRs: {os_data.get('open_prs', 0)}
        - Issues Opened: {os_data.get('issues_opened', 0)}
        - Repos Contributed To: {len(os_data.get('repos_contributed_to', []))}
        - Top Contributions: {os_data.get('top_contributions', [])}
        
        Generate a JSON object:
        {{
            "title": "A creative title for their open source persona (e.g., 'The PR Ninja', 'Community Champion')",
            "summary": "A 2-3 sentence summary of their open source impact",
            "strength": "Their biggest open source strength",
            "next_step": "Advice on how to increase their impact",
            "vibe": "A fun one-word descriptor (e.g., 'Legendary', 'Rising', 'Emerging')"
        }}
        
        Be encouraging but honest. Return ONLY valid JSON.
        """
        
        insights = call_groq_llm(prompt) or call_gemini_llm(prompt)
        
        if not insights:
            insights = {
                "title": "Open Source Contributor",
                "summary": f"Has contributed to {len(os_data.get('repos_contributed_to', []))} repositories with {os_data.get('merged_prs', 0)} merged PRs.",
                "strength": "Active community participation",
                "next_step": "Keep contributing to diverse projects",
                "vibe": "Solid"
            }
        
        os_data['insights'] = insights
    else:
        os_data['insights'] = {
            "title": "Open Source Newcomer",
            "summary": "Just getting started with open source contributions. Every journey begins with a single PR!",
            "strength": "Potential waiting to be unleashed",
            "next_step": "Find a project you care about and open your first issue or PR",
            "vibe": "Fresh"
        }
    
    return jsonify(os_data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
