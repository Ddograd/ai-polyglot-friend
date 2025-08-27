# app.py
"""
AI Polyglot Friend — Streamlit app
- Multi-user self-learning language friend
- Admin panel (password-protected) with user management + dedupe
- Optional server-side audio transcription (ASR) for uploaded files (requires SpeechRecognition + pydub + ffmpeg)
- Optional TTS playback (pyttsx3 offline or gTTS)
Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import json, os, random, difflib, tempfile, traceback
from datetime import datetime
from pathlib import Path

# ---------------- configuration ----------------
GLOBAL_FILE = "ai_friend_global.json"
ADMIN_PASSWORD = os.environ.get("AI_ADMIN_PW", "changeme")  # change this or set env var

# optional libs detection
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    ASR_AVAILABLE = True
except Exception:
    ASR_AVAILABLE = False

try:
    import pyttsx3
    TTS_ENGINE = "pyttsx3"
except Exception:
    try:
        from gtts import gTTS
        TTS_ENGINE = "gtts"
    except Exception:
        TTS_ENGINE = None

# ---------------- utilities ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def normalize(s):
    if not s: return ""
    repl = {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n","ü":"u"}
    return "".join(repl.get(ch,ch) for ch in s).lower().strip()

def fuzzy_ratio(a,b):
    return difflib.SequenceMatcher(a=normalize(a), b=normalize(b)).ratio()

# ---------------- seed prompts ----------------
SEED = {
  "es": [
    {"native":"hello","target":"hola","topic":"greet","level":1},
    {"native":"how are you?","target":"¿cómo estás?","topic":"greet","level":1},
    {"native":"i want water please","target":"quiero agua por favor","topic":"cafe","level":1},
    {"native":"where is the bathroom?","target":"¿dónde está el baño?","topic":"travel","level":1},
  ],
  "fr": [
    {"native":"hello","target":"bonjour","topic":"greet","level":1},
    {"native":"i want water please","target":"je veux de l'eau, s'il vous plaît","topic":"cafe","level":1},
  ],
  "de": [
    {"native":"hello","target":"hallo","topic":"greet","level":1},
  ]
}

# ---------------- global memory helpers ----------------
def default_user():
    return {
        "name": None,
        "lang": "es",
        "level": 1,
        "stats": {"correct":0,"wrong":0,"recent":[]},
        "custom_vocab": {},            # lang -> list of {native,target,topic,level,last_used}
        "conversation_history": []     # list of {when,prompt,your,target,score}
    }

def load_global():
    if Path(GLOBAL_FILE).exists():
        try:
            with open(GLOBAL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # ensure shape
            if "users" not in data:
                data = {"users": data}
            return data
        except Exception:
            return {"users": {}}
    return {"users": {}}

def save_global(data):
    # dedupe before write
    data = deduplicate_global(data)
    with open(GLOBAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------- deduplication ----------------
def deduplicate_user(mem):
    # custom_vocab dedupe per language
    for lang, vlist in list(mem.get("custom_vocab", {}).items()):
        seen = set(); unique=[]
        for it in vlist:
            native = (it.get("native") or "").strip()
            target = (it.get("target") or it.get("correct_answer") or "").strip()
            topic = (it.get("topic") or "").strip()
            level = int(it.get("level") or 1)
            key = (normalize(native), normalize(target), topic, level)
            if key not in seen:
                seen.add(key)
                unique.append({"native": native, "target": target, "topic": topic, "level": level, "last_used": it.get("last_used") or now_iso()})
        mem["custom_vocab"][lang] = unique
    # history dedupe
    seen=set(); unique=[]
    for turn in mem.get("conversation_history", []):
        u = (turn.get("your") or turn.get("user") or "").strip()
        a = (turn.get("ai") or turn.get("reply") or "").strip()
        l = turn.get("lang") or mem.get("lang") or ""
        key = (normalize(u), normalize(a), l)
        if key not in seen:
            seen.add(key)
            unique.append({"when": turn.get("when") or now_iso(), "prompt": turn.get("prompt") or "", "your": u, "target": turn.get("target") or "", "score": turn.get("score")})
    mem["conversation_history"] = unique[-500:]
    # trim recent stats
    mem["stats"]["recent"] = mem["stats"].get("recent", [])[-200:]
    return mem

def deduplicate_global(data):
    # data = {"users":{username: userdata}}
    users = data.get("users", {})
    for uname, udata in users.items():
        users[uname] = deduplicate_user(udata)
    data["users"] = users
    return data

# ---------------- ASR & TTS helpers ----------------
def transcribe_uploaded_audio(uploaded_file):
    if not ASR_AVAILABLE:
        raise RuntimeError("ASR libs not installed on server.")
    # save to temp file
    tmp_in = tempfile.NamedTemporaryFile(delete=False)
    tmp_in.write(uploaded_file.getvalue()); tmp_in.flush(); tmp_in.close()
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        audioseg = AudioSegment.from_file(tmp_in.name)
        audioseg.export(tmp_wav, format="wav")
    except Exception:
        # fallback: try to write raw bytes as wav (may fail)
        with open(tmp_wav, "wb") as wf:
            wf.write(open(tmp_in.name, "rb").read())
    r = sr.Recognizer()
    with sr.AudioFile(tmp_wav) as source:
        audio = r.record(source)
    text = r.recognize_google(audio)
    try:
        os.remove(tmp_in.name); os.remove(tmp_wav)
    except:
        pass
    return text

def tts_bytes(text, lang_code="en"):
    if TTS_ENGINE == "pyttsx3":
        try:
            engine = pyttsx3.init()
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            engine.save_to_file(text, tmp.name)
            engine.runAndWait()
            data = open(tmp.name, "rb").read()
            os.remove(tmp.name)
            return data, "audio/wav"
        except Exception as e:
            # fallback to gTTS if available
            pass
    if TTS_ENGINE == "gtts":
        try:
            from gtts import gTTS
            t = gTTS(text=text, lang=lang_code.split("-")[0] if "-" in lang_code else lang_code)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()
            t.save(tmp.name)
            data = open(tmp.name, "rb").read()
            os.remove(tmp.name)
            return data, "audio/mp3"
        except Exception as e:
            raise RuntimeError("TTS failed: " + str(e))
    raise RuntimeError("No TTS engine available on server.")

# ---------------- AI / learning helpers ----------------
def choose_prompt_for_user(user_mem):
    lang = user_mem.get("lang", "es")
    level = user_mem.get("level", 1)
    prompts = SEED.get(lang, []).copy()
    # include custom vocab entries
    for ent in user_mem.get("custom_vocab", {}).get(lang, []):
        prompts.append({"native": ent.get("native",""), "target": ent.get("target",""), "topic": ent.get("topic","misc"), "level": ent.get("level",1)})
    if not prompts:
        return None
    pool = [p for p in prompts if p.get("level",1) <= level]
    return random.choice(pool or prompts)

def score_answer(user_text, target):
    return fuzzy_ratio(user_text, target)

def add_attempt_to_vocab(user_mem, lang, user_ans, correct_ans, topic="misc", level=1):
    ent = {"native": user_ans, "target": correct_ans, "topic": topic, "level": level, "last_used": now_iso()}
    user_mem.setdefault("custom_vocab", {}).setdefault(lang, []).append(ent)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Polyglot Friend", layout="wide")
st.title("🌍 AI Polyglot Friend — Multiuser")

# load global once into session
if "global_data" not in st.session_state:
    st.session_state.global_data = load_global()

global_data = st.session_state.global_data
if "ui_history" not in st.session_state:
    st.session_state.ui_history = []  # simple conversation view for current session

# sidebar mode
mode = st.sidebar.radio("Mode", ["Chat", "Admin"])

if mode == "Chat":
    st.sidebar.subheader("Sign in / Create user")
    username = st.sidebar.text_input("Username (unique)", value="")
    if not username:
        st.sidebar.info("Enter a username and press 'Load user'.")
    if st.sidebar.button("Load user"):
        if username.strip() == "":
            st.sidebar.error("Enter a valid username.")
        else:
            uname = username.strip()
            if "users" not in global_data:
                global_data["users"] = {}
            if uname not in global_data["users"]:
                global_data["users"][uname] = default_user()
                global_data["users"][uname]["name"] = uname
                save_global(global_data)
            st.session_state.current_user = uname
            st.experimental_rerun()

    if "current_user" not in st.session_state:
        st.info("Load an existing user or create a new one from the sidebar.")
    else:
        uname = st.session_state.current_user
        user_mem = global_data["users"].get(uname, default_user())
        st.sidebar.write(f"Signed in as: **{uname}**")
        user_mem["name"] = st.sidebar.text_input("Display name", value=user_mem.get("name") or uname)
        user_mem["lang"] = st.sidebar.selectbox("Practice language", ["es","fr","de","it","pt","ja","en"], index=["es","fr","de","it","pt","ja","en"].index(user_mem.get("lang","es")))
        user_mem["level"] = st.sidebar.slider("Level", 1, 5, user_mem.get("level",1))
        if st.sidebar.button("Save profile"):
            global_data["users"][uname] = user_mem
            save_global(global_data)
            st.sidebar.success("Profile saved.")

        # Main chat column
        col_main, col_right = st.columns([3,1])
        with col_main:
            st.subheader(f"Chat — {user_mem.get('name') or uname}")
            for who, text in st.session_state.ui_history[-40:]:
                if who == "you":
                    st.markdown(f"<div style='text-align:right;background:#e8f7ff;padding:8px;border-radius:8px'><b>You:</b> {text}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background:#f5f5f5;padding:8px;border-radius:8px'><b>AI:</b> {text}</div>", unsafe_allow_html=True)

            user_text = st.text_input("Type your answer / message here")
            audio_upload = st.file_uploader("Or upload audio to transcribe (wav/mp3/m4a/ogg)", type=["wav","mp3","m4a","ogg"])
            action = st.selectbox("Action", ["Practice prompt", "Send message", "Add custom phrase", "Show stats"])

            if st.button("Submit"):
                transcribed = None
                if audio_upload is not None:
                    try:
                        transcribed = transcribe_uploaded_audio(audio_upload)
                        st.success("Transcribed: " + transcribed)
                    except Exception as e:
                        st.warning("Transcription failed: " + str(e))
                text = (user_text or transcribed or "").strip()
                if not text:
                    st.warning("Please type something or upload audio.")
                else:
                    if action == "Send message":
                        st.session_state.ui_history.append(("you", text))
                        reply = f"I heard: {text}"
                        st.session_state.ui_history.append(("ai", reply))
                        # try TTS
                        try:
                            data, mime = tts_bytes(reply, lang_code=user_mem.get("lang","en"))
                            st.audio(data, format=mime)
                        except Exception as e:
                            st.warning("TTS unavailable: " + str(e))
                    elif action == "Practice prompt":
                        item = choose_prompt_for_user(user_mem)
                        if not item:
                            st.info("No prompts available. Add custom phrases.")
                        else:
                            # show prompt
                            prompt_msg = f"Translate/say in {user_mem.get('lang')}: '{item.get('native')}'"
                            st.session_state.ui_history.append(("ai", prompt_msg))
                            # if user answered in input -> score now
                            if text:
                                score = score_answer(text, item.get("target",""))
                                correct = score >= 0.65
                                if correct:
                                    reply = f"✅ Good (score {round(score,2)}). Correct: {item.get('target')}"
                                    user_mem["stats"]["correct"] += 1
                                else:
                                    reply = f"❌ Not quite (score {round(score,2)}). Correct: {item.get('target')}"
                                    user_mem["stats"]["wrong"] += 1
                                    add_attempt_to_vocab(user_mem, user_mem.get("lang","es"), text, item.get("target",""), item.get("topic","misc"), user_mem.get("level",1))
                                # update recent window
                                user_mem["stats"].setdefault("recent", []).append(1 if correct else 0)
                                user_mem["stats"]["recent"] = user_mem["stats"]["recent"][-20:]
                                # auto-level adjust
                                if len(user_mem["stats"]["recent"]) >= 6:
                                    rate = sum(user_mem["stats"]["recent"]) / len(user_mem["stats"]["recent"])
                                    if rate >= 0.8 and user_mem["level"] < 5:
                                        user_mem["level"] += 1
                                        reply += f" 🎉 Level up to {user_mem['level']}"
                                    elif rate <= 0.5 and user_mem["level"] > 1:
                                        user_mem["level"] -= 1
                                        reply += f" ⚠️ Level down to {user_mem['level']}"
                                # record history
                                user_mem.setdefault("conversation_history", []).append({"when": now_iso(), "prompt": item.get("native"), "your": text, "target": item.get("target"), "score": round(score,2)})
                                st.session_state.ui_history.append(("you", text))
                                st.session_state.ui_history.append(("ai", reply))
                                # TTS reply
                                try:
                                    data, mime = tts_bytes(reply, lang_code=user_mem.get("lang","en"))
                                    st.audio(data, format=mime)
                                except Exception as e:
                                    st.warning("TTS failed: " + str(e))
                            else:
                                st.session_state.ui_history.append(("ai", "Type your answer now and press Submit to score."))
                        # save user state
                        global_data["users"][uname] = user_mem
                        save_global(global_data)
                    elif action == "Add custom phrase":
                        # expects "native || target"
                        if "||" in text:
                            native, target = [p.strip() for p in text.split("||",1)]
                            ent = {"native": native, "target": target, "topic": "user", "level": user_mem.get("level",1), "last_used": now_iso()}
                            user_mem.setdefault("custom_vocab", {}).setdefault(user_mem.get("lang","es"), []).append(ent)
                            st.success("Added custom phrase.")
                            global_data["users"][uname] = user_mem
                            save_global(global_data)
                            st.session_state.ui_history.append(("you", text))
                            st.session_state.ui_history.append(("ai", f"Added: {native} → {target}"))
                        else:
                            st.warning("Use format: native || target")
                    elif action == "Show stats":
                        st.session_state.ui_history.append(("you", text))
                        st.session_state.ui_history.append(("ai", f"Stats: correct={user_mem['stats'].get('correct',0)} wrong={user_mem['stats'].get('wrong',0)} level={user_mem.get('level',1)}"))
                        save_global(global_data)
                    # clear input
                    st.session_state.user_text = ""
        with col_right:
            st.subheader("Quick")
            if st.button("Clear view"):
                st.session_state.ui_history = []
            if st.button("Show my recent history"):
                st.write(user_mem.get("conversation_history", [])[-10:])
            if st.button("Show my vocab"):
                st.json(user_mem.get("custom_vocab", {}))

elif mode == "Admin":
    st.sidebar.subheader("Admin login")
    pw = st.sidebar.text_input("Admin password", type="password")
    if pw == ADMIN_PASSWORD:
        st.subheader("🛠 Admin Panel")
        data = global_data.get("users", {})
        users = list(data.keys())
        st.write(f"Users: {len(users)}")
        if users:
            sel = st.selectbox("Select user", users)
            udata = data.get(sel, default_user())
            st.write("### Profile")
            st.json({"name": udata.get("name"), "lang": udata.get("lang"), "level": udata.get("level"), "stats": udata.get("stats")})
            with st.expander("Vocabulary"):
                st.json(udata.get("custom_vocab", {}))
            with st.expander("History (last 50)"):
                for t in udata.get("conversation_history", [])[-50:]:
                    st.write(f"👤 {t.get('your')} → 🤖 {t.get('target')}  (score {t.get('score')})")
            if st.button("Download user data"):
                st.download_button("Download JSON", json.dumps(udata, indent=2, ensure_ascii=False), file_name=f"{sel}_backup.json")
            if st.button("Reset user data"):
                global_data["users"][sel] = default_user()
                save_global(global_data)
                st.success("Reset.")
            if st.button("Delete user"):
                del global_data["users"][sel]
                save_global(global_data)
                st.warning("Deleted.")
        else:
            st.info("No users yet.")
        st.markdown("---")
        if st.button("Deduplicate all users"):
            global_data = deduplicate_global(global_data)
            save_global(global_data)
            st.success("Deduplicated global data.")
        st.markdown("Manual global import / export")
        uploaded = st.file_uploader("Upload global JSON to merge", type=["json"])
        if uploaded is not None:
            try:
                rd = json.load(uploaded)
                # naive merge: add users from uploaded
                for k,v in rd.get("users", rd).items():
                    if k not in global_data.get("users", {}):
                        global_data.setdefault("users", {})[k] = v
                    else:
                        # merge vocab & history simple append
                        existing = global_data["users"][k]
                        existing.setdefault("conversation_history", []).extend(v.get("conversation_history",[]))
                        for lang, vv in v.get("custom_vocab", {}).items():
                            existing.setdefault("custom_vocab", {}).setdefault(lang, []).extend(vv)
                save_global(global_data)
                st.success("Merged uploaded global into current global store.")
            except Exception as e:
                st.error("Failed to merge: " + str(e))
        if Path(GLOBAL_FILE).exists():
            with open(GLOBAL_FILE, "r", encoding="utf-8") as f:
                txt = f.read()
            st.download_button("Download global JSON", data=txt, file_name=GLOBAL_FILE, mime="application/json")
    else:
        st.warning("Enter admin password to access the admin panel.")

# final save to ensure data persisted
save_global(global_data)
