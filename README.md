# Passport Seva Application Guide

Independent educational Hybrid RAG assistant for adults applying in India for an ordinary passport under the **Normal** process. Covers fresh applications, routine reissues, and appointment booking/rescheduling using indexed official Passport Seva sources with verified citations.

> **Disclaimer:** This is an independent internship project and is not affiliated with the Ministry of External Affairs or the Government of India. Never enter passport numbers, Application Reference Numbers (ARNs), or personal identity documents into the app.

---

## Prerequisites

- **Windows PowerShell**
- **Python 3.11+** with the virtual environment in `backend/.venv`
- **Node.js 20+** and **npm 10+**
- **`backend/.env`** containing `GROQ_API_KEY=<your-groq-api-key>` (and optional `GROQ_MODEL=`, which defaults to `openai/gpt-oss-20b`). Copy `backend/.env.example` if setting up from scratch.

---

## Startup Commands (Windows PowerShell)

Run the backend and frontend in two separate PowerShell terminals.

### Terminal 1 — FastAPI Backend (`http://127.0.0.1:8000`)

```powershell
cd C:\D\PassportSeva\backend
$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Interactive OpenAPI documentation: `http://127.0.0.1:8000/docs`
- Health check endpoint: `http://127.0.0.1:8000/api/health`

### Terminal 2 — React + TypeScript Frontend (`http://127.0.0.1:5173`)

```powershell
cd C:\D\PassportSeva\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open **`http://127.0.0.1:5173`** in your browser. Vite proxies `/api/*` requests (`GET /api/health` and `POST /api/ask`) to `http://127.0.0.1:8000`, so `GROQ_API_KEY` stays strictly on the backend.

---

## Rebuilding the Retrieval Index & Running Checks

### Rebuild the index (if `data/passport_seva/index/` is missing or sources changed)

```powershell
cd C:\D\PassportSeva\backend
$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\.venv\Scripts\python -m app.rag.cli build
```

This validates `data/passport_seva/source_manifest.json` and `seed_records.jsonl`, extracts `data/passport_seva/raw/appointment.pdf` (preserving numbered steps and 1-based PDF viewer pages while superseding duplicate seed paraphrases), and writes the 18-chunk BM25 + FAISS index to `data/passport_seva/index/`.

### Run backend unit + API tests, smoke check, and Phase 4 evaluation

```powershell
cd C:\D\PassportSeva\backend
$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m app.rag.cli smoke
.\.venv\Scripts\python -m app.rag.cli eval
```

### Run a single terminal retrieval query

```powershell
cd C:\D\PassportSeva\backend
$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\.venv\Scripts\python -m app.rag.cli query "I can't make it on the day I booked. Can I move it?" --mode hybrid
```

### Run frontend tests, linter, and production build

```powershell
cd C:\D\PassportSeva\frontend
npm test
npm run lint
npm run build
```

---

## Request Flow in Plain Language

1. **Question validation (`POST /api/ask`)** — FastAPI validates that the question is non-blank, within the 500-character limit, and specifies a valid retrieval mode (`hybrid` by default, or `semantic` / `rerank`).
2. **Deterministic routing policy (`backend/app/rag/policy.py`)** — Before calling any LLM, a rule-based classifier checks whether the question is:
   - **Out of scope** (`out_of_scope`) — e.g., Tatkaal, minors/children, diplomatic/official passports, PCC, or applications outside India.
   - **Live status request** (`live_status_redirect`) — e.g., passport dispatch or Speed Post tracking. Directs the user to the official portal without requesting an ARN or passport number.
   - **Disputed reschedule limit** (`conflicting_sources`) — asks *how many times* an appointment can be rescheduled. Official sources disagree (the main guide states three times for Normal applications, while the Payment FAQ and Appointment PDF state twice within a year of the first appointment), so the backend explains the discrepancy, cites the conflicting sources, and directs the user to check the remaining attempts displayed in their Passport Seva account.
   - **Vague document request** (`clarification`) — asks what documents are needed without specifying **Fresh** vs. **Reissue**. Prompts the user to choose Fresh or Reissue.
   - **Scenario-specific document checklist** (`official_redirect`) — asks for an exact checklist for Fresh or Reissue. Since no verified Advisor output is stored in the dataset, the backend links directly to the official Fresh or Reissue Document Advisor (`routing_only` records are never used as checklist evidence).
3. **Hybrid retrieval (`backend/app/rag/retrieve.py`)** — For in-scope procedural questions (`answer`), the retriever runs:
   - **Keyword search (Okapi BM25)** over shared tokenised chunk IDs, preserving acronyms (`ARN`, `PSK`, `POPSK`) and atomic portal button labels (`Pay and Schedule Appointment`, `Schedule/Reschedule Appointment`, `Track Payment Status`).
   - **Semantic search (FAISS `IndexFlatIP`)** using L2-normalised embeddings from `sentence-transformers/all-MiniLM-L6-v2` (cosine similarity).
   - **Reciprocal Rank Fusion (RRF, $k = 60$)** to merge the BM25 and semantic rankings into a single `hybrid` ranking.
   - **Optional cross-encoder reranking (`rerank` mode)** using `cross-encoder/ms-marco-MiniLM-L-6-v2` on the RRF shortlist. `hybrid` is kept as the application default because it achieves higher top-1 accuracy (85.0% vs. 65.0% across the 20 retrieval test cases) and ranks the 5-step PDF rescheduling walkthrough #1 on colloquial queries like *"Can I move it?"*.
4. **Evidence filtering & Groq generation (`backend/app/service.py`, `backend/app/llm.py`)** — Only `answer_evidence` chunks without disputed numeric reschedule limits are passed to Groq (`openai/gpt-oss-20b`, 20s bounded timeout, 0 retries) as numbered `<passage id="S1">...</passage>` blocks. The system prompt treats passages strictly as data, requires `[S#]` citations, and forbids outside knowledge or numeric reschedule limits.
5. **Citation verification** — The backend parses Groq's JSON response, verifies that every cited ID (`S1`, `S2`, …) and inline `[S#]` marker refers to a supplied passage, rejects any answer that states a numeric reschedule limit, and maps each ID to the canonical title, section, review date, 1-based PDF viewer pages (`[5, 6]`), and URL (`...#page=5`) from backend index metadata. Before rendering any link, the frontend also validates that the URL uses `https://` on `passportindia.gov.in` (or a subdomain).

---

## Short Demo Sequence

With both terminals running and **`http://127.0.0.1:5173`** open in your browser:

1. **Move a booked appointment & inspect the PDF citation**
   - Click the sample question button in the empty chat:
     > *"I can't make it on the day I booked. Can I move it?"*
   - Press **Enter** (or click **Send**).
   - Confirm that the assistant displays the 5-step rescheduling procedure and a **Sources** section citing **`[S1] New Process for Online Appointment Booking`** (`3. Cancel/Reschedule Appointment · PDF pages 5–6`), whose link points to `https://passportindia.gov.in/AppOnlineProject/pdf/New_Online_Appointment_Booking_Process.pdf#page=5`.
2. **Ask a vague document question & follow up with Reissue**
   - Enter:
     > *"What exact documents do I need?"*
   - Confirm that the assistant responds with **Clarification needed** and two choice buttons (**Fresh application** and **Reissue**).
   - Click **Reissue (previously held an ordinary passport)**. Confirm that the frontend submits the complete contextual follow-up (*"For a Normal adult Reissue passport application: What exact documents do I need?"*) and displays the verified link to the official **Reissue Document Advisor** (`https://services2.passportindia.gov.in/psp/docAdvisor/reissuePassport`).
3. **Ask how many times an appointment can be moved (conflicting sources)**
   - Enter:
     > *"How many times can I reschedule?"*
   - Confirm that the assistant flags **Official sources disagree**, gives **no definitive numeric limit**, explains that the user should check the remaining attempts shown in their Passport Seva account, and lists the three official sources (`apply`, `appointment`, `payment_faq`).

---

## Remaining Limitations

- **Dataset scope:** Covers only adults in India applying for an ordinary passport through the **Normal** route (11 curated seed passages from 8 official sources, with `reschedule_pdf` superseded by 8 chunks extracted from the official 6-page `appointment.pdf`, totaling 18 indexed chunks). Tatkaal, minor/child passports, diplomatic/official passports, PCC, and overseas applications are out of scope.
- **No stored document checklists:** The dataset intentionally contains no completed Document Advisor checklists (`fresh_advisor`, `reissue_advisor`, and `reissue_reason` are `routing_only`), so checklist requests always route to the official Advisor rather than listing documents.
- **Unverified live status tracker URL:** Following **Track Application Status** on the official portal (`https://www.passportindia.gov.in/psp`) points to the client-side SPA route `/psp/trackApplicationService` (which serves only a `<noscript>` JavaScript shell over HTTP), while the legacy endpoint `https://passportindia.gov.in/AppOnlineProject/statusTracker/trackStatusInpNew` returns HTTP `503`. Because neither URL can be verified to render the status tracker directly, the frontend omits the unverified link and instructs users to open **Track Application Status** from the official Passport Seva home page.
- **Reranker sensitivity on short colloquial queries:** The general-domain cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) favours short single-sentence FAQ paraphrases over multi-step PDF procedures when the query lacks exact lexical overlap (e.g., *"Can I move it?"* without the word *"reschedule"*). `hybrid` (BM25 + FAISS RRF) is therefore used as the default mode.
- **Browser testing environment:** Automated UI tests run in Vitest + JSDOM (`7/7` passing) alongside live HTTP verification through the Vite proxy (`http://127.0.0.1:5173/api/*`). Interactive visual layout inspection in a desktop/mobile browser window and manual server stop/restart retry should be spot-checked locally when presenting the demo.

## Android debug APK (Capacitor, PowerShell only, no Android Studio)

The APK bundles the built React app (`frontend\dist`). FastAPI must stay reachable
and needs internet for Groq. The Groq key stays in `backend\.env`; it is never in the APK.

**Toolchain for this project** (from the installed Capacitor 8.6.0): JDK 17+ (JDK 21 Temurin used),
Android Gradle Plugin 8.13.0, Gradle 8.14.3 (wrapper), compileSdk/targetSdk 36, minSdk 24, build-tools 36.0.0.

### 1. One-time SDK setup
Download "Command line tools only" (Windows) from https://developer.android.com/studio#command-tools,
extract so `sdkmanager.bat` is at `C:\Android\Sdk\cmdline-tools\latest\bin\sdkmanager.bat`, then:

```powershell
$env:ANDROID_HOME = "C:\Android\Sdk"
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-21.0.12.101-hotspot"
$env:Path = "$env:JAVA_HOME\bin;$env:ANDROID_HOME\cmdline-tools\latest\bin;$env:ANDROID_HOME\platform-tools;$env:Path"
sdkmanager.bat --sdk_root="$env:ANDROID_HOME" --licenses
sdkmanager.bat --sdk_root="$env:ANDROID_HOME" "platform-tools" "platforms;android-36" "build-tools;36.0.0"
```

### 2. Backend for the phone (separate terminal; phone and laptop on the same Wi-Fi)
```powershell
cd C:\D\PassportSeva\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Check `http://<laptop-ip>:8000/api/health` in the phone's browser first (allow Python through Windows Firewall if needed).
Find the IP with `ipconfig` (Wi-Fi IPv4; 192.168.0.106 when this was written). Use `10.0.2.2` only for the emulator.

### 3. Build
`VITE_API_BASE_URL` is an **origin only** (no trailing `/api`); the client appends `/api/ask` and `/api/health`.
`CAP_DEBUG_HTTP=1` serves the WebView over http and is for local-Wi-Fi debug builds only; omit it and use an `https://` backend for release.
```powershell
$env:ANDROID_HOME = "C:\Android\Sdk"
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-21.0.12.101-hotspot"
$env:VITE_API_BASE_URL = "http://192.168.0.106:8000"
$env:CAP_DEBUG_HTTP = "1"
cd C:\D\PassportSeva\frontend
npm test; npm run lint; npm run build
npx cap sync android
cd android
.\gradlew.bat assembleDebug
Copy-Item app\build\outputs\apk\debug\app-debug.apk C:\D\PassportSeva\Passport-Seva-Application-Guide-debug.apk
```
Cleartext HTTP is allowed only by `frontend\android\app\src\debug\AndroidManifest.xml`. FastAPI allows the WebView origins
(`http://localhost`, `https://localhost`, `capacitor://localhost`) via `CORS_ALLOW_ORIGINS` (comma separated override).

### 4. Install
USB debugging on, authorize the prompt, then:
```powershell
adb devices
adb install -r C:\D\PassportSeva\Passport-Seva-Application-Guide-debug.apk
```
Or copy the APK to the phone and open it. Manual checks: app opens with Vite stopped; header shows "Ready";
"I can't make it on the day I booked. Can I move it?" returns an answer citing the appointment PDF (PDF pages 5-6);
"How do I renew my passport?" then "What documents do I need?" shows the Reissue Document Advisor; stop FastAPI to see the retry UI.

### Rebuilding after the IP or backend URL changes
The URL is baked in at build time. Set the new `VITE_API_BASE_URL`, then rerun `npm run build`, `npx cap sync android`, `.\gradlew.bat assembleDebug` and copy the APK again.

## Deploying Backend to Render (Python Web Service)

The backend can be deployed directly to Render as a Python Web Service. Keep the service root at the repository root (`.`) because `data/` is located beside `backend/`.

### 1. Render Service Settings
- **Environment**: Python
- **Root Directory**: `.` (leave empty or set to repository root)
- **Build Command**: `python scripts/render_build.py`
- **Start Command**: `uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT --workers 1`
- **Health Check Path**: `/api/health`

### 2. Environment Variables in Render Dashboard
- `PYTHON_VERSION`: `3.11.7`
- `GROQ_API_KEY`: *(required)* Your Groq secret API key.
- `HF_HOME`: `.cache/huggingface` (persists cached model downloads in the build directory).
- `HF_HUB_DISABLE_SYMLINKS_WARNING`: `1`
- `TOKENIZERS_PARALLELISM`: `false`
- `CORS_ALLOW_ORIGINS`: `https://localhost,capacitor://localhost,http://localhost:5173` (add your deployed frontend domain).
- `PASSPORT_WARM_RERANKER`: `0` (keeps hybrid as default; avoids loading the cross-encoder into RAM at startup).

### 3. Sizing & Memory Concerns
- **Measured RAM**:
  - Python + FastAPI base: ~32 MB
  - With Embedder (`all-MiniLM-L6-v2`) warmed up (Hybrid): ~538 MB
  - With Reranker (`ms-marco-MiniLM-L-6-v2`) loaded: ~636 MB
- **Instance Recommendation**:
  - Render Free Tier has a strict **512 MB limit**. Running PyTorch on CPU exceeds 512 MB during warm-up/search (~538 MB), triggering OOM restarts (`Exit 137`).
  - Use a **Starter (512 MB with swap) or Standard (1 GB+ RAM)** instance for reliable production operation.
  - Run with **1 Uvicorn worker** (`--workers 1`) to avoid duplicating PyTorch model weights in memory.
