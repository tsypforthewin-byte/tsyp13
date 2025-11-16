# Project overview

This repository contains a small job-search app with a frontend, a backend proxy service, and Firebase Cloud Functions. Below is a short description of each top-level folder and how to run them locally and in production-like environments.

## Prerequisites

- Node.js (v16+ recommended; functions specify Node 22 in `functions/package.json`) and npm.
- Firebase CLI (if you want to run or deploy `functions/`).
- For local development: your provider API keys (Jooble key for the backend, Firebase config for the frontend).

## Folder documentation

- `backend/`
  - What it is: A small Express.js server that proxies search requests to the Jooble API and cleans job descriptions.
  - Files of interest: `server.js`, `package.json`, `.env.example` (added).
  - Environment variables:
    - `JOOBLE_API_KEY` — required for contacting the Jooble API. Do not commit this value.
  - How to run locally:
    ```bash
    cd backend
    npm install
    cp .env.example .env
    # Edit .env and set JOOBLE_API_KEY=your_actual_key
    npm start
    ```
  - Notes: The server listens on port 3001 by default. If `JOOBLE_API_KEY` is missing the server will warn and requests to Jooble will fail.

- `frontend/`
  - What it is: React app built with Create React App. Presents job search UI and calls the backend for results. Uses Firebase for optional Firestore integration.
  - Files of interest: `src/`, `public/`, `package.json`, `.env` (local placeholder), `README.md` (frontend-specific docs).
  - Environment variables (for local dev create `.env` at `frontend/.env`):
    - `REACT_APP_FIREBASE_API_KEY`
    - `REACT_APP_FIREBASE_AUTH_DOMAIN`
    - `REACT_APP_FIREBASE_PROJECT_ID`
    - `REACT_APP_FIREBASE_STORAGE_BUCKET`
    - `REACT_APP_FIREBASE_MESSAGING_SENDER_ID`
    - `REACT_APP_FIREBASE_APP_ID`
  - How to run locally:
    ```bash
    cd frontend
    npm install
    # copy example or create .env with the REACT_APP_FIREBASE_* values
    npm start
    ```
  - How to build for production:
    ```bash
    cd frontend
    npm run build
    ```

- `functions/`
  - What it is: Firebase Cloud Functions (serverless backend) used alongside Firebase products. Contains `index.js` and `package.json` with Firebase scripts.
  - Files of interest: `index.js`, `package.json`.
  - How to run locally (emulator):
    ```bash
    cd functions
    npm install
    npm run serve
    ```
  - How to deploy to Firebase:
    ```bash
    cd functions
    npm install
    firebase deploy --only functions
    ```
  - Notes: You may need to run `firebase login` and `firebase use --add` to select a project.

## Root-level files

- `firebase.json` — Firebase project configuration for hosting/emulators.
- `.gitignore` — already includes `.env` so local environment files are ignored by default.

## Security notes and recommended steps

1. Rotate any keys that were previously committed to the repository (the backend had a hardcoded Jooble key). Treat them as compromised.
2. Remove secrets from git history if needed (tools: `git filter-repo` or BFG). This is destructive and requires a force-push; follow a proper backup and coordination plan.
3. Keep `.env` files local and use `.env.example` files to document required variables.
4. For production, prefer a secrets manager or platform environment variables rather than `.env` files.

## Quick checklist to get the whole project running locally

1. Backend
   ```bash
   cd backend
   npm install
   cp .env.example .env
   # edit .env to add JOOBLE_API_KEY
   npm start
   ```
2. Frontend
   ```bash
   cd frontend
   npm install
   # create frontend/.env with REACT_APP_FIREBASE_* keys
   npm start
   ```
3. (Optional) Functions emulator
   ```bash
   cd functions
   npm install
   npm run serve
   ```

If you want, I can also create small `README.md` files inside `backend/` and `functions/` (folder-level READMEs) or add `Makefile` top-level commands to script these steps. Would you like me to add per-folder READMEs as well?
