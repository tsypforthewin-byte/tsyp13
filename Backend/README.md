# Backend Overview

This directory contains microservices and modules used by the project. Each subfolder is a small service or cloud-trigger component. This README provides a short overview and points to module READMEs for specific run/deploy instructions.

Modules:
- `AIInterviewer/` - an interview simulation service (Python)
- `CVModules/` - collection of CV-related services:
  - `CVExtractionCloudTrigger/` - cloud trigger for extraction
  - `CVExtractor/` - extractor service
  - `CVReviewer/` - reviewer service
  - `CVRewriter/` - rewriter service
- `JobMatcher/` - job matching service (Python)
 - `scrapper/` - web scraping and Firebase backend (Node.js)

General run notes:
- Most services are Python-based and include a `requirements.txt` and either `app.py` or `main.py` as an entrypoint.
- Dockerfiles are present in many modules for containerized deployment.
- Check the module README in each subfolder for exact commands and environment variables.

If you want a quick start, pick a module and follow its README.

For development:
- Create a virtual environment, install from `requirements.txt`, and run the module's `main.py` or `app.py`.
- To run in Docker, build the Dockerfile in the module folder and run the container mapping the expected port.

If you need help running a specific module, open that module's README and follow the examples there or ask for interactive help.

Scrapper (located at `../scrapper`)
----------------------------------

This project contains a separate `scrapper/` folder that houses a Node.js-based backend and Firebase function code responsible for scraping and ingesting data. The folder typically contains:

- `scrapper/backend/` - a Node.js server (contains `package.json` and `server.js`).
- `scrapper/functions/` - Firebase Functions source (contains `index.js` and `package.json`).
- `scrapper/firebase.json` - Firebase project config.
- `scrapper/README.md` - module-specific notes (check this first for any custom instructions).

How to run the scrapper backend locally

1. Inspect `scrapper/backend/package.json` to see the available npm scripts. Then from the `scrapper/backend` directory:

```bash
cd scrapper/backend
npm install
# if package.json defines a start script:
npm start
# otherwise run directly:
node server.js
```

2. The server will log which port it binds to; if a `PORT` environment variable is used you can set it before running, e.g. `PORT=3000 npm start`.

How to run or emulate Firebase functions locally

1. From the `scrapper/functions` folder install dependencies:

```bash
cd scrapper/functions
npm install
```

2. Use the Firebase Emulator Suite (recommended) to run functions locally:

```bash
# from the repo root (or where firebase.json is located)
firebase emulators:start
```

3. To deploy functions to Firebase:

```bash
# ensure you're logged in and project is selected
firebase deploy --only functions
```

Notes
- Check `scrapper/README.md` for any project-specific environment variables, credentials, or scheduling details.
- If the backend talks to the `Backend/` services, confirm URLs and ports or use environment variables to point to local instances.
- If you want, I can open `scrapper/backend/package.json` and `scrapper/functions/package.json` and add precise run commands and example curl requests to this README.

