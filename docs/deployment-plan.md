# Bloom deployment plan

This is a plan for a two-service Render deployment. It is not a deployment record.

## 1. Prepare the repository

- Keep model weights and local account data out of Git. The two checkpoints are downloaded from `mufasabrownie/glowlytics-skin-models` by `setup_ml.sh`.
- Add the ML runtime packages (`timm`, `torch`, `torchvision`, `ultralytics`, `huggingface_hub`) to the backend's production dependency definition, or keep `setup_ml.sh` in the backend build command. Confirm compatible versions and that the build fits the selected service's memory and disk limits.
- Verify `npm ci && npm run build` in `frontend/` and `python -m pytest tests/ -v --tb=short` in `backend/` before deployment.
- Push the verified source and this plan to GitHub. Keep `.env`, `backend/data/`, and downloaded model files untracked.

## 2. Create the API service

- Create a Render Python web service from this GitHub repository, using the repository root as the service root.
- Set build command to `pip install -r backend/requirements.txt && bash setup_ml.sh`. The script downloads both model files into `backend/models/` during the build.
- Set start command to `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Set `PYTHON_VERSION` to a supported Python 3.11 release. Set `AUTH_SECRET` to a long random secret and `ALLOWED_ORIGINS` to the exact frontend HTTPS origin. Set the health check path to `/api/health`.
- Decide how to preserve accounts and scan history before inviting users. With the current SQLite store, attach a persistent disk at `/var/data` and set `BLOOM_DB_PATH=/var/data/bloom.db`. Render's default filesystem is ephemeral; persistent disks require a paid service and limit scaling to one instance. An external database would require an application change.
- Size the service after measuring a real scan with both ML models. The health endpoint reports whether the two checkpoints are installed and whether both lazy-loaded models have loaded; verify a successful scan response separately.

## 3. Create the frontend site

- Create a Render static site from the same GitHub repository with root directory `frontend/`, build command `npm ci && npm run build`, and publish directory `dist`.
- Set the build-time variable `VITE_API_URL` to the API service's HTTPS URL. Set the API's `ALLOWED_ORIGINS` to the final static-site URL, then redeploy both services as needed.
- Verify sign-up, sign-in, password visibility toggle, reset flow, image upload, and the ML results panel in a production build.

## 4. Release checks

- Confirm the frontend URL and `/api/health` return HTTP 200.
- Submit a valid face image to `/api/analyse` and confirm `ml.available === true`, four finite signal scores, and an acne detection object. A 422 quality or face response will not include ML results.
- Restart the API and confirm account login and saved history still work, proving database persistence.
- Review CORS, logs, model download failures, build size, and response time. Keep the educational disclaimer visible in the UI.
- Enable automatic deployment from `main` only after these checks pass. Record the live URLs and deployment settings in the README.

## References

- [Render FastAPI deployment](https://render.com/docs/deploy-fastapi)
- [Render web services and health checks](https://render.com/docs/web-services)
- [Render static sites](https://render.com/docs/static-sites)
- [Render persistent disks](https://render.com/docs/disks)
- [Render environment variables](https://render.com/docs/configure-environment-variables)
