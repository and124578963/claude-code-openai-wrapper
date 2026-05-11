FROM python:3.12-slim

# Install system deps (curl for Poetry installer)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user. The bundled Claude Code CLI refuses to run with
# --dangerously-skip-permissions when uid=0 ("cannot be used with root/sudo
# privileges for security reasons"), and the wrapper passes that flag
# whenever permission_mode=bypassPermissions is set — which we do for ALL
# headless requests. So running as root is incompatible with serving any
# chat completion. Container runs as uid=1000 instead.
ARG WRAPPER_UID=1000
RUN useradd --create-home --shell /bin/bash --uid ${WRAPPER_UID} wrapper

# Install Poetry as the wrapper user so its bin lands in $HOME/.local
USER wrapper
ENV PATH="/home/wrapper/.local/bin:${PATH}"
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy the app code with correct ownership
USER root
COPY --chown=wrapper:wrapper . /app
WORKDIR /app

# Install Python dependencies with Poetry as the wrapper user.
# Drop the committed lock file and let poetry resolve fresh from the current
# pyproject.toml. Avoids "lock file out of sync" errors when pyproject.toml
# is bumped (e.g. claude-agent-sdk version) without regenerating the lock.
# Reproducibility is still anchored by the SemVer pins in pyproject.toml.
USER wrapper
RUN rm -f poetry.lock && poetry install --no-root

# Note: Claude Code CLI is bundled with claude-agent-sdk >= 0.1.8
# No separate Node.js/npm installation required

# Expose the port (default 8000)
EXPOSE 8000

# Run the app with Uvicorn (development mode with reload; switch to --no-reload for prod)
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
