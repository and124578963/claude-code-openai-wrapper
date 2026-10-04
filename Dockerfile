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
# poetry.lock is not committed (see .gitignore), but a developer's local one
# can still ride in via the build context — drop it and resolve fresh from
# pyproject.toml, so a stale lock can never pin the build or trip the
# "lock file out of sync" error. Everything except claude-agent-sdk is held by
# the SemVer constraints in pyproject.toml; the SDK is handled below.
USER wrapper
RUN rm -f poetry.lock && poetry install --no-root

# --- Claude Agent SDK + bundled Claude Code CLI --------------------------
# The CLI binary lives inside the SDK wheel (claude_agent_sdk/_bundled/claude)
# and the SDK's _find_bundled_cli() beats shutil.which("claude"), so this one
# dependency IS both the SDK and the CLI — no npm install needed, and one
# would be ignored anyway.
#
# Default CLAUDE_SDK_VERSION=latest makes every build pull the newest release.
# Pass a concrete version to reproduce or roll back an image:
#   docker build --build-arg CLAUDE_SDK_VERSION=0.2.95 .
# BUILD_DATE exists only to bust Docker's layer cache: without a changing arg,
# a rebuild from untouched sources would silently reuse the old SDK layer and
# "latest" would quietly mean "whatever we resolved last time".
ARG CLAUDE_SDK_VERSION=latest
ARG BUILD_DATE=unknown
RUN echo "SDK layer built ${BUILD_DATE}" \
    && if [ "${CLAUDE_SDK_VERSION}" = "latest" ]; then \
           poetry add "claude-agent-sdk@latest"; \
       else \
           poetry add "claude-agent-sdk==${CLAUDE_SDK_VERSION}"; \
       fi

# Record what actually landed and smoke-test it: assert the CLI really is
# bundled, then run it. A binary that can't print its own version here would
# otherwise surface as a 500 on the first chat completion in prod.
RUN poetry run python -c "import importlib.metadata as md, pathlib, claude_agent_sdk as s; p = pathlib.Path(s.__file__).parent / '_bundled' / 'claude'; assert p.is_file(), 'claude-agent-sdk wheel has no bundled CLI'; print(md.version('claude-agent-sdk'))" > /app/.sdk-version \
    && poetry run python -c "import pathlib, claude_agent_sdk as s; print(pathlib.Path(s.__file__).parent / '_bundled' / 'claude')" > /app/.cli-path \
    && "$(cat /app/.cli-path)" --version > /app/.cli-version \
    && echo "claude-agent-sdk $(cat /app/.sdk-version) / CLI $(cat /app/.cli-version)"

# Expose the port (default 8000)
EXPOSE 8000

# Run the app with Uvicorn. No --reload: the watchfiles reloader has no place
# in prod — it burns CPU watching /app and restarts the server (killing
# in-flight Claude sessions) on any file change. Reload is off by default;
# uvicorn has no --no-reload flag, only the enabling --reload.
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
