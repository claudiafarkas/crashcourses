# Topic: Engineering Practices

These are habits I apply across the repository rather than a standalone notebook course. They are what make a project easier to run, understand, and share.

## Checklist

- Keep the root README and topic links accurate.
- Give every case study a concise README covering the question, data, method, results, limitations, and reproducible setup.
- Use one root environment definition, such as `requirements.txt` or `pyproject.toml`.
- Extract notebook code into `src/` only when it has meaningful reuse.
- Add focused tests for reusable preprocessing, retrieval, or evaluation code.
- Use configuration and saved pipelines instead of hard-coded paths and manual reruns.
