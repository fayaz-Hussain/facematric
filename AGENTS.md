## QA / validation work on FaceMetric

These rules apply only when I ask for testing or validation work. For normal feature development, ignore this section.

- The source of truth for testing is TESTING_PLAN.md in the project root. Read it fully before any QA task.
- Never edit production code or existing tests (app code and anything already under backend/tests/). New tests go only under qa/. If you find a bug, write a failing test marked xfail(strict=True) with its bug ID, log it in qa/reports/BUGS.md, and report it. Fixes happen only after I approve them.
- Never loosen a tolerance, threshold, or assertion to make a test pass. Tolerances come from measured data and need my approval before they are locked in.
- If the PRD, the plan and the code disagree, stop and ask me. Do not choose silently.
- Work one phase at a time. Finish with a report in qa/reports/ and stop until I say "continue".
- Every claim in a report must point to a file, command or CSV that reproduces it. If you didn't run it, say so.
- Never commit or upload face images, embeddings or datasets.
- Ask before downloading datasets or installing system-level packages. Use a virtual environment.
- This is a Windows machine. Use PowerShell-compatible commands. If a phase depends on something not implemented yet, mark it BLOCKED with the reason instead of skipping it silently.
