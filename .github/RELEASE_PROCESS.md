# GitHub CI and release policy

## Continuous integration

The CI workflow runs independently on pushes, pull requests, and manual requests:

- Rules across Python 3.10, 3.11, 3.12
- Training recovery and prioritized replay
- Browser worker, JavaScript replay, and HTTP/admin integration
- Full pytest regression suite and bytecode compilation
- Docker image build and live `/health` smoke test

Each test job uploads JUnit XML results, including on failure. The `CI gate` job fails if any job fails. In repository Settings → Branches / Rulesets, require the **CI gate** status check before merging to `main`. Consider requiring review and blocking force pushes.

## Manual release only

There are **no scheduled, push-triggered, or automatic version releases**.

1. Ensure the `CI gate` is green on the commit to release.
2. Create and push a tag yourself: `git tag v1.0.0 && git push origin v1.0.0`.
3. Open Actions → **Manual Docker release** → Run workflow.
4. Enter the **existing** tag. Leave the optional GitHub release checkbox off if you only want the Docker image.
5. The workflow verifies the tag exists on `main`, builds the image, starts a container, checks `/health`, and publishes `ghcr.io/dwightmulcahy/qwixx_ai_next:<tag>`. It optionally creates a GitHub release if you explicitly check the box.

The workflow never creates or moves Git tags, and never pushes a `latest` image. It publishes only a named tag.

## Repository configuration

Ensure Actions has permission to write packages. The manual release uses the `production` environment; configure required reviewers there if you want approval before publication. GitHub Packages visibility may need to be adjusted for QNAP pulls. Container deployment still needs persistent `runs` storage.

## Limitations

GitHub Actions is the authoritative validation environment; a green run must be checked before release. Tag ancestry alone does not prove the tag's commit passed CI, so review the CI result for the exact tag commit before manually running the release.
