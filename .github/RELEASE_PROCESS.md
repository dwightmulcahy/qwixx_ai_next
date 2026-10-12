# Release process

CI runs on pushes and pull requests to main. Release publication happens only when you manually push an existing version tag (or manually dispatch the release workflow for an existing tag). No workflow creates version tags or chooses version numbers.

1. Confirm the exact commit on main has a successful CI run.
2. Configure repository Actions secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (Docker Hub access token with write permission).
3. Create and push your own version tag: `git tag v1.0.0 && git push origin v1.0.0`.
4. Actions → Release Docker image checks CI, smoke-tests the image, publishes `<DOCKERHUB_USERNAME>/qwixx-ai:v1.0.0` for amd64 and arm64, and creates a GitHub Release with changelog sections grouped from conventional commit subjects.

The release job uses the `production` environment. If you enable required reviewers, approval is needed before publication. No `latest` tag is published. Existing GitHub Releases are left unchanged on rerun. The first release includes all reachable commits; later releases compare with the previous version tag.

Dependabot checks GitHub Actions and Python dependencies weekly and groups updates into pull requests. CI validates those PRs before merging.
