# Manual Setup Required

This document outlines the manual setup steps required to complete the SDLC implementation due to GitHub App permission limitations.

## 📋 Overview

The automated SDLC implementation has created comprehensive documentation and templates, but requires manual setup of:

1. GitHub Actions workflows
2. Repository settings and branch protection
3. Security configurations
4. External service integrations

## 🔧 Required Actions

### 1. GitHub Actions Workflows

Copy the workflow templates from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 2. Repository Secrets

Configure the following secrets in GitHub repository settings (`Settings > Secrets and variables > Actions`):

#### Required Secrets
- `CODECOV_TOKEN` - For code coverage reporting
- `PYPI_API_TOKEN` - For PyPI package publishing
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `WANDB_API_KEY` - For ML experiment tracking

#### Optional Secrets (for enhanced security)
- `GITGUARDIAN_API_KEY` - For advanced secrets scanning
- `STAGING_API_KEY` - For staging environment testing
- `DOCKER_REGISTRY_TOKEN` - For enhanced Docker registry access

### 3. Branch Protection Rules

Configure branch protection for the `main` branch (`Settings > Branches > Add rule`):

#### Protection Settings
- [x] Require a pull request before merging
  - [x] Require approvals (minimum: 2)
  - [x] Dismiss stale PR approvals when new commits are pushed
  - [x] Require review from code owners
- [x] Require status checks to pass before merging
  - [x] Require branches to be up to date before merging
  - Required status checks:
    - `CI / Code Quality`
    - `CI / Tests (Python 3.10)`
    - `CI / Integration Tests`
    - `CI / Build Package`
    - `CI / Docker Build Test`
- [x] Require conversation resolution before merging
- [x] Require signed commits
- [x] Restrict pushes that create files

### 4. Repository Settings

#### General Settings (`Settings > General`)
- [x] Allow merge commits
- [x] Allow squash merging (default)
- [ ] Allow rebase merging
- [x] Always suggest updating pull request branches
- [x] Automatically delete head branches

#### Security Settings (`Settings > Security`)
- [x] Enable vulnerability alerts
- [x] Enable Dependabot alerts
- [x] Enable Dependabot security updates
- [x] Enable Dependabot version updates

### 5. Issue and PR Templates

The following templates are already created and will be automatically available:

- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/PULL_REQUEST_TEMPLATE.md`

### 6. External Service Integrations

#### Codecov Integration
1. Visit [codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Copy the repository token to `CODECOV_TOKEN` secret

#### Slack Integration
1. Create a Slack webhook URL
2. Add the URL to `SLACK_WEBHOOK_URL` secret
3. Configure desired channels for notifications

#### PyPI Integration (for releases)
1. Create a PyPI API token
2. Add the token to `PYPI_API_TOKEN` secret
3. Verify package name availability

### 7. Container Registry Setup

#### GitHub Container Registry (GHCR)
The workflows are configured to use GHCR with the `GITHUB_TOKEN`, which should work automatically.

#### Alternative Registry (Optional)
To use a different container registry:
1. Update the `REGISTRY` environment variable in workflows
2. Add registry credentials to secrets
3. Update the login action in workflows

## 🔍 Verification Steps

After completing the manual setup:

### 1. Test CI Workflow
```bash
# Create a test branch and push changes
git checkout -b test-ci-setup
echo "# Test" > test-file.md
git add test-file.md
git commit -m "test: verify CI workflow"
git push origin test-ci-setup

# Create a PR and verify workflows run
```

### 2. Verify Branch Protection
- Try to push directly to `main` (should be blocked)
- Verify PR requirements are enforced
- Check that status checks are required

### 3. Test Security Scanning
```bash
# Trigger security workflow manually
gh workflow run security.yml
```

### 4. Validate Notifications
- Check Slack channels for workflow notifications
- Verify error alerts are delivered

## 📊 Monitoring and Maintenance

### Regular Maintenance Tasks

#### Weekly
- Review failed workflow runs
- Update dependencies via automated PRs
- Check security scan results

#### Monthly
- Review and update branch protection rules
- Audit repository access and permissions
- Update workflow templates if needed

#### Quarterly
- Review and rotate secrets
- Update external service integrations
- Evaluate new security tools

### Metrics to Track
- CI/CD pipeline success rates
- Security scan coverage
- Dependency update frequency
- Time to deployment

## 🚨 Troubleshooting

### Common Issues

1. **Workflow Not Triggering**
   - Check workflow file syntax with `gh workflow view`
   - Verify branch/path filters
   - Check repository permissions

2. **Secret Access Issues**
   - Verify secret names match workflow references
   - Check secret scope (repository vs environment)
   - Ensure secrets are not exposed in logs

3. **Test Failures**
   - Review test logs in workflow runs
   - Check for environment differences
   - Verify dependency compatibility

4. **Permission Errors**
   - Review `GITHUB_TOKEN` permissions
   - Check organization/repository settings
   - Verify third-party app authorizations

### Getting Help

- Check workflow run logs in GitHub Actions tab
- Review [GitHub Actions documentation](https://docs.github.com/en/actions)
- Consult repository's `docs/workflows/README.md`
- Open an issue with `help-wanted` label

## ✅ Setup Checklist

Use this checklist to track completion:

- [ ] Copy workflow files to `.github/workflows/`
- [ ] Configure repository secrets
- [ ] Set up branch protection rules
- [ ] Configure repository settings
- [ ] Integrate external services
- [ ] Test CI workflow with test PR
- [ ] Verify branch protection
- [ ] Test security scanning
- [ ] Validate notifications
- [ ] Document any customizations

## 📝 Notes

- All workflow templates follow GitHub Actions best practices
- Security scanning is configured for comprehensive coverage
- Dependency management includes automated updates
- Release process includes semantic versioning
- Container images are built for multiple architectures

After completing these setup steps, your repository will have a production-ready SDLC with automated testing, security scanning, and deployment capabilities.