'use strict';

const REPO = 'Alian3785/jaxrep';
const OWNER = 'Alian3785';
const BOT = 'coderabbitai[bot]';
const REQUIRED_WORKFLOWS = [
  '.github/workflows/test_linters.yaml',
  '.github/workflows/run_algs.yaml',
  '.github/workflows/review_loop_tests.yml',
];

function eligible(pr) {
  return pr.state === 'open' && !pr.draft && pr.user.login === OWNER &&
    pr.head.repo?.full_name === REPO && pr.base.repo?.full_name === REPO &&
    (pr.head.ref.startsWith('codex/') || pr.labels.some(l => l.name === 'codex-auto')) &&
    !pr.labels.some(l => l.name === 'codex-stop');
}

function needsGpu(files) {
  // Fail closed for unfamiliar paths: only isolated documentation/UI/tests are exempt.
  return files.some(({filename: p, previous_filename: old}) => [p, old].filter(Boolean).some(f =>
    !(/^(docs\/|web\/|tests\/|\.github\/)/.test(f) ||
      /(^|\/)(test_[^/]+\.py|[^/]+\.test\.cjs)$/.test(f) ||
      /\.(md|rst|png|jpg|svg|html|css|js|cjs)$/.test(f) ||
      ['.coderabbit.yaml', '.gitignore', '.gitattributes', 'LICENSE'].includes(f))));
}

function findingsFor(s) {
  const findings = s.threads.filter(t => !t.isResolved &&
    [BOT, 'coderabbitai'].includes(t.comments.nodes[0]?.author?.login));
  const latest = s.reviews.filter(r => r.user.login === BOT && r.commit_id === s.pr.head.sha).at(-1);
  if (latest && (latest.state === 'CHANGES_REQUESTED' ||
      /(?:Outside diff range|Out.of.diff)[^\n]*\([1-9]\d*\)/i.test(latest.body || ''))) {
    findings.push({review: latest.html_url, body: latest.body});
  }
  return findings;
}

function decide(s) {
  const {pr, statuses, threads, reviews, runs, files, attempts, gpuApproved} = s;
  if (!eligible(pr)) return {action: 'skip', reason: 'PR is not opted in or is paused/draft'};
  // Status list is newest first. Require the real CodeRabbit status on this exact SHA.
  const review = statuses.find(x => x.context === 'CodeRabbit' && x.creator?.login === BOT);
  if (!review || review.state !== 'success' || review.description !== 'Review completed') {
    return {action: 'wait', reason: 'CodeRabbit has not completed this head'};
  }
  const open = threads.filter(t => !t.isResolved);
  const findings = findingsFor(s);
  if (findings.length) {
    if (attempts.some(a => a.head === pr.head.sha)) {
      return {action: 'wait', reason: 'This head already has a repair attempt'};
    }
    if (attempts.length >= 3) return {action: 'blocked', reason: 'Three repair attempts reached'};
    return {action: 'repair', reason: `${findings.length} unresolved CodeRabbit threads`, findings};
  }
  if (open.length) return {action: 'wait', reason: 'Unresolved review threads remain'};
  const latestReviews = new Map();
  for (const r of reviews) if (['APPROVED', 'CHANGES_REQUESTED', 'DISMISSED'].includes(r.state)) latestReviews.set(r.user.login, r);
  if ([...latestReviews.values()].some(r => r.state === 'CHANGES_REQUESTED')) {
    return {action: 'wait', reason: 'A reviewer still requests changes'};
  }
  for (const path of REQUIRED_WORKFLOWS) {
    const run = runs.filter(r => r.path === path && r.event === 'pull_request' &&
      r.head_sha === pr.head.sha && r.pull_requests.some(p => p.number === pr.number &&
        p.base.sha === pr.base.sha)).sort((a, b) => b.id - a.id)[0];
    if (!run || run.status !== 'completed' || run.conclusion !== 'success') {
      return {action: 'wait', reason: `Required CI is not green for this head/base: ${path}`};
    }
  }
  const latestRuns = [...new Map(runs.slice().sort((a, b) => a.id - b.id).map(r => [r.path, r])).values()];
  if (latestRuns.some(r => r.event === 'pull_request' && r.head_sha === pr.head.sha &&
      !r.path.endsWith('/codex_review_loop.yml') &&
      r.pull_requests.some(p => p.number === pr.number && p.base.sha === pr.base.sha) &&
      !['success', 'skipped', 'neutral'].includes(r.conclusion))) {
    return {action: 'wait', reason: 'Another CI workflow is pending or failed'};
  }
  if (statuses.some(x => x.state !== 'success' && x.context !== 'CodeRabbit')) {
    return {action: 'wait', reason: 'Another commit status is not successful'};
  }
  if (needsGpu(files) && !gpuApproved) {
    return {action: 'blocked', reason: 'Comparable GPU benchmark required by AGENTS.md'};
  }
  if (pr.mergeable !== true || !['clean', 'unstable'].includes(pr.mergeable_state)) {
    return {action: 'wait', reason: 'GitHub does not permit a clean merge yet'};
  }
  return {action: 'merge', reason: 'Current review and required CI passed'};
}

module.exports = {REPO, OWNER, BOT, REQUIRED_WORKFLOWS, eligible, needsGpu, findingsFor, decide};
