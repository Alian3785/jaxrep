const test = require('node:test');
const assert = require('node:assert/strict');
const {decide, REQUIRED_WORKFLOWS, needsGpu} = require('./policy.cjs');

function snapshot() {
  return {
    pr: {number: 2, state: 'open', draft: false, user: {login: 'Alian3785'},
      head: {sha: 'head', ref: 'codex/fix', repo: {full_name: 'Alian3785/jaxrep'}},
      base: {sha: 'base', ref: 'feature/game', repo: {full_name: 'Alian3785/jaxrep'}},
      labels: [], mergeable: true, mergeable_state: 'clean'},
    statuses: [{context: 'CodeRabbit', state: 'success', description: 'Review completed',
      creator: {login: 'coderabbitai[bot]'}}],
    threads: [], reviews: [], files: [{filename: 'docs/readme.md'}], attempts: [],
    runs: REQUIRED_WORKFLOWS.map((path, id) => ({id, path, event: 'pull_request', head_sha: 'head',
      status: 'completed', conclusion: 'success', pull_requests: [{number: 2, base: {sha: 'base'}}]})),
  };
}
function finding() {
  return {isResolved: false, comments: {nodes: [{author: {login: 'coderabbitai[bot]'}}]}};
}
test('merge after current review and CI; target may be any existing base', () => {
  assert.equal(decide(snapshot()).action, 'merge');
});
test('foreign author, fork, draft and stop label never run', () => {
  for (const change of [s => s.pr.user.login = 'stranger', s => s.pr.head.repo.full_name = 'other/repo',
    s => s.pr.draft = true, s => s.pr.labels.push({name: 'codex-stop'})]) {
    const s = snapshot(); change(s); assert.equal(decide(s).action, 'skip');
  }
});
test('missing, pending, skipped or spoofed CodeRabbit review never merges', () => {
  for (const change of [s => s.statuses = [], s => s.statuses[0].state = 'pending',
    s => s.statuses[0].description = 'Review skipped', s => s.statuses[0].creator.login = 'stranger']) {
    const s = snapshot(); change(s); assert.equal(decide(s).action, 'wait');
  }
});
test('success status with open findings repairs rather than merges', () => {
  const s = snapshot(); s.threads.push(finding()); assert.equal(decide(s).action, 'repair');
  s.attempts = [{head: 'head'}]; assert.equal(decide(s).action, 'wait');
  s.attempts = [1, 2, 3].map(head => ({head})); assert.equal(decide(s).action, 'blocked');
});
test('resolved findings allow merge, including obsolete threads', () => {
  const s = snapshot(); s.threads = [{...finding(), isResolved: true}];
  assert.equal(decide(s).action, 'merge');
});
test('failed, cancelled, missing and stale base/head CI block merge', () => {
  for (const change of [s => s.runs = [], s => s.runs[0].conclusion = 'failure',
    s => s.runs[0].conclusion = 'cancelled', s => s.runs[0].head_sha = 'old',
    s => s.runs[0].pull_requests[0].base.sha = 'old']) {
    const s = snapshot(); change(s); assert.equal(decide(s).action, 'wait');
  }
});
test('human review threads and changes requested block merge', () => {
  const s = snapshot(); s.reviews = [{state: 'CHANGES_REQUESTED', user: {login: 'reviewer'}}];
  assert.equal(decide(s).action, 'wait');
  s.reviews = []; s.threads = [{isResolved: false, comments: {nodes: [{author: {login: 'reviewer'}}]}}];
  assert.equal(decide(s).action, 'wait');
});
test('hot path and renamed hot path need GPU evidence; UI/docs do not', () => {
  assert.equal(needsGpu([{filename: 'docs/game.md'}]), false);
  assert.equal(needsGpu([{filename: 'stoix/envs/number_grid.py'}]), true);
  assert.equal(needsGpu([{filename: 'docs/old.md', previous_filename: 'stoix/envs/game.py'}]), true);
  const s = snapshot(); s.files = [{filename: 'stoix/envs/game.py'}];
  assert.equal(decide(s).action, 'blocked'); s.gpuApproved = true;
  assert.equal(decide(s).action, 'merge');
});
test('out-of-diff findings and review-only change requests are not ignored', () => {
  for (const review of [{state: 'CHANGES_REQUESTED', body: 'Fix this'},
    {state: 'COMMENTED', body: '<summary>Outside diff range comments (1)</summary>'}]) {
    const s = snapshot(); s.reviews = [{...review, commit_id: 'head', user: {login: 'coderabbitai[bot]'}}];
    assert.equal(decide(s).action, 'repair');
  }
});
test('a successful rerun supersedes a failed run of the same workflow', () => {
  const s = snapshot(); s.runs.push({...s.runs[0], id: -1, conclusion: 'failure'});
  assert.equal(decide(s).action, 'merge');
});
test('a pending review cannot clear an earlier request for changes', () => {
  const s = snapshot(); s.reviews = ['CHANGES_REQUESTED', 'PENDING'].map(state => ({state, user: {login: 'reviewer'}}));
  assert.equal(decide(s).action, 'wait');
});
