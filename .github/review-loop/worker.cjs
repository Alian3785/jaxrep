'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync, spawnSync} = require('node:child_process');
const {github, request} = require('./gh-client.cjs');
const {snapshot, buildPrompt} = require('./controller.cjs');
const {REPO, eligible, decide, findingsFor} = require('./policy.cjs');
const repo = {owner: 'Alian3785', repo: 'jaxrep'};

function command(exe, args, cwd) {
  for (let attempt = 0; ; attempt++) {
    try {return execFileSync(exe, args, {cwd, encoding: 'utf8', windowsHide: true, maxBuffer: 8 * 1024 * 1024,
      env: {...process.env, GIT_TERMINAL_PROMPT: '0'}}).trim();}
    catch (e) {
      if (attempt >= 2 || ![3221225477, -1073741819].includes(e.status) || e.stdout || e.stderr) throw e;
    }
  }
}
function sameTask(pr, task) {
  return eligible(pr) && pr.number === task.number && pr.head.sha === task.head &&
    pr.base.sha === task.base && pr.head.ref === task.branch && pr.base.ref === task.base_branch;
}
function protectedPath(p) {
  return /^(\.github\/|\.codex\/|\.git\/|AGENTS\.md$|\.coderabbit\.yaml$|docs\/CODE_REVIEW\.md$)/i.test(p) ||
    /(^|\/)(\.env(?:\..*)?|auth\.json|credentials|\.netrc)$/i.test(p);
}
function git(args, cwd) {
  // Pure Git operations; no MSYS submodule shell or credential helper is used.
  return command('git', ['-c', 'http.version=HTTP/1.1', ...args], cwd);
}
function fileChanges(checkout, files) {
  if (files.length > 20) throw Error('Repair exceeds 20 files; needs manual review');
  const changes = {additions: [], deletions: []};
  const root = fs.realpathSync(checkout);
  let bytes = 0;
  for (const file of files) {
    if (protectedPath(file)) throw Error('Protected file');
    const filename = path.join(checkout, file);
    if (!fs.existsSync(filename)) {changes.deletions.push({path: file}); continue;}
    const relative = path.relative(root, fs.realpathSync(filename));
    if (relative.startsWith('..') || path.isAbsolute(relative) || !fs.lstatSync(filename).isFile()) {
      throw Error('Repair contains a symlink or path outside its checkout');
    }
    const content = fs.readFileSync(filename);
    bytes += content.length;
    if (bytes > 2 * 1024 * 1024) throw Error('Repair exceeds 2 MiB; needs manual review');
    changes.additions.push({path: file, contents: content.toString('base64')});
  }
  return changes;
}

async function run() {
  // Use this computer's existing gh login, never the runner's read-only job token.
  delete process.env.GH_TOKEN; delete process.env.GITHUB_TOKEN;
  const task = JSON.parse(process.env.LOOP_TASK);
  if (!Number.isInteger(task.number) || !['repair', 'merge'].includes(task.action) ||
    !/^[a-f0-9]{40}$/.test(task.head) || !/^[a-f0-9]{40}$/.test(task.base)) throw Error('Invalid task');
  let s = await snapshot(github, repo, task.number);
  if (!sameTask(s.pr, task)) throw Error('PR changed while the job waited; next event will retry');
  if (task.action === 'merge') {
    const decision = decide(s);
    if (decision.action !== 'merge') throw Error(`Merge blocked: ${decision.reason}`);
    const r = request(`repos/${REPO}/pulls/${task.number}/merge`, 'PUT', {sha: task.head, merge_method: 'squash'});
    if (!r.merged) throw Error(r.message);
    console.log(`Merged #${task.number} into ${task.base_branch}: ${r.sha}`);
    return;
  }
  const findings = findingsFor(s);
  if (!findings.length) return;
  const workRoot = path.join(process.env.LOCALAPPDATA, 'jaxrep-review-loop');
  fs.mkdirSync(workRoot, {recursive: true});
  const directory = fs.mkdtempSync(path.join(workRoot, `pr-${task.number}-`));
  const checkout = path.join(directory, 'repo');
  git(['clone', '--no-checkout', '--filter=blob:none', `https://github.com/${REPO}.git`, checkout], directory);
  git(['config', 'core.filemode', 'false'], checkout);
  git(['-c', 'core.hooksPath=NUL', 'checkout', '--detach', task.head], checkout);
  const codex = command('where.exe', ['codex'], directory).split(/\r?\n/).find(p => p.endsWith('.exe'));
  if (!codex) throw Error('Codex CLI is unavailable in the runner PATH');
  const output = path.join(directory, 'result.md');
  const log = fs.openSync(path.join(directory, 'codex.log'), 'w');
  let result;
  try {
    result = spawnSync(codex, ['-a', 'never', 'exec', '--sandbox', 'workspace-write', '--json',
      '--output-last-message', output, '-'], {
      cwd: checkout, input: buildPrompt(s, findings), encoding: 'utf8', windowsHide: true,
      stdio: ['pipe', log, log], timeout: 70 * 60 * 1000,
    });
  } finally {fs.closeSync(log);}
  if (result.status !== 0) throw Error(`Codex failed (${result.status}); local log: ${directory}`);
  const summary = fs.readFileSync(output, 'utf8').slice(0, 12000);
  if (git(['rev-parse', 'HEAD'], checkout) !== task.head) throw Error('Codex changed commit history');
  const files = [...new Set([
    ...git(['diff', '--no-renames', '--name-only', '-z', 'HEAD'], checkout).split('\0'),
    ...git(['ls-files', '-z', '--others', '--exclude-standard'], checkout).split('\0'),
  ].filter(Boolean))];
  if (files.some(protectedPath)) throw Error('Repair touched protected control/auth files; manual review required');
  s = await snapshot(github, repo, task.number);
  if (!sameTask(s.pr, task)) throw Error('PR changed during repair; local work preserved without pushing');
  if (!files.length) {
    request(`repos/${REPO}/issues/${task.number}/comments`, 'POST', {
      body: `Codex проверил замечания к ${task.head.slice(0, 7)}. Изменений нет; автоматический проход остановлен.\n\n${summary}`,
    });
    return;
  }
  git(['diff', '--check', 'HEAD'], checkout);
  // Server-side compare-and-swap prevents overwriting a concurrent push.
  // The local gh user creates the commit, so normal PR/review events are emitted.
  const committed = await github.graphql(`mutation($input:CreateCommitOnBranchInput!) {
    createCommitOnBranch(input:$input) {commit {oid}}
  }`, {input: {
    branch: {repositoryNameWithOwner: REPO, branchName: task.branch},
    expectedHeadOid: task.head,
    message: {headline: `Fix verified CodeRabbit findings for #${task.number}`},
    fileChanges: fileChanges(checkout, files),
  }});
  const head = committed.createCommitOnBranch.commit.oid;
  request(`repos/${REPO}/issues/${task.number}/comments`, 'POST', {
    body: `Codex отправил исправления: ${head}. Ожидается повторное ревью CodeRabbit и CI.\n\n${summary}`,
  });
  console.log(`Published ${head}; waiting for new GitHub review/CI events.`);
}

if (require.main === module) run().catch(error => {console.error(error.message); process.exitCode = 1;});
module.exports = {sameTask, protectedPath, fileChanges, run};
