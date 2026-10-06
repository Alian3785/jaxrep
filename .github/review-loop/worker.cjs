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
  command('git', ['clone', '--no-checkout', '--filter=blob:none', `https://github.com/${REPO}.git`, checkout], directory);
  command('git', ['-c', 'core.hooksPath=NUL', 'checkout', '--detach', task.head], checkout);
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
  if (command('git', ['rev-parse', 'HEAD'], checkout) !== task.head) throw Error('Codex changed commit history');
  const files = [...new Set([
    ...command('git', ['diff', '--name-only', '-z', 'HEAD'], checkout).split('\0'),
    ...command('git', ['ls-files', '-z', '--others', '--exclude-standard'], checkout).split('\0'),
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
  command('git', ['diff', '--check'], checkout);
  command('git', ['add', '--', ...files], checkout);
  command('git', ['-c', 'core.hooksPath=NUL', '-c', 'user.name=disciplesjax',
    '-c', 'user.email=Alian3785@gmail.com', 'commit', '-m', `Fix verified CodeRabbit findings for #${task.number}`], checkout);
  // Ordinary push rejects a concurrent update. Never force-push the user's branch.
  command('git', ['-c', 'http.version=HTTP/1.1', '-c', 'credential.helper=',
    '-c', 'credential.helper=!gh auth git-credential', 'push', 'origin', `HEAD:refs/heads/${task.branch}`], checkout);
  const head = command('git', ['rev-parse', 'HEAD'], checkout);
  request(`repos/${REPO}/issues/${task.number}/comments`, 'POST', {
    body: `Codex отправил исправления: ${head}. Ожидается повторное ревью CodeRabbit и CI.\n\n${summary}`,
  });
  console.log(`Published ${head}; waiting for new GitHub review/CI events.`);
}

if (require.main === module) run().catch(error => {console.error(error.message); process.exitCode = 1;});
module.exports = {sameTask, protectedPath, run};
