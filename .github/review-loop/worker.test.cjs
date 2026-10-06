const test = require('node:test');
const assert = require('node:assert/strict');
const {protectedPath, sameTask} = require('./worker.cjs');
test('repairs cannot rewrite control files or credentials', () => {
  for (const f of ['.github/workflows/test.yml', '.codex/config.toml', 'AGENTS.md', '.coderabbit.yaml',
    'docs/CODE_REVIEW.md', '.env', 'subdir/.env.prod', 'auth.json']) assert.equal(protectedPath(f), true, f);
  assert.equal(protectedPath('stoix/envs/number_grid.py'), false);
});
test('worker rejects changed head, base and destination', () => {
  const pr = {number: 4, state: 'open', draft: false, user: {login: 'Alian3785'}, labels: [],
    head: {ref: 'codex/fix', sha: 'h', repo: {full_name: 'Alian3785/jaxrep'}},
    base: {ref: 'main', sha: 'b', repo: {full_name: 'Alian3785/jaxrep'}}};
  const task = {number: 4, head: 'h', base: 'b', branch: 'codex/fix', base_branch: 'main'};
  assert.equal(sameTask(pr, task), true);
  for (const key of ['head', 'base', 'branch', 'base_branch']) assert.equal(sameTask(pr, {...task, [key]: 'changed'}), false);
});
