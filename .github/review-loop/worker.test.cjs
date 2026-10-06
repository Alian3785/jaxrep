const test = require('node:test');
const assert = require('node:assert/strict');
const {protectedPath, sameTask, fileChanges} = require('./worker.cjs');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
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
test('publication includes additions/deletions but refuses outside files and oversized repairs', t => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'review-loop-test-'));
  const checkout = path.join(temp, 'repo'); fs.mkdirSync(checkout);
  fs.writeFileSync(path.join(checkout, 'game.py'), 'fixed\n');
  fs.writeFileSync(path.join(temp, 'outside.txt'), 'outside');
  t.after(() => {
    fs.unlinkSync(path.join(checkout, 'game.py'));
    fs.unlinkSync(path.join(temp, 'outside.txt'));
    fs.rmdirSync(checkout); fs.rmdirSync(temp);
  });
  assert.deepEqual(fileChanges(checkout, ['game.py', 'deleted.py']), {
    additions: [{path: 'game.py', contents: Buffer.from('fixed\n').toString('base64')}],
    deletions: [{path: 'deleted.py'}],
  });
  assert.throws(() => fileChanges(checkout, ['../outside.txt']), /outside/);
  assert.throws(() => fileChanges(checkout, ['.github/workflows/ci.yml']), /Protected/);
  assert.throws(() => fileChanges(checkout, Array(21).fill('game.py')), /20 files/);
});
