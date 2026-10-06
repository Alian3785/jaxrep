const {execFileSync} = require('node:child_process');

function gh(args, options = {}, retryRead = false) {
  for (let attempt = 0; ; attempt++) {
    try {return execFileSync('gh', args, {encoding: 'utf8', windowsHide: true,
      maxBuffer: 20 * 1024 * 1024, ...options});}
    catch (error) {
      if (!retryRead || attempt >= 2 || ![3221225477, -1073741819].includes(error.status)) throw error;
    }
  }
}

function request(endpoint, method = 'GET', body) {
  const args = ['api', endpoint, '--method', method];
  if (body) args.push('--input', '-');
  const output = gh(args, {input: body ? JSON.stringify(body) : undefined}, method === 'GET' || endpoint === 'graphql');
  return output.trim() ? JSON.parse(output) : null;
}
function route(path, field) {
  return Object.assign(async p => ({data: request(path(p))}), {path, field});
}
const prefix = p => `repos/${p.owner}/${p.repo}`;
const github = {
  rest: {
    pulls: {
      get: route(p => `${prefix(p)}/pulls/${p.pull_number}`),
      listReviews: route(p => `${prefix(p)}/pulls/${p.pull_number}/reviews`),
      listFiles: route(p => `${prefix(p)}/pulls/${p.pull_number}/files`),
    },
    repos: {listCommitStatusesForRef: route(p => `${prefix(p)}/commits/${p.ref}/statuses`)},
    issues: {listComments: route(p => `${prefix(p)}/issues/${p.issue_number}/comments`)},
    actions: {listWorkflowRunsForRepo: route(p => `${prefix(p)}/actions/runs?head_sha=${p.head_sha}&event=pull_request`, 'workflow_runs')},
  },
  async paginate(method, p) {
    const endpoint = method.path(p);
    const pages = JSON.parse(gh(['api', `${endpoint}${endpoint.includes('?') ? '&' : '?'}per_page=100`,
      '--paginate', '--slurp'], {}, true));
    return pages.flatMap(page => method.field ? page[method.field] : page);
  },
  async graphql(query, variables) {return request('graphql', 'POST', {query, variables}).data;},
};
module.exports = {github, request};
