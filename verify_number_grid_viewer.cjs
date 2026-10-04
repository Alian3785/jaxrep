"use strict";
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const root = "results/number-grid-5m";
const game = JSON.parse(fs.readFileSync(root + "/replays.json", "utf8"));
const script = fs.readFileSync("number_grid_viewer.js", "utf8").split("function frames()")[0];
const context = vm.createContext({document:{getElementById:()=>({getContext:()=>({})})}});
vm.runInContext("const GAME_DATA = " + JSON.stringify(game) + ";\n" + script, context);
let count = 0;
for (const episode of game.episodes) {
    context.current = game.initial;
    for (const expected of episode.frames) {
        context.action = expected.action;
        const actual = JSON.parse(JSON.stringify(vm.runInContext("simulateStep(current, action)", context)));
        for (const key of ["position","strength","alive","outcome","step","reward"]) {
            assert.deepStrictEqual(actual[key], expected[key], `Mismatch in ${key} at step ${expected.step}`);
        }
        context.current = actual;count++;
    }
}
// A weaker and an equal opponent next to the same destination: loss only.
context.current = {...game.initial,position:[8,7],strength:1,alive:[true,true,true],step:0,outcome:0};
vm.runInContext("GAME_DATA.initial.opponent_positions = [[7,8],[9,8],[2,3]]", context);
context.action = 2;
const loss = JSON.parse(JSON.stringify(vm.runInContext("simulateStep(current, action)", context)));
assert.strictEqual(loss.reward,-1);assert.strictEqual(loss.strength,1);
assert.strictEqual(loss.outcome,2);assert.deepStrictEqual(loss.alive,[true,true,true]);
console.log(`Viewer matches ${count} actual JAX/PPO transitions in ${game.episodes.length} episodes; loss priority verified.`);
