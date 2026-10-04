"use strict";
const moves = [[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]];
const $ = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
const outcomeLabels = ["Игра идёт", "Победа! Все оппоненты побеждены", "Поражение: сосед оказался сильнее или равен", "Таймаут: 2000 шагов"];
const outcomeShort = ["Идёт", "Победа", "Поражение", "Таймаут"];
let mode = "replay", episodeIndex = 0, frameIndex = 0, timer = null;
let manualFrames = [clone(GAME_DATA.initial)];
const canvas = $("board"), ctx = canvas.getContext("2d");

// The manual simulator uses exactly the same movement/encounter rules as JAX.
function simulateStep(current, action) {
    if (current.outcome) return clone(current);
    const next = clone(current), target = next.position.map((v,i) => v + moves[action][i]);
    const occupied = GAME_DATA.initial.opponent_positions.some((position,i) =>
        next.alive[i] && position[0] === target[0] && position[1] === target[1]);
    const inside = target.every(v => v >= 0 && v < GAME_DATA.size);
    if (inside && !GAME_DATA.initial.walls[target[0]][target[1]] && !occupied) next.position = target;
    const adjacent = GAME_DATA.initial.opponent_positions.map((position,i) => {
        const difference = position.map((v,k) => Math.abs(v - next.position[k]));
        const distance = GAME_DATA.adjacent_diagonals ? Math.max(...difference) : difference[0] + difference[1];
        return next.alive[i] && distance === 1;
    });
    next.step += 1; next.action = action;
    if (adjacent.some((yes,i) => yes && next.strength <= GAME_DATA.initial.opponent_strengths[i])) {
        next.reward = -1; next.outcome = 2;
    } else {
        const count = adjacent.filter(Boolean).length;
        next.alive = next.alive.map((yes,i) => yes && !adjacent[i]);
        next.strength += count; next.reward = count;
        next.outcome = next.alive.some(Boolean) ? 0 : 1;
        if (next.outcome === 1) next.reward = GAME_DATA.additive_victory_reward ? count + 3 : 3;
        else if (next.step >= GAME_DATA.max_steps) next.outcome = 3;
    }
    return next;
}

function frames() {
    return mode === "manual" ? manualFrames : [GAME_DATA.initial, ...GAME_DATA.episodes[episodeIndex].frames];
}
function state() { return frames()[frameIndex]; }
function totalReturn() { return frames().slice(0,frameIndex+1).reduce((sum,frame) => sum + frame.reward,0); }
function stop() { if (timer) clearInterval(timer); timer = null; $("play").textContent = "▶ Старт"; }
function draw() {
    const current = state(), size = GAME_DATA.size, cell = 48, offset = 16;
    ctx.clearRect(0,0,800,800); ctx.fillStyle = "#111d30"; ctx.fillRect(0,0,800,800);
    const contact = Array.from({length:size},() => Array(size).fill(0));
    if ($("danger").checked) GAME_DATA.initial.opponent_positions.forEach((position,i) => {
        if (!current.alive[i]) return;
        const level = current.strength > GAME_DATA.initial.opponent_strengths[i] ? 1 : 2;
        for (const [dy,dx] of moves) {
            const row = position[0]+dy, col=position[1]+dx;
            if (row>=0 && row<size && col>=0 && col<size) contact[row][col] = Math.max(contact[row][col],level);
        }
    });
    for (let row=0;row<size;row++) for (let col=0;col<size;col++) {
        const wall = GAME_DATA.initial.walls[row][col];
        ctx.fillStyle = wall ? "#35445c" : contact[row][col]===2 ? "#3c2337" : contact[row][col]===1 ? "#153731" : ((row+col)%2 ? "#142237" : "#17273d");
        ctx.beginPath(); ctx.roundRect(offset+col*cell+1,offset+row*cell+1,cell-2,cell-2,5); ctx.fill();
        if (wall) { ctx.fillStyle="#50617a";ctx.fillRect(offset+col*cell+12,offset+row*cell+22,24,3); }
    }
    if ($("trail").checked && frameIndex>0) {
        ctx.strokeStyle="#60a5fa66";ctx.lineWidth=3;ctx.beginPath();
        frames().slice(0,frameIndex+1).forEach((frame,i)=>{
            const x=offset+(frame.position[1]+.5)*cell,y=offset+(frame.position[0]+.5)*cell;
            i ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
        });ctx.stroke();
    }
    function token(position,number,color,isAgent) {
        const x=offset+position[1]*cell,y=offset+position[0]*cell;
        ctx.fillStyle=color;ctx.beginPath();ctx.roundRect(x+6,y+6,cell-12,cell-12,isAgent?12:7);ctx.fill();
        if(isAgent){ctx.strokeStyle="#d8ecff";ctx.lineWidth=2;ctx.stroke();}
        ctx.fillStyle="#0b1220";ctx.font="bold 25px system-ui";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(number,x+cell/2,y+cell/2+1);
    }
    GAME_DATA.initial.opponent_positions.forEach((position,i)=>{
        if(current.alive[i])token(position,GAME_DATA.initial.opponent_strengths[i],current.strength>GAME_DATA.initial.opponent_strengths[i]?"#34d399":"#fb7185",false);
    });token(current.position,current.strength,"#60a5fa",true);
    $("strength").textContent=current.strength;$("return").textContent=totalReturn();
    $("step").textContent=`${current.step} / ${GAME_DATA.max_steps}`;$("remaining").textContent=current.alive.filter(Boolean).length;
    $("status").textContent=outcomeLabels[current.outcome];$("status-dot").style.background=["#60a5fa","#34d399","#fb7185","#fbbf24"][current.outcome];
    $("frame").max=frames().length-1;$("frame").value=frameIndex;$("frame-label").textContent=`Шаг ${frameIndex} из ${frames().length-1}`;
    $("next").disabled=frameIndex>=frames().length-1;
    document.querySelectorAll("[data-action]").forEach(button=>button.disabled=!!current.outcome);
    document.querySelector(".pad .center").textContent=current.strength;
}
function nextFrame(){if(frameIndex<frames().length-1){frameIndex++;draw();}else stop();}
function play(){if(timer){stop();return;}if(frameIndex>=frames().length-1)frameIndex=0;$("play").textContent="⏸ Пауза";timer=setInterval(nextFrame,1000/Number($("speed").value));draw();}
function setMode(value){stop();mode=value;frameIndex=0;$("replay-mode").classList.toggle("active",value==="replay");$("manual-mode").classList.toggle("active",value==="manual");$("replay-controls").classList.toggle("hidden",value==="manual");$("manual-controls").classList.toggle("hidden",value==="replay");$("timeline").classList.toggle("hidden",value==="manual");if(value==="manual")manualFrames=[clone(GAME_DATA.initial)];draw();}
function manualMove(action){if(mode!=="manual"||state().outcome)return;manualFrames.push(simulateStep(state(),action));frameIndex++;draw();}

GAME_DATA.episodes.forEach((episode,index)=>{const option=document.createElement("option");option.value=index;option.textContent=`${episode.mode==="greedy"?"PPO: максимум":"PPO: игра "+(episode.episode+1)} · ${outcomeShort[episode.outcome]} · ${episode.length} шагов`;$("episode").appendChild(option);});
$("episode").onchange=()=>{stop();episodeIndex=Number($("episode").value);frameIndex=0;draw();};
$("play").onclick=play;$("next").onclick=()=>{stop();nextFrame();};$("restart").onclick=()=>{stop();frameIndex=0;draw();};
$("frame").oninput=()=>{stop();frameIndex=Number($("frame").value);draw();};
$("speed").oninput=()=>{$("speed-label").textContent=$("speed").value;if(timer){stop();play();}};
$("danger").onchange=draw;$("trail").onchange=draw;$("replay-mode").onclick=()=>setMode("replay");$("manual-mode").onclick=()=>setMode("manual");$("new-game").onclick=()=>setMode("manual");
document.querySelectorAll("[data-action]").forEach(button=>button.onclick=()=>manualMove(Number(button.dataset.action)));
document.addEventListener("keydown",event=>{if(mode!=="manual"||["INPUT","SELECT"].includes(event.target.tagName))return;const actions={w:0,e:1,d:2,c:3,s:4,z:5,a:6,q:7,ArrowUp:0,ArrowRight:2,ArrowDown:4,ArrowLeft:6};if(event.key in actions){event.preventDefault();manualMove(actions[event.key]);}});
const quality=RUN_SUMMARY.final_evaluations.policy;$("win-rate").textContent=`${(quality.win_rate*100).toFixed(1)}%`;$("evaluation-detail").textContent=`${quality.wins} побед из ${quality.episodes} эпизодов`;
$("throughput").textContent=`${Math.round(RUN_SUMMARY.training_steps_per_second/1000)} тыс./с`;
draw();
