(() => {
  'use strict';
  const data=JSON.parse(document.getElementById('run-data').textContent);
  const $=id=>document.getElementById(id), fmt=n=>new Intl.NumberFormat('ru-RU',{maximumFractionDigits:3}).format(n);
  const canvas=$('board'), ctx=canvas.getContext('2d');
  let mode='manual', session=null, manual=null, manualMap=null, busy=false, frame=0, recordIndex=0, timer=null, messages=[];
  let manualConstruction=null, manualTurnRules=null, manualCombat=null, capitalOpen=false, capitalKey=null;
  const records=data.records||[];
  const combatInfo=()=>mode==='manual'?(manualCombat||data.combat):data.combat;
  const profile=i=>i>=0&&i<12?(combatInfo()?.catalogue?.[current()?.state.unit_ids?.[i]]??(i<6?combatInfo()?.heroes[i]:combatInfo()?.enemies[current()?.state.enemy]?.[i-6])):null;
  const isMage=i=>profile(i)?.role==='area';
  const isWarrior=i=>profile(i)?.role==='melee';
  const role=i=>profile(i)?.name||'Пусто';
  const unitName=i=>(i<6?'Ваш ':'Вражеский ')+role(i).toLowerCase()+' '+(i%6+1);
  const partyText=units=>{const counts=new Map();for(const u of units||[])if(u)counts.set(u.name,(counts.get(u.name)||0)+1);return [...counts].map(([name,count])=>name+' × '+count).join(' · ');};
  const enemyParty=(map,enemy)=>partyText(current()?.state.in_battle?Array.from({length:6},(_,i)=>profile(i+6)):combatInfo()?.enemies[enemy]);
  const sourceName=key=>combatInfo()?.attack_types.find(t=>t.key===key)?.name||key;
  const protectionNames=bits=>(combatInfo()?.attack_types||[]).filter(t=>bits&t.bit).map(t=>t.name).join(', ')||'нет';
  const blockText=s=>{const parts=[];for(const [key,label] of [['last_immune','Иммунитет'],['last_ward','Защита поглотила удар']]){const names=Array.from({length:12},(_,i)=>i).filter(i=>s[key]&(1<<i)).map(unitName);if(names.length)parts.push(label+': '+names.join(', ')+'.');}return parts.length?' '+parts.join(' '):'';};
  function current(){return mode==='manual'?manual:records[recordIndex]?.frames[frame];}
  function currentMap(){return mode==='manual'?(manualMap||data.map):data.map;}
  function construction(){return mode==='manual'?(manualConstruction||data.construction):data.construction;}
  function turnRules(){return mode==='manual'?(manualTurnRules||data.turn_rules):data.turn_rules;}
  const branchNames={
    empire:['Воины','Стрелки','Маги','Целители','Титаны'],
    mountain_clans:['Воины','Стрелки','Маги','Великаны','Йети'],
    undead_hordes:['Воины','Призраки','Маги','Драконы','Оборотни'],
    legions:['Воины','Гаргульи','Маги','Демоны','Сатиры'],
    elves:['Кентавры','Стрелки','Маги','Поддержка','Грифоны'],
  };
  const buildStatus=['Доступно для строительства','Построено','Ветка или здание заблокированы',
    'Сначала постройте предшественника','Не хватает золота','Завершите ход отдыхом',
    'Строительство недоступно во время боя','Эпизод завершён','Нет у этой фракции'];
  function renderConstruction(snap){
    const info=construction();if(!info)return;
    const s=snap.state;
    $('capital-title').textContent=info.name;
    $('capital-day').textContent='Ход '+s.day+' · золото '+fmt(s.gold);
    $('capital-status').textContent=s.done?'Эпизод завершён.':s.in_battle?'Строить можно после боя.':s.built_today?'Постройка этого хода использована. Отдых откроет следующий ход.':'В этом ходу можно построить одно здание.';
    if(!$('faction').options.length){
      for(const faction of info.factions){const o=document.createElement('option');o.value=faction.id;o.textContent=faction.name;$('faction').append(o);}
      $('faction').value=currentMap().faction||'legions';
    }
    $('faction').disabled=busy||mode!=='manual';
    if(capitalKey!==info.faction){
      capitalKey=info.faction;$('building-list').replaceChildren();
      $('building-branch').replaceChildren();
      const all=document.createElement('option');all.value='all';all.textContent='Все здания';$('building-branch').append(all);
      for(const branch of [0,1,2,3,4,-1]){
        const rows=info.buildings.filter(b=>b.branch===branch);if(!rows.length)continue;
        const label=branch===-1?'Общие здания':branchNames[info.faction][branch];
        const option=document.createElement('option');option.value=String(branch);option.textContent=label;$('building-branch').append(option);
        const section=document.createElement('section');section.className='building-group';section.dataset.branch=branch;
        const heading=document.createElement('h3');heading.textContent=label;section.append(heading);
        const grid=document.createElement('div');grid.className='building-grid';section.append(grid);
        for(const row of rows){
          const card=document.createElement('article');card.className='building-card';card.dataset.building=row.action-18;
          const name=document.createElement('h4');name.textContent=row.name;card.append(name);
          const price=document.createElement('span');price.className='building-price';price.textContent=fmt(row.gold)+' золота';card.append(price);
          const unit=document.createElement('p');unit.className='building-unit';unit.textContent=row.unit?'Ветка юнита: '+row.unit:row.name==='Храм'?'Храм столицы':'Магическая служба';card.append(unit);
          const requirements=document.createElement('p');requirements.className='building-requires';requirements.textContent='Требует: '+(row.requires.join(', ')||'нет');card.append(requirements);
          if(row.blocks.length){const excludes=document.createElement('p');excludes.className='building-excludes';excludes.textContent='Закроет: '+row.blocks.join(', ');card.append(excludes);}
          const status=document.createElement('p');status.className='building-status';card.append(status);
          const button=document.createElement('button');button.dataset.action=row.action;button.textContent='Построить';button.setAttribute('aria-label','Построить '+row.name+' за '+row.gold+' золота');button.onclick=()=>sendAction(row.action);card.append(button);
          grid.append(card);
        }
        $('building-list').append(section);
      }
    }
    document.querySelectorAll('[data-building]').forEach(card=>{
      const i=Number(card.dataset.building),code=snap.building_status[i];
      card.className='building-card'+(code===1?' is-built':code===2?' is-blocked':code===0?' is-ready':'');
      card.querySelector('.building-status').textContent=info.buildings[i].unavailable_reason||buildStatus[code];
      const button=card.querySelector('button');button.textContent=code===1?'Построено':'Построить';
      button.disabled=mode!=='manual'||busy||!session||!snap.action_mask[Number(button.dataset.action)];
    });
    document.querySelectorAll('[data-branch]').forEach(group=>group.hidden=$('building-branch').value!=='all'&&group.dataset.branch!==$('building-branch').value);
  }
  function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ Смотреть';}
  function roundRect(x,y,w,h,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fill();}
  function drawMap(s){
    const map=currentMap();
    const width=canvas.clientWidth;if(!width)return;
    const ratio=Math.min(devicePixelRatio||1,2);canvas.width=width*ratio;canvas.height=width*ratio;ctx.setTransform(ratio,0,0,ratio,0,0);
    const cell=width/map.size;
    for(let r=0;r<map.size;r++)for(let c=0;c<map.size;c++){
      const wall=r===0||c===0||r===map.size-1||c===map.size-1,index=r*map.size+c;
      const seen=s.visited&&(s.visited[index>>>5]&(1<<(index&31)));
      roundRect(c*cell+1,r*cell+1,cell-2,cell-2,Math.max(2,cell*.12),wall?'#d3decc':seen?'#dfecd9':(r+c)%2?'#f3f6ef':'#ecf2e6');
    }
    const token=(p,n,hero)=>{roundRect(p[1]*cell+cell*.09,p[0]*cell+cell*.09,cell*.82,cell*.82,cell*.2,hero?'#a5d6ad':'#d5c0e8');ctx.fillStyle=hero?'#244c30':'#654777';ctx.font='650 '+Math.round(cell*.55)+'px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(n,(p[1]+.5)*cell,(p[0]+.52)*cell);};
    map.opponent_positions.forEach((p,i)=>{if(s.alive[i])token(p,map.enemy_units[i],false);});
    token(s.position,s.hp.slice(0,6).filter(h=>h>0).length,true);
  }
  const slots=[3,0,4,1,5,2,6,9,7,10,8,11];
  for(const i of slots){
    const button=document.createElement('button');button.className='unit'+(i>=6?' foe':'');button.dataset.slot=i;
    button.innerHTML='<span class="unit-name"></span><span class="archer-icon" aria-hidden="true">➶</span><span class="health"></span><span class="unit-stats"></span><span class="unit-xp"></span><span class="health-bar"><span class="health-fill"></span></span><span class="status"></span>';
    button.onclick=()=>sendAction(8+i-6);$(i<6?'allies':'enemies').append(button);
  }
  function experienceText(snap,i){
    const xp=snap.unit_experience?.[i];
    return xp?'Ур. '+xp[0]+' · опыт '+xp[3]+' / '+xp[2]+'\nЗа победу: '+xp[1]+' опыта':'';
  }
  function upgradeText(snap,i){
    const unit=profile(i),choices=unit?.upgrades;
    if(unit?.hero){
      const level=snap.unit_experience?.[i]?.[0]||1;
      const earned=(unit.level_bonuses||[]).filter(b=>b.level<=level).map(b=>b.name);
      const next=(unit.level_bonuses||[]).find(b=>b.level>level);
      return 'Герой: повышение без здания.'+(earned.length?' Бонусы: '+earned.join('; ')+'.':'')+
        (next?' На уровне '+next.level+': '+next.name+'.':'');
    }
    if(!choices)return '';
    if(!choices.length)return 'Следующий уровень: рост характеристик без здания.';
    return choices.map(u=>{
      if(!u.supported)return u.name+': недоступно — '+u.reason.toLowerCase()+'.';
      const b=construction()?.buildings.find(b=>b.name===u.building);
      const bit=b?1<<(b.action-18):0;
      const status=!b?'нужна столица родной фракции':snap.state.buildings&bit?'построено':snap.state.blocked_buildings&bit?'заблокировано':'не построено';
      return u.name+' ← '+u.building+' ('+status+').';
    }).join(' ');
  }
  function renderExperience(snap){
    const cards=[];
    for(let i=0;i<6;i++){
      if(!snap.max_hp[i])continue;
      const card=document.createElement('article');card.className='experience-card';
      const heading=document.createElement('b');heading.textContent=role(i)+' '+(i+1);card.append(heading);
      const hp=document.createElement('span');hp.textContent=snap.state.hp[i]+' / '+snap.max_hp[i]+' HP';card.append(hp);
      const xp=document.createElement('p');xp.textContent=experienceText(snap,i);card.append(xp);
      const next=document.createElement('small');next.textContent=upgradeText(snap,i);card.append(next);
      cards.push(card);
    }
    $('hero-experience').replaceChildren(...cards);
  }
  function xpSummary(snap){
    const parts=[];
    for(let i=0;i<12;i++){
      if(snap.state.last_promoted&(1<<i))parts.push(unitName(i)+' → уровень '+snap.unit_experience[i][0]+', полное лечение, опыт сброшен');
      else if(snap.state.last_xp?.[i])parts.push(unitName(i)+': опыт '+snap.unit_experience[i][3]+' / '+snap.unit_experience[i][2]);
    }
    return parts.length?' '+parts.join('; ')+'.':'';
  }
  function eventText(snapshot){
    const s=snapshot.state,a=unitName(s.last_actor),t=unitName(s.last_target);
    switch(s.last_event){
      case 1:return 'Начался бой с отрядом № '+(s.enemy+1)+'.';
      case 2:return isMage(s.last_actor)?a+': заклинание по всем противникам (суммарный урон '+s.last_damage+').'+blockText(s):a+(isWarrior(s.last_actor)?': удар мечом по ':': попадание в ')+t.toLowerCase()+' (−'+s.last_damage+').'+blockText(s);
      case 3:return a+': промах.';
      case 4:return a+' встал в защиту.';
      case 5:return a+' ждёт конца раунда.';
      case 6:return a+' готовится отступить.';
      case 7:return a+' покинул бой.';
      case 8:return 'Победа! Погибшие воскрешены с 1 HP.'+xpSummary(snapshot);
      case 9:return 'Ваш отряд погиб. Игра завершена.';
      case 10:return 'Отступление завершено. Ранения сохранены, погибшие воскрешены с 1 HP.';
      case 11:return 'Бой достиг лимита раундов. Эпизод завершён.';
      case 12:{const b=construction()?.buildings[s.last_building];return b?'Построено: '+b.name+' (−'+b.gold+' золота).':null;}
      case 13:return 'Отдых. Начался ход '+s.day+', +'+turnRules().income+' золота; очки перемещения восстановлены, живые бойцы получили регенерацию 10% HP. Штраф: '+fmt(Math.max(0,-snapshot.reward))+'.';
      case 14:return 'Атака заблокирована иммунитетом.'+blockText(s);
      case 15:return 'Атака поглощена защитой.'+blockText(s);
      default:return null;
    }
  }
  function render(){
    const snap=current();if(!snap)return;const s=snap.state,mask=snap.action_mask,map=currentMap();
    $('version').textContent=map.name;
    $('map-panel').hidden=s.in_battle||capitalOpen;$('battle-panel').hidden=!s.in_battle||capitalOpen;
    $('capital-panel').hidden=!capitalOpen;
    $('world-view').setAttribute('aria-pressed',String(!capitalOpen));$('capital-view').setAttribute('aria-pressed',String(capitalOpen));
    renderConstruction(snap);
    $('phase-label').textContent=s.in_battle?'Бой · отряд № '+(s.enemy+1):'Карта '+map.size+' × '+map.size;
    $('remaining').textContent=s.alive.filter(Boolean).length;
    $('wins').textContent=s.alive.filter(v=>!v).length;
    $('steps').textContent=fmt(s.step_count);$('reward').textContent=fmt(snap.total_reward);
    $('gold').textContent=fmt(s.gold);
    const turns=turnRules(),points=s.movement_points;
    $('turn-summary').textContent='Ход '+s.day+' · '+points+' / '+turns.movement_points+' очков перемещения';
    $('movement-summary').textContent=points+' / '+turns.movement_points+' очков · '+Math.floor(points/turns.move_cost)+' перемещений';
    $('rest').dataset.action=turns.rest_action;
    $('rest-hint').textContent=s.in_battle?'Отдых доступен после завершения боя.':snap.rest_penalty>0?'Штраф за оставшиеся очки: −'+fmt(snap.rest_penalty)+'. Следующий ход: +'+turns.income+' золота и полный запас очков.':'Все очки использованы: отдых без штрафа. Следующий ход: +'+turns.income+' золота и полный запас очков.';
    if(!s.in_battle)$('rest-hint').textContent+=' Живым бойцам: +'+turns.regeneration_percent+'% максимального HP с округлением вверх (до максимума).';
    const heroes=Array.from({length:6},(_,i)=>profile(i));
    renderExperience(snap);
    $('party-health').textContent=partyText(heroes.filter((u,i)=>u&&s.hp[i]>0))+' · '+s.hp.slice(0,6).reduce((a,b)=>a+b,0)+' здоровья';
    $('party-title').textContent=partyText(heroes)+'.';
    $('mage-rule').hidden=!heroes.some(u=>u?.role==='area');
    $('warrior-rule').hidden=!heroes.some(u=>u?.role==='melee');
    $('enemy-warrior-rule').hidden=!combatInfo()?.enemies.some(row=>row.some(u=>u?.role==='melee'));
    const basic=true;
    $('basic-combat-rule').hidden=false;
    $('mage-rule').innerHTML='<b>Массовая атака.</b> Маг атакует всех живых противников. Попадание, прибавка к урону, иммунитет и защита проверяются отдельно для каждой цели.';
    $('battle-hint').textContent='Источник атаки, иммунитеты и оставшиеся защиты указаны на карточках. Точность 80% соответствует фактическому шансу попадания 92,2%. Броня и действие защиты уменьшают прошедший урон.';
    $('attack-hint').textContent=isMage(s.actor)?'Массовая атака: выберите любого живого врага.':isWarrior(s.actor)?(mask.slice(8,14).some(Boolean)?'Ход воина: выберите подсвеченного врага для удара мечом.':'Воин не достаёт до врагов. Можно защищаться, ждать или отступить.'):'Нажмите на живого противника, чтобы выстрелить.';
    const message=s.won?'Победа! Карта очищена.':s.lost?'Ваш отряд погиб. Начните новую игру.':s.done?'Достигнут лимит. Начните новую игру.':s.in_battle?'Выбирайте цели и берегите свой отряд.':s.movement_points<turns.move_cost?'Очки перемещения закончились. Можно построить здание или отдохнуть.':'Подойдите к любому вражескому отряду.';
    $('message').textContent=message;
    $('map-controls').hidden=s.in_battle||capitalOpen;$('battle-controls').hidden=!s.in_battle||capitalOpen;
    $('continue').hidden=!s.in_battle||!mask[17];
    document.querySelectorAll('[data-action]').forEach(b=>b.disabled=mode!=='manual'||busy||s.done||!session||!mask[Number(b.dataset.action)]);
    $('reset').disabled=busy;
    if(s.in_battle){
      $('battle-heading').textContent='Отряд № '+(s.enemy+1)+' · '+enemyParty(map,s.enemy);
      $('round').textContent='Раунд '+Math.min(s.round,snap.battle_max_rounds ?? map.battle_max_rounds);
      $('turn-message').textContent=s.done?message:s.actor<6&&!s.retreating[s.actor]?'Ваш ход: '+role(s.actor).toLowerCase()+' '+(s.actor+1):s.actor>=6?'Ход противника: '+role(s.actor).toLowerCase()+' '+(s.actor-5):unitName(s.actor)+' завершает отступление';
      document.querySelectorAll('[data-slot]').forEach(b=>{
        const i=Number(b.dataset.slot),max=snap.max_hp[i],hp=s.hp[i];
        const targetTurn=i>=6&&s.actor<6&&!s.retreating[s.actor]&&!s.done;
        const unreachable=targetTurn&&hp>0&&!s.escaped[i]&&!mask[8+i-6];
        b.className='unit'+(i>=6?' foe':'')+(isMage(i)?' mage':'')+(isWarrior(i)?' warrior':'')+(i===s.actor&&!s.done?' active':'')+(!max?' empty':!hp?' dead':s.escaped[i]?' escaped':'')+(unreachable?' unreachable':targetTurn&&mask[8+i-6]?' reachable':'');
        b.querySelector('.unit-name').textContent=max?role(i)+' '+(i%6+1):'Пусто';
        b.querySelector('.archer-icon').textContent=isMage(i)?'✦':isWarrior(i)?'⚔':'➶';
        b.querySelector('.health').textContent=max?hp+' / '+max+' HP':'—';
        const values=snap.unit_stats?.[i]??[max,isMage(i)?map.mage_damage:isWarrior(i)?map.warrior_damage:map.archer_damage,100*(isWarrior(i)?(map.warrior_accuracy??.8):map.archer_accuracy),0,isWarrior(i)?map.warrior_initiative:60];
        b.querySelector('.unit-stats').textContent=max?'Урон '+fmt(values[1])+' · Точн. '+fmt(values[2])+'%\nБроня '+fmt(values[3])+'% · Иниц. '+fmt(values[4])+'\nИсточник: '+sourceName(profile(i)?.attack_type)+'\nИммунитеты: '+protectionNames(profile(i)?.immunities||0)+'\nЗащиты: '+protectionNames((profile(i)?.protections||0)&~s.wards_used[i])+((profile(i)?.protections||0)&s.wards_used[i]?' (израсходовано: '+protectionNames(profile(i).protections&s.wards_used[i])+')':''):'';
        b.querySelector('.unit-xp').textContent=max?experienceText(snap,i):'';
        b.querySelector('.health-fill').style.width=(max?hp/max*100:0)+'%';
        b.querySelector('.status').textContent=!max?'':!hp?'Погиб':s.escaped[i]?'Отступил':unreachable?'Вне досягаемости':s.retreating[i]?'Побег':s.defended[i]?'Защита':s.turn_phase[i]===1?'Ожидание':s.turn_phase[i]===2?'Ход завершён':'';
        b.disabled=i<6||mode!=='manual'||busy||s.done||!session||!mask[8+i-6];
        b.setAttribute('aria-label',(i>=6?(isMage(s.actor)?'Заклинание по всем врагам: противник ':isWarrior(s.actor)?'Удар мечом: противник ':'Стрелять: противник ')+(i%6+1):unitName(i))+', '+hp+' из '+max+' здоровья'+(unreachable?', вне досягаемости':''));
      });
      const queue=Array.from({length:12},(_,i)=>i).filter(i=>s.hp[i]>0&&!s.escaped[i]&&s.turn_phase[i]<2);
      const priority=i=>s.turn_phase[i]===0?s.priority[i]:-s.priority[i];queue.sort((a,b)=>priority(b)-priority(a)||a-b);
      $('queue').replaceChildren(...queue.map(i=>{const e=document.createElement('span');e.className='queue-unit'+(i>=6?' foe':'')+(i===s.actor?' current':'');e.textContent=(isMage(i)?'М':isWarrior(i)?'⚔':i<6?'Л':'П')+(i%6+1);e.title=unitName(i)+(basic?' · инициатива раунда '+Math.floor(s.priority[i]):'');return e;}));
    }else drawMap(s);
    $('events').replaceChildren(...messages.slice(-6).reverse().map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
    if(records.length){const last=records[recordIndex].frames.length-1;$('scrubber').max=last;$('scrubber').value=frame;$('frame-count').textContent=frame+' / '+last;$('next').disabled=frame>=last;}
  }
  async function request(path,payload){
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const value=await response.json();if(!response.ok)throw Error(value.error||'Не удалось выполнить действие.');return value;
  }
  function showError(error){$('connection').textContent='Игра недоступна';$('message').textContent=error.message+' Для игры запустите open-numbergrid.cmd.';}
  async function resetGame(){
    if(busy)return;busy=true;render();
    try{const result=await request('/api/reset',{seed:42,faction:$('faction').value||data.map.faction||'legions'});session=result.session;manualMap=result.map;manualConstruction=result.construction;manualTurnRules=result.turn_rules;manualCombat=result.combat;manual=result.snapshot;messages=[];$('connection').textContent='Игра готова · локально';}
    catch(error){busy=false;render();showError(error);return;}
    busy=false;render();
  }
  async function sendAction(action){
    if(mode!=='manual'||busy||!session||!manual||manual.state.done||!manual.action_mask[action])return;
    busy=true;render();
    try{const result=await request('/api/step',{session,action});
      for(const snapshot of result.events){manual=snapshot;const text=eventText(snapshot);if(text)messages.push(text);render();if(result.events.length>1)await new Promise(r=>setTimeout(r,180));}
      manual=result.snapshot;
    }catch(error){busy=false;render();showError(error);return;}
    busy=false;render();
  }
  function setMode(next){if(busy)return;stop();mode=next;messages=[];$('manual-tab').setAttribute('aria-selected',String(mode==='manual'));$('replay-tab').setAttribute('aria-selected',String(mode==='replay'));$('manual-controls').hidden=mode!=='manual';$('replay-controls').hidden=mode!=='replay';render();}
  function nextFrame(){const last=records[recordIndex].frames.length-1;if(frame<last){frame++;const text=eventText(current());if(text)messages.push(text);}if(frame>=last)stop();render();}
  $('manual-tab').onclick=()=>setMode('manual');$('replay-tab').onclick=()=>setMode('replay');$('replay-tab').disabled=!records.length;
  $('world-view').onclick=()=>{capitalOpen=false;render();};$('capital-view').onclick=()=>{capitalOpen=true;render();};
  $('building-branch').onchange=render;
  $('reset').onclick=resetGame;document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>sendAction(Number(b.dataset.action)));
  $('play').onclick=()=>{if(timer){stop();return;}if(frame===records[recordIndex].frames.length-1){frame=0;messages=[];}timer=setInterval(nextFrame,230);$('play').textContent='Ⅱ Пауза';};
  $('next').onclick=nextFrame;$('scrubber').oninput=()=>{stop();frame=Number($('scrubber').value);messages=[];render();};
  $('record').onchange=()=>{stop();recordIndex=Number($('record').value);frame=0;messages=[];render();};
  records.forEach((r,i)=>{const option=document.createElement('option');option.value=i;option.textContent=r.label;$('record').append(option);});
  const keys={ArrowUp:0,ArrowRight:2,ArrowDown:4,ArrowLeft:6,KeyW:0,KeyE:1,KeyD:2,KeyC:3,KeyX:4,KeyS:4,KeyZ:5,KeyA:6,KeyQ:7};
  window.addEventListener('keydown',event=>{if(event.target.matches('select,input,textarea')||event.ctrlKey||event.metaKey||event.altKey)return;if(mode==='manual'&&!capitalOpen&&!manual?.state.in_battle&&keys[event.code]!==undefined){event.preventDefault();sendAction(keys[event.code]);}});
  window.addEventListener('resize',()=>{const snap=current();if(snap&&!snap.state.in_battle)drawMap(snap.state);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  $('version').textContent=data.map.name;
  if(data.result){$('run-card').hidden=false;$('run-info').textContent='Запись: '+data.map.opponent_positions.length+' отрядов'+' сквайров и лучников'+' · '+fmt(data.result.training_steps)+' шагов · '+fmt(Math.round(data.result.mean_steps_per_second))+' шагов/с';$('run-evaluation').textContent='Победы argmax на карте записи: '+data.result.final_evaluation.successes+' / '+data.result.final_evaluation.episodes+'.';}
  window.numberGridApp={snapshot:()=>JSON.parse(JSON.stringify({mode,frame,...current()}))};
  if(records.length){manual=records[0].frames[0];render();}
  resetGame();
})();
