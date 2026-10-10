(() => {
  'use strict';
  const data=JSON.parse(document.getElementById('run-data').textContent);
  const $=id=>document.getElementById(id), fmt=n=>new Intl.NumberFormat('ru-RU',{maximumFractionDigits:3}).format(n);
  const canvas=$('board'), ctx=canvas.getContext('2d');
  let mode='manual', session=null, manual=null, manualMap=null, busy=false, frame=0, recordIndex=0, timer=null, messages=[];
  let manualConstruction=null, manualTurnRules=null, manualCombat=null, manualCapital=null, manualPotions=null, capitalOpen=false, potionsOpen=false, selectedPotion=0, capitalKey=null;
  let manualEquipment=null;
  const records=data.records||[];
  const combatInfo=()=>mode==='manual'?(manualCombat||data.combat):data.combat;
  const profile=i=>{
    const original=i>=0&&i<12?(combatInfo()?.catalogue?.[current()?.state.unit_ids?.[i]]??(i<6?combatInfo()?.heroes[i]:combatInfo()?.enemies[current()?.state.enemy]?.[i-6])):null;
    const lowered=combatInfo()?.catalogue?.[current()?.state.decay_form?.[i]];
    let base=lowered?{...lowered,name:original.name+' → '+lowered.name}:original;
    if(base&&current()?.state.copied?.[i])base={...base,name:'Двойник → '+base.name,size:1};
    if(base&&(current()?.state.summon_owner?.[i]??-1)>=0)base={...base,name:base.name+' · призван'};
    if(base&&current()?.state.imp?.[i])base={...base,name:base.name+' · '+(base.size===2?'Толстый бес':'Бес'),role:'melee',attack_type:'weapon',immunities:0,protections:0,unit_type:'Warrior'};
    const owners=current()?.state.healer_wards?.[i]||[];
    const temporary=[4,256,8,2].reduce((bits,bit,j)=>bits|(owners[j]?bit:0),0);
    const s=current()?.state,potionWard=s&&!s.imp?.[i]&&!s.decay_form?.[i]&&(s.summon_owner?.[i]??-1)<0?(s.copied?.[i]?(s.copy_traits?.[i]?.[3]||0):(s.potion_wards?.[i]||0)):0;
    const bookWard=base?.hero&&i<6&&!s?.imp?.[i]&&!s?.decay_form?.[i]&&!s?.copied?.[i]?(equipmentInfo()?.items?.[s?.equipped?.[3]]?.ward||0):0;
    if(base)base={...base,protections:(base.protections||0)|temporary|potionWard|bookWard};
    return base&&current()?.state.fenrir?.[i]&&!current()?.state.imp?.[i]?{...base,name:'Дух Фенрира',role:'melee',attack_type:'weapon',unit_type:'Warrior'}:base;
  };
  const isMage=i=>profile(i)?.role==='area';
  const isWarrior=i=>profile(i)?.role==='melee';
  const isHealer=i=>profile(i)?.role==='healer';
  const isCopier=i=>profile(i)?.unit_type==='Doppelganger';
  const isSummoner=i=>['Summoner','Occultmaster','Lyf','Laclaan'].includes(profile(i)?.unit_type);
  const targetAction=i=>i<6&&isCopier(current()?.state.actor)?57+i:8+i%6;
  const isPowerSupport=i=>['Travnitsa','Novice','Dwarfdruid','Arhidruid'].includes(profile(i)?.unit_type);
  const isMassHealer=i=>['Profit','Deva roshi'].includes(profile(i)?.unit_type);
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
  const potionInfo=()=>mode==='manual'?(manualPotions||data.potions):data.potions;
  const equipmentInfo=()=>mode==='manual'?(manualEquipment||data.equipment):data.equipment;
  const movementCap=snap=>snap.movement_cap;
  function renderEquipment(snap){
    const info=equipmentInfo(),s=snap.state;
    $('equipment-panel').hidden=!info;
    if(!info)return;
    $('equipment-rule').textContent=(info.policy==='price'?'Более дорогие находки автоматически заменяют дешёвые в своём слоте.':'Автовыбор по правилам референса: сапоги — по движению, книга сохраняется, артефакты — по цене.')+' Заменённые вещи остаются в инвентаре.';
    $('equipment-slots').replaceChildren(...info.slots.map((name,slot)=>{
      const item=info.items[s.equipped?.[slot]],row=document.createElement('div');row.className='equipment-slot';
      const title=document.createElement('span');title.textContent=name;
      const value=document.createElement('b');value.textContent=item?item.name:'Пусто';
      const detail=document.createElement('small');detail.textContent=item?item.label+' · '+fmt(item.price)+' золота':'';
      row.append(title,value,detail);return row;
    }));
    const counts=new Map();for(const id of s.item_inventory||[])if(id>=0)counts.set(id,(counts.get(id)||0)+1);
    $('equipment-count').textContent='Все предметы · '+[...counts.values()].reduce((a,b)=>a+b,0);
    $('equipment-stock').replaceChildren(...[...counts].map(([id,count])=>{
      const item=info.items[id],row=document.createElement('li');
      const worn=(s.equipped||[]).filter(i=>i===id).length;
      const cost=item.sell_price!=null?'продажа: '+fmt(item.sell_price):fmt(item.price);
      row.textContent=item.name+' × '+count+' · '+cost+' золота'+(worn?' · надето: '+worn:'');
      row.title=item.label;return row;
    }));
  }
  const capitalInfo=()=>mode==='manual'?(manualCapital||data.capital):data.capital;
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
  function renderCapitalServices(snap){
    const info=capitalInfo(),section=$('capital-services');section.hidden=!info;if(!info)return;
    const s=snap.state,atCapital=s.position.every((n,i)=>n===info.position[i]);
    const temple=snap.building_status?.[info.temple_action-18]===1;
    $('capital-location').textContent='Столица · клетка '+info.position.join(', ')+' · без стража';
    $('service-status').textContent=s.done?'Эпизод завершён.':s.in_battle?'Услуги доступны после боя.':!atCapital?'Вернитесь на золотую клетку столицы, чтобы лечить и воскрешать бойцов.':!temple?'Вы в столице. Постройте Храм за 300 золота, чтобы открыть лечение и воскрешение.':'Вы в столице. Храм открыт; услуги не расходуют очки перемещения и не завершают ход.';
    const cards=[];
    for(let i=0;i<6;i++){
      if(!snap.max_hp[i])continue;
      const q=snap.capital_quotes?.[i];if(!q)continue;
      const card=document.createElement('article');card.className='recovery-card'+(!s.hp[i]?' fallen':'');
      const name=document.createElement('h3');name.textContent=role(i)+' '+(i+1);card.append(name);
      const hp=document.createElement('p');hp.className='recovery-health';hp.textContent=s.hp[i]+' / '+snap.max_hp[i]+' HP'+(!s.hp[i]?' · погиб':'');card.append(hp);
      const prices=document.createElement('p');prices.className='recovery-price';prices.textContent='Лечение: '+q[0]+' золото / HP · воскрешение: '+q[4]+' золота';card.append(prices);
      const action=(!s.hp[i]?info.revive_start:info.heal_start)+i;
      const button=document.createElement('button');button.dataset.action=action;
      button.textContent=!s.hp[i]?'Воскресить · '+q[4]+' золота':q[1]===0?'Здоровье полное':q[2]>0?'Лечить +'+q[2]+' HP · '+q[3]+' золота':'Не хватает золота';
      button.dataset.agentAction=action;button.onclick=()=>sendAction(action);card.append(button);
      cards.push(card);
    }
    $('recovery-list').replaceChildren(...cards);
  }
  function renderPotions(snap){
    const items=potionInfo()||[],s=snap.state;
    $('potion-status').textContent=s.done?'Эпизод завершён.':s.in_battle?'Зелья можно использовать после завершения боя.':'Доступны на любой клетке карты, даже без очков перемещения. Выберите зелье и бойца.';
    $('potion-stock').replaceChildren(...items.map((item,i)=>{
      const button=document.createElement('button');button.className='potion-choice';
      button.setAttribute('aria-pressed',String(selectedPotion===i));
      button.textContent=item.name+' · '+item.label+' · '+s.potions[i]+' шт.';
      button.onclick=()=>{selectedPotion=i;render();};return button;
    }));
    const item=items[selectedPotion];if(!item)return;
    $('potion-detail').textContent=item.effect==='revive'?'Воскрешает одного погибшего бойца с 1 HP.':item.effect==='heal'?'Восстанавливает до '+item.amount+' HP одному живому раненому бойцу. Здоровье не превышает максимум.':item.label+' · '+(item.duration==='permanent'?'Постоянный эффект; сохраняется при повышении.':'До следующего отдыха. Один раз на бойца за ход для каждого вида зелья.');
    const cards=[];
    for(let i=0;i<6;i++){
      if(!snap.max_hp[i])continue;
      const card=document.createElement('article');card.className='recovery-card'+(!s.hp[i]?' fallen':'');
      const name=document.createElement('h3');name.textContent=role(i)+' '+(i+1);card.append(name);
      const hp=document.createElement('p');hp.className='recovery-health';hp.textContent=s.hp[i]+' / '+snap.max_hp[i]+' HP'+(!s.hp[i]?' · погиб':'');card.append(hp);
      const button=document.createElement('button'),amount=snap.potion_quotes?.[selectedPotion]?.[i]||0;
      button.dataset.action=item.action_start+i;
      button.textContent=!s.potions[selectedPotion]?'Зелья закончились':item.effect==='revive'?(s.hp[i]?'Боец жив':'Воскресить · 1 бутылка'):!s.hp[i]?'Сначала воскресите бойца':item.effect==='heal'?(!amount?'Здоровье полное':'Лечить +'+amount+' HP · 1 бутылка'):snap.action_mask[item.action_start+i]?'Применить · 1 бутылка':s.in_battle?'Завершите бой':'Эффект уже действует';
      button.dataset.agentAction=item.action_start+i;button.onclick=()=>sendAction(item.action_start+i);card.append(button);cards.push(card);
    }
    $('potion-targets').replaceChildren(...cards);
  }
  function renderMapCommands(snap){
    const attacks=[];
    for(const [action,command] of (snap.map_commands||[]).entries()){
      const [enemy,cost]=command, direction=document.querySelector('.dpad [data-action="'+action+'"]');
      direction.classList.toggle('attack-direction',enemy>=0);
      direction.title=enemy>=0?'Атаковать отряд № '+(enemy+1)+' · '+cost+' очков':'Перемещение · '+cost+' очка';
      if(enemy<0)continue;
      const button=document.createElement('button');button.dataset.action=action;
      button.textContent='Атаковать отряд № '+(enemy+1)+' · '+enemyParty(currentMap(),enemy)+' · '+cost+' очков';
      button.dataset.agentAction=action;button.onclick=()=>sendAction(action);attacks.push(button);
    }
    $('map-targets').replaceChildren(...attacks);
    $('map-targets').hidden=!attacks.length;
  }
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
    (map.chests||[]).forEach((chest,i)=>{
      if(!s.chest_alive?.[i])return;
      const [r,c]=chest.position,x=c*cell,y=r*cell;
      roundRect(x+cell*.12,y+cell*.24,cell*.76,cell*.6,cell*.1,'#9b6230');
      roundRect(x+cell*.12,y+cell*.17,cell*.76,cell*.28,cell*.1,'#d8ad64');
      ctx.fillStyle='#f5dfa1';ctx.fillRect(x+cell*.44,y+cell*.35,cell*.12,cell*.24);
    });
    const capital=capitalInfo()?.position;
    if(capital){
      roundRect(capital[1]*cell+1,capital[0]*cell+1,cell-2,cell-2,cell*.15,'#dfb963');
      ctx.fillStyle='#604515';ctx.font='650 '+Math.round(cell*.62)+'px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('♜',(capital[1]+.5)*cell,(capital[0]+.52)*cell);
    }
    token(s.position,s.hp.slice(0,6).filter(h=>h>0).length,true);
    if(capital){ctx.strokeStyle='#aa7623';ctx.lineWidth=2;ctx.strokeRect(capital[1]*cell+1,capital[0]*cell+1,cell-2,cell-2);}
  }
  const slots=[3,0,4,1,5,2,6,9,7,10,8,11];
  for(const i of slots){
    const button=document.createElement('button');button.className='unit'+(i>=6?' foe':'');button.dataset.slot=i;
    button.innerHTML='<span class="unit-name"></span><span class="archer-icon" aria-hidden="true">➶</span><span class="health"></span><span class="unit-stats"></span><span class="unit-xp"></span><span class="health-bar"><span class="health-fill"></span></span><span class="status"></span>';
    button.dataset.agentAction=targetAction(i);button.onclick=()=>sendAction(targetAction(i));$(i<6?'allies':'enemies').append(button);
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
    const ticks=[...['poison','burn','water'].flatMap(kind=>(snapshot.state['last_'+kind+'_damage']||[]).map((amount,i)=>amount?unitName(i)+': '+({poison:'яд',burn:'огонь',water:'вода'}[kind])+' −'+amount+' HP':'').filter(Boolean))];
    return actionEventText(snapshot)+(ticks.length?' '+ticks.join('; ')+'.':'');
  }
  function actionEventText(snapshot){
    const s=snapshot.state,a=unitName(s.last_actor),t=unitName(s.last_target);
    switch(s.last_event){
      case 1:return 'Атака отряда № '+(s.enemy+1)+'. Отряд остаётся на своей клетке; очков перемещения: '+s.movement_points+'.';
      case 2:return isMage(s.last_actor)?a+': заклинание по всем противникам (суммарный урон '+s.last_damage+').'+blockText(s):a+(isWarrior(s.last_actor)?': удар мечом по ':': попадание в ')+t.toLowerCase()+' (−'+s.last_damage+').'+blockText(s);
      case 3:return a+': промах.';
      case 22:return a+': превращение в Дух Фенрира. Доля здоровья сохранена.';
      case 33:return a+': принял облик выбранного бойца.';
      case 34:return a+': призвал подкрепление. Очередь показывает готовность новых бойцов.';
      case 23:return a+(s.last_target<0?': массовое превращение в бесов.':': '+t+' превращён в беса.')+' Здоровье сохранено.'+blockText(s);
      case 25:return a+(s.last_target<0?': массовый паралич.':': паралич — '+t+'.')+blockText(s);
      case 26:return a+': пропускает ход из-за паралича.';
      case 27:return a+(s.last_target<0?': массовый страх.':': страх — '+t+'.')+blockText(s);
      case 28:return 'Противник побеждён. Лекари получают по одному действию перед выдачей опыта.';
      case 24:return a+': понижение формы '+t+' (урон '+s.last_damage+'). Доля HP пересчитана.';
      case 4:return a+' встал в защиту.';
      case 5:return a+(s.post_victory?' пропускает заключительное лечение.':' ждёт конца раунда.');
      case 6:return a+' готовится отступить.';
      case 7:return a+' покинул бой.';
      case 8:return 'Победа! Ранения и потери сохранены.'+xpSummary(snapshot);
      case 9:return 'Ваш отряд погиб. Игра завершена.';
      case 10:return 'Отступление завершено. Ранения и потери сохранены.';
      case 11:return 'Бой достиг лимита раундов. Эпизод завершён.';
      case 12:{const b=construction()?.buildings[s.last_building];return b?'Построено: '+b.name+' (−'+b.gold+' золота).':null;}
      case 13:return 'Отдых. Начался ход '+s.day+', +'+turnRules().income+' золота; очки перемещения восстановлены, живые бойцы получили регенерацию 10% HP. Штраф: '+fmt(Math.max(0,-snapshot.reward))+'.';
      case 14:return 'Атака заблокирована иммунитетом.'+blockText(s);
      case 32:return a+': атака и замедление инициативы противников (урон '+s.last_damage+').';
      case 31:return a+': дополнительный ход — '+t+'.';
      case 30:return a+': воскрешение — '+t+' (+'+s.last_damage+' HP). Повторное воскрешение в этом бою недоступно.';
      case 29:return a+': усиление — '+t+'. Сила атаки: '+s.last_damage+'.';
      case 16:return a+(s.last_target<0?': лечение союзников':': лечение — '+t.toLowerCase())+' (+'+s.last_damage+' HP).';
      case 17:return 'Храм: '+t.toLowerCase()+' восстановил '+s.last_damage+' HP за '+s.last_service_cost+' золота.';
      case 18:return 'Храм: '+t.toLowerCase()+' воскрешён с 1 HP за '+s.last_service_cost+' золота.';
      case 19:return potionInfo()?.[s.last_potion]?.name+': '+t.toLowerCase()+' восстановил '+s.last_damage+' HP. Осталось: '+s.potions[s.last_potion]+'.';
      case 21:return 'Сундук: '+[...s.last_loot.map((n,i)=>n?(potionInfo()?.[i]?.name+' × '+n):''),...(s.last_item_loot||[]).map((n,i)=>n?(equipmentInfo()?.items?.[i]?.name+' × '+n):'')].filter(Boolean).join(', ')+'. Предметы получены, экипировка обновлена.';
      case 20:return 'Зелье воскрешения: '+t.toLowerCase()+' вернулся с 1 HP. Осталось: '+s.potions[s.last_potion]+'.';
      case 37:return potionInfo()?.[s.last_potion]?.name+': '+t.toLowerCase()+' получил усиление. Осталось: '+s.potions[s.last_potion]+'.';
      case 15:return 'Атака поглощена защитой.'+blockText(s);
      default:return null;
    }
  }
  function render(){
    const snap=current();if(!snap)return;const s=snap.state,mask=snap.action_mask,map=currentMap();
    $('version').textContent=map.name;
    $('map-panel').hidden=s.in_battle||capitalOpen||potionsOpen;$('battle-panel').hidden=!s.in_battle||capitalOpen||potionsOpen;
    $('capital-panel').hidden=!capitalOpen;$('potions-panel').hidden=!potionsOpen;
    $('world-view').setAttribute('aria-pressed',String(!capitalOpen&&!potionsOpen));$('capital-view').setAttribute('aria-pressed',String(capitalOpen));$('potions-view').setAttribute('aria-pressed',String(potionsOpen));
    renderConstruction(snap);
    renderCapitalServices(snap);
    renderPotions(snap);
    renderEquipment(snap);
    renderMapCommands(snap);
    $('phase-label').textContent=s.in_battle?'Бой · отряд № '+(s.enemy+1):'Карта '+map.size+' × '+map.size;
    $('remaining').textContent=s.alive.filter(Boolean).length;
    $('chest-status').hidden=!(map.chests||[]).length;
    $('chest-status').textContent=(s.last_event===21?eventText(snap)+' ':'')+'Сундуков осталось: '+(s.chest_alive||[]).filter(Boolean).length+' / '+(map.chests||[]).length+'. Подойдите на соседнюю клетку, включая диагональ: всё содержимое попадёт в инвентарь.';
    $('wins').textContent=s.alive.filter(v=>!v).length;
    $('steps').textContent=fmt(s.step_count);$('reward').textContent=fmt(snap.total_reward);
    $('gold').textContent=fmt(s.gold);
    const turns=turnRules(),points=s.movement_points;
    $('turn-summary').textContent='Ход '+s.day+' · '+points+' / '+movementCap(snap)+' очков перемещения';
    $('movement-summary').textContent=points+' / '+movementCap(snap)+' очков · '+Math.floor(points/turns.move_cost)+' перемещений';
    $('rest').dataset.action=turns.rest_action;
    $('rest-hint').textContent=s.in_battle?'Отдых доступен после завершения боя.':snap.rest_penalty>0?'Штраф за оставшиеся очки: −'+fmt(snap.rest_penalty)+'. Следующий ход: +'+turns.income+' золота и полный запас очков.':'Все очки использованы: отдых без штрафа. Следующий ход: +'+turns.income+' золота и полный запас очков.';
    if(!s.in_battle)$('rest-hint').textContent+=' Живым бойцам: +'+(turns.regeneration_percent+10*(equipmentInfo()?.items?.[s.equipped?.[2]]?.regeneration||0))+'% максимального HP с округлением вверх (до максимума).';
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
    $('battle-hint').textContent='Источник атаки, иммунитеты и оставшиеся защиты указаны на карточках. Точность 80% соответствует фактическому шансу попадания 92,2%. Броня и действие защиты уменьшают прошедший урон. Титан занимает обе клетки линии. Служка лечит одного живого союзника на 20 HP.';
    $('fenrir').hidden=!s.in_battle||profile(s.actor)?.unit_type!=='Wolf Lord';
    if(s.in_battle&&s.round===0)$('battle-hint').textContent='Подготовка двойников перед первым раундом: выберите доступного бойца любой стороны или пропустите копирование кнопкой защиты.';
    $('attack-hint').textContent=profile(s.actor)?.unit_type==='Succub'?'Массовое превращение в бесов без урона: выберите любого доступного живого противника.':profile(s.actor)?.unit_type==='Witch'?'Превращение: выберите живого противника.':profile(s.actor)?.unit_type==='Alchemist'?'Дополнительный ход: выберите живого союзника, кроме алхимика и отступающего.':profile(s.actor)?.unit_type==='Patriach'?'Выберите союзника: лечение живого или однократное воскрешение погибшего с 50% HP.':isPowerSupport(s.actor)?(['Dwarfdruid','Arhidruid'].includes(profile(s.actor)?.unit_type)?'Выберите союзника для усиления и очищения; себя — только для очищения.':'Усиление: выберите другого живого союзника.'):isMassHealer(s.actor)?'Массовое лечение всех живых союзников, кроме самого лекаря. Выберите любую доступную карточку.':isHealer(s.actor)?'Лечение: выберите живого союзника.':isMage(s.actor)?'Массовая атака: выберите любого живого врага.':isWarrior(s.actor)?(mask.slice(8,14).some(Boolean)?'Ход воина: выберите подсвеченного врага для удара мечом.':'Воин не достаёт до врагов. Можно защищаться, ждать или отступить.'):'Нажмите на живого противника, чтобы выстрелить.';
    if(isCopier(s.actor))$('attack-hint').textContent='Копирование: выберите доступного бойца любой стороны. Большие юниты и стражи недоступны.';
    if(isSummoner(s.actor))$('attack-hint').textContent='Призыв: выберите свободную клетку своего отряда. Большое существо занимает переднюю и заднюю клетки линии.';
    const message=s.won?'Победа! Карта очищена.':s.lost?'Ваш отряд погиб. Начните новую игру.':s.done?'Достигнут лимит. Начните новую игру.':s.in_battle?'Выбирайте цели и берегите свой отряд.':!s.movement_points?'Очки перемещения закончились. Можно использовать зелья, строить или отдохнуть.':'Шаг на клетку врага — атака. Рядом с отрядом можно пройти без боя.';
    $('message').textContent=message;
    $('map-controls').hidden=s.in_battle||capitalOpen||potionsOpen;$('battle-controls').hidden=!s.in_battle||capitalOpen||potionsOpen;
    $('continue').hidden=!s.in_battle||!mask[17];
    document.querySelectorAll('[data-action]').forEach(b=>b.disabled=mode!=='manual'||busy||s.done||!session||!mask[Number(b.dataset.action)]);
    $('reset').disabled=busy;
    if(s.in_battle){
      $('battle-heading').textContent='Отряд № '+(s.enemy+1)+' · '+enemyParty(map,s.enemy);
      $('round').textContent='Раунд '+Math.min(s.round,snap.battle_max_rounds ?? map.battle_max_rounds);
      $('turn-message').textContent=s.done?message:s.actor<6&&(!s.retreating[s.actor]||s.post_victory)?'Ваш ход: '+role(s.actor).toLowerCase()+' '+(s.actor+1):s.actor>=6?'Ход противника: '+role(s.actor).toLowerCase()+' '+(s.actor-5):unitName(s.actor)+' завершает отступление';
      if(s.post_victory&&!s.done)$('turn-message').textContent='Лечение после боя: '+unitName(s.actor)+(s.actor<6?' · выберите союзника или пропустите действие кнопкой ожидания.':'');
      if(s.second_strike&&!s.done)$('turn-message').textContent+=' · второй удар';
      document.querySelectorAll('[data-slot]').forEach(b=>{
        const i=Number(b.dataset.slot),max=snap.max_hp[i],hp=s.hp[i];
        const local=i%6,big=profile(i)?.size===2&&(s.hp[i]>0||!s.hp[i+3]),covered=local>=3&&profile(i-3)?.size===2&&s.hp[i-3]>0;
        b.hidden=covered;
        b.style.gridRow=String(local%3+1);
        b.style.gridColumn=big?'1 / span 2':String(i<6?(local<3?2:1):(local<3?1:2));
        const action=targetAction(i);
        const targetTurn=(isCopier(s.actor)?true:(isHealer(s.actor)||isSummoner(s.actor))?i<6:i>=6)&&s.actor<6&&(!s.retreating[s.actor]||s.post_victory)&&!s.done;
        const unreachable=targetTurn&&hp>0&&!s.escaped[i]&&!mask[action];
        b.className='unit'+(i>=6?' foe':'')+(big?' large':'')+(isHealer(i)?' healer':'')+(isMage(i)?' mage':'')+(isWarrior(i)?' warrior':'')+(i===s.actor&&!s.done?' active':'')+(!max?' empty':!hp?' dead':s.escaped[i]?' escaped':'')+(unreachable?' unreachable':targetTurn&&mask[action]?' reachable':'');
        b.querySelector('.unit-name').textContent=max?role(i)+' '+(i%6+1)+(big?' · 2 клетки':''):'Пусто';
        b.querySelector('.archer-icon').textContent=isHealer(i)?'✚':isMage(i)?'✦':isWarrior(i)?'⚔':'➶';
        b.querySelector('.health').textContent=max?hp+' / '+max+' HP':'—';
        const values=snap.unit_stats?.[i]??[max,isMage(i)?map.mage_damage:isWarrior(i)?map.warrior_damage:map.archer_damage,100*(isWarrior(i)?(map.warrior_accuracy??.8):map.archer_accuracy),0,isWarrior(i)?map.warrior_initiative:60];
        b.querySelector('.unit-stats').textContent=max?(profile(i)?.unit_type==='Succub'?'Массовое превращение · без урона':profile(i)?.unit_type==='Witch'?'Превращение · без урона':(profile(i)?.unit_type==='Alchemist'?'Дополнительный ход':isPowerSupport(i)?'Усиление ×'+({Travnitsa:1.25,Novice:1.5,Dwarfdruid:1.75,Arhidruid:2}[profile(i).unit_type]):(isMassHealer(i)?'Массовое лечение ':isHealer(i)?'Лечение ':'Урон ')+fmt(values[1])))+(profile(i)?.unit_type==='Centaur Savage'?' + критический удар 5% (целые HP)':['Demon','Elfarcher'].includes(profile(i)?.unit_type)?' × 2 удара':'')+' · Точн. '+fmt(values[2])+'%\nБроня '+fmt(values[3])+'% · Иниц. '+fmt(values[4])+'\nИсточник: '+sourceName(profile(i)?.attack_type)+'\nИммунитеты: '+protectionNames(profile(i)?.immunities||0)+'\nЗащиты: '+protectionNames((profile(i)?.protections||0)&~s.wards_used[i])+((profile(i)?.protections||0)&s.wards_used[i]?' (израсходовано: '+protectionNames(profile(i).protections&s.wards_used[i])+')':''):'';
        b.querySelector('.unit-xp').textContent=max?experienceText(snap,i):'';
        b.querySelector('.health-fill').style.width=(max?hp/max*100:0)+'%';
        b.querySelector('.status').textContent=!max?'':!hp?'Погиб':s.escaped[i]?'Отступил':unreachable?'Вне досягаемости':s.paralyzed?.[i]?'Паралич':s.long_paralyzed?.[i]?'Долгий паралич':s.feared?.[i]?'Страх':s.retreating[i]?'Побег':s.defended[i]?'Защита':s.imp?.[i]?'Временное превращение':s.decay_form?.[i]?'Понижение формы':s.weakened?.[i]?'Урон ослаблен':s.armor_shreds?.[i]?'Броня повреждена':s.turn_phase[i]===1?'Ожидание':s.turn_phase[i]===2?'Ход завершён':'';
        if((s.slow_original?.[i]??-1)>=0)b.querySelector('.status').textContent+=' · Инициатива снижена';
        if(s.powerup?.[i])b.querySelector('.status').textContent+=' · Урон усилен';
        if(s.poison_turns?.[i])b.querySelector('.status').textContent+=' · Яд '+s.poison_damage[i]+' HP, ходов: '+s.poison_turns[i];
        if(s.water_turns?.[i])b.querySelector('.status').textContent+=' · Вода '+(s.water_damage[i]||10)+' HP, ходов: '+s.water_turns[i];
        if(s.burn_turns?.[i])b.querySelector('.status').textContent+=' · Огонь '+(s.burn_damage[i]||10)+' HP, ходов: '+s.burn_turns[i];
        b.disabled=!targetTurn||mode!=='manual'||busy||s.done||!session||!mask[action];
        b.setAttribute('aria-label',(isHealer(s.actor)&&i<6?(profile(s.actor)?.unit_type==='Alchemist'?'Дать дополнительный ход: ':profile(s.actor)?.unit_type==='Patriach'&&!hp?'Воскресить: ':isPowerSupport(s.actor)?'Усилить: ':'Лечить: ')+unitName(i):i>=6?(profile(s.actor)?.unit_type==='Succub'?'Превращение всех доступных врагов: противник ':profile(s.actor)?.unit_type==='Witch'?'Превращение: противник ':isMage(s.actor)?'Заклинание по всем врагам: противник ':isWarrior(s.actor)?'Удар мечом: противник ':'Стрелять: противник ')+(i%6+1):unitName(i))+', '+hp+' из '+max+' здоровья'+(unreachable?', вне досягаемости':''));
      });
      const queue=Array.from({length:12},(_,i)=>i).filter(i=>s.round===0?s.preparation?.[i]:s.post_victory?s.pending_healers[i]:s.hp[i]>0&&!s.escaped[i]&&s.turn_phase[i]<2);
      const priority=i=>s.turn_phase[i]===0?s.priority[i]:-s.priority[i];queue.sort((a,b)=>s.post_victory?a-b:priority(b)-priority(a)||a-b);
      $('queue').replaceChildren(...queue.map(i=>{const e=document.createElement('span');e.className='queue-unit'+(i>=6?' foe':'')+(i===s.actor?' current':'');e.textContent=(isHealer(i)?'✚':isMage(i)?'М':isWarrior(i)?'⚔':i<6?'Л':'П')+(i%6+1);e.title=unitName(i)+(basic?' · инициатива раунда '+Math.floor(s.priority[i]):'');return e;}));
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
    try{const result=await request('/api/reset',{seed:42,faction:$('faction').value||data.map.faction||'legions'});session=result.session;manualMap=result.map;manualConstruction=result.construction;manualTurnRules=result.turn_rules;manualCombat=result.combat;manualCapital=result.capital;manualPotions=result.potions;manualEquipment=result.equipment;manual=result.snapshot;messages=[];$('connection').textContent='Игра готова · локально';}
    catch(error){busy=false;render();showError(error);return;}
    busy=false;render();
  }
  async function sendAction(action){
    if(mode!=='manual'||busy||!session||!manual||manual.state.done||!manual.action_mask[action])return false;
    busy=true;render();
    try{const result=await request('/api/step',{session,action});
      for(const snapshot of result.events){manual=snapshot;const text=eventText(snapshot);if(text)messages.push(text);render();if(result.events.length>1)await new Promise(r=>setTimeout(r,180));}
      manual=result.snapshot;
    }catch(error){busy=false;render();showError(error);return false;}
    busy=false;render();return true;
  }
  function setMode(next){if(busy)return;stop();mode=next;messages=[];$('manual-tab').setAttribute('aria-selected',String(mode==='manual'));$('replay-tab').setAttribute('aria-selected',String(mode==='replay'));$('manual-controls').hidden=mode!=='manual';$('replay-controls').hidden=mode!=='replay';render();}
  function nextFrame(){const last=records[recordIndex].frames.length-1;if(frame<last){frame++;const text=eventText(current());if(text)messages.push(text);}if(frame>=last)stop();render();}
  $('map-zoom').onchange=()=>{
    canvas.style.width=(Number($('map-zoom').value)*100)+'%';
    const snap=current();if(!snap)return;drawMap(snap.state);
    const viewport=$('map-viewport'),cell=canvas.clientWidth/currentMap().size;
    viewport.scrollLeft=(snap.state.position[1]+.5)*cell-viewport.clientWidth/2;
    viewport.scrollTop=(snap.state.position[0]+.5)*cell-viewport.clientHeight/2;
  };
  canvas.onmousemove=event=>{
    const snap=current();if(!snap)return;
    const rect=canvas.getBoundingClientRect(),map=currentMap();
    const row=Math.floor((event.clientY-rect.top)/rect.height*map.size),col=Math.floor((event.clientX-rect.left)/rect.width*map.size);
    const chest=(map.chests||[]).find((c,i)=>snap.state.chest_alive?.[i]&&c.position[0]===row&&c.position[1]===col);
    canvas.title=chest?'Сундук: '+[...Object.entries(chest.potions||{}),...Object.entries(chest.items||{})].filter(([,n])=>n).map(([key,n])=>(potionInfo()?.find(p=>p.key===key)?.name||equipmentInfo()?.items.find(p=>p.key===key)?.name||key)+' × '+n).join(', '):'';
  };
  canvas.ondblclick=event=>{
    const snap=current();if(mode!=='manual'||!snap||snap.state.in_battle||busy||snap.state.done)return;
    const rect=canvas.getBoundingClientRect(),map=currentMap();
    const row=Math.floor((event.clientY-rect.top)/rect.height*map.size),col=Math.floor((event.clientX-rect.left)/rect.width*map.size);
    const enemy=map.opponent_positions.findIndex((p,i)=>snap.state.alive[i]&&p[0]===row&&p[1]===col);
    if(enemy<0)return;
    const action=(snap.map_commands||[]).findIndex(command=>command[0]===enemy);
    if(action>=0&&snap.action_mask[action])sendAction(action);
  };
  $('manual-tab').onclick=()=>setMode('manual');$('replay-tab').onclick=()=>setMode('replay');$('replay-tab').disabled=!records.length;
  $('world-view').onclick=()=>{capitalOpen=false;potionsOpen=false;render();};$('capital-view').onclick=()=>{capitalOpen=true;potionsOpen=false;render();};
  $('potions-view').onclick=()=>{capitalOpen=false;potionsOpen=true;render();};
  $('building-branch').onchange=render;
  $('reset').onclick=resetGame;document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>sendAction(Number(b.dataset.action)));
  $('play').onclick=()=>{if(timer){stop();return;}if(frame===records[recordIndex].frames.length-1){frame=0;messages=[];}timer=setInterval(nextFrame,230);$('play').textContent='Ⅱ Пауза';};
  $('next').onclick=nextFrame;$('scrubber').oninput=()=>{stop();frame=Number($('scrubber').value);messages=[];render();};
  $('record').onchange=()=>{stop();recordIndex=Number($('record').value);frame=0;messages=[];render();};
  records.forEach((r,i)=>{const option=document.createElement('option');option.value=i;option.textContent=r.label;$('record').append(option);});
  const keys={ArrowUp:0,ArrowRight:2,ArrowDown:4,ArrowLeft:6,KeyW:0,KeyE:1,KeyD:2,KeyC:3,KeyX:4,KeyS:4,KeyZ:5,KeyA:6,KeyQ:7};
  window.addEventListener('keydown',event=>{if(event.target.matches('select,input,textarea')||event.ctrlKey||event.metaKey||event.altKey)return;if(mode==='manual'&&!capitalOpen&&!potionsOpen&&!manual?.state.in_battle&&keys[event.code]!==undefined){event.preventDefault();sendAction(keys[event.code]);}});
  window.addEventListener('resize',()=>{const snap=current();if(snap&&!snap.state.in_battle)drawMap(snap.state);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  $('version').textContent=data.map.name;
  if(data.result){$('run-card').hidden=false;$('run-info').textContent='Запись: '+data.map.opponent_positions.length+' вражеских отрядов'+' · '+fmt(data.result.training_steps)+' шагов · '+fmt(Math.round(data.result.mean_steps_per_second))+' шагов/с';$('run-evaluation').textContent='Победы argmax на карте записи: '+data.result.final_evaluation.successes+' / '+data.result.final_evaluation.episodes+'.';}
  let agentPick=null,agentAuto=false,actionCount=0,autoRun=0;
  function showPick(){
    document.querySelectorAll('.agent-pick').forEach(b=>b.classList.remove('agent-pick'));
    if(mode!=='manual'||!agentPick||agentPick.count!==actionCount)return;
    document.querySelectorAll('[data-action="'+agentPick.action+'"],[data-agent-action="'+agentPick.action+'"]').forEach(b=>b.classList.add('agent-pick'));
  }
  const baseRender=render;render=function(){baseRender();showPick();};
  const baseSend=sendAction;sendAction=async function(action){const ok=await baseSend(action);if(ok)actionCount++;showPick();return ok;};
  async function askAgent(){
    if(!session||!manual||manual.state.done)return null;
    const askedSession=session,askedCount=actionCount;
    try{const pick=await request('/api/agent',{session});
      if(session!==askedSession||actionCount!==askedCount)return null;
      agentPick={...pick,count:actionCount};
      $('agent-hint').textContent='Агент выбирает: '+pick.label+' (вероятность '+Math.round(pick.probability*100)+'%).';showPick();return agentPick;}
    catch(error){if(session===askedSession)$('agent-hint').textContent=error.message;return null;}
  }
  async function agentMove(){if(busy)return false;const pick=await askAgent();if(!pick||busy)return false;return await sendAction(pick.action);}
  function setAuto(on){agentAuto=on;$('agent-auto').textContent=on?'Ⅱ Остановить агента':'▶ Агент играет сам';}
  $('reset').addEventListener('click',()=>{agentPick=null;setAuto(false);showPick();});
  $('agent-suggest').onclick=askAgent;$('agent-move').onclick=agentMove;
  $('agent-auto').onclick=async()=>{if(agentAuto){setAuto(false);return;}const run=++autoRun;setAuto(true);
    while(run===autoRun&&agentAuto&&mode==='manual'&&manual&&!manual.state.done){if(busy){await new Promise(r=>setTimeout(r,60));continue;}if(!await agentMove())break;await new Promise(r=>setTimeout(r,120));}
    if(run===autoRun)setAuto(false);};
  fetch('/api/health').then(r=>r.json()).then(h=>{if(!h.agent)return;$('agent-controls').hidden=false;
    $('agent-info').textContent='Модель: '+fmt(h.agent.training_steps)+' шагов обучения · побед argmax '+h.agent.argmax_wins+' / '+h.agent.argmax_episodes+'.';}).catch(()=>{});
  window.numberGridApp={snapshot:()=>JSON.parse(JSON.stringify({mode,frame,...current()}))};
  if(records.length){manual=records[0].frames[0];render();}
  resetGame();
})();
