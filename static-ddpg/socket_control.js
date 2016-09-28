(function(){
  var socket = io('http://127.0.0.1:5000');
  var jStatus = Dom.get('server-status');

  socket.on('connect', function(){
    jStatus.innerHTML = '<p>connect</p>';
  });
  
  socket.on('disconnect', function(){
    jStatus.innerHTML += '<p>disconnect</p>';
  });

  socket.on('action', function(data){
    playerX = data.playerX;
    speed = data.speed;
  });

  Dom.get('j-btn-start').onclick = function(){
    var space = {
      playerX_space: [-1.0, 1.0],
      speed_space: [0, maxSpeed],
    };
    // tell server the range of action, for normalization
    socket.emit('action_space', space);
    // for avoiding exceptions in replay buffer on server
    // START_FRAME = true;
    Game.run(gameParams);
  };

  Dom.get('j-btn-stop').onclick = function(){
    Game.stop();
  };

  /*----------- the above is controller----------------------------------*/
  var timeIntervalID = null;
  // dynamically create a smaller canvas for preview
  var smallCanvas = document.createElement('canvas');
  var smallCtx = smallCanvas.getContext('2d');
  var smallImage = new Image();
  var smallWidth = 84;
  var smallHeight = 84;
  smallCanvas.width = smallWidth;
  smallCanvas.height = smallHeight;
  smallCanvas.zIndex = '100';
  Dom.get('preview').appendChild(smallCanvas);

  function capture() {
    // scale the snapshot of main canvas to a smaller one
    smallCtx.drawImage(canvas, 0, 0, smallWidth, smallHeight);
    var prefix = 'data:image/png;base64,';
    return smallCanvas.toDataURL().substr(prefix.length);
  }

  function getReward() {
    var COLLISION_COST = -1.0;
    var OFF_ROAD_COST = -0.8;
    var LANE_PENALTY = 0.5;

    if (COLLISION_OCCURED) {
      return COLLISION_COST;
    }
    var pos = Math.abs(playerX);
    if (pos > 1.0) {
      return OFF_ROAD_COST;
    }

    var inLane = pos <= 0.1 || (pos >= 0.6 && pos <= 0.8)
    var penalty = inLane ? 1 : LANE_PENALTY;
    return penalty * (speed / maxSpeed);
  }

//=========================================================================
// THE GAME LOOP
//=========================================================================
  
  var sampleCount = 0;
  var gameParams = {
    canvas: canvas, render: render, update: update, stats: stats, step: step,
    images: ["background", "sprites"],
    keys: [
      { keys: [KEY.LEFT,  KEY.A], mode: 'down', action: function() { keyLeft   = true;  } },
      { keys: [KEY.RIGHT, KEY.D], mode: 'down', action: function() { keyRight  = true;  } },
      { keys: [KEY.UP,    KEY.W], mode: 'down', action: function() { keyFaster = true;  } },
      { keys: [KEY.DOWN,  KEY.S], mode: 'down', action: function() { keySlower = true;  } },
      { keys: [KEY.LEFT,  KEY.A], mode: 'up',   action: function() { keyLeft   = false; } },
      { keys: [KEY.RIGHT, KEY.D], mode: 'up',   action: function() { keyRight  = false; } },
      { keys: [KEY.UP,    KEY.W], mode: 'up',   action: function() { keyFaster = false; } },
      { keys: [KEY.DOWN,  KEY.S], mode: 'up',   action: function() { keySlower = false; } }
    ],

    ready: function(images) {
      background = images[0];
      sprites    = images[1];
      reset();
      Dom.storage.fast_lap_time = Dom.storage.fast_lap_time || 180;
      // updateHud('fast_lap_time', formatTime(Util.toFloat(Dom.storage.fast_lap_time)));
    },

    afterUpdate: function(){ 
      // if collision or off-road occurs, restart the game
      var terminal = false;
      if (COLLISION_OCCURED || Math.abs(playerX) > 1.0){
        terminal = true; 
      }

      var img = capture();
      var reward = getReward();
      // console.log([playerX, speed, maxSpeed]);
      // [0, 9840, 12000]
      var data = {
        img: img,
        reward: reward,
        terminal: terminal,
        playerX: playerX,
        speed: speed,
        start_frame: START_FRAME,
      }

      // console.log(data);
      socket.emit('message', data);

      if (START_FRAME){
        START_FRAME = false;
      }

      if(terminal){
        Game.restart();
      }

    }
  };

  



})();