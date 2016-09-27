(function(){
  var socket = io('http://127.0.0.1:5000');
  var jStatus = Dom.get('server-status');

  socket.on('connect', function(){
    jStatus.innerHTML = '<p>connect</p>';
  });
  
  socket.on('disconnect', function(){
    jStatus.innerHTML += '<p>disconnect</p>';
  });

  socket.on('message', function(outPlayerX, outSpeed){
    playerX = outPlayerX;
    speed = outSpeed;
  });

  Dom.get('j-btn-start').onclick = function(){
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

//=========================================================================
// THE GAME LOOP
//=========================================================================
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
      var pos = Math.abs(playerX);
      var img = capture();
      var terminal = false;
      if (COLLISION_OCCURED || pos > 1.0){
        terminal = true; 
      }
      // console.log([playerX, speed, maxSpeed]);
      // [0, 9840, 12000]
      var data = {
        img: img,
        terminal: terminal,
        start_frame: START_FRAME,
        collision: COLLISION_OCCURED,
        playerX: playerX,
        playerX_space: [-1.0, 1.0],
        speed: speed,
        speed_space: [0, maxSpeed]
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