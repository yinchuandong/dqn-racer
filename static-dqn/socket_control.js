(function(){
  var socket = io('http://127.0.0.1:5000');
  var jStatus = Dom.get('server-status');

  socket.on('connect', function(){
    jStatus.innerHTML = '<p>connect</p>';
  });
  
  socket.on('disconnect', function(){
    jStatus.innerHTML += '<p>disconnect</p>';
  });

  socket.on('init', function(action){
    var telemetry = [];
    // the timer for collecting snapshots
    timeIntervalID = setInterval(function(){
      telemetry.push({
        collision: COLLISION_OCCURED,
        player_x: playerX,
        speed: speed,
        max_speed: maxSpeed
      });

      if(telemetry.length % 3 == 0) {
        var json = capture(false);
        json['telemetry'] = telemetry;
        socket.emit('message', json);
        telemetry = []
        // for testing
        // clearInterval(timeIntervalID);
      }
    }, 1000/30);
  });

  socket.on('message', function(action){
    keyLeft = action['keyLeft']
    keyRight = action['keyRight']
    keyFaster = action['keyFaster']
    keySlower = action['keySlower']
  });
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

  function capture(terminal) {
    function scaleImage(){
      smallCtx.drawImage(canvas, 0, 0, smallWidth, smallHeight);
      var prefix = 'data:image/png;base64,';
      return smallCanvas.toDataURL().substr(prefix.length);
    }
    var data = scaleImage();
    // send status to sever for calculating reward
    var json = {
      'img': data,
      terminal: terminal,
      start_frame: START_FRAME,
      action: [keyLeft, keyRight, keyFaster, keySlower]
    }
    if (START_FRAME){
      START_FRAME = false;
    }
    return json;
  };

  setTimeout(function(){
    Game.run(gameParams);
    socket.emit('init', '----init-----');
  }, 1000);

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
      if (COLLISION_OCCURED || pos > 1.0){
          var json = capture(true);
          json['telemetry'] = [{
            collision: COLLISION_OCCURED,
            player_x: playerX,
            speed: speed,
            max_speed: maxSpeed
          }];
          socket.emit('message', json);
          if(!COLLISION_OCCURED){
            console.log('collision');
            Game.restart();
          }
          // Game.restart();
          COLLISION_OCCURED=false;
      }
    }
  };







})();